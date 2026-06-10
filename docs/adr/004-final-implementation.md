# ADR-004: Final Implementation Snapshot

**Status**: Implemented
**Date**: 2026-06-01
**Supersedes**: This ADR records the realized architecture as the project ships
for thesis defense. ADR-001 set the original course-project scope (Last.fm-2k,
artist-level), ADR-002 is **deprecated**, ADR-003 (Spotify Playlists, track-level)
defined the v2 design intent. The implementation has evolved beyond ADR-003 in
several places — captured below — all driven by paper-faithfulness or empirical
necessity.

---

## Context

The project delivers a **paper-faithful KGAT** (Wang et al., KDD 2019) for
**explainable track-level music recommendation** on the Spotify Playlists
dataset. Two thesis questions are answered for every user:
1. **What** to recommend? — top-K tracks ranked by dot product over learned
   KGAT embeddings.
2. **Why** these recommendations? — attention-weighted paths through the
   knowledge graph (`direct`, `via_artist`, `via_playlist`), with a
   hub-masking fidelity test for faithfulness.

This ADR locks in the final architecture and records every deliberate
deviation from ADR-003 with its justification.

---

## Decision Record

### 1. Knowledge graph schema — as in ADR-003

| Node type | Source | Approx count (post K-core) |
|---|---|---|
| `user` | `user_id` | ~14K |
| `track` | deduped `(artistname, trackname)` lowercased | ~381K |
| `artist` | deduped `artistname` lowercased | ~62K |
| `playlist` | deduped `(user_id, playlistname)` lowercased | ~73K |

Six directed relations: `(user, liked, track)`, `(track, in_playlist, playlist)`,
`(track, performed_by, artist)`, plus the three reverse edges.
**No `(user, owns, playlist)` edge** (leak path with L≥2 — ADR-003 §"Why no owns").

K-core ≥5 iterated until stable. Per-user 80/10/10 split on `(user, liked, track)`
only, seeded with `np.random.default_rng(42)`. Pure-KG relations (track↔artist,
track↔playlist) have no split — every edge is a training triple in KGE Phase II.

### 2. Model — paper-faithful KGAT (extends ADR-003)

ADR-003 said "BPR unchanged from v1". Implementation goes further to match the
paper:

- **TransR attention** (paper eq 6): `π(h, r, t) = (W_r·e_t)ᵀ · tanh(W_r·e_h + e_r)`,
  with `h` = ego (PyG dst) and `t` = neighbor (PyG src).
- **Single softmax denominator across all relations** terminating at each ego
  (paper `kgat_paper.py:384`). Per-relation attention is only meaningful after
  the joint softmax.
- **Aggregated message = un-projected neighbor embedding `e_t`** (paper line 316).
  The TransR projection only enters the attention scoring.
- **GCN aggregator (no residual)**: `LeakyReLU(W_gc^(l) · scatter_sum)`. The paper's
  alternative Bi-interaction aggregator was not implemented (orthogonal extension).
- **Per-layer ordering: dropout → L2-normalize** (paper line 289 then 292).
- **Layer-0 unnormalized in concat** (paper line 263); layers 1..L are normalized.
  Final per-node embedding has dim `(L+1) × embed_dim = 256` (`L=3`, `D=64`).
- **Shared `W_r ∈ ℝ^{R × D × K}` and `relation_emb ∈ ℝ^{R × K}` across layers and
  across the two training phases.** Paper has only one `trans_W`. KGE gradient
  directly shapes the CF attention.
- **KGE Phase II loss** added (`||W_r h + r − W_r t||²`, softplus margin, L2 reg).
  All 6 relations enter KGE so `W_r[liked]` and `relation_emb[liked]` get direct
  gradient on the most important relation. ADR-003 omitted this; we judged it
  load-bearing for paper-faithfulness.

### 3. Training (extends ADR-003 considerably)

ADR-003 sketched a single full-graph BPR loop. The implementation diverges as
follows — every divergence is documented in CLAUDE.md:

- **`LinkNeighborLoader` not full-graph.** ADR-003 said "full-graph, fall back
  to NeighborLoader if OOM". 17M edges OOM full-graph on MPS, so NeighborLoader
  is unconditional. `num_neighbors=[3, 3, 3]`, triplet negative sampling.
- **Disjoint MP / supervision split** (70 / 30, seed 42, `disjoint_mp_sup_split`).
  Without this, `LinkNeighborLoader` leaves the supervision edge in the
  message-passing graph for its own batch and BPR collapses to ~0.06 from epoch 1
  — the model trivially copies the positive track's embedding into the user.
  KGE phase uses the full `train_mask` (no GNN forward, no leak risk).
- **Alternating BPR + KGE phases per epoch** (CF over sampled subgraphs, KGE
  over raw embedding tables, all 6 relations, batch_size_kg=2048, uniform
  tail-corruption within type).
- **One KGE warmup pass before the first CF epoch.** Cold-start mitigation:
  Xavier-init `W_r` / `relation_emb` give near-uniform attention; we shape them
  once before BPR sees them. Skipped on `--resume`.
- **AdamW(lr=1e-3, weight_decay=1e-5).** 10× the paper's 1e-4 — deliberate
  given our graph is much larger and we want a tractable wall-clock. A previous
  5e-3 overshot the BPR minimum past epoch ~10.
- **`edges_per_epoch = 1_000_000`** caps per-epoch wall-clock; loader reshuffles
  every epoch, so each cap is a fresh random subset of ~6.3M train edges.
- **No `update_attentive_A` cache.** Paper recomputes attention once per epoch
  via a cached sparse `A`. Impossible here — `LinkNeighborLoader` rebuilds
  subgraphs per batch on a 17M-edge graph. Attention is computed per-batch
  inside `forward_one_layer`. Trade: per-batch compute vs. paper semantics with
  no staleness.
- **Early stopping on NDCG@10, patience=3** (eval every 5 epochs).
  Dict-format checkpoint (`model + optimizer + best_ndcg + epoch`); `--resume`
  loads all four. Legacy raw-state-dict checkpoints fall back to weights-only
  with an automatic backup.
- **Loss accumulated as 0-dim on-device tensor.** `.item()` only at heartbeat
  (every 50 batches) and end-of-epoch — avoids the MPS sync that drains the
  kernel queue.

### 4. Evaluation — sampled per-user candidate protocol

Full-corpus eval is `~5.3B` scores per pass — infeasible for the per-epoch eval
that drives early stopping.

- **Per-user candidate set**: `(user's eval positives) ∪ (shared neg_pool \
  train_pos_u \ eval_gt_u)`. Each user is ranked only against their own
  candidates.
- **Train-only message-passing graph for eval** (`make_train_only_graph`).
  Without it, val/test edges leak into the GNN's view of users/tracks.
- **Train-time eval (500 / 1000)** for early stopping. **Final eval
  (2000 / 5000)** for thesis-quality numbers.
- **Determinism**: seeds **both** `numpy` (user / neg-pool sampling) and
  `torch` (NeighborLoader uses the torch global RNG). Without both, two runs
  with the same `seed` produced different sub-graphs and different NDCG.
- **NeighborLoader subgraph forwards** for `compute_final_embeddings`, batched
  scoring at `eval_user_batch_size=128`.

### 5. Baseline parity

Track-popularity baseline (`src/baselines.py`) uses the **same seed, same
eligible users, same per-user candidate sets** as KGAT. Popularity tie-breaking
uses seeded uniform jitter (~1e-6) so `torch.topk` does not deterministically
advantage GT tracks by `track_id`. `final_eval.py` runs both side-by-side and
prints the relative lift.

### 6. Explainability — three path types + fidelity & coverage

- **Per-edge attention extraction** replays `forward_one_layer` per layer and
  per relation, concatenates logits per ego, runs the **single** softmax, then
  splits back per relation — so the per-relation attention used for path
  scoring is post the joint softmax, faithful to the paper.
- **Three path types**: `direct` (1-hop liked edge), `via_artist`
  (`user → track_A → artist → target_track` where `track_A` shares the target's
  artist), `via_playlist` (analogous through a shared playlist). Cap
  `max_user_tracks=50` per user; cap of 5 shared playlists per `(track_a, target)`.
- **Path score = geometric mean** of edge attentions along the path
  (`(a1·a2·a3)^(1/3)`).
- **Fidelity test** zeros the hub embedding (`path[2]`) and replays the forward
  layers preserving the same `dropout → norm → concat` chain as `forward()`
  (with `model.eval()` making dropout a no-op). Returns
  **`(fidelity, coverage)`** — fidelity = fraction of testable samples whose
  score dropped after masking; coverage = fraction of sampled pairs whose top-1
  explanation was multi-hop (only those are testable). ADR-003 specified
  fidelity alone; reporting both lets the thesis distinguish "masking does
  change the score" from "we couldn't even mask".
- **`pick_explain_pair.py`** auto-selects a `(user, track)` where all three
  path types apply, so the demo lands on a pair where the explanations
  actually compete.

### 7. Demo (Streamlit + pyvis)

- **`@st.cache_resource load_everything()`** loads graph + model + every-user /
  every-track embeddings once per process. Per-user scoring after that is a
  single matmul `user_emb[uid] @ track_emb.T` with train positives masked.
- **Device: `cuda → cpu`, MPS skipped.** PyG `NeighborLoader` triggers
  `aten::_convert_indices_from_coo_to_csr` which is unimplemented on MPS, so
  `app.py` and `pick_explain_pair.py` use the CUDA-or-CPU guard.
- **Two-column UI**: KGAT recs (left) vs. popularity recs (right), both
  applying the same `train_pos` mask.
- **Cold-user fallback**: users with 0 train likes get global popularity
  top-K instead of KGAT (their embedding is essentially Xavier noise).
- **Pyvis explanation graph** with edge weight = attention; legend colors
  user=green, track=blue, artist=orange, playlist=purple.

---

## Drift from ADR-003 (with justification)

| ADR-003 | Realised | Why |
|---|---|---|
| BPR loss only | BPR + KGE Phase II alternating | Paper-faithfulness; `W_r[liked]` and `relation_emb[liked]` need direct gradient signal. |
| Full-graph forward, fall back to NeighborLoader if OOM | `LinkNeighborLoader` unconditional | 17M edges OOM full-graph on MPS. |
| Fidelity = score-drop fraction | `(fidelity, coverage)` tuple | Distinguishes "score changed" from "untestable" — more honest. |
| One artist per track guaranteed | Same, but enforced via dedup at edge construction | `drop_duplicates(subset=["track_key"])` in `build_edge_indices`. |
| Tags via HetRec considered as v3 extension | Not implemented | Out of thesis scope; kept the framing focused on playlist co-occurrence as **the** new structural signal. |
| Bi-interaction aggregator considered | GCN aggregator only | Orthogonal extension; the paper reports comparable results. |

---

## Empirical results

Both KGAT and the popularity baseline scored under the **same protocol** (same
eligible users, same per-user candidate sets, same seed=42).

| Split | Sample (users / negs) | Metric | KGAT | Popularity | Lift |
| --- | --- | --- | ---: | ---: | ---: |
| test | 500 / 1 000 | NDCG@10 | 0.6418 | 0.4469 | +43.6% |
| test | 2 000 / 5 000 | NDCG@10 | 0.3808 | 0.3079 | +23.7% |
| val | 500 / 1 000 | NDCG@10 (best, epoch 30) | 0.7002 | — | — |

Numbers grow as the candidate pool shrinks (fewer distractors per user). The
2 000 / 5 000 setting is the thesis-quality reference; 500 / 1 000 matches the
training-time eval used for early stopping.

---

## Risks & accepted limitations

| Risk / limitation | Treatment |
|---|---|
| Per-batch attention compute (no `update_attentive_A` cache) | Accepted as the cost of `LinkNeighborLoader`; no staleness. |
| Random per-user split (dataset has no timestamps) | Documented as the best available proxy; temporal split impossible. |
| String-based track dedup leaves some duplicates | Accepted; could be improved with fuzzy matching or Spotify ID resolution. |
| MPS PyG limitation (`aten::_convert_indices_from_coo_to_csr`) | `app.py` and `pick_explain_pair.py` skip MPS; only model + train_data on device, NeighborLoader fed CPU tensors. |
| Cold users (0 train likes) | Fallback to popularity in `app.py`; filtered out of training graph by K-core. |

---

## Verification (as shipped)

```bash
# 1. Manually download spotify_dataset.csv to data/ from Kaggle.
python -m src.build_graph                             # CKG → graph.pt + id_mappings.json
python smoke_test.py                                  # plumbing check
python -m src.train                                   # KGAT → kgat_best.pt
python -m src.train --resume                          # continue from last best
python -m src.final_eval --split test                 # 2000 / 5000 thesis-quality
python -m src.pick_explain_pair                       # auto-pick (user, track)
python -m src.explain --user N --track M --fidelity-samples 100
streamlit run app.py                                  # demo
```

---

## Status: implemented and thesis-ready

Every load-bearing deliverable is in place:
data pipeline (`src/build_graph.py`),
model (`src/model.py`),
training loop (`src/train.py`),
sampled evaluation (`src/evaluate.py`),
popularity baseline + side-by-side eval (`src/baselines.py`, `src/final_eval.py`),
explanation paths + fidelity test (`src/explain.py`, `src/pick_explain_pair.py`),
interactive demo (`app.py`),
documentation (`README.md`, `CLAUDE.md`, `docs/architecture.md`, ADRs 001/003/004).

## References

- Wang et al., *KGAT: Knowledge Graph Attention Network for Recommendation*, KDD 2019.
- Reference implementation: https://github.com/xiangwang1223/knowledge_graph_attention_network
- Spotify Playlists dataset: https://www.kaggle.com/datasets/andrewmvd/spotify-playlists
- ADR-001 (course-project scope), ADR-003 (v2 design intent).
