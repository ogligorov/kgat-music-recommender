# CLAUDE.md — Project Map for AI Collaborators

## Main idea

A **paper-faithful KGAT** (Knowledge Graph Attention Network, Wang et al.,
KDD 2019) for **explainable track-level music recommendation**. The model is
trained on a heterogeneous Collaborative Knowledge Graph derived from the
[Spotify Playlists dataset](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists),
beating a track-popularity baseline by ~+24% NDCG@10 on the held-out test
split (n=2000 users, 5000-track candidate pool) and surfacing
attention-weighted explanation paths through artist and playlist hubs.

The thesis question is two-pronged:
1. **What** to recommend? — top-K tracks ranked by dot-product over learned
   KGAT embeddings.
2. **Why** these recommendations? — top attention paths
   (`user → track_A → artist → target` or
   `user → track_A → playlist → target`), with a fidelity score that masks
   the hub node and confirms the score drops.

## Project structure

```
src/
  config.py            # hyperparameters + device auto-detect (cuda → mps → cpu)
  build_graph.py       # Spotify CSV → K-core filter → ID maps → HeteroData
                       # + per-user 80/10/10 split on (user, liked, track)
  model.py             # KGAT: TransR attention, GCN aggregator, BPR + KGE losses
  train.py             # disjoint MP/sup split + alternating BPR/KGE loop
                       # with KGE warmup, AdamW, dict-format checkpoints, --resume
  evaluate.py          # sampled NDCG@K / Recall@K with per-user candidate sets
  baselines.py         # track-popularity baseline under the identical protocol
  final_eval.py        # KGAT vs baseline side-by-side, same seed/users/candidates
  explain.py           # extract per-edge attention, find top paths,
                       # hub-mask fidelity test (returns fidelity + coverage)
  pick_explain_pair.py # auto-select (user, track) where all 3 path types apply
docs/
  architecture.md      # full data + training + eval + explain pipeline
  adr/                 # design decisions
app.py                 # Streamlit demo: cached load_everything() →
                       # KGAT top-K vs popularity top-K + pyvis explanation graph
smoke_test.py          # 1-batch forward — plumbing check before long runs
```

## Schema

**4 node types**: `user`, `track`, `artist`, `playlist`.

**6 directed relations** (`EDGE_TYPES` in `src/model.py`):

| idx | src | rel | dst | source |
| --- | --- | --- | --- | --- |
| 0 | user | liked | track | one (user, track) per CSV row, deduped |
| 1 | track | rev_liked | user | reverse |
| 2 | track | in_playlist | playlist | one (track, playlist) per CSV row |
| 3 | playlist | rev_in_playlist | track | reverse |
| 4 | track | performed_by | artist | one artist per track |
| 5 | artist | rev_performed_by | track | reverse |

No `(user, owns, playlist)` relation — playlists are treated as **content
hubs**, not user property. The model learns track→playlist→track shortcuts
without conflating ownership with affinity.

## Key decisions

### Data
- **K-core ≥ 5** iterated until stable. Drops sparse users / tracks /
  playlists; reduces ~12M raw rows to a tractable graph (~14k users,
  ~381k tracks).
- **Per-user 80/10/10 split** on `(user, liked, track)` only. Pure-KG
  relations (track↔artist, track↔playlist) have no split — every edge is
  a training triple in KGE Phase II.

### Model
- **TransR attention** (paper eq 6): `pi(h, r, t) = (W_r·e_t)ᵀ tanh(W_r·e_h + e_r)`,
  with `h` = ego (PyG dst) and `t` = neighbor (PyG src).
- **Single softmax denominator across all relations** terminating at each
  destination (paper kgat_paper.py:384). Per-relation attention is only
  meaningful after the joint softmax.
- **Aggregated message = un-projected neighbor embedding** `e_t` (paper
  line 316). The TransR projection only enters the attention scoring.
- **GCN aggregator (no residual)**: `LeakyReLU(W_gc^(l) · agg)`.
  Bi-interaction was the paper's alternative; we deliberately skipped it.
- **Per-layer ordering: dropout → L2-normalize** (paper lines 289 then 292).
  L2 makes dot-product scoring scale-invariant across layers; without it,
  layer-0 (Xavier init magnitude) dominates the concat.
- **Layer-0 unnormalized in concat** (paper line 263). Final per-node embed
  has dim `(L+1) * embed_dim = 256` with `L=3, D=64`.
- **Shared `W_r` and `relation_emb` across layers and across the two phases.**
  Paper has only one `trans_W`; KGE loss directly shapes the CF attention.

### Training
- **AdamW, lr=1e-3** — 10× the paper's 1e-4 (deliberate; our graph is much
  larger and we want a tractable wall-clock). The previous 5e-3 overshot
  the BPR minimum past epoch ~10 (val NDCG@10 0.487 → 0.45 collapse).
- **Disjoint MP / supervision split** (70 / 30, seed=42). Without this,
  `LinkNeighborLoader` leaves the supervision edge in the message-passing
  graph for its own batch and BPR collapses to ~0.06 from epoch 1 — model
  trivially copies the positive track's embedding into the user node.
- **Alternating BPR + KGE phases per epoch.** All 6 relations enter KGE
  (including `liked`) so `W_r[liked]` and `relation_emb[liked]` get direct
  gradient signal on the most important relation.
- **One KGE warmup pass before the first CF epoch.** Cold-start mitigation:
  Xavier-init `W_r` / `relation_emb` give near-uniform attention, so we
  shape them once before BPR sees them. Skipped on `--resume`.
- **No `update_attentive_A` cache.** Paper recomputes attention once per
  epoch via a cached sparse `A`. We can't — `LinkNeighborLoader` rebuilds
  subgraphs per batch on a 17M-edge graph. We compute attention per-batch
  inside `forward_one_layer`. Trade-off: per-batch compute vs paper
  semantics with no staleness.
- **`edges_per_epoch = 1M`** caps per-epoch wall-clock. Loader reshuffles
  every epoch, so each cap is a fresh random subset of ~6.3M train edges.

### Evaluation
- **Sampled metrics**: per eval pass, sample `n_eval_users` users with ≥1
  held-out positive; score each against a per-user candidate set
  `(gt_u) ∪ (shared_neg_pool \ train_pos_u \ gt_u)`. The shared pool is
  `n_eval_negatives` random tracks. Cuts eval wall-clock from ~12 min to
  ~30 s while preserving rank-ordering across models.
- **Train-only message-passing graph for eval** (`make_train_only_graph`).
  Without it, val/test edges leak into the GNN's view of users/tracks.
- **Train-time eval (500 / 1000)** for early stopping with patience 3.
  **Final eval (2000 / 5000)** for thesis-quality numbers.
- **Both KGAT and the popularity baseline use the same seed, eligible
  users, and per-user candidate sets** (`final_eval.py`) so deltas are
  apples-to-apples.
- **Popularity tie-breaking jitter**: popularity scores have heavy ties
  (long tail). Without seeded uniform jitter, `torch.topk` orders by
  candidate id, deterministically advantaging/disadvantaging GT tracks.

### Explainability
- **Three path types**: `direct`, `via_artist`, `via_playlist`. Path score
  is the geometric mean of edge attentions along the path.
- **Per-edge attention extraction replays `forward_one_layer`** so we can
  split the joint softmax back per relation.
- **Fidelity test reports `(fidelity, coverage)`** — fraction of testable
  pairs whose score dropped after hub masking, AND fraction of sampled
  pairs whose top-1 explanation was multi-hop (only those are testable).

### App
- **Streamlit + pyvis**, runs on `cuda → cpu` (skips MPS — PyG
  `NeighborLoader` triggers `aten::_convert_indices_from_coo_to_csr` which
  is unimplemented on MPS).
- **`@st.cache_resource`** loads graph + model + every-user / every-track
  embedding once per process. Per-user scoring after that is a single matmul.
- **Cold-user fallback**: users with 0 training likes get global popularity
  top-K instead of KGAT (their embedding is essentially Xavier noise).
