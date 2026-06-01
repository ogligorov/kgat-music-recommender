# ADR-003: V2 Track-Level Upgrade — Spotify Playlists

**Status**: Proposed
**Date**: 2026-05-31

---

## Context

V1 of the recommender (Last.fm-2k, artist-level) was constrained by the dataset shipping only `user → artist` play counts. The thesis goal is **track-level recommendation with explainability**, which requires a dataset with explicit user → track signals.

Play counts are also a weak preference signal — a user can play a song accidentally, on shuffle, or in the background. We want a stronger, more explicit indication that a user actually likes a track.

The **[Spotify Playlists dataset](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists)** offers exactly that: curating a track into a playlist is an explicit user action. It also gives us a **new structural signal** — playlist co-occurrence — which yields a novel and intuitive explanation type ("users who curated Track A also curated Track B").

---

## Decision

V2 will use the Spotify Playlists dataset with a four-node knowledge graph that includes `playlist` as a first-class node type.

V1 is **replaced in place**. The current `main` is tagged `v1` before v2 work begins so that the v1 implementation remains reachable for thesis comparison.

---

## Dataset

**Source**: https://www.kaggle.com/datasets/andrewmvd/spotify-playlists

**Schema** (single CSV, ~12M rows): `user_id`, `artistname`, `trackname`, `playlistname`.

**Approximate raw scale**: ~15K users, ~287K artists, ~2M unique tracks, ~232K playlists.

**Preference signal**: a track appearing in any playlist owned by a user is interpreted as `liked = 1`. Implicit but stronger than play-count signals — the user actively curated it.

**Joining**: tracks are deduplicated by `(lower(artistname).strip(), lower(trackname).strip())`. Some duplicates from unicode/typo variations will leak through; acceptable for the v2 baseline.

---

## KG Design

| Node Type | Source | Approx Count (post-filter) | Init |
|---|---|---|---|
| `user` | `user_id` column | ~10–15K | learnable embedding |
| `track` | deduped `(artistname, trackname)` | ~150–300K | learnable embedding |
| `artist` | deduped `artistname` | ~30–50K | learnable embedding |
| `playlist` | deduped `(user_id, playlistname)` | ~30–50K | learnable embedding |

| Forward Relation | Reverse | Notes |
|---|---|---|
| `(user, liked, track)` | `rev_liked` | Recommendation target. Train/val/test masks (80/10/10 per user). Deduplicated across the user's playlists. |
| `(track, in_playlist, playlist)` | `rev_in_playlist` | Many-to-many. Carries the **playlist co-occurrence signal**. |
| `(track, performed_by, artist)` | `rev_performed_by` | One artist per track (dataset constraint). |

**Total: 6 edge types** (3 forward + 3 reverse).

### Why no `(user, owns, playlist)` edge

A direct `owns` edge introduces a leakage path during training. With L≥2 layers and the `rev_in_playlist` edge present, the propagation chain `user → owns → playlist → rev_in_playlist → track` lets a user's hidden test track flow back into the user embedding through message-passing — even when the `liked` edge to that track is masked. The `track ↔ playlist` edges already carry the playlist co-occurrence signal we want; the `owns` edge adds nothing the model can't recover and contaminates the eval. Drop it.

### K-core filter (applied iteratively until stable)

- Drop users with <5 liked tracks.
- Drop tracks with <5 distinct likers.
- Drop playlists with <5 tracks (filters single-song "playlists" and noise).

---

## Architecture Changes from V1

### Model (`src/model.py`)

```python
class KGAT(nn.Module):
    def __init__(self, n_users, n_tracks, n_artists, n_playlists,
                 embed_dim=64, n_layers=3, n_heads=4, dropout=0.1):
        self.user_emb     = nn.Embedding(n_users, embed_dim)
        self.track_emb    = nn.Embedding(n_tracks, embed_dim)
        self.artist_emb   = nn.Embedding(n_artists, embed_dim)
        self.playlist_emb = nn.Embedding(n_playlists, embed_dim)

        edge_types = [
            ('user',     'liked',          'track'),
            ('track',    'rev_liked',      'user'),
            ('track',    'in_playlist',    'playlist'),
            ('playlist', 'rev_in_playlist','track'),
            ('track',    'performed_by',   'artist'),
            ('artist',   'rev_performed_by','track'),
        ]
```

**Layer depth**: `L = 3`. This matches the empirical sweet spot reported in the original KGAT paper (Wang et al., KDD 2019, §4.4 / Table 3): L=1 is competitive, L=2 improves, L=3 is consistently best across all three of their datasets, and L≥4 starts to over-smooth. V2 sticks with L=3 rather than re-running a depth ablation that the literature already settles.

**Scoring**: dot product of user and track embeddings. BPR loss unchanged from v1.

### Scale & Training

- ~5–10M edges after filtering. Start with full-graph propagation (matches v1 and the original KGAT paper, which trains full-graph on Amazon-Book at comparable scale). Fall back to `NeighborLoader` only if MPS OOM hits during training.
- **Evaluation**: cannot materialize a 15K × 300K score matrix; users scored in batches of 128, top-K extracted per batch. This is independent of the training memory question.
- **Negative sampling**: random tracks rejected if already in user's positive set.

---

## Cold-Start Strategy

V2 handles **user-side cold-start only**. Track cold-start is out of scope.

| Case | Treatment |
|---|---|
| **Warm user** (≥5 likes, in training graph) | Standard KGAT scoring. |
| **Cold user** (filtered by K-core, never seen) | **Track-popularity fallback** — top-K tracks ranked by unique-liker count. |
| **Sparse user** (1–4 likes) | Same fallback — they were filtered out of the training graph by K-core. |

The boundary is the K-core threshold (≥5 likes); no separate tunable cold-user threshold. The popularity baseline already exists in v1's `src/baselines.py` and gets reused at inference.

---

## Eval Split

**Per-user random 80/10/10** of the user's `liked` edges. Reuses v1's `split_interactions()` unchanged. The dataset has no timestamps, so a temporal split is not possible. Per-user random keeps every warm user represented in train/val/test.

---

## Explanation Path Types (V2)

| Path | Meaning |
|---|---|
| **Direct** | `User → Track` if a `liked` edge exists |
| **Via shared artist** | `User → Track_A → Artist → Track_B` ("you liked another song by this artist") |
| **Via shared playlist** | `User → Track_A → Playlist → Track_B` ("users who curated Track_A also curated Track_B") |

Fidelity test: same as v1 — zero out the mid-node embedding, re-run forward, check if the recommendation score drops.

---

## Trade-offs

**Gains:**
- Cleaner preference signal (curation > play count).
- New structural signal (playlist co-occurrence) yielding a novel and intuitive explanation type.
- Single-CSV ingest; no metadata-joining pipeline.

**Losses / accepted limitations:**
- No genre signal in the baseline.
- No timestamps → splits are random per-user, not temporal.
- ~15K users is moderate (~7× larger than v1's 1.9K, within the typical range for academic music-rec benchmarks).
- String-based dedup leaves some duplicate tracks. Could be improved later with fuzzy matching or Spotify ID resolution.

### Why we don't merge the HetRec or Last.fm_data datasets

The repo also ships `data/raw/*.dat` (HetRec LastFM-2k, v1's source) and `data/Last.fm_data.csv` (a separate single-month scrobble dump). Neither is merged into v2:

- **`Last.fm_data.csv`** has only **11 unique users** (single-month scrobble dump). 76K (artist, track) pairs but mean 1.7 listeners per track — no collaborative signal possible from 11 users. Reject as a user/edge source. Album column would face the same problem.
- **HetRec users** are a disjoint cohort from Spotify users — can't merge user IDs.
- **HetRec tags joined onto Spotify artists by lowercased artist name** is the only structurally interesting option. Measured overlap on the full Spotify CSV: HetRec has 17,615 artists (12,127 with ≥1 tag); Spotify has 282,490 unique artists; the tagged-HetRec ∩ Spotify intersection is **9,998 artists = 3.5% of Spotify artists**. Post-K-core the overlap is likely higher (popular artists dominate both datasets), but at 3.5% raw the tag subgraph would be sparse and disconnected for ~96% of artists, biasing tag-mediated propagation toward a small popular subset. Adds an extra node type and two edge types for a noisy partial signal, and dilutes the thesis framing around playlist co-occurrence being **the** new structural signal.

Tag enrichment via HetRec is a viable v3 extension, not a v2 baseline.

---

## Implementation Order

1. Tag current `main` as `v1`.
2. `src/config.py` — drop `dataset_url`; add CSV path, k-core thresholds, `eval_user_batch_size`, `n_layers = 3`.
3. `src/build_graph.py` — full rewrite for the new schema. Reuses existing `split_interactions()` unchanged.
4. `src/model.py` — new `__init__` signature, 6 edge types, L=3.
5. `smoke_test.py` — verify forward pass on the new graph; flag if memory pressure suggests `NeighborLoader` is needed.
6. `src/train.py` — full-graph BPR loop over `(user, track)` pairs (fallback to `NeighborLoader` if OOM).
7. `src/evaluate.py` — per-user batched scoring; reuses NDCG/Recall functions unchanged. Cold-user routing to popularity fallback.
8. `src/baselines.py` — track popularity = unique-user count per track.
9. `src/explain.py` — new path types (via artist, via playlist); reuse fidelity-test scaffolding.
10. `app.py` — switch artist picker to track picker; render playlist nodes in explanation graph.
11. `docs/architecture.md` — update node/edge tables, sequence diagrams, §6.4 V2 description.

---

## Verification

```bash
# Manual: download spotify_dataset.csv to data/ from Kaggle
python -m src.build_graph           # check post-filter counts and split sums
python smoke_test.py                # KGAT forward on graph_v2; flags memory pressure
python -m src.train                 # loss decreases; NDCG@10 beats popularity by epoch ~20
python -m src.evaluate              # final NDCG@10/20, Recall@10/20
python -m src.explain --user 0 --track 5
                                    # at least one playlist-typed path appears;
                                    # fidelity ratio prints
streamlit run app.py                # demo loads; explanation graph shows
                                    # user/track/artist/playlist nodes
```

**Numerical bars** (treat as smoke checks, not targets):
- KGAT NDCG@10 ≥ track-popularity baseline + 0.02 absolute.
- Fidelity ≥60% (mid-node masking causes score drop in ≥60% of explained pairs).
- Training wall-clock <30 min on MPS for 100 epochs.
