# Music Recommender (KGAT, v2)

Track-level music recommendation using a Knowledge Graph Attention Network on the
Spotify Playlists dataset. 4 node types (user, track, artist, playlist) and 6 edge
types. Design rationale: `docs/adr/003-v2-spotify-playlists.md`. Architecture:
`docs/architecture.md`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# CPU / MPS (Apple Silicon)
pip install torch
# CUDA 12.8 (e.g. RTX 40/50 series). Use the nightly index if your GPU is
# newer than the latest stable wheel:
# pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install torch_geometric pandas numpy
```

`src/config.py` auto-detects `cuda` → `mps` → `cpu`.

## Data

The raw CSV (`data/spotify_dataset.csv`, ~1.1 GB) is gitignored. Download it once
into `data/`. The processed graph (`data/processed/graph.pt`, `id_mappings.json`)
is built from the CSV — either rebuild on the training machine or copy the
processed files across.

## Pipeline

Run from the repo root.

### 1. Build the graph

```bash
python -m src.build_graph
```

Reads `data/spotify_dataset.csv`, applies a 5-core filter (configurable in
`src/config.py`), builds the heterogeneous graph, and writes:

- `data/processed/graph.pt` — PyG `HeteroData` with train/val/test masks on
  `(user, liked, track)` edges (per-user 80/10/10 split).
- `data/processed/id_mappings.json` — global ID ↔ index mappings.

### 2. Smoke test

```bash
python smoke_test.py
```

Loads the graph, instantiates the model, runs a sub-graph forward, prints
shapes. Use this to verify the install before launching a long training run.

### 3. Train

```bash
python -m src.train
```

BPR loss with `LinkNeighborLoader` over `(user, liked, track)` edges. Evaluates
every 5 epochs on the val split (sampled metrics) with patience-3 early
stopping. Writes the best checkpoint to `data/processed/kgat_best.pt`.

Per-batch wall-clock baselines:

| Device | ms/batch | Epoch (~954 batches) |
| --- | --- | --- |
| Apple M-series (MPS) | ~900 ms | ~15 min |
| RTX 50-series (CUDA) | expected ~100–200 ms | ~2–3 min |

Hyperparameters live in `src/config.py`.

### 4. Evaluate

```bash
python -m src.evaluate
```

Sampled-metrics NDCG@10/20 and Recall@10/20 against the val split. With no
trained checkpoint loaded, the run only verifies plumbing — numbers are
meaningless. Edit the script to load `kgat_best.pt` for real metrics.

### 5. Popularity baseline

```bash
python -m src.baselines
```

Track-popularity baseline under the identical sampled-metrics protocol as
`evaluate.py`, so the numbers are directly comparable to KGAT.

## Cross-machine workflow

Code: push/pull via git.

Data: either ship `data/processed/graph.pt` + `id_mappings.json` between
machines (~few hundred MB), or copy the raw CSV and re-run `src.build_graph`.
The training output (`kgat_best.pt`) is not committed; copy it manually if you
need it elsewhere.

## Layout

```
src/
  config.py          # all hyperparameters + device detection
  build_graph.py     # CSV → HeteroData → graph.pt
  model.py           # KGAT (HeteroConv + GATConv, per-edge-type)
  train.py           # BPR loop with LinkNeighborLoader
  evaluate.py        # sampled NDCG@K / Recall@K
  baselines.py       # track-popularity baseline
docs/
  architecture.md
  adr/               # design decisions
data/
  spotify_dataset.csv      # raw, gitignored
  processed/               # graph.pt, id_mappings.json, kgat_best.pt
```
