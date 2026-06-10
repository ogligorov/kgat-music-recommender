# Music Recommender — KGAT

Paper-faithful **Knowledge Graph Attention Network** (Wang et al., KDD 2019) for
explainable track-level recommendation on the
[Spotify Playlists dataset](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists).

Heterogeneous graph: 4 node types (`user`, `track`, `artist`, `playlist`) and 6
directed relations. TransR attention + KGE Phase II loss + GCN aggregator. BPR
on `(user, liked, track)`.

Architecture: `docs/architecture.md`. Project map and key decisions: `CLAUDE.md`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch                                    # CPU / MPS
# CUDA wheels: pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
# PyG sampler backend (match your torch + CUDA build):
# pip install pyg-lib torch-sparse -f https://data.pyg.org/whl/torch-<X.Y.Z>+<cpu|cuXXX>.html
```

`src/config.py` auto-detects `cuda` → `mps` → `cpu`.

## Data

Place the raw CSV at `data/spotify_dataset.csv` (~1.1 GB, gitignored). Processed
artifacts (`data/processed/graph.pt`, `id_mappings.json`, `kgat_best.pt`) are
also gitignored — rebuild them on each machine, or copy the processed files.

## Workflow

Run from the repo root.

```bash
# 1. Build the heterogeneous graph from the CSV (5-core filter, 80/10/10 split).
python -m src.build_graph

# 2. Smoke test (forward pass on a tiny subgraph).
python smoke_test.py

# 3. Train. AdamW + BPR + KGE alternating phases. Writes data/processed/kgat_best.pt.
python -m src.train               # fresh run
python -m src.train --resume      # continue from last best checkpoint

# 4. Final eval — KGAT vs popularity baseline, side-by-side, same protocol.
python -m src.final_eval --split test --n-users 2000 --n-negatives 5000

# 5. Pick a (user, track) pair worth explaining (top-K hit + artist + playlist hub).
python -m src.pick_explain_pair

# 6. Explain that pair: top attention paths + fidelity score.
python -m src.explain --user N --track M --fidelity-samples 100

# 7. Demo UI (KGAT vs popularity top-K + interactive explanation graph).
streamlit run app.py
```

Hyperparameters: `src/config.py`.

## Metrics

Sampled-metrics protocol (per-user candidate sets: held-out positives ∪ shared
negative pool, train positives masked). Both KGAT and the popularity baseline
use the same eligible users, same per-user candidates, same seed.

| Split | Sample (users / negs) | Metric | KGAT | Popularity | Lift |
| --- | --- | --- | ---: | ---: | ---: |
| test | 500 / 1 000 | NDCG@10 | 0.6418 | 0.4469 | +43.6% |
| test | 500 / 1 000 | Recall@10 | — | — | ~+85% |
| test | 2 000 / 5 000 | NDCG@10 | 0.3808 | 0.3079 | +23.7% |
| val | 500 / 1 000 | NDCG@10 (best, epoch 30) | 0.7002 | — | — |

Numbers grow as the candidate pool shrinks (fewer distractors per user). The
2 000 / 5 000 setting is the thesis-quality reference; 500 / 1 000 matches the
training-time eval used for early stopping.

## Layout

```
src/
  config.py            # hyperparameters + device detection
  build_graph.py       # CSV → HeteroData + 80/10/10 split → graph.pt
  model.py             # KGAT (TransR attention, GCN aggregator, KGE loss)
  train.py             # disjoint MP/sup split + BPR + KGE alternating loop
  evaluate.py          # sampled NDCG@K / Recall@K
  baselines.py         # track-popularity baseline (same protocol)
  final_eval.py        # KGAT vs baseline side-by-side
  explain.py           # top-K attention paths + fidelity test
  pick_explain_pair.py # auto-pick a (user, track) worth explaining
docs/
  architecture.md
  adr/                 # design decisions
app.py                 # Streamlit demo (KGAT vs popularity + path viz)
smoke_test.py          # plumbing check
```
