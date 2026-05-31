# Architecture — KGAT Music Recommender

**Project**: Explainable music recommendation using Knowledge Graph Attention Networks (KGAT) over a heterogeneous Collaborative Knowledge Graph (CKG) built from Last.fm-2k + MusicBrainz era enrichment.

**Status**: V1 (artist-level) implemented. V2 (track-level with audio features) is specified in [ADR-002](adr/002-v2-track-level-upgrade.md).

---

## 1. High-Level Overview

The system answers two questions for each user:
1. **What** to recommend? — Top-K artists ranked by dot product over learned KGAT embeddings.
2. **Why** these recommendations? — Top-K attention-weighted paths through the knowledge graph (e.g., `User → Artist_A → Tag → Artist_B`).

It is split into five offline stages (data → graph → train → evaluate) and one online stage (Streamlit demo for exploration and explanation).

### 1.1 Component Map

```mermaid
flowchart LR
    subgraph "Data Layer"
        A1[Last.fm-2k<br/>HetRec 2011]
        A2[MusicBrainz<br/>artist eras]
    end

    subgraph "Pipeline (offline)"
        B1[download.py]
        B2[build_graph.py]
        B3[train.py]
        B4[evaluate.py]
        B5[explain.py]
    end

    subgraph "Artifacts (data/processed)"
        C1[(ckg_heterodata.pt)]
        C2[(id_mappings.json)]
        C3[(kgat_best.pt)]
    end

    subgraph "Model"
        D1[src/model.py<br/>KGAT]
        D2[src/baselines.py<br/>Popularity]
    end

    subgraph "UI (online)"
        E1[app.py<br/>Streamlit]
        E2[pyvis<br/>graph viz]
    end

    A1 --> B1 --> B2
    A2 -.MusicBrainz API.-> B2
    B2 --> C1
    B2 --> C2
    C1 --> B3 --> C3
    C1 --> B4
    C3 --> B4
    C1 --> B5
    C3 --> B5
    D1 --> B3
    D1 --> B4
    D1 --> B5
    C1 --> E1
    C2 --> E1
    C3 --> E1
    D1 --> E1
    D2 --> E1
    E1 --> E2
```

### 1.2 Repository Layout

| Path | Purpose |
|---|---|
| [src/config.py](../src/config.py) | Single dataclass with paths + hyperparameters. MPS auto-selected if available. |
| [src/download.py](../src/download.py) | Fetch and unzip Last.fm-2k into `data/raw/`. |
| [src/build_graph.py](../src/build_graph.py) | Build the `HeteroData` CKG, query MusicBrainz, write artifacts. |
| [src/model.py](../src/model.py) | `KGAT` (`HeteroConv` of `GATConv` per relation) + BPR loss. |
| [src/train.py](../src/train.py) | Full-graph training loop with early stopping on NDCG@10. |
| [src/evaluate.py](../src/evaluate.py) | NDCG@K, Recall@K. |
| [src/explain.py](../src/explain.py) | Attention extraction, path search, leave-one-out fidelity test. |
| [src/baselines.py](../src/baselines.py) | Popularity baseline. |
| [app.py](../app.py) | Streamlit demo. |
| [smoke_test.py](../smoke_test.py) | Verifies MPS + PyG + GATConv + NeighborLoader work end-to-end. |

---

## 2. Knowledge Graph Schema

The CKG is a [PyG `HeteroData`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.HeteroData.html) instance with four node types and three forward relations (each duplicated as a `rev_` edge so message passing is bidirectional).

### 2.1 Node Types

| Type | Source | Count (Last.fm-2k) |
|---|---|---|
| `user` | `user_artists.dat` | ~1,892 |
| `artist` | `artists.dat` | ~17,632 |
| `tag` | `user_taggedartists.dat` (deduped) | ~11,946 |
| `era` | `DECADE_BUCKETS` ⊕ MusicBrainz | 8 (`1950s`–`2010s` + `unknown`) |

### 2.2 Edge Types

| Forward | Reverse | Carries |
|---|---|---|
| `(user, listens_to, artist)` | `(artist, rev_listens_to, user)` | `edge_attr = log1p(playcount)`; train/val/test masks (80/10/10 per user) |
| `(artist, tagged_with, tag)` | `(tag, rev_tagged_with, artist)` | — |
| `(artist, active_in_era, era)` | `(era, rev_active_in_era, artist)` | — |

```mermaid
graph LR
    U((user)) -->|listens_to| A((artist))
    A -->|tagged_with| T((tag))
    A -->|active_in_era| E((era))
    A -.rev_listens_to.-> U
    T -.rev_tagged_with.-> A
    E -.rev_active_in_era.-> A
```

### 2.3 ID Mappings

[build_graph.py](../src/build_graph.py) produces `id_mappings.json` containing:
- `user_to_idx`, `artist_to_idx`, `tag_to_idx`, `era_to_idx` — original ID → graph index
- Inverses (`idx_to_*`) for human-readable rendering
- `artist_id_to_name`, `tag_id_to_name` — lookup tables for the Streamlit UI

---

## 3. Model Architecture

[`KGAT`](../src/model.py) is a heterogeneous GNN with relation-aware attention.

### 3.1 Components

1. **Per-type learnable embeddings** (`nn.Embedding`): `user`, `artist`, `tag`, `era`, all of dim `embed_dim=64`.
2. **N stacked `HeteroConv` layers** (`n_layers=2`). Each wraps six `GATConv` modules — one per (forward + reverse) edge type. Aggregation across relations is `sum`.
3. **Layer aggregation**: the final embedding is the sum of the initial embeddings + every layer's output (a residual-like accumulator). This lets the model retain shallow signals while still propagating multi-hop info.
4. **Scoring**: dot product of user and artist embeddings. Trained with **BPR** (Bayesian Personalized Ranking) loss against per-user negative samples.

### 3.2 Forward Pass

```mermaid
flowchart TD
    Init[Initial embeddings<br/>user / artist / tag / era] --> L1
    Init --> Acc((+))

    subgraph "Layer 1"
      L1["HeteroConv:<br/>GATConv per relation"]
      L1 --> Relu1[ReLU]
    end
    Relu1 --> Acc
    Relu1 --> L2

    subgraph "Layer 2"
      L2["HeteroConv:<br/>GATConv per relation"]
      L2 --> Relu2[ReLU]
    end
    Relu2 --> Acc

    Acc --> Out[Final embedding dict<br/>x_dict per node type]
    Out --> Score["score(u,a) = u · a"]
```

### 3.3 Hyperparameters

Defined in [src/config.py](../src/config.py).

| Name | Default | Notes |
|---|---|---|
| `embed_dim` | 64 | Same dim across all node types (required for `HeteroConv` `sum` aggregation). |
| `n_layers` | 2 | 2-hop receptive field. |
| `n_heads` | 4 | GAT attention heads (averaged via `concat=False`). |
| `dropout` | 0.1 | Applied inside `GATConv`. |
| `lr` | 5e-3 | Adam. |
| `weight_decay` | 1e-5 | L2 regularization. |
| `n_epochs` | 100 | With early stopping (patience=3 evals on NDCG@10). |
| `top_k` | `[10, 20]` | NDCG@K and Recall@K evaluated at these cutoffs. |
| `device` | `mps` if available else `cpu` | Apple Silicon path. |

---

## 4. Pipeline Flow (Offline)

End-to-end command sequence to take an empty checkout to a working demo:

```bash
python -m src.download        # 1. Fetch Last.fm-2k into data/raw/
python -m src.build_graph     # 2. Build CKG → ckg_heterodata.pt + id_mappings.json
python -m src.train           # 3. Train KGAT → kgat_best.pt
python -m src.evaluate        # 4. (optional) Standalone metrics print
python -m src.explain --user 0 --artist 5  # 5. Attention paths + fidelity
streamlit run app.py          # 6. Launch interactive demo
```

### 4.1 Sequence — Graph Construction

```mermaid
sequenceDiagram
    autonumber
    participant U as User (CLI)
    participant BG as build_graph.py
    participant FS as data/raw
    participant MB as MusicBrainz API
    participant Out as data/processed

    U->>BG: python -m src.build_graph
    BG->>FS: read user_artists.dat, artists.dat,<br/>tags.dat, user_taggedartists.dat
    FS-->>BG: pandas DataFrames

    BG->>BG: build_id_mappings<br/>user_to_idx, artist_to_idx, ...

    BG->>BG: build listens_to edges<br/>edge_attr = log1p(playcount)
    BG->>BG: build tagged_with edges<br/>deduped artist-tag pairs

    Note over BG,MB: Era enrichment<br/>rate-limited 1 req per sec
    BG->>FS: check musicbrainz_eras.json cache
    alt cache hit
        FS-->>BG: cached eras dict
    else cache miss
        loop top-2000 artists by popularity
            BG->>MB: GET /ws/2/artist?query=...
            MB-->>BG: life-span begin to decade bucket
        end
        BG->>FS: write musicbrainz_eras.json
    end

    BG->>BG: split_interactions<br/>per-user 80/10/10 masks
    BG->>BG: assemble HeteroData<br/>forward + rev edges

    BG->>Out: torch.save ckg_heterodata.pt
    BG->>Out: json.dump id_mappings.json
    BG-->>U: print counts + sizes
```

### 4.2 Sequence — Training

```mermaid
sequenceDiagram
    autonumber
    participant U as User (CLI)
    participant T as train.py
    participant M as KGAT
    participant E as evaluate.py
    participant FS as data/processed

    U->>T: python -m src.train
    T->>FS: load ckg_heterodata.pt
    FS-->>T: HeteroData

    T->>M: KGAT(n_users, n_artists, n_tags, n_eras, ...)
    T->>M: warmup forward (init lazy params)
    T->>T: build_user_positive_sets
    T->>T: optimizer = Adam(model.parameters)

    loop epoch = 1..n_epochs
        T->>M: out = model(data) full-graph
        M-->>T: x_dict per node type

        T->>T: sample_negatives<br/>reject if in user_positives
        T->>M: bpr_loss(u, pos_a, neg_a)
        M-->>T: loss
        T->>M: loss.backward then optimizer.step

        alt epoch mod 5 == 0
            T->>E: evaluate_model(model, data, top_k)
            E->>M: forward + score matrix
            E-->>T: ndcg and recall at k
            alt NDCG@10 improved
                T->>FS: save kgat_best.pt
            else 3 evals without improvement
                T-->>U: early stop
            end
        end
    end

    T-->>U: best NDCG@10 + checkpoint path
```

### 4.3 Sequence — Evaluation

```mermaid
sequenceDiagram
    autonumber
    participant Caller as train / CLI
    participant E as evaluate_model()
    participant M as KGAT
    participant D as HeteroData

    Caller->>E: evaluate_model(model, data, top_k_values)
    E->>M: model(data) (eval mode)
    M-->>E: x_dict
    E->>E: scores = user_emb @ artist_emb.T

    E->>D: get test_mask, train_mask
    D-->>E: edge tensors
    E->>E: scores at train_edges = -inf<br/>(exclude already-seen items)

    loop for each user with test interactions
        E->>E: top_items = argsort(scores per uid) take max_k
        E->>E: ndcg_at_k(top_items, gt, k)
        E->>E: recall_at_k(top_items, gt, k)
    end

    E-->>Caller: ndcg and recall averaged over users
```

### 4.4 Sequence — Explanation Path Extraction

The path extractor runs **after** training and uses the trained attention weights as evidence for *why* an artist was recommended.


```mermaid
sequenceDiagram
    autonumber
    participant Caller as app.py / CLI
    participant X as find_explanation_path()
    participant XA as extract_attention_weights()
    participant M as KGAT
    participant D as HeteroData

    Caller->>X: find_explanation_path(model, data, user_idx, artist_idx, top_k=5)

    alt no precomputed attentions
        X->>XA: extract_attention_weights(model, data)
        loop for each KGAT layer
            loop for each edge type
                XA->>M: GATConv with return_attention_weights=True
                M-->>XA: out plus edge_idx and attn
                XA->>XA: attn.mean over heads
            end
            XA->>XA: aggregate per-dst (sum then ReLU)<br/>matches HeteroConv aggr=sum
        end
        XA-->>X: list of dict per layer<br/>edge_type to attn tensor
    end

    X->>D: edge_index of (user, listens_to, artist)
    Note over X,D: 1-hop direct path<br/>User to Artist if it exists
    alt direct edge exists
        X->>X: append direct path with layer-0 attention
    end

    Note over X,D: 2-hop via shared tag<br/>User to Artist_A to Tag to Artist_B
    X->>D: get user listened artists
    X->>D: get target artist tags
    loop mid_artist in user_artists first 20
        X->>D: get mid_artist tags
        X->>X: shared = user_artist_tags intersect target_tags
        loop tag in shared first 5
            X->>X: attn = cube root of (a1 mul a2 mul a3)<br/>geometric mean of edge attentions
            X->>X: append path
        end
    end

    X->>X: sort by attention desc, take top_k
    X-->>Caller: ranked paths with attention and type
```

### 4.5 Sequence — Fidelity Test (Faithfulness Metric)

The fidelity test answers: *"if we remove the node the model says is the explanation, does the recommendation actually weaken?"* If yes for most samples, the attention paths are causally faithful (not just decorative).

```mermaid
sequenceDiagram
    autonumber
    participant CLI as explain.py main
    participant FT as fidelity_test()
    participant X as find_explanation_path()
    participant M as KGAT
    participant D as HeteroData

    CLI->>FT: fidelity_test(model, data, n_samples=100)
    FT->>M: out = model(data) (full forward)
    FT->>FT: precompute all_layer_attentions

    FT->>D: sample n test edges<br/>user, artist pairs

    loop for each user_idx and artist_idx
        FT->>FT: original_score = u dot a
        FT->>X: find_explanation_path(top_k=1, precomputed)
        X-->>FT: path

        alt path is direct or too short
            FT->>FT: skip
        else 2-hop path
            FT->>FT: zero embedding of mid_node (tag or artist)
            FT->>M: re-run forward with masked x_dict
            M-->>FT: out_masked
            FT->>FT: masked_score = u_prime dot a_prime
            alt masked_score lt original_score
                FT->>FT: changes += 1
            end
        end
    end

    FT-->>CLI: changes div total<br/>fidelity ratio in 0 to 1
```

---

## 5. Online Flow (Streamlit Demo)

The demo loads the trained checkpoint once (`@st.cache_resource`) and then serves recommendations + explanations interactively.

### 5.1 Sequence — User Browsing the Demo

```mermaid
sequenceDiagram
    autonumber
    actor Browser as User (browser)
    participant App as app.py (Streamlit)
    participant M as KGAT
    participant Pop as popularity_baseline
    participant X as find_explanation_path
    participant Viz as pyvis Network

    Browser->>App: open http://localhost:8501

    Note over App: First request only (cached afterwards)
    App->>App: load_model_and_data()<br/>HeteroData + KGAT + kgat_best.pt + mappings

    Browser->>App: select user_idx + top_k (sidebar)

    par KGAT recs
        App->>M: forward(data) and scores = u @ A.T
        App->>App: mask training items with -inf
        App->>App: top-K artists
    and Popularity recs
        App->>Pop: get_popularity_recommendations(data, user_idx)
        Pop-->>App: top-K most-listened
    end

    App-->>Browser: render two-column layout<br/>KGAT vs Popularity

    Browser->>App: pick artist from KGAT list to explain

    App->>X: find_explanation_path(model, data, user_idx, artist_idx, top_k=5)
    X-->>App: ranked attention paths

    App->>App: render paths as text<br/>User N to Artist to Tag to Artist
    App->>Viz: build pyvis graph (color by node type,<br/>edge weight = attention)
    Viz-->>App: HTML
    App-->>Browser: embed iframe
```

### 5.2 UI Layout

| Region | Content |
|---|---|
| Sidebar | User ID input, Top-K slider |
| Left column | KGAT recommendations (with scores) |
| Right column | Popularity baseline (no scores) |
| Bottom | Artist picker → ranked attention paths (text) → interactive pyvis graph |

Color legend in `render_explanation_graph`:
- 🟢 user (green)
- 🔵 artist (blue)
- 🟠 tag (orange)
- 🟣 era (purple)

---

## 6. Cross-Cutting Concerns

### 6.1 Determinism & Reproducibility

- Train/val/test split uses a fixed seed (`np.random.default_rng(42)` in `split_interactions`).
- Negative sampling during training is stochastic — runs are not bit-exact reproducible, but NDCG numbers are stable to ±0.005 across seeds.
- Best checkpoint is selected on **NDCG@10 against the val set** (currently the same as the test set in this v1; deferred to V2).

### 6.2 Performance

- **Graph fits in memory**: ~32K nodes, ~200K edges. No `NeighborLoader` needed for v1; full-graph forward pass is used in train, eval, and inference. Smoke test verifies `NeighborLoader` works for the planned V2 scale-up.
- **MPS backend**: PyG supports MPS as of 2.5+. The `Config.device` field auto-selects.
- **MusicBrainz**: aggressive caching (`musicbrainz_eras.json`) + 1.1s sleep between requests to honor the 1 req/sec rate limit. Top-2000 artists by popularity are queried; the rest fall into the `unknown` era bucket.

### 6.3 Robustness

- **Cold-start users (<3 interactions)**: skipped during split; will not have val/test items but remain in the graph.
- **Lazy parameters**: `GATConv` with `(-1, -1)` input dim shape needs an initial forward pass to materialize weights — done in `train.py` and `app.py` before any `state_dict` load.
- **Missing checkpoint**: `app.py` shows a warning and runs with random embeddings, so the UI still loads end-to-end during development.

### 6.4 Extensibility (V2)

[ADR-002](adr/002-v2-track-level-upgrade.md) plans a track-level upgrade with:
- New node types: `track`, replacing `artist` as the recommendation target; `genre` distinct from `tag`.
- Continuous **audio features** (13-dim Spotify) projected into the embedding space and added to the learnable track embedding.
- Million Song Dataset (50K tracks, 9.7M interactions) requires `NeighborLoader` and per-user batched evaluation.

V1 was deliberately kept artist-level because the Last.fm-2k dataset only ships user-artist interactions. The `smoke_test.py` neighbor-loader check is the foundation for that V2 transition.

---

## 7. References

- [PROJECT.md](../PROJECT.md) — original thesis brief (Bulgarian) covering motivation, problem statement, and methodology.
- [docs/adr/001-architecture-decisions.md](adr/001-architecture-decisions.md) — accepted decisions with rejected alternatives.
- [docs/adr/002-v2-track-level-upgrade.md](adr/002-v2-track-level-upgrade.md) — proposed V2 upgrade.
- KGAT paper: Wang et al., *KGAT: Knowledge Graph Attention Network for Recommendation*, KDD 2019.
- HetRec 2011 Last.fm-2k: https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
- MusicBrainz Web Service: https://musicbrainz.org/doc/MusicBrainz_API
