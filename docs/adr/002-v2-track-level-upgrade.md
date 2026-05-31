# ADR-002: V2 Track-Level Upgrade Plan

**Status**: Superseded by [ADR-003](003-v2-spotify-playlists.md)
**Date**: 2026-05-11
**Context**: Upgrade path from artist-level KGAT (v1) to track-level KGAT with audio features (v2).

---

## Dataset

**Source**: Million Song Dataset + Spotify + Last.fm  
https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm

**Music_Info.csv** (50,683 tracks):
- `track_id`, `name`, `artist`, `spotify_preview_url`, `spotify_id`
- `tags`, `genre`, `year`
- Audio features (13): `duration_ms`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`

**User_Listening_History.csv** (9.7M records):
- `track_id`, `user_id`, `playcount`

Both files linked by `track_id` — no fuzzy matching needed.

---

## New KG Design

| Node Type | Source | Count (approx) | Features |
|-----------|--------|----------------|----------|
| User | Listening History | ~100K+ users | Learnable embedding |
| Track | Music_Info | 50,683 | Learnable embedding + projected audio features (13-dim → embed_dim) |
| Artist | Music_Info (unique artists) | ~20K | Learnable embedding |
| Genre | Music_Info `genre` col | ~15-20 | Learnable embedding |

| Relation | Connects | Source |
|----------|----------|--------|
| listened_to | User → Track | User_Listening_History (edge weight = log1p(playcount)) |
| performed_by | Track → Artist | Music_Info |
| has_genre | Track → Genre | Music_Info |
| + 3 reverse edges for bidirectional message passing |

**Available for future enrichment** (present in dataset): `tags` column → Tag nodes, `year` column → Era nodes.

---

## Architecture Changes from V1

### Model (`src/model.py`)

```python
class KGAT(nn.Module):
    def __init__(self, n_users, n_tracks, n_artists, n_genres,
                 n_audio_features=13, embed_dim=64, n_layers=2,
                 n_heads=4, dropout=0.1):
        # Learnable embeddings
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.track_emb = nn.Embedding(n_tracks, embed_dim)
        self.artist_emb = nn.Embedding(n_artists, embed_dim)
        self.genre_emb = nn.Embedding(n_genres, embed_dim)

        # Audio feature projection (13 → embed_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(n_audio_features, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # HeteroConv layers (6 edge types)
        edge_types = [
            ('user', 'listened_to', 'track'),
            ('track', 'rev_listened_to', 'user'),
            ('track', 'performed_by', 'artist'),
            ('artist', 'rev_performed_by', 'track'),
            ('track', 'has_genre', 'genre'),
            ('genre', 'rev_has_genre', 'track'),
        ]
```

### Feature Fusion

```python
def get_initial_embeddings(self, data):
    audio_feat = data['track'].x  # (n_tracks, 13), z-score normalized
    projected = self.feature_proj(audio_feat)
    track_init = self.track_emb.weight + projected  # Addition fusion
    return {'user': self.user_emb.weight, 'track': track_init,
            'artist': self.artist_emb.weight, 'genre': self.genre_emb.weight}
```

---

## Scale Considerations

- ~10M edges total → NeighborLoader required for training
- Evaluation: cannot materialize 100K × 50K score matrix → batch per-user
- Training: BPR loss over user-track pairs, negative tracks sampled randomly
- Audio feature normalization: z-score (mean=0, std=1)
- Users filtered to ≥5 interactions for valid train/val/test splits

---

## Implementation Order

1. `src/config.py` — new config fields
2. `src/download.py` — validation script (manual Kaggle download)
3. `src/build_graph.py` — full rewrite (largest effort)
4. `src/model.py` — new architecture with feature projection
5. Smoke test forward pass
6. `src/train.py` — track-level BPR with NeighborLoader
7. `src/evaluate.py` — batched track-level metrics
8. `src/explain.py` — new path types (via artist, via genre)
9. `src/baselines.py` — track popularity baseline
10. Update ADR-001

---

## Explanation Path Types (V2)

- **Via shared artist**: User → Track_A → Artist → Track_B
- **Via shared genre**: User → Track_A → Genre → Track_B
- Fidelity test: same zero-embedding + re-forward approach as V1
