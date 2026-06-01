"""Build the Collaborative Knowledge Graph from the Spotify Playlists CSV.

Produces a PyG HeteroData with:
  Node types: user, track, artist, playlist
  Edge types: liked, in_playlist, performed_by (+ reverses)
  Train/val/test split on (user, liked, track) edges (80/10/10 per user)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.config import Config

EXPECTED_COLUMNS = {"user_id", "artistname", "trackname", "playlistname"}


def load_spotify_csv(path: Path) -> pd.DataFrame:
    """Load the Spotify Playlists CSV. Header has leading-space-quoted columns,
    so skipinitialspace=True is required. Drops malformed rows."""
    df = pd.read_csv(
        path,
        skipinitialspace=True,
        on_bad_lines="skip",
        dtype={"user_id": str, "artistname": str, "trackname": str, "playlistname": str},
    )
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}; found {df.columns.tolist()}")
    df = df.dropna(subset=["user_id", "artistname", "trackname", "playlistname"])

    # Lowercase + strip for dedup
    df["artist_lc"] = df["artistname"].str.lower().str.strip()
    df["track_lc"] = df["trackname"].str.lower().str.strip()
    df["playlist_lc"] = df["playlistname"].str.lower().str.strip()

    df = df[(df["artist_lc"] != "") & (df["track_lc"] != "") & (df["playlist_lc"] != "")]

    # Composite keys for grouping
    df["track_key"] = df["artist_lc"] + "|||" + df["track_lc"]
    df["playlist_key"] = df["user_id"] + "|||" + df["playlist_lc"]
    return df


def apply_kcore_filter(
    df: pd.DataFrame,
    min_user_likes: int,
    min_track_likers: int,
    min_playlist_size: int,
) -> pd.DataFrame:
    """Iterative K-core: drop sparse users/tracks/playlists until stable.

    - User: must have >= min_user_likes distinct liked tracks.
    - Track: must have >= min_track_likers distinct likers.
    - Playlist: must have >= min_playlist_size distinct tracks.
    """
    iteration = 0
    while True:
        iteration += 1
        before = len(df)

        # User filter: distinct tracks per user
        user_track_counts = df.groupby("user_id")["track_key"].nunique()
        valid_users = user_track_counts[user_track_counts >= min_user_likes].index
        df = df[df["user_id"].isin(valid_users)]

        # Track filter: distinct users per track
        track_user_counts = df.groupby("track_key")["user_id"].nunique()
        valid_tracks = track_user_counts[track_user_counts >= min_track_likers].index
        df = df[df["track_key"].isin(valid_tracks)]

        # Playlist filter: distinct tracks per playlist
        playlist_track_counts = df.groupby("playlist_key")["track_key"].nunique()
        valid_playlists = playlist_track_counts[playlist_track_counts >= min_playlist_size].index
        df = df[df["playlist_key"].isin(valid_playlists)]

        after = len(df)
        print(f"  K-core iter {iteration}: {before:,} -> {after:,} rows")
        if after == before:
            break
    return df


def build_id_mappings(df: pd.DataFrame) -> dict:
    """Build dense int indices for users, tracks, artists, playlists."""
    user_ids = sorted(df["user_id"].unique())
    track_keys = sorted(df["track_key"].unique())
    artist_keys = sorted(df["artist_lc"].unique())
    playlist_keys = sorted(df["playlist_key"].unique())

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    track_to_idx = {t: i for i, t in enumerate(track_keys)}
    artist_to_idx = {a: i for i, a in enumerate(artist_keys)}
    playlist_to_idx = {p: i for i, p in enumerate(playlist_keys)}

    # Display strings for the UI: pick the first observed casing for each canonical key
    track_display = (
        df.drop_duplicates("track_key")
        .set_index("track_key")[["artistname", "trackname"]]
        .to_dict(orient="index")
    )
    artist_display = (
        df.drop_duplicates("artist_lc").set_index("artist_lc")["artistname"].to_dict()
    )
    playlist_display = (
        df.drop_duplicates("playlist_key").set_index("playlist_key")["playlistname"].to_dict()
    )

    return {
        "user_to_idx": user_to_idx,
        "track_to_idx": track_to_idx,
        "artist_to_idx": artist_to_idx,
        "playlist_to_idx": playlist_to_idx,
        "track_display": track_display,
        "artist_display": artist_display,
        "playlist_display": playlist_display,
    }


def build_edge_indices(df: pd.DataFrame, mappings: dict) -> dict:
    """Build the three forward edge_index tensors. Reverse edges are added in main()."""
    # (user, liked, track): one edge per distinct (user, track) pair across all playlists
    liked = df[["user_id", "track_key"]].drop_duplicates()
    liked_src = liked["user_id"].map(mappings["user_to_idx"]).to_numpy(dtype=np.int64)
    liked_dst = liked["track_key"].map(mappings["track_to_idx"]).to_numpy(dtype=np.int64)
    liked_edge_index = torch.from_numpy(np.stack([liked_src, liked_dst]))

    # (track, in_playlist, playlist): one edge per distinct (track, playlist) pair
    in_pl = df[["track_key", "playlist_key"]].drop_duplicates()
    in_pl_src = in_pl["track_key"].map(mappings["track_to_idx"]).to_numpy(dtype=np.int64)
    in_pl_dst = in_pl["playlist_key"].map(mappings["playlist_to_idx"]).to_numpy(dtype=np.int64)
    in_pl_edge_index = torch.from_numpy(np.stack([in_pl_src, in_pl_dst]))

    # (track, performed_by, artist): one edge per unique track -> its artist
    perf = df[["track_key", "artist_lc"]].drop_duplicates(subset=["track_key"])
    perf_src = perf["track_key"].map(mappings["track_to_idx"]).to_numpy(dtype=np.int64)
    perf_dst = perf["artist_lc"].map(mappings["artist_to_idx"]).to_numpy(dtype=np.int64)
    perf_edge_index = torch.from_numpy(np.stack([perf_src, perf_dst]))

    return {
        "liked": liked_edge_index,
        "in_playlist": in_pl_edge_index,
        "performed_by": perf_edge_index,
    }


def split_interactions(
    edge_index: torch.Tensor, n_users: int, val_ratio: float = 0.1, test_ratio: float = 0.1
) -> dict[str, torch.Tensor]:
    """Per-user 80/10/10 random split. Carries forward unchanged from v1."""
    rng = np.random.default_rng(42)
    train_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
    val_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    test_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)

    for uid in range(n_users):
        user_edges = (edge_index[0] == uid).nonzero(as_tuple=True)[0].numpy()
        if len(user_edges) < 3:
            continue
        rng.shuffle(user_edges)
        n_val = max(1, int(len(user_edges) * val_ratio))
        n_test = max(1, int(len(user_edges) * test_ratio))
        val_edges = user_edges[:n_val]
        test_edges = user_edges[n_val : n_val + n_test]
        val_mask[val_edges] = True
        test_mask[test_edges] = True
        train_mask[val_edges] = False
        train_mask[test_edges] = False

    return {"train": train_mask, "val": val_mask, "test": test_mask}


def make_train_only_graph(
    data: HeteroData, mp_mask: torch.Tensor | None = None
) -> HeteroData:
    """Build a HeteroData whose (user, liked, track) and (track, rev_liked, user)
    edge_indices are filtered to a chosen subset. All other edge types and node
    counts are shared by reference.

    This is required because PyG's `train_mask` is metadata: it is NOT applied
    automatically during message passing. Forwarding `data` directly leaks val/test
    labels into the GNN, contaminating both training (model sees the answer) and
    evaluation (embeddings encode the held-out edges).

    `mp_mask` controls which edges survive into the message-passing graph:
      - None (default): use `train_mask`. Correct for evaluation, where every
        train edge should inform the model's view of users/tracks.
      - Custom bool mask of length |liked edges|: use during supervised training
        to disjoint-split train edges into a message-passing pool (mp_mask=True)
        and a supervision pool (mp_mask=False, fed as edge_label_index). Without
        this disjoint split, LinkNeighborLoader leaves the supervision edge in
        the MP graph for its own batch and the model trivially copies the
        positive track's embedding into the user node before scoring it.
    """
    if mp_mask is None:
        mp_mask = data["user", "liked", "track"].train_mask
    new_data = HeteroData()
    for nt in data.node_types:
        new_data[nt].num_nodes = data[nt].num_nodes
    for et in data.edge_types:
        if et in (("user", "liked", "track"), ("track", "rev_liked", "user")):
            new_data[et].edge_index = data[et].edge_index[:, mp_mask]
        else:
            new_data[et].edge_index = data[et].edge_index
    return new_data


def main():
    cfg = Config()

    print(f"Loading {cfg.csv_path}...")
    df = load_spotify_csv(cfg.csv_path)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Pre-filter: users={df['user_id'].nunique():,}, "
          f"tracks={df['track_key'].nunique():,}, "
          f"artists={df['artist_lc'].nunique():,}, "
          f"playlists={df['playlist_key'].nunique():,}")

    print(f"\nApplying K-core filter "
          f"(user>={cfg.min_user_likes}, track>={cfg.min_track_likers}, "
          f"playlist>={cfg.min_playlist_size})...")
    df = apply_kcore_filter(
        df,
        min_user_likes=cfg.min_user_likes,
        min_track_likers=cfg.min_track_likers,
        min_playlist_size=cfg.min_playlist_size,
    )
    print(f"  Post-filter: {len(df):,} rows")
    print(f"  Post-filter: users={df['user_id'].nunique():,}, "
          f"tracks={df['track_key'].nunique():,}, "
          f"artists={df['artist_lc'].nunique():,}, "
          f"playlists={df['playlist_key'].nunique():,}")

    print("\nBuilding ID mappings...")
    mappings = build_id_mappings(df)
    n_users = len(mappings["user_to_idx"])
    n_tracks = len(mappings["track_to_idx"])
    n_artists = len(mappings["artist_to_idx"])
    n_playlists = len(mappings["playlist_to_idx"])

    print("Building edge indices...")
    edges = build_edge_indices(df, mappings)

    print("Assembling HeteroData...")
    data = HeteroData()
    data["user"].num_nodes = n_users
    data["track"].num_nodes = n_tracks
    data["artist"].num_nodes = n_artists
    data["playlist"].num_nodes = n_playlists

    data["user", "liked", "track"].edge_index = edges["liked"]
    data["track", "in_playlist", "playlist"].edge_index = edges["in_playlist"]
    data["track", "performed_by", "artist"].edge_index = edges["performed_by"]

    data["track", "rev_liked", "user"].edge_index = edges["liked"].flip(0)
    data["playlist", "rev_in_playlist", "track"].edge_index = edges["in_playlist"].flip(0)
    data["artist", "rev_performed_by", "track"].edge_index = edges["performed_by"].flip(0)

    print(f"Splitting (user, liked, track) per-user 80/10/10...")
    split_masks = split_interactions(edges["liked"], n_users)
    data["user", "liked", "track"].train_mask = split_masks["train"]
    data["user", "liked", "track"].val_mask = split_masks["val"]
    data["user", "liked", "track"].test_mask = split_masks["test"]

    cfg.processed_data_dir.mkdir(parents=True, exist_ok=True)
    graph_path = cfg.processed_data_dir / "graph.pt"
    torch.save(data, graph_path)

    # Persist mappings (keys can be tuples-as-strings; store a JSON-safe form)
    mappings_serializable = {
        "user_to_idx": mappings["user_to_idx"],
        "track_to_idx": mappings["track_to_idx"],
        "artist_to_idx": mappings["artist_to_idx"],
        "playlist_to_idx": mappings["playlist_to_idx"],
        "track_display": {k: v for k, v in mappings["track_display"].items()},
        "artist_display": mappings["artist_display"],
        "playlist_display": mappings["playlist_display"],
    }
    mappings_path = cfg.processed_data_dir / "id_mappings.json"
    with open(mappings_path, "w") as f:
        json.dump(mappings_serializable, f)

    print("\n" + "=" * 50)
    print("GRAPH CONSTRUCTION COMPLETE")
    print("=" * 50)
    print(f"  Users:     {n_users:,}")
    print(f"  Tracks:    {n_tracks:,}")
    print(f"  Artists:   {n_artists:,}")
    print(f"  Playlists: {n_playlists:,}")
    print(f"  Edges (liked):        {edges['liked'].shape[1]:,}")
    print(f"  Edges (in_playlist):  {edges['in_playlist'].shape[1]:,}")
    print(f"  Edges (performed_by): {edges['performed_by'].shape[1]:,}")
    print(f"  Train liked: {split_masks['train'].sum().item():,}")
    print(f"  Val liked:   {split_masks['val'].sum().item():,}")
    print(f"  Test liked:  {split_masks['test'].sum().item():,}")
    print(f"  Saved graph    -> {graph_path}")
    print(f"  Saved mappings -> {mappings_path}")


if __name__ == "__main__":
    main()
