"""Build the Collaborative Knowledge Graph (CKG) from Last.fm-2k + Wikidata era data.

Produces a PyG HeteroData with:
  Node types: user, artist, tag, era
  Edge types: listens_to, tagged_with, active_in_era (+ reverses)
  Train/val/test split on user-artist interactions (80/10/10 per user)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.config import Config

DECADE_BUCKETS = ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "unknown"]


def load_user_artists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="latin-1")


def load_artists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="latin-1")


def load_tags(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="latin-1")


def load_user_tagged_artists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="latin-1")


def query_musicbrainz_eras(artist_names: list[str], cache_path: Path, max_queries: int = 2000) -> dict[str, str]:
    """Query MusicBrainz for artist formation decades. Returns {artist_name_lower: decade_bucket}.

    Only queries the first `max_queries` artists (should be sorted by popularity).
    Rest default to "unknown" era.
    """
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if cached:
            print(f"Loading cached MusicBrainz eras from {cache_path} ({len(cached)} artists)")
            return cached

    import time
    import requests as req

    to_query = artist_names[:max_queries]
    est_minutes = len(to_query) * 1.1 / 60
    print(f"Querying MusicBrainz for top {len(to_query)} artists (~{est_minutes:.0f} minutes)...")
    print("  Tip: you can Ctrl+C at any time — progress is saved to cache.")

    results: dict[str, str] = {}
    consecutive_failures = 0
    headers = {"User-Agent": "KGAT-Music-Recommender/0.1 (university course project)"}

    try:
        for i, name in enumerate(tqdm(to_query, desc="MusicBrainz")):
            if not name.strip():
                continue

            try:
                resp = req.get(
                    "https://musicbrainz.org/ws/2/artist",
                    params={"query": f'artist:"{name}"', "fmt": "json", "limit": "1"},
                    headers=headers,
                    timeout=10,
                )

                if resp.status_code == 503:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        print(f"\n  Too many 503 errors, stopping.")
                        break
                    time.sleep(2)
                    continue

                resp.raise_for_status()
                data = resp.json()
                consecutive_failures = 0

                artists = data.get("artists", [])
                if artists:
                    best = artists[0]
                    # Check name matches reasonably
                    if best.get("score", 0) >= 90:
                        life_span = best.get("life-span", {})
                        begin = life_span.get("begin", "")
                        if begin and len(begin) >= 4:
                            year = int(begin[:4])
                            decade = f"{(year // 10) * 10}s"
                            if decade in DECADE_BUCKETS:
                                results[name.lower()] = decade

            except (req.RequestException, ValueError, KeyError):
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print(f"\n  Too many consecutive failures, stopping.")
                    break
                continue

            time.sleep(1.1)  # Respect MusicBrainz 1 req/sec limit

            # Save progress every 500 artists
            if (i + 1) % 500 == 0:
                with open(cache_path, "w") as f:
                    json.dump(results, f)
                tqdm.write(f"  Progress saved: {len(results)} eras found so far")

    except KeyboardInterrupt:
        print(f"\n  Interrupted. Saving {len(results)} results to cache...")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f)

    print(f"  MusicBrainz coverage: {len(results)}/{len(artist_names)} ({100*len(results)/len(artist_names):.1f}%)")
    return results


def build_id_mappings(
    user_artists_df: pd.DataFrame,
    artists_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    artist_tag_pairs: pd.DataFrame,
) -> dict:
    user_ids = sorted(user_artists_df["userID"].unique())
    artist_ids = sorted(artists_df["id"].unique())
    tag_ids = sorted(artist_tag_pairs["tagID"].unique())

    return {
        "user_to_idx": {uid: i for i, uid in enumerate(user_ids)},
        "artist_to_idx": {aid: i for i, aid in enumerate(artist_ids)},
        "tag_to_idx": {tid: i for i, tid in enumerate(tag_ids)},
        "era_to_idx": {era: i for i, era in enumerate(DECADE_BUCKETS)},
        "idx_to_user": {i: int(uid) for i, uid in enumerate(user_ids)},
        "idx_to_artist": {i: int(aid) for i, aid in enumerate(artist_ids)},
        "idx_to_tag": {i: int(tid) for i, tid in enumerate(tag_ids)},
        "idx_to_era": {i: era for i, era in enumerate(DECADE_BUCKETS)},
        "artist_id_to_name": dict(zip(artists_df["id"], artists_df["name"])),
        "tag_id_to_name": dict(zip(tags_df["tagID"], tags_df["tagValue"])),
    }


def split_interactions(
    edge_index: torch.Tensor, n_users: int, val_ratio: float = 0.1, test_ratio: float = 0.1
) -> dict[str, torch.Tensor]:
    """Per-user split of user-artist interactions into train/val/test."""
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


def main():
    cfg = Config()

    print("Loading raw data...")
    user_artists_df = load_user_artists(cfg.raw_data_dir / "user_artists.dat")
    artists_df = load_artists(cfg.raw_data_dir / "artists.dat")
    tags_df = load_tags(cfg.raw_data_dir / "tags.dat")
    user_tagged_df = load_user_tagged_artists(cfg.raw_data_dir / "user_taggedartists.dat")

    # Deduplicate artist-tag pairs (multiple users may tag same artist with same tag)
    artist_tag_pairs = user_tagged_df[["artistID", "tagID"]].drop_duplicates()

    print("Building ID mappings...")
    mappings = build_id_mappings(user_artists_df, artists_df, tags_df, artist_tag_pairs)

    n_users = len(mappings["user_to_idx"])
    n_artists = len(mappings["artist_to_idx"])
    n_tags = len(mappings["tag_to_idx"])
    n_eras = len(DECADE_BUCKETS)

    # Build listens_to edges
    print("Building edge indices...")
    ua_users = user_artists_df["userID"].map(mappings["user_to_idx"]).values
    ua_artists = user_artists_df["artistID"].map(mappings["artist_to_idx"]).values
    valid = ~(np.isnan(ua_users) | np.isnan(ua_artists))
    listens_edge_index = torch.tensor(
        np.stack([ua_users[valid].astype(np.int64), ua_artists[valid].astype(np.int64)]), dtype=torch.long
    )
    listens_weights = torch.tensor(
        np.log1p(user_artists_df["weight"].values[valid]), dtype=torch.float
    )

    # Build tagged_with edges
    at_artists = artist_tag_pairs["artistID"].map(mappings["artist_to_idx"]).values
    at_tags = artist_tag_pairs["tagID"].map(mappings["tag_to_idx"]).values
    valid = ~(np.isnan(at_artists) | np.isnan(at_tags))
    tagged_edge_index = torch.tensor(
        np.stack([at_artists[valid].astype(np.int64), at_tags[valid].astype(np.int64)]), dtype=torch.long
    )

    # Build active_in_era edges via MusicBrainz
    # Sort artists by popularity (listen count) so we query the most important ones first
    artist_popularity = user_artists_df.groupby("artistID").size().to_dict()
    sorted_aids = sorted(mappings["artist_to_idx"].keys(), key=lambda x: artist_popularity.get(x, 0), reverse=True)
    artist_names = [str(mappings["artist_id_to_name"].get(aid, "")) for aid in sorted_aids]
    era_results = query_musicbrainz_eras(artist_names, cfg.raw_data_dir / "musicbrainz_eras.json")

    era_src, era_dst = [], []
    for aid, idx in mappings["artist_to_idx"].items():
        name = str(mappings["artist_id_to_name"].get(aid, "")).lower()
        decade = era_results.get(name, "unknown")
        era_src.append(idx)
        era_dst.append(mappings["era_to_idx"][decade])

    era_edge_index = torch.tensor(np.stack([era_src, era_dst]), dtype=torch.long)

    # Assemble HeteroData
    print("Assembling HeteroData...")
    data = HeteroData()
    data["user"].num_nodes = n_users
    data["artist"].num_nodes = n_artists
    data["tag"].num_nodes = n_tags
    data["era"].num_nodes = n_eras

    data["user", "listens_to", "artist"].edge_index = listens_edge_index
    data["user", "listens_to", "artist"].edge_attr = listens_weights

    data["artist", "tagged_with", "tag"].edge_index = tagged_edge_index
    data["artist", "active_in_era", "era"].edge_index = era_edge_index

    # Reverse edges
    data["artist", "rev_listens_to", "user"].edge_index = listens_edge_index.flip(0)
    data["tag", "rev_tagged_with", "artist"].edge_index = tagged_edge_index.flip(0)
    data["era", "rev_active_in_era", "artist"].edge_index = era_edge_index.flip(0)

    # Train/val/test split on listens_to edges
    print("Splitting interactions (80/10/10 per user)...")
    split_masks = split_interactions(listens_edge_index, n_users)
    data["user", "listens_to", "artist"].train_mask = split_masks["train"]
    data["user", "listens_to", "artist"].val_mask = split_masks["val"]
    data["user", "listens_to", "artist"].test_mask = split_masks["test"]

    # Save
    cfg.processed_data_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, cfg.processed_data_dir / "ckg_heterodata.pt")

    mappings_serializable = {
        k: {str(k2): v2 for k2, v2 in v.items()} if isinstance(v, dict) else v
        for k, v in mappings.items()
    }
    with open(cfg.processed_data_dir / "id_mappings.json", "w") as f:
        json.dump(mappings_serializable, f)

    # Verification
    print("\n" + "=" * 50)
    print("CKG CONSTRUCTION COMPLETE")
    print("=" * 50)
    print(f"  Users:   {n_users}")
    print(f"  Artists: {n_artists}")
    print(f"  Tags:    {n_tags}")
    print(f"  Eras:    {n_eras}")
    print(f"  Edges (listens_to):    {listens_edge_index.shape[1]}")
    print(f"  Edges (tagged_with):   {tagged_edge_index.shape[1]}")
    print(f"  Edges (active_in_era): {era_edge_index.shape[1]}")
    print(f"  Train interactions: {split_masks['train'].sum().item()}")
    print(f"  Val interactions:   {split_masks['val'].sum().item()}")
    print(f"  Test interactions:  {split_masks['test'].sum().item()}")
    print(f"  Saved to: {cfg.processed_data_dir / 'ckg_heterodata.pt'}")


if __name__ == "__main__":
    main()
