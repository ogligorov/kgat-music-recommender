import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import requests
import torch

from dotenv import load_dotenv

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ID_MAPPINGS_PATH = PROJECT_ROOT / "data" / "processed" / "id_mappings.json"
GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "graph.pt"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "artist_genres.json"

LASTFM_URL = "http://ws.audioscrobbler.com/2.0/"
MUSICBRAINZ_URL = "https://musicbrainz.org/ws/2/artist"

LASTFM_TAG_MIN_COUNT = 20
LASTFM_MIN_TAGS = 2
MB_MIN_SCORE = 80
MIN_TRACK_COUNT = 5 # skip artists with fewer tracks in the graph
WORKERS = 4
PROGRESS_EVERY = 100

USER_AGENT = "music_recommender_thesis/1.0 (academic research)"


def fetch_lastfm_tags(artist_name: str, api_key: str) -> list[str]:
    params = {
        "method": "artist.gettoptags",
        "artist": artist_name,
        "api_key": api_key,
        "format": "json",
    }
    try:
        resp = requests.get(LASTFM_URL, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    toptags = data.get("toptags")
    if not isinstance(toptags, dict):
        return []
    raw_tags = toptags.get("tag", [])
    if isinstance(raw_tags, dict):
        raw_tags = [raw_tags]
    if not isinstance(raw_tags, list):
        return []

    tags: list[str] = []
    for t in raw_tags:
        if not isinstance(t, dict):
            continue
        try:
            count = int(t.get("count", 0))
        except (TypeError, ValueError):
            count = 0
        name = t.get("name")
        if count >= LASTFM_TAG_MIN_COUNT and isinstance(name, str) and name.strip():
            tags.append(name.strip().lower())
    return tags


def fetch_musicbrainz_tags(artist_name: str) -> list[str]:
    params = {"query": f'artist:"{artist_name}"', "limit": 1, "fmt": "json"}
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(MUSICBRAINZ_URL, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    artists = data.get("artists", [])
    if not isinstance(artists, list) or not artists:
        return []
    top = artists[0]
    if not isinstance(top, dict):
        return []
    try:
        score = int(top.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    if score < MB_MIN_SCORE:
        return []

    raw_tags = top.get("tags", [])
    if not isinstance(raw_tags, list):
        return []
    return [t["name"].strip().lower() for t in raw_tags
            if isinstance(t, dict) and isinstance(t.get("name"), str) and t["name"].strip()]


def fetch_one(artist_lc: str, display: str, api_key: str) -> tuple[str, list[str], bool]:
    """Fetch tags for one artist. Returns (artist_lc, tags, used_fallback)."""
    tags = fetch_lastfm_tags(display, api_key)
    used_fallback = False
    if len(tags) < LASTFM_MIN_TAGS:
        mb_tags = fetch_musicbrainz_tags(display)
        if mb_tags:
            tags = mb_tags
            used_fallback = True
    seen: set[str] = set()
    unique: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return artist_lc, unique, used_fallback


def save_output(path: Path, data: dict[str, list[str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--api-key",
        default=os.environ.get("LASTFM_API_KEY"),
        help="Last.fm API key (or set LASTFM_API_KEY env var)",
    )
    p.add_argument(
        "--save-every", type=int, default=100,
        help="Persist output every N processed artists (default: 100)",
    )
    p.add_argument(
        "--workers", type=int, default=WORKERS,
        help=f"Concurrent workers (default: {WORKERS})",
    )
    p.add_argument(
        "--min-tracks", type=int, default=MIN_TRACK_COUNT,
        help=f"Skip artists with fewer than N tracks in the graph (default: {MIN_TRACK_COUNT})",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print("ERROR: Last.fm API key required. Set LASTFM_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    with ID_MAPPINGS_PATH.open("r", encoding="utf-8") as f:
        mappings: dict[str, Any] = json.load(f)

    artist_to_idx: dict[str, int] = mappings["artist_to_idx"]
    artist_display: dict[str, str] = mappings.get("artist_display", {})

    # Filter to artists with enough tracks to ever surface in top-K
    print(f"Loading graph to filter artists with < {args.min_tracks} tracks...")
    graph = torch.load(GRAPH_PATH, weights_only=False)
    edge_index = graph["track", "performed_by", "artist"].edge_index
    artist_track_counts: dict[int, int] = collections.Counter(edge_index[1].tolist())
    artist_lcs = [
        a for a, idx in artist_to_idx.items()
        if artist_track_counts.get(idx, 0) >= args.min_tracks
    ]
    print(f"Filtered: {len(artist_lcs)}/{len(artist_to_idx)} artists with >= {args.min_tracks} tracks")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[str]] = {}
    if OUTPUT_PATH.exists():
        try:
            with OUTPUT_PATH.open("r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Resuming: {len(results)} artists already done")
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARN: could not load existing output ({e}); starting fresh")

    todo = [a for a in artist_lcs if a not in results]
    total = len(artist_lcs)
    print(f"Remaining: {len(todo)} artists to fetch")

    results_lock = Lock()
    fallbacks = 0
    processed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, a, artist_display.get(a, a), args.api_key): a
            for a in todo
        }
        for future in as_completed(futures):
            try:
                artist_lc, tags, used_fallback = future.result()
            except Exception as e:
                artist_lc = futures[future]
                print(f"WARN: error fetching {artist_lc!r}: {e}")
                tags, used_fallback = [], False

            with results_lock:
                results[artist_lc] = tags
                if used_fallback:
                    fallbacks += 1
                processed += 1
                if processed % PROGRESS_EVERY == 0:
                    print(f"{len(results)}/{total} done, {fallbacks} fallbacks so far")
                if processed % args.save_every == 0:
                    save_output(OUTPUT_PATH, results)

    save_output(OUTPUT_PATH, results)
    print(f"Done: {len(results)}/{total} artists written ({fallbacks} MusicBrainz fallbacks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
