"""Build a stratified Suno subset for the AI-music extension demo.

Reads the pre-downloaded nyuuzyou/suno parquet file, applies quality filters,
assigns a canonical genre via substring matching on metadata_tags, then keeps
the top-600 tracks per genre bucket by upvote_count.
"""
from pathlib import Path

import pandas as pd

SUNO_RAW = Path("data/raw/suno/data/train-00000-of-00001.parquet")
OUTPUT_PATH = Path("data/processed/suno_subset.parquet")

KEEP_COLUMNS = [
    "id",
    "title",
    "audio_url",
    "image_url",
    "metadata_tags",
    "metadata_has_vocal",
    "upvote_count",
    "play_count",
]

GENRE_KEYWORDS = {
    "rock": "rock",
    "pop": "pop",
    "rap": "hip-hop",
    "hip hop": "hip-hop",
    "hip-hop": "hip-hop",
    "electronic": "electronic",
    "edm": "electronic",
    "jazz": "jazz",
    "classical": "classical",
    "metal": "metal",
    "r&b": "r&b",
    "rnb": "r&b",
    "indie": "indie",
    "folk": "folk",
    "country": "country",
}

PER_GENRE_CAP = 600


def primary_genre(tags_str):
    if not isinstance(tags_str, str):
        return None
    for tag in tags_str.lower().split(","):
        tag = tag.strip()
        for keyword, canonical in GENRE_KEYWORDS.items():
            if keyword in tag:
                return canonical
    return None


def main():
    if not SUNO_RAW.exists():
        raise FileNotFoundError(
            f"Raw Suno file not found at {SUNO_RAW}. "
            "Download it from huggingface - nyuuzyou/suno"
        )
    print(f"Loading {SUNO_RAW} ...")
    df = pd.read_parquet(SUNO_RAW)
    total_loaded = len(df)
    print(f"Total rows loaded: {total_loaded:,}")

    df = df[
        df["is_public"]
        & ~df["is_trashed"]
        & (df["status"] == "complete")
        & df["metadata_tags"].notna()
        & df["audio_url"].notna()
    ]
    print(f"After quality filter: {len(df):,}")

    df = df[KEEP_COLUMNS].copy()

    df["canonical_genre"] = df["metadata_tags"].apply(primary_genre)
    df = df[df["canonical_genre"].notna()]
    print(f"After genre assignment: {len(df):,}")

    df = (
        df.sort_values("upvote_count", ascending=False)
        .groupby("canonical_genre")
        .head(PER_GENRE_CAP)
    )
    print(f"Final subset size: {len(df):,}")

    print("\nRows per genre:")
    counts = df["canonical_genre"].value_counts().sort_values(ascending=False)
    for genre, count in counts.items():
        print(f"  {genre:12s} {count:>6,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
