from collections import defaultdict

import torch
import pandas as pd

GENRE_KEYWORDS = {
    "rock": "rock",
    "pop": "pop",
    "rap": "hip-hop", "hip hop": "hip-hop", "hip-hop": "hip-hop",
    "electronic": "electronic", "edm": "electronic",
    "jazz": "jazz",
    "classical": "classical",
    "metal": "metal",
    "r&b": "r&b", "rnb": "r&b",
    "indie": "indie",
    "folk": "folk",
    "country": "country",
}


def _tag_to_genre(tag: str) -> str | None:
    tag_lower = tag.lower().strip()
    for keyword, canonical in GENRE_KEYWORDS.items():
        if keyword in tag_lower:
            return canonical
    return None


def get_user_genre_profile(user_idx, artist_genres, artist_emb, user_emb,
                           idx_to_artist, top_k=10):
    scores = user_emb[user_idx] @ artist_emb.T
    top_artist_indices = scores.topk(top_k).indices

    genre_weights = defaultdict(float)
    top_artist_names = []
    for idx in top_artist_indices:
        artist_lc = idx_to_artist[int(idx)]
        top_artist_names.append(artist_lc)
        weight = max(scores[idx].item(), 0.0)
        for tag in artist_genres.get(artist_lc, []):
            genre = _tag_to_genre(tag)
            if genre:
                genre_weights[genre] += weight

    total = sum(genre_weights.values())
    profile = {g: w / total for g, w in genre_weights.items()} if total > 0 else {}
    return profile, top_artist_names


def query_suno(genre_profile, suno_df, top_k=5):
    if not genre_profile:
        return _dedup(suno_df.nlargest(top_k * 3, "upvote_count"), top_k), True

    def overlap_score(tags_str):
        if not tags_str:
            return 0.0
        score = 0.0
        seen_genres: set[str] = set()
        for tag in tags_str.lower().split(","):
            genre = _tag_to_genre(tag)
            if genre and genre not in seen_genres:
                seen_genres.add(genre)
                score += genre_profile.get(genre, 0.0)
        return score

    suno_df = suno_df.copy()
    suno_df["overlap"] = suno_df["metadata_tags"].apply(overlap_score)
    suno_df["quality"] = suno_df["upvote_count"] / (suno_df["play_count"] + 1)
    suno_df["score"] = suno_df["overlap"] * (1 + suno_df["quality"])

    matched = suno_df[suno_df["overlap"] > 0].nlargest(top_k * 3, "score")
    if matched.empty:
        return _dedup(suno_df.nlargest(top_k * 3, "upvote_count"), top_k), True
    return _dedup(matched, top_k), False


def _dedup(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Return top_k rows with unique titles (case-insensitive)."""
    seen: set[str] = set()
    rows = []
    for _, row in df.iterrows():
        key = row["title"].strip().lower() if isinstance(row["title"], str) else str(row.name)
        if key not in seen:
            seen.add(key)
            rows.append(row)
        if len(rows) == top_k:
            break
    return pd.DataFrame(rows) if rows else df.head(top_k)


def _clean_tags(tags_str: str) -> str:
    """Deduplicate comma-separated tags, preserving first-occurrence order."""
    if not isinstance(tags_str, str):
        return ""
    seen: set[str] = set()
    unique = []
    for tag in tags_str.split(","):
        t = tag.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)
    return ", ".join(unique)


def explain_suno_recs(genre_profile, top_artist_names, artist_display,
                      fallback=False, top_n_genres=3):
    if fallback:
        return "Popular AI-generated tracks this week."
    top_genres = sorted(genre_profile, key=genre_profile.get, reverse=True)[:top_n_genres]
    display_names = [artist_display.get(a, a) for a in top_artist_names[:3]]
    return (f"Based on your affinity for {', '.join(display_names)}, "
            f"you seem to enjoy {', '.join(top_genres)}.")
