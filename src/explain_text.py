"""Render KGAT explanation paths as natural-language Bulgarian sentences.

Takes the structured path dicts produced by `find_explanation_path` and
turns each into one sentence describing why the target track was
recommended. Three path types map to three sentence templates:

  * via_artist   — "...защото си харесал X, която e от същия артист — Y."
  * via_playlist — "...защото си харесал X, която споделя плейлиста *Y*..."
"""

from collections.abc import Mapping


def _track_label(track_idx: int, mappings: Mapping, idx_to_key: Mapping) -> str:
    """Reuses the same resolution chain as app.track_label so labels match
    the rest of the UI."""
    key = idx_to_key.get("track", {}).get(track_idx)
    if not key:
        return f"track #{track_idx}"
    disp = mappings.get("track_display", {}).get(key, {})
    if disp:
        return f"{disp.get('artistname', '?')} — {disp.get('trackname', '?')}"
    return key


def _artist_label(artist_idx: int, mappings: Mapping, idx_to_key: Mapping) -> str:
    key = idx_to_key.get("artist", {}).get(artist_idx, "")
    return mappings.get("artist_display", {}).get(key, key) or f"артист #{artist_idx}"


def _playlist_label(playlist_idx: int, mappings: Mapping, idx_to_key: Mapping) -> str:
    key = idx_to_key.get("playlist", {}).get(playlist_idx, "")
    return mappings.get("playlist_display", {}).get(key, key) or f"плейлист #{playlist_idx}"


def render_path_bg(
    path_info: dict,
    mappings: Mapping,
    idx_to_key: Mapping,
) -> str:
    """Render one explanation path as a Bulgarian sentence.

    `path_info` matches the dict shape produced by `find_explanation_path`:
    keys are `"type"` ('direct' | 'via_artist' | 'via_playlist') and
    `"path"` (list of `(node_type, node_idx)` tuples)."""
    path = path_info["path"]
    ptype = path_info["type"]

    target_track = _track_label(path[-1][1], mappings, idx_to_key)

    if ptype == "direct":
        # path = [(user, u), (track, target)]
        return (
            f"Препоръчваме **{target_track}**, защото вече си я харесал/а "
            f"и моделът открива силно сходство между нея и останалите ти "
            f"харесвания."
        )

    # Multi-hop: [(user, u), (track, t_A), (hub_type, hub_idx), (track, target)]
    src_track = _track_label(path[1][1], mappings, idx_to_key)

    if ptype == "via_artist":
        artist = _artist_label(path[2][1], mappings, idx_to_key)
        return (
            f"Препоръчваме **{target_track}**, защото си харесал/а "
            f"**{src_track}** — двете песни са от един и същ изпълнител, "
            f"*{artist}*."
        )

    if ptype == "via_playlist":
        playlist = _playlist_label(path[2][1], mappings, idx_to_key)
        return (
            f"Препоръчваме **{target_track}**, защото си харесал/а "
            f"**{src_track}**, а двете песни се срещат заедно в плейлиста "
            f"*\"{playlist}\"*."
        )

    return f"Препоръчваме **{target_track}**."


def render_paths_bg(
    paths: list[dict],
    mappings: Mapping,
    idx_to_key: Mapping,
    max_paths: int = 3,
) -> list[str]:
    """Render the top-N paths. Capped at `max_paths` to avoid wall-of-text."""
    return [
        render_path_bg(p, mappings, idx_to_key)
        for p in paths[:max_paths]
    ]
