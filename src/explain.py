"""Explainability for the v2 4-node KG (user / track / artist / playlist).

Two responsibilities:
  - `find_explanation_path` extracts top-k attention-weighted paths from a
    user to a recommended track. Three path types:
        * direct       : (user, liked, track) edge present in the train graph.
        * via_artist   : user → track_A → artist → target_track, where track_A
                         is liked by the user and shares the target's artist.
        * via_playlist : user → track_A → playlist → target_track, where the
                         playlist contains both track_A and the target.
    Path score is the geometric mean of edge attentions along the path,
    head-averaged and layer-averaged.
  - `fidelity_test` masks the explanation's hub node (artist or playlist)
    and re-forwards. Reports the fraction of test pairs where the score drops.

All forwards run on the train-only graph (`make_train_only_graph`) to match
the message-passing the model actually saw at training time. Running on the
full graph would let val/test edges leak into the explanations.
"""

import json

import torch
from torch_geometric.data import HeteroData

from src.build_graph import make_train_only_graph
from src.model import KGAT


def extract_attention_weights(model: KGAT, data: HeteroData) -> list[dict[tuple, torch.Tensor]]:
    """Run a full forward and capture per-layer attention weights, head-averaged
    and aligned with `data[edge_type].edge_index` (one scalar per edge).

    Replicates `model.forward` step-by-step rather than calling it, because PyG
    only exposes attention through `subconv(..., return_attention_weights=True)`
    and `HeteroConv` doesn't surface that. The aggregation here MUST match
    `model.forward`'s behavior (sum across edge types, NO ReLU between layers,
    consistent with v2's KGAT-paper concat aggregation)."""
    model.eval()
    x_dict = model.get_initial_embeddings(data)
    edge_index_dict = data.edge_index_dict

    all_layer_attentions: list[dict[tuple, torch.Tensor]] = []

    with torch.no_grad():
        for conv in model.convs:
            layer_attn: dict[tuple, torch.Tensor] = {}
            out_per_dst: dict[str, list[torch.Tensor]] = {}

            for edge_type, subconv in conv.convs.items():
                src_type, _, dst_type = edge_type
                edge_index = edge_index_dict[edge_type]
                src_x = x_dict[src_type]
                dst_x = x_dict[dst_type]

                out, (_, attn) = subconv(
                    (src_x, dst_x), edge_index, return_attention_weights=True
                )
                # Per-head attention → mean over heads to get a single scalar
                # per edge that we can compare across edge types.
                layer_attn[edge_type] = attn.mean(dim=-1) if attn.dim() > 1 else attn
                out_per_dst.setdefault(dst_type, []).append(out)

            all_layer_attentions.append(layer_attn)

            # HeteroConv aggr="sum" across the edge types incident to each dst.
            x_dict = {
                ntype: torch.stack(outs).sum(dim=0)
                for ntype, outs in out_per_dst.items()
            }

    return all_layer_attentions


def _get_edge_attention(
    all_layer_attentions: list[dict[tuple, torch.Tensor]],
    edge_type: tuple,
    src_idx: int,
    dst_idx: int,
    data: HeteroData,
) -> float:
    """Mean attention across layers for the (src_idx → dst_idx) edge of
    `edge_type`. Returns 0.0 if the edge isn't in the graph."""
    edge_index = data[edge_type].edge_index
    mask = (edge_index[0] == src_idx) & (edge_index[1] == dst_idx)
    if not mask.any():
        return 0.0
    edge_idx = mask.nonzero(as_tuple=True)[0][0].item()

    total = 0.0
    count = 0
    for layer_attn in all_layer_attentions:
        attn = layer_attn.get(edge_type)
        if attn is not None and edge_idx < len(attn):
            total += attn[edge_idx].item()
            count += 1
    return total / count if count > 0 else 0.0


def find_explanation_path(
    model: KGAT,
    data: HeteroData,
    user_idx: int,
    track_idx: int,
    top_k: int = 3,
    max_user_tracks: int = 50,
    precomputed_attentions: list[dict[tuple, torch.Tensor]] | None = None,
) -> list[dict]:
    """Top-k paths user → target_track ranked by geometric-mean edge attention.

    `data` should be the train-only graph (see `make_train_only_graph`); a
    direct edge will only register if it was in train, which is what we want
    when explaining a held-out recommendation."""
    all_layer_attentions = precomputed_attentions or extract_attention_weights(model, data)
    paths: list[dict] = []

    liked = data["user", "liked", "track"].edge_index
    perf = data["track", "performed_by", "artist"].edge_index
    in_pl = data["track", "in_playlist", "playlist"].edge_index

    # 1-hop direct edge.
    direct_mask = (liked[0] == user_idx) & (liked[1] == track_idx)
    if direct_mask.any():
        edge_idx = direct_mask.nonzero(as_tuple=True)[0][0].item()
        attn = all_layer_attentions[0][("user", "liked", "track")][edge_idx].item()
        paths.append({
            "type": "direct",
            "path": [("user", user_idx), ("track", track_idx)],
            "relation": "liked",
            "attention": attn,
        })

    # User's liked tracks — capped for tractability on power users.
    user_mask = liked[0] == user_idx
    user_tracks = liked[1, user_mask].unique()[:max_user_tracks].tolist()
    if not user_tracks:
        paths.sort(key=lambda p: p["attention"], reverse=True)
        return paths[:top_k]

    # via_artist: user → track_A → artist → target_track. Each track has one
    # artist (one row in `performed_by` per track), so the only candidate
    # mid-artist is the target's artist; we just check whether each user-liked
    # track shares it.
    target_artist_mask = perf[0] == track_idx
    if target_artist_mask.any():
        target_artist = perf[1, target_artist_mask][0].item()
        for track_a in user_tracks:
            if track_a == track_idx:
                continue
            shares_artist = ((perf[0] == track_a) & (perf[1] == target_artist)).any()
            if not shares_artist:
                continue
            a1 = _get_edge_attention(
                all_layer_attentions, ("user", "liked", "track"),
                user_idx, track_a, data,
            )
            a2 = _get_edge_attention(
                all_layer_attentions, ("track", "performed_by", "artist"),
                track_a, target_artist, data,
            )
            a3 = _get_edge_attention(
                all_layer_attentions, ("artist", "rev_performed_by", "track"),
                target_artist, track_idx, data,
            )
            paths.append({
                "type": "via_artist",
                "path": [
                    ("user", user_idx),
                    ("track", track_a),
                    ("artist", target_artist),
                    ("track", track_idx),
                ],
                "attention": (a1 * a2 * a3) ** (1 / 3),
            })

    # via_playlist: user → track_A → playlist → target_track. A playlist
    # qualifies if it contains both track_A and the target. Many tracks live
    # on dozens of playlists, so cap shared playlists per (track_a) pair.
    target_pl_mask = in_pl[0] == track_idx
    target_playlists = set(in_pl[1, target_pl_mask].cpu().tolist())
    if target_playlists:
        for track_a in user_tracks:
            if track_a == track_idx:
                continue
            track_a_pl_mask = in_pl[0] == track_a
            shared = list(set(in_pl[1, track_a_pl_mask].cpu().tolist()) & target_playlists)[:5]
            for pl_idx in shared:
                a1 = _get_edge_attention(
                    all_layer_attentions, ("user", "liked", "track"),
                    user_idx, track_a, data,
                )
                a2 = _get_edge_attention(
                    all_layer_attentions, ("track", "in_playlist", "playlist"),
                    track_a, pl_idx, data,
                )
                a3 = _get_edge_attention(
                    all_layer_attentions, ("playlist", "rev_in_playlist", "track"),
                    pl_idx, track_idx, data,
                )
                paths.append({
                    "type": "via_playlist",
                    "path": [
                        ("user", user_idx),
                        ("track", track_a),
                        ("playlist", pl_idx),
                        ("track", track_idx),
                    ],
                    "attention": (a1 * a2 * a3) ** (1 / 3),
                })

    paths.sort(key=lambda p: p["attention"], reverse=True)
    return paths[:top_k]


@torch.no_grad()
def fidelity_test(model: KGAT, data: HeteroData, n_samples: int = 100) -> float:
    """For sampled (user, target_track) test edges, find the top explanation
    path; if it's via_artist or via_playlist, zero the hub node's initial
    embedding and re-forward. Fidelity = fraction of samples where the
    masked score is lower than the original. `data` is the FULL graph (we
    need test_mask); message passing happens on the train-only graph."""
    model.eval()
    train_data = make_train_only_graph(data)

    out = model(train_data)
    user_emb = out["user"]
    track_emb = out["track"]

    all_layer_attentions = extract_attention_weights(model, train_data)

    test_mask = data["user", "liked", "track"].test_mask
    test_edges = data["user", "liked", "track"].edge_index[:, test_mask]
    n_test = test_edges.shape[1]
    sample_indices = torch.randperm(n_test)[: min(n_samples, n_test)]

    changes = 0
    total = 0

    for idx in sample_indices.tolist():
        user_idx = test_edges[0, idx].item()
        track_idx = test_edges[1, idx].item()

        original_score = (user_emb[user_idx] * track_emb[track_idx]).sum().item()

        paths = find_explanation_path(
            model, train_data, user_idx, track_idx, top_k=1,
            precomputed_attentions=all_layer_attentions,
        )
        if not paths or paths[0]["type"] == "direct":
            continue

        # Hub node sits at index 2 in via_artist / via_playlist paths.
        path = paths[0]["path"]
        if len(path) < 4:
            continue
        mid_node_type, mid_node_idx = path[2]

        # Replicate `model.forward` with one row of the hub embedding zeroed.
        # Must match v2 aggregation exactly: concat of (initial + each layer
        # output) along feature dim, no ReLU between layers.
        x_dict = model.get_initial_embeddings(train_data)
        x_dict = {k: v.clone() for k, v in x_dict.items()}
        x_dict[mid_node_type][mid_node_idx] = 0.0

        layer_outputs = {k: [v] for k, v in x_dict.items()}
        for conv in model.convs:
            x_dict = conv(x_dict, train_data.edge_index_dict)
            for k in layer_outputs:
                if k in x_dict:
                    layer_outputs[k].append(x_dict[k])
        masked = {k: torch.cat(layers, dim=-1) for k, layers in layer_outputs.items()}

        masked_score = (masked["user"][user_idx] * masked["track"][track_idx]).sum().item()

        if masked_score < original_score:
            changes += 1
        total += 1

    return changes / total if total > 0 else 0.0


def _label_for(node_type: str, node_idx: int, mappings: dict, idx_to_key: dict) -> str:
    """Best-effort human-readable label for a path node."""
    if node_type == "track":
        key = idx_to_key.get("track", {}).get(node_idx)
        disp = mappings.get("track_display", {}).get(key, {}) if key else {}
        if disp:
            return f"{disp.get('artistname', '?')} — {disp.get('trackname', '?')}"
        return key or ""
    if node_type == "artist":
        key = idx_to_key.get("artist", {}).get(node_idx)
        return mappings.get("artist_display", {}).get(key, key or "")
    if node_type == "playlist":
        key = idx_to_key.get("playlist", {}).get(node_idx)
        return mappings.get("playlist_display", {}).get(key, key or "")
    if node_type == "user":
        return idx_to_key.get("user", {}).get(node_idx, str(node_idx))
    return ""


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, default=0)
    parser.add_argument("--track", type=int, default=5)
    parser.add_argument("--fidelity-samples", type=int, default=100)
    args = parser.parse_args()

    from torch_geometric.loader import NeighborLoader

    from src.config import Config

    cfg = Config()

    print(f"Loading graph from {cfg.processed_data_dir / 'graph.pt'}...")
    data = torch.load(cfg.processed_data_dir / "graph.pt", weights_only=False)
    train_data = make_train_only_graph(data)

    model = KGAT(
        n_users=data["user"].num_nodes,
        n_tracks=data["track"].num_nodes,
        n_artists=data["artist"].num_nodes,
        n_playlists=data["playlist"].num_nodes,
        embed_dim=cfg.embed_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
    )

    # Initialize lazy GATConv params via one tiny sub-graph forward (matches
    # the pattern in evaluate.py / train.py).
    init_loader = NeighborLoader(
        train_data, num_neighbors=[3, 3, 3], input_nodes="user", batch_size=8,
    )
    with torch.no_grad():
        model(next(iter(init_loader)))

    checkpoint = cfg.processed_data_dir / "kgat_best.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        print(f"Loaded checkpoint from {checkpoint}")
    else:
        print(f"WARNING: no checkpoint at {checkpoint}; attention weights and "
              f"fidelity numbers will be meaningless on an untrained model.")

    print(f"\nExplanation paths for user={args.user} -> track={args.track}:")
    paths = find_explanation_path(model, train_data, args.user, args.track, top_k=5)

    mappings_path = cfg.processed_data_dir / "id_mappings.json"
    mappings: dict = {}
    if mappings_path.exists():
        with open(mappings_path) as f:
            mappings = json.load(f)

    idx_to_key = {
        "user": {v: k for k, v in mappings.get("user_to_idx", {}).items()},
        "track": {v: k for k, v in mappings.get("track_to_idx", {}).items()},
        "artist": {v: k for k, v in mappings.get("artist_to_idx", {}).items()},
        "playlist": {v: k for k, v in mappings.get("playlist_to_idx", {}).items()},
    }

    if not paths:
        print("  (no paths found — user and track may share no artist or playlist hub)")
    for i, p in enumerate(paths):
        print(f"\n  Path {i+1} (type={p['type']}, attention={p['attention']:.4f}):")
        for node_type, node_idx in p["path"]:
            label = _label_for(node_type, node_idx, mappings, idx_to_key)
            print(f"    {node_type}[{node_idx}] {label}")

    print(f"\nRunning fidelity test ({args.fidelity_samples} samples)...")
    fidelity = fidelity_test(model, data, n_samples=args.fidelity_samples)
    print(f"Fidelity score: {fidelity:.4f}")


if __name__ == "__main__":
    main()
