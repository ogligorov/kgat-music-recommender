"""Explainability for the 4-node KG (user / track / artist / playlist).

Two responsibilities:
  - `find_explanation_path` extracts top-k attention-weighted paths from a
    user to a recommended track. Three path types:
        * direct       : (user, liked, track) edge present in the train graph.
        * via_artist   : user -> track_A -> artist -> target_track, where track_A
                         is liked by the user and shares the target's artist.
        * via_playlist : user -> track_A -> playlist -> target_track, where the
                         playlist contains both track_A and the target.
    Path score is the geometric mean of edge attentions along the path,
    head-averaged and layer-averaged.
  - `fidelity_test` masks the explanation's hub node (artist or playlist)
    and re-forwards. Reports the fraction of test pairs where the score drops.
"""

import json
import time
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax
from torch_geometric.loader import NeighborLoader

from src.config import Config
    
from src.build_graph import make_train_only_graph
from src.model import EDGE_TYPES, KGAT


LOOKUP_EDGE_TYPES: tuple[tuple[str, str, str], ...] = (
    ("user", "liked", "track"),
    ("track", "performed_by", "artist"),
    ("artist", "rev_performed_by", "track"),
    ("track", "in_playlist", "playlist"),
    ("playlist", "rev_in_playlist", "track"),
)


def build_edge_indexes(data: HeteroData) -> dict[tuple, dict[int, dict[int, int]]]:
    indexes: dict[tuple, dict[int, dict[int, int]]] = {}
    for et in LOOKUP_EDGE_TYPES:
        if et not in data.edge_types:
            indexes[et] = {}
            continue
        ei = data[et].edge_index
        if ei is None or ei.numel() == 0:
            indexes[et] = {}
            continue
        src_list = ei[0].tolist()
        dst_list = ei[1].tolist()
        idx: dict[int, dict[int, int]] = {}
        for i, (s, d) in enumerate(zip(src_list, dst_list)):
            bucket = idx.get(s)
            if bucket is None:
                bucket = {}
                idx[s] = bucket
            bucket.setdefault(d, i)
        indexes[et] = idx
    return indexes


def extract_attention_weights(model: KGAT, data: HeteroData) -> list[dict[tuple, torch.Tensor]]:
    """Run a full forward and capture per-layer, per-edge-type attention"""
    model.eval()
    x_dict = model.get_initial_embeddings(data)
    edge_index_dict = data.edge_index_dict

    all_layer_attentions: list[dict[tuple, torch.Tensor]] = []

    with torch.no_grad():
        for layer_idx in range(model.n_layers):
            per_dst: dict[str, dict] = {
                nt: {"logits": [], "rel_data": []}
                for nt in x_dict
            }
            cursors: dict[str, int] = {nt: 0 for nt in x_dict}

            for r_idx, et in enumerate(EDGE_TYPES):
                s_type, _, d_type = et
                ei = edge_index_dict.get(et)
                if ei is None or ei.numel() == 0:
                    per_dst[d_type]["rel_data"].append((et, None, None, None))
                    continue
                src_local, dst_local = ei[0], ei[1]
                x_src = x_dict[s_type][src_local]
                x_dst = x_dict[d_type][dst_local]
                Wr = model.W_r[r_idx]
                src_proj = x_src @ Wr
                dst_proj = x_dst @ Wr
                r_e = model.relation_emb.weight[r_idx]
                logit = (src_proj * torch.tanh(dst_proj + r_e)).sum(-1)
                start = cursors[d_type]
                end = start + logit.numel()
                cursors[d_type] = end
                per_dst[d_type]["logits"].append(logit)
                per_dst[d_type]["rel_data"].append((et, (start, end), x_src, dst_local))

            layer_attn: dict[tuple, torch.Tensor] = {}
            out_dict: dict[str, torch.Tensor] = {}
            for nt, buf in per_dst.items():
                n_dst = x_dict[nt].size(0)
                if not buf["logits"]:
                    out_dict[nt] = torch.zeros_like(x_dict[nt])
                    for et, _, _, _ in buf["rel_data"]:
                        layer_attn[et] = torch.zeros(0)
                    continue
                all_logits = torch.cat(buf["logits"], dim=0)
                all_dst = torch.cat(
                    [d for _, sl, _, d in buf["rel_data"] if sl is not None],
                    dim=0,
                )
                attn = softmax(all_logits, all_dst, num_nodes=n_dst)
                first_x_src = next(x for _, sl, x, _ in buf["rel_data"] if sl is not None)
                embed_dim = first_x_src.size(-1)
                out = torch.zeros(
                    (n_dst, embed_dim),
                    device=x_dict[nt].device, dtype=x_dict[nt].dtype,
                )
                for et, sl, x_src_r, dst_r in buf["rel_data"]:
                    if sl is None:
                        layer_attn[et] = torch.zeros(0)
                        continue
                    attn_r = attn[sl[0]:sl[1]]
                    out.index_add_(0, dst_r, x_src_r * attn_r.unsqueeze(-1))
                    layer_attn[et] = attn_r
                out_dict[nt] = out

            all_layer_attentions.append(layer_attn)

            x_dict = {
                nt: F.leaky_relu(model.W_gc[layer_idx](v), negative_slope=model.leaky_relu_slope)
                for nt, v in out_dict.items()
            }
            x_dict = {k: F.normalize(v, p=2, dim=-1) for k, v in x_dict.items()}

    return all_layer_attentions


def _get_edge_attention(
    all_layer_attentions: list[dict[tuple, torch.Tensor]],
    edge_type: tuple,
    src_idx: int,
    dst_idx: int,
    indexes: dict[tuple, dict[int, dict[int, int]]],
) -> float:
    """Mean attention across layers for the (src_idx -> dst_idx) edge of
    `edge_type`. Returns 0.0 if the edge isn't in the graph."""
    bucket = indexes.get(edge_type, {}).get(src_idx)
    if bucket is None:
        return 0.0
    edge_idx = bucket.get(dst_idx)
    if edge_idx is None:
        return 0.0

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
    indexes: dict[tuple, dict[int, dict[int, int]]] | None = None,
) -> list[dict]:
    all_layer_attentions = precomputed_attentions or extract_attention_weights(model, data)
    if indexes is None:
        indexes = build_edge_indexes(data)
    paths: list[dict] = []

    liked_idx = indexes[("user", "liked", "track")]
    perf_idx = indexes[("track", "performed_by", "artist")]
    in_pl_idx = indexes[("track", "in_playlist", "playlist")]

    user_liked = liked_idx.get(user_idx, {})

    if track_idx in user_liked:
        attn = _get_edge_attention(
            all_layer_attentions, ("user", "liked", "track"),
            user_idx, track_idx, indexes,
        )
        paths.append({
            "type": "direct",
            "path": [("user", user_idx), ("track", track_idx)],
            "relation": "liked",
            "attention": attn,
        })

    # User's liked tracks — capped for tractability on power users. 
    # Sorted for deterministic ordering across runs
    user_tracks = sorted(user_liked.keys())[:max_user_tracks]
    if not user_tracks:
        paths.sort(key=lambda p: p["attention"], reverse=True)
        return paths[:top_k]

    # via_artist: user -> track_A -> artist -> target_track.
    target_artist_bucket = perf_idx.get(track_idx)
    if target_artist_bucket:
        target_artist = next(iter(target_artist_bucket))
        for track_a in user_tracks:
            if track_a == track_idx:
                continue
            track_a_bucket = perf_idx.get(track_a)
            if not track_a_bucket or target_artist not in track_a_bucket:
                continue
            a1 = _get_edge_attention(
                all_layer_attentions, ("user", "liked", "track"),
                user_idx, track_a, indexes,
            )
            a2 = _get_edge_attention(
                all_layer_attentions, ("track", "performed_by", "artist"),
                track_a, target_artist, indexes,
            )
            a3 = _get_edge_attention(
                all_layer_attentions, ("artist", "rev_performed_by", "track"),
                target_artist, track_idx, indexes,
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

    target_pl_bucket = in_pl_idx.get(track_idx, {})
    target_playlists = set(target_pl_bucket.keys())
    if target_playlists:
        for track_a in user_tracks:
            if track_a == track_idx:
                continue
            track_a_pl_bucket = in_pl_idx.get(track_a, {})
            shared = list(set(track_a_pl_bucket.keys()) & target_playlists)[:5]
            for pl_idx in shared:
                a1 = _get_edge_attention(
                    all_layer_attentions, ("user", "liked", "track"),
                    user_idx, track_a, indexes,
                )
                a2 = _get_edge_attention(
                    all_layer_attentions, ("track", "in_playlist", "playlist"),
                    track_a, pl_idx, indexes,
                )
                a3 = _get_edge_attention(
                    all_layer_attentions, ("playlist", "rev_in_playlist", "track"),
                    pl_idx, track_idx, indexes,
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
def fidelity_test(model: KGAT, data: HeteroData, n_samples: int = 100,
                  seed: int = 42,
                  precomputed_attentions: list[dict[tuple, torch.Tensor]] | None = None,
                  indexes: dict[tuple, dict[int, dict[int, int]]] | None = None,
                  ) -> tuple[float, float]:
    """For sampled (user, target_track) test edges, find the top explanation
    path, if it's via_artist or via_playlist, drop every edge incident on the
    hub node from the message-passing graph and re-forward. 
    Fidelity = fraction of samples where the masked score is lower than the original.
    `data` is the FULL graph (we need test_mask); message passing happens on
    the train-only graph.

    Returns (fidelity, coverage):
      fidelity = changes / total
      coverage = total / n_samples — fraction of sampled pairs whose top-1
                 explanation was a non-direct path (only those are testable
                 by hub-masking). Reporting both lets the thesis distinguish
                 'masking does change the score' from 'we can't even mask'.
    """
    model.eval()
    device = next(model.parameters()).device
    train_data = make_train_only_graph(data).to(device)

    out = model(train_data)
    user_emb = out["user"]
    track_emb = out["track"]

    if precomputed_attentions is None:
        precomputed_attentions = extract_attention_weights(model, train_data)
    else:
        first_layer = precomputed_attentions[0]
        for et in EDGE_TYPES:
            n_edges = train_data[et].edge_index.shape[1] if et in train_data.edge_types else 0
            attn_len = first_layer.get(et, torch.zeros(0)).numel()
            if n_edges > 0 and attn_len != n_edges:
                raise ValueError(
                    f"precomputed_attentions[{et}] has {attn_len} entries but "
                    f"train_data[{et}] has {n_edges} edges — were they built "
                    f"against different graphs?"
                )

    if indexes is None:
        indexes = build_edge_indexes(train_data)

    test_mask = data["user", "liked", "track"].test_mask
    test_edges = data["user", "liked", "track"].edge_index[:, test_mask]
    n_test = test_edges.shape[1]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    sample_indices = torch.randperm(n_test, generator=gen)[: min(n_samples, n_test)]

    changes = 0
    total = 0
    n_attempted = len(sample_indices)
    t_start = time.perf_counter()

    for i, idx in enumerate(sample_indices.tolist(), 1):
        user_idx = test_edges[0, idx].item()
        track_idx = test_edges[1, idx].item()

        original_score = (user_emb[user_idx] * track_emb[track_idx]).sum().item()

        paths = find_explanation_path(
            model, train_data, user_idx, track_idx, top_k=1,
            precomputed_attentions=precomputed_attentions,
            indexes=indexes,
        )
        if not paths or paths[0]["type"] == "direct":
            elapsed = time.perf_counter() - t_start
            eta = elapsed / i * (n_attempted - i)
            print(f"  [{i}/{n_attempted}] direct/no path — skipped "
                  f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)
            continue

        path = paths[0]["path"]
        if len(path) < 4:
            continue
        mid_node_type, mid_node_idx = path[2]

        # Edge-level masking: drop every edge incident on the hub from the
        # message-passing graph, then re-forward. Stronger than zeroing the
        # hub's initial embedding (which gets reconstructed at layer 1+ from
        # the hub's incoming messages), cutting edges severs the hub entirely.
        masked_edge_index_dict = {}
        for et, ei in train_data.edge_index_dict.items():
            s_type, _, d_type = et
            if s_type != mid_node_type and d_type != mid_node_type:
                masked_edge_index_dict[et] = ei
                continue
            keep = torch.ones(ei.shape[1], dtype=torch.bool, device=ei.device)
            if s_type == mid_node_type:
                keep &= ei[0] != mid_node_idx
            if d_type == mid_node_type:
                keep &= ei[1] != mid_node_idx
            masked_edge_index_dict[et] = ei[:, keep]

        x_dict = model.get_initial_embeddings(train_data)
        layer_outputs = {k: [v] for k, v in x_dict.items()}
        for layer_idx in range(model.n_layers):
            x_dict = model.forward_one_layer(
                x_dict, masked_edge_index_dict, layer_idx,
            )
            x_dict = {k: F.normalize(v, p=2, dim=-1) for k, v in x_dict.items()}
            for k in layer_outputs:
                if k in x_dict:
                    layer_outputs[k].append(x_dict[k])
        masked = {k: torch.cat(layers, dim=-1) for k, layers in layer_outputs.items()}

        masked_score = (masked["user"][user_idx] * masked["track"][track_idx]).sum().item()

        if masked_score < original_score:
            changes += 1
        total += 1
        elapsed = time.perf_counter() - t_start
        eta = elapsed / i * (n_attempted - i)
        print(f"  [{i}/{n_attempted}] {paths[0]['type']:<13} "
              f"orig={original_score:+.3f} masked={masked_score:+.3f} "
              f"{'DROP' if masked_score < original_score else 'keep'} | "
              f"changes={changes}/{total} | "
              f"elapsed {elapsed:.0f}s ETA {eta:.0f}s", flush=True)

    fidelity = changes / total if total > 0 else 0.0
    coverage = total / n_attempted if n_attempted > 0 else 0.0
    return fidelity, coverage


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, default=0)
    parser.add_argument("--track", type=int, default=5)
    parser.add_argument("--fidelity-samples", type=int, default=100)
    args = parser.parse_args()

    cfg = Config()
    device = torch.device("cpu")

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
        mess_dropout=cfg.mess_dropout,
        kge_dim=cfg.kge_dim,
        kge_reg=cfg.kge_reg,
        leaky_relu_slope=cfg.leaky_relu_slope,
    ).to(device)

    init_loader = NeighborLoader(
        train_data, num_neighbors=[3, 3, 3], input_nodes="user", batch_size=8,
    )
    with torch.no_grad():
        model(next(iter(init_loader)).to(device))

    train_data = train_data.to(device)

    checkpoint = cfg.processed_data_dir / "kgat_best.pt"
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint from {checkpoint}")
    else:
        print(f"WARNING: no checkpoint at {checkpoint}; attention weights and "
              f"fidelity numbers will be meaningless on an untrained model.")

    print("\nExtracting per-edge attention weights (one-time)...")
    t_attn = time.perf_counter()
    all_layer_attentions = extract_attention_weights(model, train_data)
    print(f"  done in {time.perf_counter() - t_attn:.1f}s")

    print("Building edge-index lookup tables (one-time)...")
    t_idx = time.perf_counter()
    indexes = build_edge_indexes(train_data)
    print(f"  done in {time.perf_counter() - t_idx:.1f}s")

    print(f"\nExplanation paths for user={args.user} -> track={args.track}:")
    paths = find_explanation_path(
        model, train_data, args.user, args.track, top_k=5,
        precomputed_attentions=all_layer_attentions,
        indexes=indexes,
    )

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
    fidelity, coverage = fidelity_test(
        model, data, n_samples=args.fidelity_samples,
        precomputed_attentions=all_layer_attentions,
        indexes=indexes,
    )
    print(f"Fidelity score: {fidelity:.4f} (coverage: {coverage:.2%})")
    print("  fidelity   = fraction of testable samples where masking the hub "
          "node lowered the score")
    print("  coverage   = fraction of sampled test edges whose top-1 explanation "
          "was multi-hop (only those are testable; direct paths are skipped)")


if __name__ == "__main__":
    main()
