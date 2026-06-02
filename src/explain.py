"""Explainability for the 4-node KG (user / track / artist / playlist).

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
the message-passing the model actually saw at training time.
"""

import json
import time

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter, softmax

from src.build_graph import make_train_only_graph
from src.model import EDGE_TYPES, KGAT


def extract_attention_weights(model: KGAT, data: HeteroData) -> list[dict[tuple, torch.Tensor]]:
    """Run a full forward and capture per-layer, per-edge-type attention,
    aligned with `data[edge_type].edge_index` (one scalar per edge).

    Replicates `KGAT.forward_one_layer` step by step so we can split the
    post-softmax attention tensor back per relation. The softmax denominator
    runs across all incoming relations to a destination, so per-relation
    attention numbers are only meaningful AFTER the joint softmax.
    """
    model.eval()
    x_dict = model.get_initial_embeddings(data)
    edge_index_dict = data.edge_index_dict

    all_layer_attentions: list[dict[tuple, torch.Tensor]] = []

    with torch.no_grad():
        for layer_idx in range(model.n_layers):
            # Per-destination buffers; track each relation's slice so we can
            # split the softmax output back.
            per_dst: dict[str, dict[str, list]] = {
                nt: {"logits": [], "msgs": [], "dst": [], "rel_slices": []}
                for nt in x_dict
            }
            cursors: dict[str, int] = {nt: 0 for nt in x_dict}

            for r_idx, et in enumerate(EDGE_TYPES):
                s_type, _, d_type = et
                ei = edge_index_dict.get(et)
                if ei is None or ei.numel() == 0:
                    per_dst[d_type]["rel_slices"].append((et, None))
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
                per_dst[d_type]["msgs"].append(x_src)
                per_dst[d_type]["dst"].append(dst_local)
                per_dst[d_type]["rel_slices"].append((et, (start, end)))

            layer_attn: dict[tuple, torch.Tensor] = {}
            out_dict: dict[str, torch.Tensor] = {}
            for nt, buf in per_dst.items():
                n_dst = x_dict[nt].size(0)
                if not buf["logits"]:
                    out_dict[nt] = torch.zeros_like(x_dict[nt])
                    for et, _ in buf["rel_slices"]:
                        layer_attn[et] = torch.zeros(0)
                    continue
                all_logits = torch.cat(buf["logits"], dim=0)
                all_msgs = torch.cat(buf["msgs"], dim=0)
                all_dst = torch.cat(buf["dst"], dim=0)
                attn = softmax(all_logits, all_dst, num_nodes=n_dst)
                weighted = all_msgs * attn.unsqueeze(-1)
                out_dict[nt] = scatter(weighted, all_dst, dim=0, dim_size=n_dst, reduce="sum")
                for et, sl in buf["rel_slices"]:
                    layer_attn[et] = attn[sl[0]:sl[1]] if sl is not None else torch.zeros(0)

            all_layer_attentions.append(layer_attn)

            # GCN aggregator + dropout (off in eval) + L2-norm — the same chain
            # forward() applies, so x_dict matches what the next layer sees.
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

    # 1-hop direct edge. Use the same cross-layer mean as via_artist /
    # via_playlist so direct-path scores are comparable to multi-hop ones.
    direct_mask = (liked[0] == user_idx) & (liked[1] == track_idx)
    if direct_mask.any():
        attn = _get_edge_attention(
            all_layer_attentions, ("user", "liked", "track"),
            user_idx, track_idx, data,
        )
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
def fidelity_test(model: KGAT, data: HeteroData, n_samples: int = 100,
                  seed: int = 42) -> tuple[float, float]:
    """For sampled (user, target_track) test edges, find the top explanation
    path; if it's via_artist or via_playlist, zero the hub node's initial
    embedding and re-forward. Fidelity = fraction of samples where the
    masked score is lower than the original. `data` is the FULL graph (we
    need test_mask); message passing happens on the train-only graph.

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

    all_layer_attentions = extract_attention_weights(model, train_data)

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
            precomputed_attentions=all_layer_attentions,
        )
        if not paths or paths[0]["type"] == "direct":
            elapsed = time.perf_counter() - t_start
            eta = elapsed / i * (n_attempted - i)
            print(f"  [{i}/{n_attempted}] direct/no path — skipped "
                  f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)
            continue

        # Hub node sits at index 2 in via_artist / via_playlist paths.
        path = paths[0]["path"]
        if len(path) < 4:
            continue
        mid_node_type, mid_node_idx = path[2]

        # Replicate `model.forward` with one row of the hub embedding zeroed.
        # forward_one_layer → mess_dropout (no-op in eval) → L2-norm, concat
        # layers including UN-normalized initial. Must match forward() exactly.
        x_dict = model.get_initial_embeddings(train_data)
        x_dict = {k: v.clone() for k, v in x_dict.items()}
        x_dict[mid_node_type][mid_node_idx] = 0.0

        layer_outputs = {k: [v] for k, v in x_dict.items()}
        for layer_idx in range(model.n_layers):
            x_dict = model.forward_one_layer(
                x_dict, train_data.edge_index_dict, layer_idx,
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, default=0)
    parser.add_argument("--track", type=int, default=5)
    parser.add_argument("--fidelity-samples", type=int, default=100)
    args = parser.parse_args()

    from torch_geometric.loader import NeighborLoader

    from src.config import Config

    cfg = Config()
    # Full-graph attention extraction OOMs on consumer GPUs at this scale (17M
    # edges × 6 relations). CPU is slower but reliable.
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

    # Initialize lazy params via one tiny sub-graph forward. NeighborLoader
    # requires CPU tensors (MPS lacks CSR conversion), so we sample on CPU
    # and move the batch to device.
    init_loader = NeighborLoader(
        train_data, num_neighbors=[3, 3, 3], input_nodes="user", batch_size=8,
    )
    with torch.no_grad():
        model(next(iter(init_loader)).to(device))

    # Move the train-only graph to device for find_explanation_path /
    # fidelity_test forwards.
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
    fidelity, coverage = fidelity_test(model, data, n_samples=args.fidelity_samples)
    print(f"Fidelity score: {fidelity:.4f} (coverage: {coverage:.2%})")
    print("  fidelity   = fraction of testable samples where masking the hub "
          "node lowered the score")
    print("  coverage   = fraction of sampled test edges whose top-1 explanation "
          "was multi-hop (only those are testable; direct paths are skipped)")


if __name__ == "__main__":
    main()
