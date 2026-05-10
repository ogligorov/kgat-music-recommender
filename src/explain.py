"""Explainability: attention path extraction and fidelity testing."""

import json

import torch
from torch_geometric.data import HeteroData

from src.model import KGAT


def extract_attention_weights(model: KGAT, data: HeteroData) -> list[dict[tuple, torch.Tensor]]:
    """Run forward pass and capture attention weights from all GATConv layers."""
    model.eval()
    x_dict = model.get_initial_embeddings(data)
    edge_index_dict = data.edge_index_dict

    all_layer_attentions: list[dict[tuple, torch.Tensor]] = []

    with torch.no_grad():
        for layer_idx, conv in enumerate(model.convs):
            layer_attn = {}
            out_per_dst: dict[str, list[torch.Tensor]] = {}

            for edge_type, subconv in conv.convs.items():
                src_type, rel_type, dst_type = edge_type
                edge_index = edge_index_dict[edge_type]

                src_x = x_dict[src_type]
                dst_x = x_dict[dst_type]

                out, (edge_idx, attn) = subconv(
                    (src_x, dst_x), edge_index, return_attention_weights=True
                )
                if attn.dim() == 1:
                    layer_attn[edge_type] = attn
                else:
                    layer_attn[edge_type] = attn.mean(dim=-1)
                out_per_dst.setdefault(dst_type, []).append(out)

            all_layer_attentions.append(layer_attn)

            # Aggregate outputs per destination type (matches HeteroConv aggr="sum")
            x_dict = {
                ntype: torch.relu(torch.stack(outs).sum(dim=0))
                for ntype, outs in out_per_dst.items()
            }

    return all_layer_attentions


def find_explanation_path(
    model: KGAT,
    data: HeteroData,
    user_idx: int,
    artist_idx: int,
    top_k: int = 3,
    precomputed_attentions: list[dict[tuple, torch.Tensor]] | None = None,
) -> list[dict]:
    """Find the top-k attention-weighted paths from user to recommended artist.

    Returns paths as list of dicts with nodes and attention scores.
    """
    all_layer_attentions = precomputed_attentions or extract_attention_weights(model, data)

    paths = []

    # 1-hop path: User -> listens_to -> Artist (direct if exists in graph)
    edge_index = data["user", "listens_to", "artist"].edge_index
    mask = (edge_index[0] == user_idx) & (edge_index[1] == artist_idx)
    if mask.any():
        edge_idx = mask.nonzero(as_tuple=True)[0][0].item()
        attn = all_layer_attentions[0][("user", "listens_to", "artist")]
        paths.append({
            "type": "direct",
            "path": [("user", user_idx), ("artist", artist_idx)],
            "relation": "listens_to",
            "attention": attn[edge_idx].item() if edge_idx < len(attn) else 0.0,
        })

    # 2-hop paths: User -> Artist_A -> Tag -> Artist_B (via shared tags)
    # Find artists the user listens to
    user_artist_edges = data["user", "listens_to", "artist"].edge_index
    user_mask = user_artist_edges[0] == user_idx
    user_artists = user_artist_edges[1, user_mask].unique()

    # Find tags of the target artist
    artist_tag_edges = data["artist", "tagged_with", "tag"].edge_index
    target_tag_mask = artist_tag_edges[0] == artist_idx
    target_tags = artist_tag_edges[1, target_tag_mask].unique()

    if len(target_tags) > 0 and len(user_artists) > 0:
        # Find shared tags between user's artists and target artist
        for mid_artist in user_artists[:20]:  # Limit search
            mid_tag_mask = artist_tag_edges[0] == mid_artist.item()
            mid_tags = artist_tag_edges[1, mid_tag_mask]

            shared_tags = set(mid_tags.cpu().tolist()) & set(target_tags.cpu().tolist())
            for tag_idx in list(shared_tags)[:5]:
                # Compute path attention as product of edge attentions
                attn_user_artist = _get_edge_attention(
                    all_layer_attentions, ("user", "listens_to", "artist"),
                    user_idx, mid_artist.item(), data
                )
                attn_artist_tag = _get_edge_attention(
                    all_layer_attentions, ("artist", "tagged_with", "tag"),
                    mid_artist.item(), tag_idx, data
                )
                attn_tag_target = _get_edge_attention(
                    all_layer_attentions, ("tag", "rev_tagged_with", "artist"),
                    tag_idx, artist_idx, data
                )

                path_attention = (attn_user_artist * attn_artist_tag * attn_tag_target) ** (1/3)
                paths.append({
                    "type": "via_tag",
                    "path": [
                        ("user", user_idx),
                        ("artist", mid_artist.item()),
                        ("tag", tag_idx),
                        ("artist", artist_idx),
                    ],
                    "attention": path_attention,
                })

    # Sort by attention and return top-k
    paths.sort(key=lambda p: p["attention"], reverse=True)
    return paths[:top_k]


def _get_edge_attention(
    all_layer_attentions: list[dict],
    edge_type: tuple,
    src_idx: int,
    dst_idx: int,
    data: HeteroData,
) -> float:
    """Get attention weight for a specific edge."""
    edge_index = data[edge_type].edge_index
    mask = (edge_index[0] == src_idx) & (edge_index[1] == dst_idx)
    if not mask.any():
        return 0.0
    edge_idx = mask.nonzero(as_tuple=True)[0][0].item()

    # Average attention across layers
    total_attn = 0.0
    count = 0
    for layer_attn in all_layer_attentions:
        if edge_type in layer_attn:
            attn = layer_attn[edge_type]
            if edge_idx < len(attn):
                total_attn += attn[edge_idx].item()
                count += 1
    return total_attn / count if count > 0 else 0.0


@torch.no_grad()
def fidelity_test(model: KGAT, data: HeteroData, n_samples: int = 100) -> float:
    """Leave-one-out fidelity: zero top-attention node embedding, re-forward, check score change.

    Returns fidelity score (fraction of cases where masking the explanation node
    reduces the recommendation score).
    """
    model.eval()
    out = model(data)
    user_emb = out["user"]
    artist_emb = out["artist"]

    # Precompute attention weights once
    all_layer_attentions = extract_attention_weights(model, data)

    test_mask = data["user", "listens_to", "artist"].test_mask
    test_edges = data["user", "listens_to", "artist"].edge_index[:, test_mask]

    n_test = test_edges.shape[1]
    sample_indices = torch.randperm(n_test)[:n_samples]

    changes = 0
    total = 0

    for idx in sample_indices:
        user_idx = test_edges[0, idx].item()
        artist_idx = test_edges[1, idx].item()

        original_score = (user_emb[user_idx] * artist_emb[artist_idx]).sum().item()

        paths = find_explanation_path(
            model, data, user_idx, artist_idx, top_k=1,
            precomputed_attentions=all_layer_attentions,
        )
        if not paths or paths[0]["type"] == "direct":
            continue

        path = paths[0]["path"]
        if len(path) < 3:
            continue

        mid_node_type, mid_node_idx = path[1]

        # Re-forward with zeroed explanation node embedding
        x_dict_masked = model.get_initial_embeddings(data)
        x_dict_masked[mid_node_type] = x_dict_masked[mid_node_type].clone()
        x_dict_masked[mid_node_type][mid_node_idx] = 0.0

        out_masked = {k: v.clone() for k, v in x_dict_masked.items()}
        for conv in model.convs:
            x_dict_masked = conv(x_dict_masked, data.edge_index_dict)
            x_dict_masked = {k: torch.relu(v) for k, v in x_dict_masked.items()}
            for k in out_masked:
                if k in x_dict_masked:
                    out_masked[k] = out_masked[k] + x_dict_masked[k]

        masked_score = (
            out_masked["user"][user_idx] * out_masked["artist"][artist_idx]
        ).sum().item()

        if masked_score < original_score:
            changes += 1
        total += 1

    return changes / total if total > 0 else 0.0


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, default=0)
    parser.add_argument("--artist", type=int, default=5)
    parser.add_argument("--fidelity-samples", type=int, default=100)
    args = parser.parse_args()

    from src.config import Config
    cfg = Config()

    data = torch.load(cfg.processed_data_dir / "ckg_heterodata.pt", weights_only=False)
    checkpoint = cfg.processed_data_dir / "kgat_best.pt"

    model = KGAT(
        n_users=data["user"].num_nodes,
        n_artists=data["artist"].num_nodes,
        n_tags=data["tag"].num_nodes,
        n_eras=data["era"].num_nodes,
        embed_dim=cfg.embed_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
    )
    with torch.no_grad():
        model(data)

    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        print(f"Loaded checkpoint from {checkpoint}")

    # Extract explanation paths
    print(f"\nExplanation paths for user={args.user} -> artist={args.artist}:")
    paths = find_explanation_path(model, data, args.user, args.artist, top_k=5)

    mappings_path = cfg.processed_data_dir / "id_mappings.json"
    if mappings_path.exists():
        with open(mappings_path) as f:
            mappings = json.load(f)
    else:
        mappings = {}

    for i, path in enumerate(paths):
        print(f"\n  Path {i+1} (attention={path['attention']:.4f}):")
        for node_type, node_idx in path["path"]:
            name = ""
            if node_type == "artist" and "idx_to_artist" in mappings:
                orig_id = mappings["idx_to_artist"].get(str(node_idx), "?")
                name = mappings.get("artist_id_to_name", {}).get(str(orig_id), "")
            elif node_type == "tag" and "idx_to_tag" in mappings:
                tag_id = mappings["idx_to_tag"].get(str(node_idx), "?")
                name = mappings.get("tag_id_to_name", {}).get(str(tag_id), f"tag_{tag_id}")
            print(f"    {node_type}[{node_idx}] {name}")

    # Fidelity test
    print(f"\nRunning fidelity test ({args.fidelity_samples} samples)...")
    fidelity = fidelity_test(model, data, n_samples=args.fidelity_samples)
    print(f"Fidelity score: {fidelity:.4f}")


if __name__ == "__main__":
    main()
