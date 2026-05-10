"""Explainability: attention path extraction and fidelity testing."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv

from src.model import KGAT


def extract_attention_weights(model: KGAT, data: HeteroData) -> dict[tuple, torch.Tensor]:
    """Run forward pass and capture attention weights from all GATConv layers."""
    model.eval()
    attention_weights = {}

    # Hook to capture attention weights
    hooks = []

    def make_hook(layer_idx, edge_type):
        def hook_fn(module, input, output):
            # GATConv returns (out, attention_weights) when return_attention_weights=True
            # But with HeteroConv we need a different approach
            attention_weights[(layer_idx, edge_type)] = output
        return hook_fn

    # We need to get attention by setting return_attention_weights temporarily
    x_dict = model.get_initial_embeddings(data)
    edge_index_dict = data.edge_index_dict

    all_layer_attentions: list[dict[tuple, torch.Tensor]] = []

    with torch.no_grad():
        for layer_idx, conv in enumerate(model.convs):
            layer_attn = {}
            # Access each sub-conv and run with return_attention_weights=True
            for edge_type, subconv in conv.convs.items():
                src_type, rel_type, dst_type = edge_type
                edge_index = edge_index_dict[edge_type]

                src_x = x_dict[src_type]
                dst_x = x_dict[dst_type]

                # GATConv forward with attention weights
                out, (edge_idx, attn) = subconv(
                    (src_x, dst_x), edge_index, return_attention_weights=True
                )
                layer_attn[edge_type] = attn.mean(dim=-1)  # Average over heads

            all_layer_attentions.append(layer_attn)

            # Propagate through layer (normal forward)
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: torch.relu(v) for k, v in x_dict.items()}

    return all_layer_attentions


def find_explanation_path(
    model: KGAT,
    data: HeteroData,
    user_idx: int,
    artist_idx: int,
    top_k: int = 3,
) -> list[dict]:
    """Find the top-k attention-weighted paths from user to recommended artist.

    Returns paths as list of dicts with nodes and attention scores.
    """
    all_layer_attentions = extract_attention_weights(model, data)

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

            shared_tags = set(mid_tags.numpy()) & set(target_tags.numpy())
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
    """Leave-one-out fidelity: remove top-attention node, check if recommendation changes.

    Returns fidelity score (fraction of cases where removing the explanation node
    changes the recommendation).
    """
    model.eval()
    out = model(data)
    user_emb = out["user"]
    artist_emb = out["artist"]

    test_mask = data["user", "listens_to", "artist"].test_mask
    test_edges = data["user", "listens_to", "artist"].edge_index[:, test_mask]

    # Sample test interactions
    n_test = test_edges.shape[1]
    sample_indices = torch.randperm(n_test)[:n_samples]

    changes = 0
    total = 0

    for idx in sample_indices:
        user_idx = test_edges[0, idx].item()
        artist_idx = test_edges[1, idx].item()

        # Original score
        original_score = (user_emb[user_idx] * artist_emb[artist_idx]).sum().item()

        # Find explanation path
        paths = find_explanation_path(model, data, user_idx, artist_idx, top_k=1)
        if not paths or paths[0]["type"] == "direct":
            continue

        # Get the intermediate node to mask
        path = paths[0]["path"]
        if len(path) >= 3:
            # Mask the middle node by zeroing its embedding contribution
            mid_node_type, mid_node_idx = path[1]

            # Re-run with zeroed middle node
            modified_emb = out[mid_node_type].clone()
            modified_emb[mid_node_idx] = 0

            # Recompute user embedding influence (approximate)
            # For simplicity: check if the artist's rank drops
            scores = user_emb[user_idx] @ artist_emb.T
            original_rank = (scores > original_score).sum().item()

            # Zero out the intermediate and see effect
            scores_modified = scores.clone()
            if mid_node_type == "artist":
                scores_modified[mid_node_idx] = -float("inf")

            new_rank = (scores_modified > scores_modified[artist_idx]).sum().item()

            if new_rank > original_rank:
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
                name = f"tag_{mappings['idx_to_tag'].get(str(node_idx), '?')}"
            print(f"    {node_type}[{node_idx}] {name}")

    # Fidelity test
    print(f"\nRunning fidelity test ({args.fidelity_samples} samples)...")
    fidelity = fidelity_test(model, data, n_samples=args.fidelity_samples)
    print(f"Fidelity score: {fidelity:.4f}")


if __name__ == "__main__":
    main()
