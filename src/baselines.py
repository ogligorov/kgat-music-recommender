"""Baseline recommenders for comparison."""

import numpy as np
import torch
from torch_geometric.data import HeteroData


@torch.no_grad()
def popularity_baseline(data: HeteroData, top_k_values: list[int]) -> dict:
    """Recommend most-listened artists globally (ignoring user preferences)."""
    train_mask = data["user", "listens_to", "artist"].train_mask
    train_edges = data["user", "listens_to", "artist"].edge_index[:, train_mask]

    # Count artist popularity from training set
    n_artists = data["artist"].num_nodes
    artist_counts = torch.bincount(train_edges[1], minlength=n_artists).float()

    # Get test ground truth
    test_mask = data["user", "listens_to", "artist"].test_mask
    test_edges = data["user", "listens_to", "artist"].edge_index[:, test_mask]

    test_ground_truth: dict[int, set[int]] = {}
    for i in range(test_edges.shape[1]):
        uid = test_edges[0, i].item()
        aid = test_edges[1, i].item()
        test_ground_truth.setdefault(uid, set()).add(aid)

    # For each user, rank by popularity (excluding train items)
    max_k = max(top_k_values)
    pop_ranking = torch.argsort(artist_counts, descending=True).numpy()

    from src.evaluate import ndcg_at_k, recall_at_k

    ndcg_results = {k: [] for k in top_k_values}
    recall_results = {k: [] for k in top_k_values}

    # Build train items per user for exclusion
    train_items: dict[int, set[int]] = {}
    for i in range(train_edges.shape[1]):
        uid = train_edges[0, i].item()
        aid = train_edges[1, i].item()
        train_items.setdefault(uid, set()).add(aid)

    for uid, gt_items in test_ground_truth.items():
        user_train = train_items.get(uid, set())
        # Filter out train items from popularity ranking
        filtered = [a for a in pop_ranking if a not in user_train][:max_k]
        ranked = np.array(filtered)

        for k in top_k_values:
            ndcg_results[k].append(ndcg_at_k(ranked, gt_items, k))
            recall_results[k].append(recall_at_k(ranked, gt_items, k))

    return {
        "ndcg": {k: np.mean(v) for k, v in ndcg_results.items()},
        "recall": {k: np.mean(v) for k, v in recall_results.items()},
    }
