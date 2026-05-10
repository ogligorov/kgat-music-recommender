"""Evaluation metrics: NDCG@K and Recall@K for recommendation."""

import torch
import numpy as np
from src.model import KGAT
from torch_geometric.data import HeteroData


def ndcg_at_k(ranked_items: np.ndarray, ground_truth: set[int], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_items: np.ndarray, ground_truth: set[int], k: int) -> float:
    hits = sum(1 for item in ranked_items[:k] if item in ground_truth)
    return hits / len(ground_truth) if ground_truth else 0.0


@torch.no_grad()
def evaluate_model(model: KGAT, data: HeteroData, top_k_values: list[int]) -> dict:
    model.eval()
    out = model(data)
    user_emb = out["user"]
    artist_emb = out["artist"]

    # All scores: (n_users, n_artists)
    scores = user_emb @ artist_emb.T

    # Get test edges
    test_mask = data["user", "listens_to", "artist"].test_mask
    test_edges = data["user", "listens_to", "artist"].edge_index[:, test_mask]

    # Get train edges (to exclude from ranking)
    train_mask = data["user", "listens_to", "artist"].train_mask
    train_edges = data["user", "listens_to", "artist"].edge_index[:, train_mask]

    # Build ground truth per user
    n_users = data["user"].num_nodes
    test_ground_truth: dict[int, set[int]] = {}
    for i in range(test_edges.shape[1]):
        uid = test_edges[0, i].item()
        aid = test_edges[1, i].item()
        test_ground_truth.setdefault(uid, set()).add(aid)

    # Mask out training items from scores
    for i in range(train_edges.shape[1]):
        scores[train_edges[0, i], train_edges[1, i]] = -float("inf")

    max_k = max(top_k_values)
    ndcg_results = {k: [] for k in top_k_values}
    recall_results = {k: [] for k in top_k_values}

    for uid, gt_items in test_ground_truth.items():
        user_scores = scores[uid].cpu().numpy()
        top_items = np.argsort(user_scores)[::-1][:max_k]

        for k in top_k_values:
            ndcg_results[k].append(ndcg_at_k(top_items, gt_items, k))
            recall_results[k].append(recall_at_k(top_items, gt_items, k))

    model.train()
    return {
        "ndcg": {k: np.mean(v) for k, v in ndcg_results.items()},
        "recall": {k: np.mean(v) for k, v in recall_results.items()},
    }
