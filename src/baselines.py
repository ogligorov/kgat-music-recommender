"""Baseline recommenders for comparison.

Track-popularity baseline mirrors the sampled-metrics protocol from
evaluate.py so KGAT vs. baseline numbers are directly comparable
(same eligible users, same per-user candidate sets, same train-positive masking).
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.config import Config
from src.evaluate import ndcg_at_k, recall_at_k


@torch.no_grad()
def track_popularity_baseline(
    data: HeteroData,
    top_k_values: list[int],
    split: str = "val",
    seed: int = 42,
    cfg: Config | None = None,
) -> dict:
    """Score each user by global track popularity (likers per track) under the
    same per-user-candidate sampled-metrics protocol as evaluate_model.

    Pass `cfg` to override n_eval_users / n_eval_negatives.
    """
    assert split in ("val", "test")
    if cfg is None:
        cfg = Config()
    rng = np.random.default_rng(seed)

    n_users = data["user"].num_nodes
    n_tracks = data["track"].num_nodes

    liked = data["user", "liked", "track"]
    train_edges = liked.edge_index[:, liked.train_mask]
    eval_mask = liked.val_mask if split == "val" else liked.test_mask
    eval_edges = liked.edge_index[:, eval_mask]

    track_pop = torch.bincount(train_edges[1], minlength=n_tracks).float()

    train_pos_per_user: list[set[int]] = [set() for _ in range(n_users)]
    for u, t in zip(train_edges[0].tolist(), train_edges[1].tolist()):
        train_pos_per_user[u].add(t)
    eval_gt_per_user: list[set[int]] = [set() for _ in range(n_users)]
    for u, t in zip(eval_edges[0].tolist(), eval_edges[1].tolist()):
        eval_gt_per_user[u].add(t)

    eligible = np.array(
        [u for u in range(n_users) if eval_gt_per_user[u]], dtype=np.int64
    )
    if len(eligible) > cfg.n_eval_users:
        eligible = rng.choice(eligible, size=cfg.n_eval_users, replace=False)
    eligible.sort()
    eligible_list = eligible.tolist()

    neg_pool = rng.choice(n_tracks, size=min(cfg.n_eval_negatives, n_tracks), replace=False)
    neg_pool_set: set[int] = {int(t) for t in neg_pool}

    max_k = max(top_k_values)
    ndcg_results: dict[int, list[float]] = {k: [] for k in top_k_values}
    recall_results: dict[int, list[float]] = {k: [] for k in top_k_values}

    for uid in eligible_list:
        gt = eval_gt_per_user[uid]
        train_pos = train_pos_per_user[uid]
        cand = (neg_pool_set - train_pos - gt) | gt
        cand_tensor = torch.tensor(sorted(cand), dtype=torch.long)
        # Seeded jitter (<<1) breaks popularity ties uniformly.
        jitter = torch.from_numpy(rng.uniform(0, 1e-6, size=cand_tensor.numel())).float()
        scores = track_pop[cand_tensor] + jitter

        k_actual = min(max_k, scores.numel())
        top_idx = torch.topk(scores, k_actual).indices
        top_global = cand_tensor[top_idx].numpy()
        for k in top_k_values:
            ndcg_results[k].append(ndcg_at_k(top_global, gt, k))
            recall_results[k].append(recall_at_k(top_global, gt, k))

    return {
        "ndcg": {k: float(np.mean(v)) if v else 0.0 for k, v in ndcg_results.items()},
        "recall": {k: float(np.mean(v)) if v else 0.0 for k, v in recall_results.items()},
    }


@torch.no_grad()
def score_cold_user(data: HeteroData, top_k: int) -> list[int]:
    """Top-K most popular tracks (by train likers). Used by app.py for users
    not in the training-time mapping (cold-start fallback)."""
    train_mask = data["user", "liked", "track"].train_mask
    train_edges = data["user", "liked", "track"].edge_index[:, train_mask]
    n_tracks = data["track"].num_nodes
    track_pop = torch.bincount(train_edges[1], minlength=n_tracks)
    return torch.topk(track_pop, top_k).indices.tolist()


def main():
    """Sanity-check the baseline against val (sampled metrics)."""
    cfg = Config()
    print(f"Loading graph from {cfg.processed_data_dir / 'graph.pt'}...")
    data = torch.load(cfg.processed_data_dir / "graph.pt", weights_only=False)

    print(f"Running track_popularity_baseline (split=val, "
          f"n_eval_users={cfg.n_eval_users}, n_eval_negatives={cfg.n_eval_negatives})...")
    metrics = track_popularity_baseline(data, cfg.top_k, split="val")
    for k in cfg.top_k:
        print(f"  NDCG@{k}: {metrics['ndcg'][k]:.4f}   Recall@{k}: {metrics['recall'][k]:.4f}")

    print("\nCold-user fallback top-10:")
    print(f"  {score_cold_user(data, top_k=10)}")


if __name__ == "__main__":
    main()
