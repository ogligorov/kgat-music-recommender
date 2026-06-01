"""Evaluation metrics: NDCG@K and Recall@K for track-level recommendation.

Sampled-metrics design (chosen to fit the 30-min wall-clock budget):
  - Each eval pass scores `cfg.n_eval_users` randomly sampled users with
    >=1 eval positive against a PER-USER candidate set:
        (user's eval positives) ∪ (shared neg_pool \\ user's train positives \\ user's eval positives)
    The shared neg_pool is `cfg.n_eval_negatives` random tracks. Each user is
    ranked only over their own candidate set, so other users' positives never
    appear as "negatives" for them. This matches BPR/NCF/LightGCN sampled-eval.
  - Final embeddings are computed over the UNION of candidate tracks via a
    single NeighborLoader pass; per-user ranking gathers from that pool.
  - Message passing uses a train-only graph copy so val/test edges never
    contaminate the GNN's view of the user/track nodes.
  - Absolute numbers are not comparable to full-corpus eval, but trends and
    rank-ordering between models are preserved — which is what we need for
    early stopping and for comparing KGAT to the popularity baseline.
"""

import time

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.build_graph import make_train_only_graph
from src.config import Config
from src.model import KGAT


def ndcg_at_k(ranked_items: np.ndarray, ground_truth: set[int], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if int(item) in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_items: np.ndarray, ground_truth: set[int], k: int) -> float:
    hits = sum(1 for item in ranked_items[:k] if int(item) in ground_truth)
    return hits / len(ground_truth) if ground_truth else 0.0


@torch.no_grad()
def compute_final_embeddings(
    model: KGAT,
    data: HeteroData,
    node_type: str,
    cfg: Config,
    device: torch.device,
    input_nodes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sub-graph forwards over the requested seeds and gather their final
    embeddings. If `input_nodes` is None, embeds every node of `node_type`;
    otherwise embeds only the listed global indices, returned in input order.

    `data` should be a train-only graph (see `make_train_only_graph`) so that
    sampled neighborhoods cannot reach val/test edges."""
    if input_nodes is None:
        n_target = data[node_type].num_nodes
        loader_input = node_type
        global_to_row: dict[int, int] | None = None
    else:
        n_target = input_nodes.numel()
        loader_input = (node_type, input_nodes)
        global_to_row = {int(g): i for i, g in enumerate(input_nodes.tolist())}

    embeddings = torch.zeros(n_target, model.out_dim)
    loader = NeighborLoader(
        data,
        num_neighbors=cfg.num_neighbors,
        input_nodes=loader_input,
        batch_size=cfg.eval_inference_batch_size,
        shuffle=False,
    )
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        seed_size = batch[node_type].batch_size
        seed_global = batch[node_type].n_id[:seed_size].cpu()
        seed_emb = out[node_type][:seed_size].cpu()

        if global_to_row is None:
            embeddings[seed_global] = seed_emb
        else:
            for j, g in enumerate(seed_global.tolist()):
                embeddings[global_to_row[g]] = seed_emb[j]

    return embeddings


@torch.no_grad()
def evaluate_model(
    model: KGAT,
    data: HeteroData,
    top_k_values: list[int],
    split: str = "val",
    seed: int = 42,
    verbose: bool = True,
    cfg: Config | None = None,
) -> dict:
    """Sampled-metrics NDCG@K and Recall@K. See module docstring for design.

    Pass `cfg` to override n_eval_users / n_eval_negatives without mutating
    the global default — used by final_eval.py for tighter thesis numbers.
    """
    assert split in ("val", "test"), f"split must be 'val' or 'test', got {split!r}"

    if cfg is None:
        cfg = Config()
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    # PyG's NeighborLoader uses the torch global RNG for neighbor sampling, so
    # without this two runs with the same `seed` produce different embeddings
    # and different NDCG. Seed both numpy (for user/neg_pool sampling) and
    # torch (for sampler) for end-to-end reproducibility.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    timings: dict[str, float] = {}

    # Train-only graph for message passing — val/test edges must not reach the GNN.
    t0 = time.perf_counter()
    train_data = make_train_only_graph(data)
    timings["train_data"] = time.perf_counter() - t0

    n_users = data["user"].num_nodes
    n_tracks = data["track"].num_nodes

    t0 = time.perf_counter()
    liked = data["user", "liked", "track"]
    train_edges = liked.edge_index[:, liked.train_mask]
    eval_mask = liked.val_mask if split == "val" else liked.test_mask
    eval_edges = liked.edge_index[:, eval_mask]

    train_pos_per_user: list[set[int]] = [set() for _ in range(n_users)]
    for u, t in zip(train_edges[0].tolist(), train_edges[1].tolist()):
        train_pos_per_user[u].add(t)
    eval_gt_per_user: list[set[int]] = [set() for _ in range(n_users)]
    for u, t in zip(eval_edges[0].tolist(), eval_edges[1].tolist()):
        eval_gt_per_user[u].add(t)

    # Eligible eval users: those with at least one eval positive.
    eligible = np.array(
        [u for u in range(n_users) if eval_gt_per_user[u]], dtype=np.int64
    )
    if len(eligible) > cfg.n_eval_users:
        eligible = rng.choice(eligible, size=cfg.n_eval_users, replace=False)
    eligible.sort()
    eligible_list = eligible.tolist()

    # Shared neg_pool. Per-user candidate = (gt_u) ∪ (neg_pool \ train_pos_u \ gt_u).
    # Clamp to n_tracks so rng.choice(replace=False) doesn't raise on small datasets.
    neg_pool_size = min(cfg.n_eval_negatives, n_tracks)
    neg_pool = rng.choice(n_tracks, size=neg_pool_size, replace=False)
    neg_pool_set: set[int] = {int(t) for t in neg_pool}

    # Embedding pool = union of every user's candidate set = neg_pool ∪ all eligible gt.
    candidate_set: set[int] = set(neg_pool_set)
    for u in eligible_list:
        candidate_set.update(eval_gt_per_user[u])
    candidate_tracks = sorted(candidate_set)
    track_idx_map = {t: i for i, t in enumerate(candidate_tracks)}

    # Per-user candidate column positions within the union.
    user_cand_cols: list[torch.Tensor] = []
    for uid in eligible_list:
        gt = eval_gt_per_user[uid]
        train_pos = train_pos_per_user[uid]
        cand = (neg_pool_set - train_pos - gt) | gt
        cols = [track_idx_map[t] for t in cand]
        user_cand_cols.append(torch.tensor(cols, dtype=torch.long))
    timings["candidate_setup"] = time.perf_counter() - t0

    user_index = torch.tensor(eligible_list, dtype=torch.long)
    track_index = torch.tensor(candidate_tracks, dtype=torch.long)

    t0 = time.perf_counter()
    user_emb = compute_final_embeddings(
        model, train_data, "user", cfg, device, input_nodes=user_index
    )
    timings["user_emb"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    track_emb = compute_final_embeddings(
        model, train_data, "track", cfg, device, input_nodes=track_index
    )
    timings["track_emb"] = time.perf_counter() - t0

    max_k = max(top_k_values)
    ndcg_results: dict[int, list[float]] = {k: [] for k in top_k_values}
    recall_results: dict[int, list[float]] = {k: [] for k in top_k_values}

    track_emb_t = track_emb.T.contiguous()

    t0 = time.perf_counter()
    for start in range(0, len(eligible_list), cfg.eval_user_batch_size):
        end = min(start + cfg.eval_user_batch_size, len(eligible_list))
        scores = user_emb[start:end] @ track_emb_t  # (B, |union|)

        for i in range(end - start):
            uid = eligible_list[start + i]
            cols = user_cand_cols[start + i]
            user_scores = scores[i, cols]  # (|cand_u|,)
            k_actual = min(max_k, user_scores.numel())
            top_idx_in_cand = torch.topk(user_scores, k_actual).indices
            top_local = cols[top_idx_in_cand].numpy()
            top_global = np.array([candidate_tracks[ti] for ti in top_local])
            gt = eval_gt_per_user[uid]
            for k in top_k_values:
                ndcg_results[k].append(ndcg_at_k(top_global, gt, k))
                recall_results[k].append(recall_at_k(top_global, gt, k))
    timings["scoring"] = time.perf_counter() - t0

    if verbose:
        total = sum(timings.values())
        parts = " | ".join(f"{k}: {v:.1f}s" for k, v in timings.items())
        print(f"  eval breakdown ({total:.1f}s total) | {parts} | "
              f"users: {len(eligible_list)}, |union|: {len(candidate_tracks)}", flush=True)

    if was_training:
        model.train()
    return {
        "ndcg": {k: float(np.mean(v)) if v else 0.0 for k, v in ndcg_results.items()},
        "recall": {k: float(np.mean(v)) if v else 0.0 for k, v in recall_results.items()},
    }


def main():
    """Plumbing check: untrained model + sampled scoring should run quickly.
    Metrics are meaningless without training — this just exercises the path."""
    import time

    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Loading graph from {cfg.processed_data_dir / 'graph.pt'}...")
    data = torch.load(cfg.processed_data_dir / "graph.pt", weights_only=False)

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

    # Initialize lazy GATConv params via one tiny sub-graph forward (train-only graph).
    train_data = make_train_only_graph(data)
    init_loader = NeighborLoader(
        train_data, num_neighbors=[5, 5, 5], input_nodes="user", batch_size=8,
    )
    with torch.no_grad():
        model(next(iter(init_loader)).to(device))

    print(f"Running evaluate_model on untrained model "
          f"(n_eval_users={cfg.n_eval_users}, n_eval_negatives={cfg.n_eval_negatives})...")
    t0 = time.time()
    metrics = evaluate_model(model, data, cfg.top_k, split="val")
    elapsed = time.time() - t0

    print(f"Eval wall-clock: {elapsed:.1f}s")
    for k in cfg.top_k:
        print(f"  NDCG@{k}: {metrics['ndcg'][k]:.4f}   Recall@{k}: {metrics['recall'][k]:.4f}")
    print("Plumbing OK.")


if __name__ == "__main__":
    main()
