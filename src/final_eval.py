"""Final evaluation: load the trained KGAT, score it on the chosen split with
larger sampled-eval sizes, and run the popularity baseline through the same
protocol for a side-by-side comparison.

Both KGAT and baseline use the SAME seed, eligible-user sample, and per-user
candidate sets, so the deltas are apples-to-apples.

Usage:
  .venv/bin/python -m src.final_eval --split test
  .venv/bin/python -m src.final_eval --split val --n-users 1000 --n-negatives 2000
"""

import argparse
import time

import torch
from torch_geometric.loader import NeighborLoader

from src.baselines import track_popularity_baseline
from src.build_graph import make_train_only_graph
from src.config import Config
from src.evaluate import evaluate_model
from src.model import KGAT


def load_trained_model(cfg: Config, data, device: torch.device) -> KGAT:
    """Construct KGAT, run a tiny init forward, then load weights from kgat_best.pt."""
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

    train_data = make_train_only_graph(data)
    init_loader = NeighborLoader(
        train_data, num_neighbors=[3, 3, 3], input_nodes="user", batch_size=8,
    )
    with torch.no_grad():
        model(next(iter(init_loader)).to(device))

    checkpoint_path = cfg.processed_data_dir / "kgat_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train the model first via "
            f"`python -m src.train`."
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        ckpt_epoch = ckpt.get("epoch", "?")
        ckpt_best = ckpt.get("best_ndcg", None)
        meta = f"epoch={ckpt_epoch}"
        if ckpt_best is not None:
            meta += f", train-time best NDCG@10={ckpt_best:.4f}"
        print(f"Loaded dict-format checkpoint ({meta}) from {checkpoint_path}")
    else:
        model.load_state_dict(ckpt)
        print(f"Loaded legacy-format checkpoint from {checkpoint_path}")
    model.eval()
    return model


def print_comparison(kgat: dict, baseline: dict, top_k_values: list[int],
                     split: str, n_users: int, n_negs: int) -> None:
    """Side-by-side table for KGAT vs. popularity baseline. Lift is relative."""
    print(f"\n{'=' * 60}")
    print(f"FINAL EVAL — split={split}, n_eval_users={n_users}, "
          f"n_eval_negatives={n_negs}")
    print("=" * 60)
    header = f"{'Metric':<14} {'KGAT':>10} {'Popularity':>12} {'Lift':>10}"
    print(header)
    print("-" * len(header))
    for k in top_k_values:
        for name in ("ndcg", "recall"):
            k_val = kgat[name][k]
            b_val = baseline[name][k]
            lift = (k_val - b_val) / b_val * 100 if b_val > 0 else float("inf")
            label = f"{name.upper()}@{k}"
            print(f"{label:<14} {k_val:>10.4f} {b_val:>12.4f} "
                  f"{lift:>+9.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="test",
                       help="Which held-out split to evaluate on. (default: test)")
    parser.add_argument("--n-users", type=int, default=2000,
                       help="Sampled eligible users per metric (default: 2000)")
    parser.add_argument("--n-negatives", type=int, default=5000,
                       help="Shared neg-pool size (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="RNG seed; both KGAT and baseline use the same one "
                            "so candidate sets line up exactly. (default: 42)")
    args = parser.parse_args()

    cfg = Config()
    # Override only the two sampling knobs.
    cfg.n_eval_users = args.n_users
    cfg.n_eval_negatives = args.n_negatives
    device = torch.device(cfg.device)

    print(f"Loading graph from {cfg.processed_data_dir / 'graph.pt'}...")
    data = torch.load(cfg.processed_data_dir / "graph.pt", weights_only=False)

    model = load_trained_model(cfg, data, device)

    t0 = time.perf_counter()
    kgat_metrics = evaluate_model(
        model, data, cfg.top_k, split=args.split, seed=args.seed,
        verbose=True, cfg=cfg,
    )
    t_kgat = time.perf_counter() - t0
    print(f"  KGAT eval done in {t_kgat:.1f}s", flush=True)

    t0 = time.perf_counter()
    baseline_metrics = track_popularity_baseline(
        data, cfg.top_k, split=args.split, seed=args.seed, cfg=cfg,
    )
    t_baseline = time.perf_counter() - t0
    print(f"  Baseline eval done in {t_baseline:.1f}s", flush=True)

    print_comparison(
        kgat_metrics, baseline_metrics, cfg.top_k,
        split=args.split, n_users=args.n_users, n_negs=args.n_negatives,
    )


if __name__ == "__main__":
    main()
