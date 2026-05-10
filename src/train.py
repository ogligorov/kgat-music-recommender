"""Training loop for KGAT with NeighborLoader and BPR loss."""

import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from src.config import Config
from src.evaluate import evaluate_model
from src.model import KGAT


def sample_negatives(user_indices: torch.Tensor, n_artists: int, pos_artist_indices: torch.Tensor) -> torch.Tensor:
    """Sample one negative artist per user (uniform, not in positive set for that user)."""
    neg = torch.randint(0, n_artists, (len(user_indices),))
    # Simple rejection: re-sample collisions (rare given n_artists >> batch)
    collision = neg == pos_artist_indices
    neg[collision] = torch.randint(0, n_artists, (collision.sum().item(),))
    return neg


def train_epoch(model: KGAT, data, optimizer: torch.optim.Optimizer, cfg: Config) -> float:
    model.train()

    # Full-graph forward pass
    out = model(data)
    user_emb = out["user"]
    artist_emb = out["artist"]

    # Get training edges
    train_mask = data["user", "listens_to", "artist"].train_mask
    edge_index = data["user", "listens_to", "artist"].edge_index[:, train_mask]

    # Sample negatives for all training edges
    pos_users = edge_index[0]
    pos_artists = edge_index[1]
    neg_artists = sample_negatives(pos_users, data["artist"].num_nodes, pos_artists)

    # BPR loss over all training edges
    loss = model.bpr_loss(
        user_emb[pos_users],
        artist_emb[pos_artists],
        artist_emb[neg_artists],
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    cfg = Config()
    device = torch.device(cfg.device)

    print(f"Loading CKG from {cfg.processed_data_dir / 'ckg_heterodata.pt'}...")
    data = torch.load(cfg.processed_data_dir / "ckg_heterodata.pt", weights_only=False)

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

    # Initialize lazy parameters
    with torch.no_grad():
        model(data)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {cfg.device}, Layers: {cfg.n_layers}, Embed dim: {cfg.embed_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_ndcg = 0.0
    patience = 10
    patience_counter = 0
    checkpoint_path = cfg.processed_data_dir / "kgat_best.pt"

    print(f"\nTraining for up to {cfg.n_epochs} epochs (patience={patience})...")
    print("-" * 60)

    for epoch in range(1, cfg.n_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, data, optimizer, cfg)
        train_time = time.time() - t0

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate_model(model, data, cfg.top_k)
            ndcg_10 = metrics["ndcg"][10]
            recall_10 = metrics["recall"][10]

            print(
                f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                f"NDCG@10: {ndcg_10:.4f} | Recall@10: {recall_10:.4f} | "
                f"Time: {train_time:.1f}s"
            )

            if ndcg_10 > best_ndcg:
                best_ndcg = ndcg_10
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 5

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best NDCG@10: {best_ndcg:.4f})")
                break
        else:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Time: {train_time:.1f}s")

    print("-" * 60)
    print(f"Training complete. Best NDCG@10: {best_ndcg:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
