"""Training loop for KGAT with BPR loss."""

import time

import torch

from src.config import Config
from src.evaluate import evaluate_model
from src.model import KGAT


def build_user_positive_sets(edge_index: torch.Tensor, n_users: int) -> list[set[int]]:
    """Build per-user sets of positive artist indices from training edges."""
    user_positives: list[set[int]] = [set() for _ in range(n_users)]
    users = edge_index[0].numpy()
    artists = edge_index[1].numpy()
    for u, a in zip(users, artists):
        user_positives[u].add(int(a))
    return user_positives


def sample_negatives(
    user_indices: torch.Tensor,
    n_artists: int,
    user_positives: list[set[int]],
) -> torch.Tensor:
    """Sample one negative artist per user, excluding all positives for that user."""
    neg = torch.randint(0, n_artists, (len(user_indices),))
    for i in range(len(user_indices)):
        uid = user_indices[i].item()
        positives = user_positives[uid]
        while neg[i].item() in positives:
            neg[i] = torch.randint(0, n_artists, (1,))
    return neg


def train_epoch(model: KGAT, data, optimizer: torch.optim.Optimizer, cfg: Config, user_positives: list[set[int]]) -> float:
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
    neg_artists = sample_negatives(pos_users, data["artist"].num_nodes, user_positives)

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

    # Pre-build per-user positive sets for negative sampling
    train_mask = data["user", "listens_to", "artist"].train_mask
    train_edges = data["user", "listens_to", "artist"].edge_index[:, train_mask]
    user_positives = build_user_positive_sets(train_edges, data["user"].num_nodes)

    best_ndcg = 0.0
    patience = 3
    patience_counter = 0
    checkpoint_path = cfg.processed_data_dir / "kgat_best.pt"

    print(f"\nTraining for up to {cfg.n_epochs} epochs (patience={patience})...")
    print("-" * 60)

    for epoch in range(1, cfg.n_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, data, optimizer, cfg, user_positives)
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
                patience_counter += 1

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
