"""MPS + PyTorch Geometric compatibility smoke test.

Verifies the full stack works before any model code is written:
1. MPS device availability
2. GATConv forward + backward on MPS/CPU
3. HeteroData construction
4. NeighborLoader mini-batch sampling on heterogeneous graph
"""

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv


def check_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        x = torch.randn(4, 4, device="mps")
        _ = x @ x.T
        print("[OK] MPS available and functional")
        return "mps"
    print("[WARN] MPS not available, falling back to CPU")
    return "cpu"


def check_gat_forward_backward(device: str) -> None:
    conv = GATConv(16, 8, heads=4, concat=False).to(device)
    x = torch.randn(20, 16, device=device)
    edge_index = torch.randint(0, 20, (2, 50), device=device)

    out = conv(x, edge_index)
    assert out.shape == (20, 8), f"Unexpected shape: {out.shape}"

    loss = out.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in conv.parameters())
    assert has_grad, "No gradients computed"
    print(f"[OK] GATConv forward + backward on {device}")


def check_hetero_data(device: str) -> HeteroData:
    data = HeteroData()
    data["user"].x = torch.randn(10, 16)
    data["artist"].x = torch.randn(30, 16)
    data["tag"].x = torch.randn(20, 16)
    data["era"].x = torch.randn(8, 16)

    data["user", "listens_to", "artist"].edge_index = torch.randint(0, 10, (1, 40)).repeat(2, 1)
    data["user", "listens_to", "artist"].edge_index[1] = torch.randint(0, 30, (40,))

    data["artist", "tagged_with", "tag"].edge_index = torch.stack([
        torch.randint(0, 30, (50,)),
        torch.randint(0, 20, (50,)),
    ])

    data["artist", "active_in_era", "era"].edge_index = torch.stack([
        torch.randint(0, 30, (30,)),
        torch.randint(0, 8, (30,)),
    ])

    assert len(data.node_types) == 4
    assert len(data.edge_types) == 3
    print(f"[OK] HeteroData created: {len(data.node_types)} node types, {len(data.edge_types)} edge types")
    return data


def check_neighbor_loader(data: HeteroData) -> None:
    loader = NeighborLoader(
        data,
        num_neighbors=[5, 3],
        input_nodes="user",
        batch_size=4,
    )
    batch = next(iter(loader))
    assert "user" in batch.node_types
    assert batch["user"].batch_size == 4
    print(f"[OK] NeighborLoader produces mini-batches (batch_size=4, got {batch['user'].num_nodes} user nodes in subgraph)")


def main():
    print("=" * 50)
    print("KGAT Music Recommender — Smoke Test")
    print("=" * 50)

    device = check_device()
    check_gat_forward_backward(device)
    data = check_hetero_data(device)
    check_neighbor_loader(data)

    print("=" * 50)
    print(f"ALL CHECKS PASSED (device={device})")
    print("=" * 50)


if __name__ == "__main__":
    main()
