"""Smoke test: load the graph, instantiate KGAT, verify a forward path that fits in memory.

Verifies:
1. Device availability (CUDA → MPS → CPU)
2. graph.pt loads with the expected 4 node types / 6 edge types
3. Full-graph forward — attempted; OOM is expected at our scale and triggers fallback
4. NeighborLoader sub-graph forward + backward works on the selected device
5. Reports memory used by the sampled forward (CUDA / MPS only)
"""

import torch
from torch_geometric.loader import NeighborLoader

from src.config import Config
from src.model import KGAT, EDGE_TYPES

EXPECTED_NODE_TYPES = {"user", "track", "artist", "playlist"}


def check_device() -> str:
    if torch.cuda.is_available():
        x = torch.randn(4, 4, device="cuda")
        _ = x @ x.T
        name = torch.cuda.get_device_name(0)
        print(f"[OK] CUDA available and functional ({name})")
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        x = torch.randn(4, 4, device="mps")
        _ = x @ x.T
        print("[OK] MPS available and functional")
        return "mps"
    print("[WARN] No GPU available, falling back to CPU")
    return "cpu"


def check_graph(cfg: Config):
    path = cfg.processed_data_dir / "graph.pt"
    data = torch.load(path, weights_only=False)
    assert set(data.node_types) == EXPECTED_NODE_TYPES, (
        f"Expected {EXPECTED_NODE_TYPES}, got {set(data.node_types)}"
    )
    assert set(data.edge_types) == set(EDGE_TYPES), (
        f"Expected {len(EDGE_TYPES)} edge types, got {len(data.edge_types)}: {data.edge_types}"
    )
    liked = data["user", "liked", "track"]
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        assert hasattr(liked, mask_name), f"Missing {mask_name} on (user, liked, track)"
    n = {nt: data[nt].num_nodes for nt in data.node_types}
    print("[OK] graph.pt schema verified")
    print(f"     nodes: users={n['user']:,}  tracks={n['track']:,}  "
          f"artists={n['artist']:,}  playlists={n['playlist']:,}")
    return data, n


def try_full_graph_forward(model: KGAT, data, device: str) -> bool:
    """Attempt a full-graph forward. Returns True on success, False on OOM.

    OOM is expected at our scale and is the trigger for switching the trainer
    to LinkNeighborLoader.
    """
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    try:
        out = model(data)
        for nt, t in out.items():
            assert not torch.isnan(t).any(), f"{nt} embeddings contain NaN"
        print("[OK] Full-graph forward fits in memory; can train without sampling.")
        return True
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "mps backend out of memory" in msg:
            print("[INFO] Full-graph forward OOMs (expected at our scale).")
            print("       -> Falling back to NeighborLoader-based training.")
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            return False
        raise


def check_neighbor_loader_forward(model: KGAT, data, n: dict, device: str):
    """Sample a sub-graph with NeighborLoader and run forward+backward through it."""
    loader = NeighborLoader(
        data,
        num_neighbors=[10, 10, 5],
        input_nodes="user",
        batch_size=256,
        shuffle=True,
    )
    batch = next(iter(loader)).to(device)

    out = model(batch)
    for nt in EXPECTED_NODE_TYPES:
        assert nt in out, f"Forward missing {nt}"
        assert not torch.isnan(out[nt]).any(), f"{nt} embeddings contain NaN"
    print(f"[OK] NeighborLoader sub-graph forward — non-NaN for all 4 node types")
    print(f"     batch user nodes: {batch['user'].num_nodes:,}, "
          f"track nodes: {batch['track'].num_nodes:,}")

    # Dummy BPR backward over a few real positive (user, track) pairs in this sub-graph
    seed_users = torch.arange(batch["user"].batch_size, device=device)
    pos_track = torch.randint(0, batch["track"].num_nodes, (len(seed_users),), device=device)
    neg_track = torch.randint(0, batch["track"].num_nodes, (len(seed_users),), device=device)
    loss = model.bpr_loss(out["user"][seed_users], out["track"][pos_track], out["track"][neg_track])
    loss.backward()
    print(f"[OK] Backward completed, dummy BPR loss = {loss.item():.4f}")

    if device == "mps":
        cur_gb = torch.mps.current_allocated_memory() / (1024 ** 3)
        recommended_gb = torch.mps.recommended_max_memory() / (1024 ** 3)
        print(f"[INFO] MPS current allocated: {cur_gb:.2f} GB / recommended {recommended_gb:.2f} GB")
    elif device == "cuda":
        cur_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[INFO] CUDA current allocated: {cur_gb:.2f} GB / peak {peak_gb:.2f} GB / total {total_gb:.2f} GB")


def main():
    print("=" * 60)
    print("KGAT — Smoke Test")
    print("=" * 60)

    cfg = Config()
    device = check_device()
    data, n = check_graph(cfg)

    model = KGAT(
        n_users=n["user"], n_tracks=n["track"],
        n_artists=n["artist"], n_playlists=n["playlist"],
        embed_dim=cfg.embed_dim, n_layers=cfg.n_layers,
        n_heads=cfg.n_heads, mess_dropout=cfg.mess_dropout,
    ).to(device)

    full_graph_ok = try_full_graph_forward(model, data.to(device), device)
    if not full_graph_ok:
        # Move data back to CPU for the loader (it samples on CPU then ships to device)
        data = data.to("cpu")
        check_neighbor_loader_forward(model, data, n, device)

    print("=" * 60)
    print(f"ALL CHECKS PASSED (device={device}, "
          f"full_graph={'yes' if full_graph_ok else 'no, using NeighborLoader'})")
    print("=" * 60)


if __name__ == "__main__":
    main()
