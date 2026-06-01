"""Training loop for KGAT with BPR loss, using LinkNeighborLoader over (user, liked, track) edges."""

import random
import time

import torch
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

from src.build_graph import make_train_only_graph
from src.config import Config
from src.evaluate import evaluate_model
from src.model import KGAT


def build_user_positive_sets(edge_index: torch.Tensor, n_users: int) -> list[set[int]]:
    """Per-user sets of positive track indices, used for neg-sample rejection."""
    user_positives: list[set[int]] = [set() for _ in range(n_users)]
    users = edge_index[0].numpy()
    tracks = edge_index[1].numpy()
    for u, t in zip(users, tracks):
        user_positives[u].add(int(t))
    return user_positives


def sample_negatives_in_subgraph(
    user_globals: list[int],
    sub_track_globals: list[int],
    user_positives: list[set[int]],
    max_retries: int = 10,
) -> torch.Tensor:
    """Sample one negative track LOCAL index per seed user, drawing from the sub-graph's
    track nodes and rejecting any track in the user's TRAIN positive set. After
    `max_retries` rejected draws we accept whatever we have — at our scale (>10K
    candidate tracks per sub-graph, hundreds of train positives per user) the
    residual collision rate is negligible and a hard retry cap removes the
    pathological loop risk for power users.

    Pure-Python implementation: per-batch we make ~batch_size `random.randrange`
    calls instead of allocating per-retry torch tensors. Eliminates the
    `tensor.item()` MPS sync points and `torch.randint(...,(1,))` allocator hits
    in the inner retry loop, which dominated wall-clock at batch_size=1048."""
    n_sub_tracks = len(sub_track_globals)
    n_seeds = len(user_globals)
    neg_local = [random.randrange(n_sub_tracks) for _ in range(n_seeds)]
    for i in range(n_seeds):
        positives = user_positives[user_globals[i]]
        for _ in range(max_retries):
            if sub_track_globals[neg_local[i]] not in positives:
                break
            neg_local[i] = random.randrange(n_sub_tracks)
    return torch.tensor(neg_local, dtype=torch.long)


def train_epoch(model, loader, optimizer, user_positives, device, max_batches: int | None = None,
                heartbeat_every: int = 50, epoch_label: str = "") -> float:
    model.train()
    # Accumulate loss on-device. `.item()` is only called at heartbeat (every
    # `heartbeat_every` batches) and at end-of-epoch. Per-batch `.item()`
    # forces an MPS sync that drains the kernel queue and stalls overlap
    # between forward, backward, and the loader's next batch.
    total_loss_t = torch.zeros((), device=device)
    n_batches = 0
    total = max_batches if max_batches is not None else len(loader)

    t_epoch_start = time.perf_counter()

    # Sync between phases ONLY for the first 3 debug batches. MPS dispatches
    # async, so without a sync, phase timings get attributed to whichever
    # later sync point drains the queue.
    is_mps = device.type == "mps"

    def sync():
        if is_mps:
            torch.mps.synchronize()

    # Use explicit iterator so we can time loader.next() separately for the
    # first 3 batches (debugging the 4 s/batch wall-clock).
    loader_iter = iter(loader)
    while True:
        if max_batches is not None and n_batches >= max_batches:
            break

        debug_phase = n_batches < 3
        phases: dict[str, float] = {}

        t = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        if debug_phase:
            phases["loader_next"] = time.perf_counter() - t
            sub_n_users = batch["user"].num_nodes
            sub_n_tracks = batch["track"].num_nodes
            sub_n_artists = batch["artist"].num_nodes
            sub_n_playlists = batch["playlist"].num_nodes
            sub_n_edges = sum(
                batch[et].edge_index.shape[1] for et in batch.edge_types
            )

        t = time.perf_counter()
        batch = batch.to(device)
        if debug_phase:
            sync()
            phases["to_device"] = time.perf_counter() - t

        t = time.perf_counter()
        out = model(batch)
        if debug_phase:
            sync()
            phases["forward"] = time.perf_counter() - t

        edge_label_index = batch["user", "liked", "track"].edge_label_index
        user_local = edge_label_index[0]
        pos_track_local = edge_label_index[1]

        t = time.perf_counter()
        user_globals = batch["user"].n_id[user_local].cpu().tolist()
        sub_track_globals = batch["track"].n_id.cpu().tolist()
        if debug_phase:
            phases["sync_to_cpu"] = time.perf_counter() - t

        t = time.perf_counter()
        neg_track_local = sample_negatives_in_subgraph(
            user_globals, sub_track_globals, user_positives
        ).to(device)
        if debug_phase:
            phases["neg_sample"] = time.perf_counter() - t

        t = time.perf_counter()
        loss = model.bpr_loss(
            out["user"][user_local],
            out["track"][pos_track_local],
            out["track"][neg_track_local],
        )
        if debug_phase:
            sync()
            phases["bpr_loss"] = time.perf_counter() - t

        t = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        if debug_phase:
            sync()
            phases["backward"] = time.perf_counter() - t

        t = time.perf_counter()
        optimizer.step()
        if debug_phase:
            sync()
            phases["opt_step"] = time.perf_counter() - t

        # Accumulate detached loss on-device. Only debug batches sync via
        # `.item()` here, so per-phase totals stay realistic; steady-state
        # batches keep the GPU queue full.
        if debug_phase:
            t = time.perf_counter()
            loss_val = loss.item()
            phases["loss_item"] = time.perf_counter() - t
            total_loss_t = total_loss_t + loss.detach()
        else:
            total_loss_t = total_loss_t + loss.detach()

        n_batches += 1

        if debug_phase:
            parts = ", ".join(f"{k}={v*1000:.0f}ms" for k, v in phases.items())
            total_ms = sum(phases.values()) * 1000
            print(f"  [{epoch_label}batch {n_batches} subgraph: "
                  f"u={sub_n_users}, t={sub_n_tracks}, a={sub_n_artists}, "
                  f"p={sub_n_playlists}, edges={sub_n_edges}]", flush=True)
            print(f"  [{epoch_label}batch {n_batches} phases | total {total_ms:.0f}ms] {parts}",
                  flush=True)
        elif n_batches % heartbeat_every == 0:
            elapsed = time.perf_counter() - t_epoch_start
            avg_ms = (elapsed / n_batches) * 1000
            remaining = (total - n_batches) * (elapsed / n_batches)
            avg_loss = (total_loss_t / n_batches).item()  # one MPS sync per heartbeat
            print(f"  [{epoch_label}batch {n_batches}/{total}] avg {avg_ms:.0f} ms/batch | "
                  f"loss {avg_loss:.4f} | ETA {remaining:.0f}s", flush=True)

    return (total_loss_t / max(1, n_batches)).item()


def main():
    cfg = Config()
    device = torch.device(cfg.device)

    graph_path = cfg.processed_data_dir / "graph.pt"
    print(f"Loading graph from {graph_path}...", flush=True)
    t0 = time.perf_counter()
    data = torch.load(graph_path, weights_only=False)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    # Train-only graph for message passing. PyG's train_mask is metadata only —
    # passing the full `data` to the loader leaks val/test edges through the GNN.
    t0 = time.perf_counter()
    train_data = make_train_only_graph(data)
    print(f"Built train-only graph in {time.perf_counter() - t0:.1f}s", flush=True)

    n_users = data["user"].num_nodes
    n_tracks = data["track"].num_nodes
    n_artists = data["artist"].num_nodes
    n_playlists = data["playlist"].num_nodes

    model = KGAT(
        n_users=n_users, n_tracks=n_tracks,
        n_artists=n_artists, n_playlists=n_playlists,
        embed_dim=cfg.embed_dim, n_layers=cfg.n_layers,
        n_heads=cfg.n_heads, dropout=cfg.dropout,
    ).to(device)

    # Initialize lazy GATConv parameters via one tiny sub-graph forward
    with torch.no_grad():
        init_loader = NeighborLoader(
            train_data, num_neighbors=[3, 3, 3], input_nodes="user", batch_size=8,
        )
        init_batch = next(iter(init_loader)).to(device)
        model(init_batch)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print(f"Device: {cfg.device}, Layers: {cfg.n_layers}, Embed dim: {cfg.embed_dim}, "
          f"Out dim: {model.out_dim}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    liked = data["user", "liked", "track"]
    train_mask = liked.train_mask
    train_edges = liked.edge_index[:, train_mask]
    print(f"Building user-positive sets ({train_edges.shape[1]:,} train edges)...", flush=True)
    t0 = time.perf_counter()
    user_positives = build_user_positive_sets(train_edges, n_users)
    print(f"  built in {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"Constructing LinkNeighborLoader (batch_size={cfg.batch_size}, "
          f"num_neighbors={cfg.num_neighbors})...", flush=True)
    t0 = time.perf_counter()
    loader = LinkNeighborLoader(
        train_data,
        num_neighbors=cfg.num_neighbors,
        edge_label_index=(("user", "liked", "track"), train_edges),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    n_batches_per_epoch = len(loader)
    max_batches_per_epoch = cfg.edges_per_epoch // cfg.batch_size
    actual_batches_per_epoch = min(n_batches_per_epoch, max_batches_per_epoch)
    print(f"  loader built in {time.perf_counter() - t0:.1f}s; "
          f"has {n_batches_per_epoch:,} batches; "
          f"capping to {actual_batches_per_epoch:,} per epoch "
          f"(edges_per_epoch={cfg.edges_per_epoch:,})", flush=True)

    best_ndcg = 0.0
    patience = 3
    patience_counter = 0
    checkpoint_path = cfg.processed_data_dir / "kgat_best.pt"

    print(f"\nTraining for up to {cfg.n_epochs} epochs (patience={patience})...", flush=True)
    print("-" * 60, flush=True)

    for epoch in range(1, cfg.n_epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_epoch(
            model, loader, optimizer, user_positives, device,
            max_batches=max_batches_per_epoch,
            epoch_label=f"e{epoch} ",
        )
        train_time = time.perf_counter() - t0

        if epoch % 5 == 0 or epoch == 1:
            t_eval = time.perf_counter()
            metrics = evaluate_model(model, data, cfg.top_k)
            eval_time = time.perf_counter() - t_eval
            ndcg_10 = metrics["ndcg"][10]
            recall_10 = metrics["recall"][10]
            print(
                f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                f"NDCG@10: {ndcg_10:.4f} | Recall@10: {recall_10:.4f} | "
                f"Train: {train_time:.1f}s | Eval: {eval_time:.1f}s",
                flush=True,
            )
            if ndcg_10 > best_ndcg:
                best_ndcg = ndcg_10
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best NDCG@10: {best_ndcg:.4f})",
                      flush=True)
                break
        else:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Train: {train_time:.1f}s",
                  flush=True)

    print("-" * 60)
    print(f"Training complete. Best NDCG@10: {best_ndcg:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
