"""Training loop for KGAT with BPR loss, using LinkNeighborLoader over (user, liked, track) edges.

Negative sampling and message-passing leak prevention:

  - Negative sampling is delegated to PyG's `NegativeSampling(mode="triplet")`
    so each positive (u, t+) is paired with a randomly-corrupted dst (u, t-)
    that shares the same user. The triplet API exposes aligned `src_index`,
    `dst_pos_index`, `dst_neg_index` — no positional-zip assumption needed.

  - LEAK FIX: `LinkNeighborLoader` does NOT auto-strip supervision edges from
    the message-passing graph (PyG docs: "by default supervision edges in
    `edge_label_index` will not get masked out during sampling"). If we use
    every train edge as both an MP edge AND a supervision seed, the loader
    samples the seed (u, t+) edge as a 1-hop neighbor of u, message passing
    aggregates t+'s embedding directly into u's, and the model learns the
    trivial "I'm connected → score high" rule. BPR loss then collapses to
    ~0.06 from epoch 1 (vs the ~0.69 random-init starting point) and val
    NDCG never escapes the popularity prior.

    Fix: disjoint-split the train edges. 70% become MP-only (the loader's
    edge_index, used purely for message passing), 30% become supervision-only
    (the loader's edge_label_index, never appearing in the MP graph). The
    split is deterministic via a fixed RNG seed so eval is reproducible.
"""

import time

import torch
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.sampler import NegativeSampling

from src.build_graph import make_train_only_graph
from src.config import Config
from src.evaluate import evaluate_model
from src.model import KGAT


def train_epoch(model, loader, optimizer, device, max_batches: int | None = None,
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

        # NegativeSampling(mode="triplet", amount=1) exposes the seed as three
        # aligned local-index tensors. In HETEROGENEOUS mode PyG attaches them
        # to the node stores (NOT the edge store): src_index goes on the source
        # node type, dst_pos_index/dst_neg_index on the dst node type. See
        # torch_geometric/loader/link_loader.py L325-327. The user is the same
        # for the pos and neg pair by construction (dst-only corruption). No
        # edge_label / edge_label_index in triplet mode.
        src_local = batch["user"].src_index
        pos_track_local = batch["track"].dst_pos_index
        neg_track_local = batch["track"].dst_neg_index

        t = time.perf_counter()
        loss = model.bpr_loss(
            out["user"][src_local],
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

        if debug_phase:
            t = time.perf_counter()
            _ = loss.item()
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
            avg_loss = (total_loss_t / n_batches).item()
            print(f"  [{epoch_label}batch {n_batches}/{total}] avg {avg_ms:.0f} ms/batch | "
                  f"loss {avg_loss:.4f} | ETA {remaining:.0f}s", flush=True)

    return (total_loss_t / max(1, n_batches)).item()


def disjoint_mp_sup_split(
    train_mask: torch.Tensor, sup_ratio: float = 0.3, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a train_mask of shape (E,) into (mp_mask, sup_mask), both of
    shape (E,), disjoint, summing to train_mask. `sup_ratio` of train edges
    go to supervision-only. Deterministic via `seed` for reproducible runs."""
    train_idx = train_mask.nonzero(as_tuple=True)[0]
    n_train = train_idx.numel()
    n_sup = int(n_train * sup_ratio)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_train, generator=g)
    sup_idx = train_idx[perm[:n_sup]]
    mp_idx = train_idx[perm[n_sup:]]
    mp_mask = torch.zeros_like(train_mask)
    sup_mask = torch.zeros_like(train_mask)
    mp_mask[mp_idx] = True
    sup_mask[sup_idx] = True
    return mp_mask, sup_mask


def main():
    cfg = Config()
    device = torch.device(cfg.device)

    graph_path = cfg.processed_data_dir / "graph.pt"
    print(f"Loading graph from {graph_path}...", flush=True)
    t0 = time.perf_counter()
    data = torch.load(graph_path, weights_only=False)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    # Disjoint MP/supervision split within the train edges. See module docstring
    # for the motivation. 30% of train edges become supervision-only seeds; the
    # other 70% are the message-passing graph the loader samples neighbors from.
    liked = data["user", "liked", "track"]
    train_mask = liked.train_mask
    mp_mask, sup_mask = disjoint_mp_sup_split(train_mask, sup_ratio=0.3, seed=42)
    n_train = int(train_mask.sum())
    n_mp = int(mp_mask.sum())
    n_sup = int(sup_mask.sum())
    print(f"Disjoint split: train={n_train:,} -> MP={n_mp:,} ({n_mp/n_train:.0%}) | "
          f"sup={n_sup:,} ({n_sup/n_train:.0%})", flush=True)

    # Train-only MP graph using the MP subset of train edges (NOT all train edges).
    # PyG's train_mask is metadata only — passing the full `data` to the loader
    # leaks val/test edges through the GNN, and even with all train edges in MP
    # the supervision seed remains visible to its own batch's neighborhood.
    t0 = time.perf_counter()
    train_data = make_train_only_graph(data, mp_mask=mp_mask)
    print(f"Built train-only MP graph in {time.perf_counter() - t0:.1f}s", flush=True)

    n_users = data["user"].num_nodes
    n_tracks = data["track"].num_nodes
    n_artists = data["artist"].num_nodes
    n_playlists = data["playlist"].num_nodes

    model = KGAT(
        n_users=n_users, n_tracks=n_tracks,
        n_artists=n_artists, n_playlists=n_playlists,
        embed_dim=cfg.embed_dim, n_layers=cfg.n_layers,
        n_heads=cfg.n_heads, mess_dropout=cfg.mess_dropout,
    ).to(device)

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

    # AdamW = Adam with decoupled weight decay (paper's effective regularizer
    # is L2 on parameters, applied separately from the gradient step). Keeping
    # `weight_decay` aligned with the paper's `regs[0]=1e-5`. Single optimizer
    # for both phases — Stage B reuses this when KGE batches are added.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Supervision seeds = train edges NOT in the MP graph. By construction these
    # cannot appear as neighbors when sampling around themselves, so the loader
    # cannot leak the answer into the user/track embeddings before scoring.
    sup_edges = liked.edge_index[:, sup_mask]

    print(f"Constructing LinkNeighborLoader (batch_size={cfg.batch_size}, "
          f"num_neighbors={cfg.num_neighbors}, neg_sampling=triplet/amount=1)...", flush=True)
    t0 = time.perf_counter()
    # neg_sampling=NegativeSampling(mode="triplet", amount=1) corrupts the dst
    # only — each positive (u, t+) yields a (u, t-) with t- a uniform random
    # track. PyG attaches src_index, dst_pos_index, dst_neg_index to the seed
    # storage; user alignment between pos and neg is guaranteed by construction.
    # Residual collisions (t- happens to also be a true positive for u) are
    # negligible at this corpus size — well within BPR's robustness to a small
    # fraction of mislabeled negatives, so we don't reject them.
    loader = LinkNeighborLoader(
        train_data,
        num_neighbors=cfg.num_neighbors,
        edge_label_index=(("user", "liked", "track"), sup_edges),
        batch_size=cfg.batch_size,
        shuffle=True,
        neg_sampling=NegativeSampling(mode="triplet", amount=1),
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
            model, loader, optimizer, device,
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
