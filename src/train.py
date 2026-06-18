import time

import torch
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.sampler import NegativeSampling

from src.build_graph import make_train_only_graph
from src.config import Config
from src.evaluate import evaluate_model
from src.model import EDGE_TYPES, KGAT


def train_epoch(model, loader, optimizer, device, max_batches: int | None = None,
                heartbeat_every: int = 50, epoch_label: str = "") -> float:
    model.train()
    total_loss_t = torch.zeros((), device=device)
    n_batches = 0
    total = max_batches if max_batches is not None else len(loader)

    t_epoch_start = time.perf_counter()

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


def build_kge_triples(data, train_mask: torch.Tensor) -> list[torch.Tensor]:
    triples: list[torch.Tensor] = []
    liked_et = ("user", "liked", "track")
    rev_liked_et = ("track", "rev_liked", "user")
    for et in EDGE_TYPES:
        ei = data[et].edge_index
        if et == liked_et:
            ei = ei[:, train_mask]
        elif et == rev_liked_et:
            ei = ei[:, train_mask]
        triples.append(ei.contiguous())
    return triples


def train_epoch_kge(
    model, optimizer, kge_triples: list[torch.Tensor], type_sizes: dict[str, int],
    batch_size_kg: int, device, max_batches_per_relation: int | None = None,
    epoch_label: str = "",
) -> float:
    model.train()
    total_loss_t = torch.zeros((), device=device)
    n_batches = 0

    for r_idx, et in enumerate(EDGE_TYPES):
        s_type, _, d_type = et
        ei = kge_triples[r_idx]
        n_edges = ei.shape[1]
        if n_edges == 0:
            continue
        n_t = type_sizes[d_type]

        n_steps = (n_edges + batch_size_kg - 1) // batch_size_kg
        if max_batches_per_relation is not None:
            n_steps = min(n_steps, max_batches_per_relation)

        perm = torch.randperm(n_edges)

        for step in range(n_steps):
            sl = perm[step * batch_size_kg : (step + 1) * batch_size_kg]
            h = ei[0, sl].to(device)
            t_pos = ei[1, sl].to(device)
            t_neg = torch.randint(0, n_t, (sl.numel(),), device=device)

            loss = model.kge_loss_one_relation(r_idx, h, t_pos, t_neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_t = total_loss_t + loss.detach()
            n_batches += 1

    if n_batches == 0:
        return 0.0
    avg = (total_loss_t / n_batches).item()
    print(f"  [{epoch_label}KGE] {n_batches} batches | avg loss {avg:.4f}", flush=True)
    return avg


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from data/processed/kgat_best.pt if it exists. Loads model "
             "weights (and optimizer state + best NDCG + last epoch if the "
             "checkpoint is in the new dict format).",
    )
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(cfg.device)

    graph_path = cfg.processed_data_dir / "graph.pt"
    print(f"Loading graph from {graph_path}...", flush=True)
    t0 = time.perf_counter()
    data = torch.load(graph_path, weights_only=False)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    # Disjoint MP/supervision split within train edges: 70% message-passing, 30% supervision
    liked = data["user", "liked", "track"]
    train_mask = liked.train_mask
    mp_mask, sup_mask = disjoint_mp_sup_split(train_mask, sup_ratio=0.3, seed=42)
    n_train = int(train_mask.sum())
    n_mp = int(mp_mask.sum())
    n_sup = int(sup_mask.sum())
    print(f"Disjoint split: train={n_train:,} -> MP={n_mp:,} ({n_mp/n_train:.0%}) | "
          f"sup={n_sup:,} ({n_sup/n_train:.0%})", flush=True)

    # Train-only MP graph using the MP subset of train edges.
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
        mess_dropout=cfg.mess_dropout,
        kge_dim=cfg.kge_dim, kge_reg=cfg.kge_reg,
        leaky_relu_slope=cfg.leaky_relu_slope,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    sup_edges = liked.edge_index[:, sup_mask]

    print(f"Constructing LinkNeighborLoader (batch_size={cfg.batch_size}, "
          f"num_neighbors={cfg.num_neighbors}, neg_sampling=triplet/amount=1)...", flush=True)
    t0 = time.perf_counter()

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

    kge_triples = build_kge_triples(data, train_mask)
    type_sizes = {"user": n_users, "track": n_tracks,
                  "artist": n_artists, "playlist": n_playlists}
    total_kge_edges = sum(t.shape[1] for t in kge_triples)
    print(f"KGE triples: {total_kge_edges:,} across {len(EDGE_TYPES)} relations "
          f"(batch_size_kg={cfg.batch_size_kg})", flush=True)

    best_ndcg = 0.0
    patience = 3
    patience_counter = 0
    start_epoch = 1
    checkpoint_path = cfg.processed_data_dir / "kgat_best.pt"

    # Resume restores the model weights from a previous run
    resume_legacy = False
    if args.resume:
        if not checkpoint_path.exists():
            print(f"--resume passed but no checkpoint at {checkpoint_path}; "
                  f"starting fresh.", flush=True)
        else:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                best_ndcg = ckpt.get("best_ndcg", 0.0)
                start_epoch = ckpt.get("epoch", 0) + 1
                print(f"Resumed from {checkpoint_path}: epoch {start_epoch}, "
                      f"best NDCG@10 so far {best_ndcg:.4f}", flush=True)
            else:
                model.load_state_dict(ckpt)
                resume_legacy = True
                import shutil
                backup_path = checkpoint_path.with_name(
                    checkpoint_path.stem + "_legacy_backup.pt"
                )
                if not backup_path.exists():
                    shutil.copy(checkpoint_path, backup_path)
                    print(f"Backed up legacy checkpoint to {backup_path}", flush=True)
                print(f"Resumed model weights from {checkpoint_path} "
                      f"(legacy format — running an eval to seed best_ndcg).",
                      flush=True)
                seed_metrics = evaluate_model(model, data, cfg.top_k, verbose=False)
                best_ndcg = seed_metrics["ndcg"][10]
                print(f"  seeded best NDCG@10 = {best_ndcg:.4f} from loaded weights",
                      flush=True)

    print(f"\nTraining for up to {cfg.n_epochs} epochs (patience={patience})...", flush=True)
    print("-" * 60, flush=True)

    # Cold-start: W_r and relation_emb are Xavier-init at epoch 1 — attention
    # is near-uniform until KGE has shaped them. One KGE warmup pass before
    # the first CF epoch gives the attention something to differentiate on.
    if start_epoch == 1 and not args.resume:
        print("KGE warmup before first CF epoch...", flush=True)
        t0 = time.perf_counter()
        train_epoch_kge(
            model, optimizer, kge_triples, type_sizes,
            batch_size_kg=cfg.batch_size_kg, device=device,
            epoch_label="warmup ",
        )
        print(f"  warmup done in {time.perf_counter() - t0:.1f}s", flush=True)
    elif args.resume:
        print(f"Resuming{' (legacy)' if resume_legacy else ''}; skipping KGE warmup.",
              flush=True)

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        t0 = time.perf_counter()
        cf_loss = train_epoch(
            model, loader, optimizer, device,
            max_batches=max_batches_per_epoch,
            epoch_label=f"e{epoch} ",
        )
        cf_time = time.perf_counter() - t0

        t_kge = time.perf_counter()
        kge_loss = train_epoch_kge(
            model, optimizer, kge_triples, type_sizes,
            batch_size_kg=cfg.batch_size_kg, device=device,
            epoch_label=f"e{epoch} ",
        )
        kge_time = time.perf_counter() - t_kge
        train_time = cf_time + kge_time

        if epoch % 5 == 0 or epoch == 1:
            t_eval = time.perf_counter()
            metrics = evaluate_model(model, data, cfg.top_k)
            eval_time = time.perf_counter() - t_eval
            ndcg_10 = metrics["ndcg"][10]
            recall_10 = metrics["recall"][10]
            print(
                f"Epoch {epoch:3d} | CF: {cf_loss:.4f} | KGE: {kge_loss:.4f} | "
                f"NDCG@10: {ndcg_10:.4f} | Recall@10: {recall_10:.4f} | "
                f"Train: {train_time:.1f}s (cf {cf_time:.1f} + kge {kge_time:.1f}) | "
                f"Eval: {eval_time:.1f}s",
                flush=True,
            )
            if ndcg_10 > best_ndcg:
                best_ndcg = ndcg_10
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_ndcg": best_ndcg,
                    "epoch": epoch,
                }, checkpoint_path)
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best NDCG@10: {best_ndcg:.4f})",
                      flush=True)
                break
        else:
            print(f"Epoch {epoch:3d} | CF: {cf_loss:.4f} | KGE: {kge_loss:.4f} | "
                  f"Train: {train_time:.1f}s (cf {cf_time:.1f} + kge {kge_time:.1f})",
                  flush=True)

    print("-" * 60)
    print(f"Training complete. Best NDCG@10: {best_ndcg:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
