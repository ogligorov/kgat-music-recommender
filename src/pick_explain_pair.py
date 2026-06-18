import torch
from torch_geometric.loader import NeighborLoader

from src.build_graph import make_train_only_graph
from src.config import Config
from src.evaluate import compute_final_embeddings
from src.model import KGAT


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading graph from {cfg.processed_data_dir / 'graph.pt'}...")
    data = torch.load(cfg.processed_data_dir / "graph.pt", weights_only=False)
    train_data = make_train_only_graph(data)

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

    init_loader = NeighborLoader(
        train_data, num_neighbors=[3, 3, 3], input_nodes="user", batch_size=8,
    )
    with torch.no_grad():
        model(next(iter(init_loader)).to(device))

    ckpt_path = cfg.processed_data_dir / "kgat_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path}")

    liked = data["user", "liked", "track"]
    train_edges = liked.edge_index[:, liked.train_mask]
    test_edges = liked.edge_index[:, liked.test_mask]

    n_users = data["user"].num_nodes
    n_tracks = data["track"].num_nodes

    train_pos: dict[int, set[int]] = {}
    for u, t in zip(train_edges[0].tolist(), train_edges[1].tolist()):
        train_pos.setdefault(u, set()).add(t)

    test_pos: dict[int, set[int]] = {}
    for u, t in zip(test_edges[0].tolist(), test_edges[1].tolist()):
        test_pos.setdefault(u, set()).add(t)

    perf = data["track", "performed_by", "artist"].edge_index
    in_pl = data["track", "in_playlist", "playlist"].edge_index
    track_to_artist: dict[int, int] = {}
    for t, a in zip(perf[0].tolist(), perf[1].tolist()):
        track_to_artist[t] = a
    track_to_playlists: dict[int, set[int]] = {}
    for t, p in zip(in_pl[0].tolist(), in_pl[1].tolist()):
        track_to_playlists.setdefault(t, set()).add(p)

    print("Computing user embeddings...")
    user_emb = compute_final_embeddings(
        model, train_data, "user", cfg, device,
        input_nodes=torch.arange(n_users),
    )
    print("Computing track embeddings...")
    track_emb = compute_final_embeddings(
        model, train_data, "track", cfg, device,
        input_nodes=torch.arange(n_tracks),
    )

    eligible = sorted(test_pos.keys())
    print(f"Scanning {len(eligible)} users with held-out positives...")

    chosen = None
    for uid in eligible:
        gt = test_pos[uid]
        tr_pos = train_pos.get(uid, set())

        scores = user_emb[uid] @ track_emb.T
        scores_masked = scores.clone()
        if tr_pos:
            scores_masked[torch.tensor(sorted(tr_pos), dtype=torch.long)] = -1e9

        top20 = torch.topk(scores_masked, k=20).indices.tolist()
        hits = [t for t in top20 if t in gt]
        if not hits:
            continue

        for target in hits:
            artist = track_to_artist.get(target)
            target_pls = track_to_playlists.get(target, set())
            if artist is None or not target_pls:
                continue
            # Caps to first 50 likes so power users with hundreds of likes won't blow up the hub scan
            user_liked_tracks = sorted(tr_pos)[:50]
            shares_artist = any(
                track_to_artist.get(t) == artist
                for t in user_liked_tracks
                if t != target
            )
            shares_playlist = any(
                track_to_playlists.get(t, set()) & target_pls
                for t in user_liked_tracks
                if t != target
            )
            if shares_artist and shares_playlist:
                rank = top20.index(target) + 1
                chosen = (uid, target, rank, len(user_liked_tracks),
                          artist, sorted(target_pls)[:3])
                break
        if chosen is not None:
            break

    if chosen is None:
        print("\nNo user found with all 3 path types active for a top-20 hit.")
        print("Falling back to the first user whose top-20 contains a test hit.")
        for uid in eligible:
            gt = test_pos[uid]
            tr_pos = train_pos.get(uid, set())
            scores = user_emb[uid] @ track_emb.T
            scores_masked = scores.clone()
            if tr_pos:
                scores_masked[torch.tensor(sorted(tr_pos), dtype=torch.long)] = -1e9
            top20 = torch.topk(scores_masked, k=20).indices.tolist()
            hits = [t for t in top20 if t in gt]
            if hits:
                target = hits[0]
                rank = top20.index(target) + 1
                chosen = (uid, target, rank, len(tr_pos), None, [])
                break

    if chosen is None:
        print("No suitable user found at all. Run with --user 0 --track 5 defaults.")
        return

    uid, target, rank, n_train, artist, pls = chosen
    print("\n" + "=" * 60)
    print("Picked explanation pair:")
    print(f"  user={uid}  (has {n_train} train positives)")
    print(f"  track={target}  (held-out test positive)")
    print(f"  rank in user's top-20 (after masking train pos): #{rank}")
    if artist is not None:
        print(f"  artist hub: {artist}; sample shared playlists: {pls}")
    print("=" * 60)
    print(f"\nNext: .venv/bin/python -m src.explain --user {uid} --track {target} --fidelity-samples 100")


if __name__ == "__main__":
    main()
