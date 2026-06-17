import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from pyvis.network import Network
from torch_geometric.loader import NeighborLoader

from src.baselines import score_cold_user
from src.build_graph import make_train_only_graph
from src.config import Config
from src.evaluate import compute_final_embeddings
from src.explain import build_edge_indexes, extract_attention_weights, find_explanation_path
from src.explain_text import render_paths_bg
from src.genre_bridge import _clean_tags, explain_suno_recs, get_user_genre_profile, query_suno
from src.model import KGAT


@st.cache_resource(show_spinner="Loading graph + model + embeddings (one-time, ~1 min)...")
def load_everything():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(cfg.processed_data_dir / "graph.pt", weights_only=False)
    train_data = make_train_only_graph(data)
    train_data_dev = train_data.to(device)

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
    if not ckpt_path.exists():
        st.error(f"No trained checkpoint at {ckpt_path}. Run `python -m src.train` first.")
        st.stop()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Embed every user and every track once; CPU tensors for the matmul.
    n_users = data["user"].num_nodes
    n_tracks = data["track"].num_nodes
    user_emb = compute_final_embeddings(
        model, train_data, "user", cfg, device,
        input_nodes=torch.arange(n_users),
    )
    track_emb = compute_final_embeddings(
        model, train_data, "track", cfg, device,
        input_nodes=torch.arange(n_tracks),
    )
    artist_emb = compute_final_embeddings(
        model, train_data, "artist", cfg, device,
    )

    # Popularity vector over train edges only.
    liked = data["user", "liked", "track"]
    train_edges = liked.edge_index[:, liked.train_mask]
    track_pop = torch.bincount(train_edges[1], minlength=n_tracks).float()

    train_pos_per_user: list[set[int]] = [set() for _ in range(n_users)]
    for u, t in zip(train_edges[0].tolist(), train_edges[1].tolist()):
        train_pos_per_user[u].add(t)

    with open(cfg.processed_data_dir / "id_mappings.json") as f:
        mappings = json.load(f)
    idx_to_key = {
        nt: {v: k for k, v in mappings[f"{nt}_to_idx"].items()}
        for nt in ("user", "track", "artist", "playlist")
    }

    genres_path = cfg.processed_data_dir / "artist_genres.json"
    artist_genres = json.load(open(genres_path)) if genres_path.exists() else {}
    suno_path = cfg.processed_data_dir / "suno_subset.parquet"
    suno_df = pd.read_parquet(suno_path) if suno_path.exists() else pd.DataFrame()

    with torch.no_grad():
        attentions = extract_attention_weights(model, train_data_dev)
    indexes = build_edge_indexes(train_data_dev)

    return {
        "cfg": cfg,
        "data": data,
        "train_data": train_data,
        "train_data_dev": train_data_dev,
        "model": model,
        "user_emb": user_emb,
        "track_emb": track_emb,
        "artist_emb": artist_emb,
        "track_pop": track_pop,
        "train_pos_per_user": train_pos_per_user,
        "mappings": mappings,
        "idx_to_key": idx_to_key,
        "artist_genres": artist_genres,
        "suno_df": suno_df,
        "attentions": attentions,
        "indexes": indexes,
        "device": device,
    }


def kgat_topk(user_emb: torch.Tensor, track_emb: torch.Tensor,
              user_idx: int, train_pos: set[int], top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    scores = user_emb[user_idx] @ track_emb.T
    if train_pos:
        scores = scores.clone()
        scores[torch.tensor(sorted(train_pos), dtype=torch.long)] = -1e9
    top = torch.topk(scores, k=top_k)
    return top.indices, top.values


def popularity_topk(track_pop: torch.Tensor, train_pos: set[int], top_k: int) -> torch.Tensor:
    scores = track_pop.clone()
    if train_pos:
        scores[torch.tensor(sorted(train_pos), dtype=torch.long)] = -1.0
    return torch.topk(scores, k=top_k).indices


def track_label(track_idx: int, mappings: dict, idx_to_key: dict) -> str:
    key = idx_to_key["track"].get(track_idx)
    if not key:
        return f"track #{track_idx}"
    disp = mappings.get("track_display", {}).get(key, {})
    if disp:
        return f"{disp.get('artistname', '?')} — {disp.get('trackname', '?')}"
    return key


def artist_label(artist_idx: int, mappings: dict, idx_to_key: dict) -> str:
    key = idx_to_key["artist"].get(artist_idx, "")
    return mappings.get("artist_display", {}).get(key, key or f"artist #{artist_idx}")


def playlist_label(playlist_idx: int, mappings: dict, idx_to_key: dict) -> str:
    key = idx_to_key["playlist"].get(playlist_idx, "")
    return mappings.get("playlist_display", {}).get(key, key or f"playlist #{playlist_idx}")


def node_label(node_type: str, node_idx: int, mappings: dict, idx_to_key: dict) -> str:
    if node_type == "track":
        return track_label(node_idx, mappings, idx_to_key)
    if node_type == "artist":
        return artist_label(node_idx, mappings, idx_to_key)
    if node_type == "playlist":
        return playlist_label(node_idx, mappings, idx_to_key)
    if node_type == "user":
        return idx_to_key["user"].get(node_idx, f"user #{node_idx}")
    return f"{node_type} #{node_idx}"


def render_explanation_graph(paths: list[dict], mappings: dict, idx_to_key: dict) -> Network:
    net = Network(height="450px", width="100%", directed=True, notebook=False)
    net.barnes_hut()
    color = {"user": "#4CAF50", "track": "#2196F3", "artist": "#FF9800", "playlist": "#9C27B0"}
    size = {"user": 25, "track": 20, "artist": 18, "playlist": 18}

    added: set[str] = set()
    for path_info in paths:
        path = path_info["path"]
        attn = path_info["attention"]
        for node_type, node_idx in path:
            node_id = f"{node_type}_{node_idx}"
            if node_id in added:
                continue
            added.add(node_id)
            net.add_node(
                node_id,
                label=node_label(node_type, node_idx, mappings, idx_to_key),
                color=color.get(node_type, "#888"),
                size=size.get(node_type, 15),
            )
        for i in range(len(path) - 1):
            src_type, src_idx = path[i]
            dst_type, dst_idx = path[i + 1]
            net.add_edge(
                f"{src_type}_{src_idx}", f"{dst_type}_{dst_idx}",
                value=float(attn), title=f"attention: {attn:.4f}",
            )
    return net


def main():
    st.set_page_config(page_title="KGAT Music Recommender", layout="wide")
    st.title("KGAT Music Recommender")
    st.caption("4-node KG (user / track / artist / playlist) — TransR attention")

    state = load_everything()
    n_users = state["data"]["user"].num_nodes

    st.sidebar.header("User & display options")
    user_idx = st.sidebar.number_input(
        "User index", min_value=0, max_value=n_users - 1, value=0,
    )
    top_k = st.sidebar.slider("Recommendations to show", 5, 20, 10)
    user_key = state["idx_to_key"]["user"].get(user_idx, "?")
    n_train = len(state["train_pos_per_user"][user_idx])
    st.sidebar.markdown(f"**User key:** `{user_key}`")
    st.sidebar.markdown(f"**Train likes:** {n_train}")

    if n_train == 0:
        st.warning(
            "This user has no training likes — KGAT recommendations may be poor. "
            "Falling back to global popularity for context."
        )
        cold_top = score_cold_user(state["data"], top_k=top_k)
        for i, t in enumerate(cold_top, 1):
            st.write(f"**{i}.** {track_label(t, state['mappings'], state['idx_to_key'])}")
    else:
        train_pos = state["train_pos_per_user"][user_idx]
        kgat_idx, kgat_scores = kgat_topk(state["user_emb"], state["track_emb"],
                                           user_idx, train_pos, top_k)
        pop_idx = popularity_topk(state["track_pop"], train_pos, top_k)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("KGAT")
            for i, (idx, score) in enumerate(zip(kgat_idx.tolist(), kgat_scores.tolist()), 1):
                st.write(f"**{i}.** {track_label(idx, state['mappings'], state['idx_to_key'])} "
                         f"  *(score {score:.3f})*")
        with col2:
            st.subheader("Popularity baseline")
            for i, idx in enumerate(pop_idx.tolist(), 1):
                st.write(f"**{i}.** {track_label(idx, state['mappings'], state['idx_to_key'])}")

        st.divider()
        st.subheader("Explanation: why did KGAT pick this track?")
        selected = st.selectbox(
            "Track to explain",
            options=kgat_idx.tolist(),
            format_func=lambda x: track_label(int(x), state["mappings"], state["idx_to_key"]),
        )
        if selected is not None:
            paths = find_explanation_path(
                state["model"], state["train_data_dev"],
                int(user_idx), int(selected), top_k=5,
                precomputed_attentions=state["attentions"],
                indexes=state["indexes"],
            )
            if not paths:
                st.info(
                    "No explanation paths found. The model likely picked this track via "
                    "embedding similarity rather than a direct artist/playlist hop."
                )
            else:
                st.write(f"**Top-{len(paths)} attention paths** "
                         f"(user {user_idx} → {track_label(int(selected), state['mappings'], state['idx_to_key'])}):")
                for i, p in enumerate(paths, 1):
                    path_str = " → ".join(
                        node_label(nt, idx, state["mappings"], state["idx_to_key"])
                        for nt, idx in p["path"]
                    )
                    st.write(f"{i}. **[{p['type']}]** {path_str}  *(attention {p['attention']:.4f})*")

                # Natural-language Bulgarian renderings of the top paths — one sentence
                # per path, template-driven (no LLM). Uses the same id_mappings as the
                # path strings above so labels match.
                sentences = render_paths_bg(paths, state["mappings"], state["idx_to_key"])
                if sentences:
                    st.markdown("**Защо тези препоръки (на български):**")
                    for s in sentences:
                        st.markdown(f"- {s}")

                net = render_explanation_graph(paths, state["mappings"], state["idx_to_key"])
                st.markdown(
                    '<div style="display:flex;gap:1.2em;font-size:0.9em;margin-bottom:0.4em;">'
                    '<span><span style="display:inline-block;width:0.9em;height:0.9em;background:#4CAF50;border-radius:50%;vertical-align:middle;"></span> user</span>'
                    '<span><span style="display:inline-block;width:0.9em;height:0.9em;background:#2196F3;border-radius:50%;vertical-align:middle;"></span> track</span>'
                    '<span><span style="display:inline-block;width:0.9em;height:0.9em;background:#FF9800;border-radius:50%;vertical-align:middle;"></span> artist</span>'
                    '<span><span style="display:inline-block;width:0.9em;height:0.9em;background:#9C27B0;border-radius:50%;vertical-align:middle;"></span> playlist</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                    html_path = Path(tmp.name)
                net.save_graph(str(html_path))
                with open(html_path) as f:
                    st.components.v1.html(f.read(), height=480)
                html_path.unlink(missing_ok=True)

    st.markdown("---")
    st.subheader("AI-Generated Tracks You Might Like")
    if state["suno_df"].empty:
        st.info("Suno dataset not loaded — run scripts/build_suno_subset.py first.")
    else:
        genre_profile, top_artist_names = get_user_genre_profile(
            user_idx,
            state["artist_genres"],
            state["artist_emb"],
            state["user_emb"],
            state["idx_to_key"]["artist"],
        )
        suno_recs, fallback = query_suno(genre_profile, state["suno_df"])
        st.caption(explain_suno_recs(
            genre_profile, top_artist_names,
            state["mappings"]["artist_display"], fallback,
        ))
        for _, row in suno_recs.iterrows():
            col1, col2 = st.columns([1, 4])
            col1.image(row["image_url"], width=80)
            col2.write(f"**{row['title']}**  \n_{_clean_tags(row['metadata_tags'])}_")
            col2.audio(row["audio_url"])


if __name__ == "__main__":
    main()
