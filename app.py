"""Streamlit demo for KGAT Music Recommender."""

import json
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from pyvis.network import Network

from src.config import Config
from src.model import KGAT
from src.explain import find_explanation_path
from src.baselines import popularity_baseline


@st.cache_resource
def load_model_and_data():
    cfg = Config()
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
    with torch.no_grad():
        model(data)

    checkpoint = cfg.processed_data_dir / "kgat_best.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    else:
        st.warning("No trained checkpoint found. Showing untrained model outputs.")

    with open(cfg.processed_data_dir / "id_mappings.json") as f:
        mappings = json.load(f)

    return model, data, mappings, cfg


def get_recommendations(model, data, user_idx, top_k=10):
    model.eval()
    with torch.no_grad():
        out = model(data)
        user_emb = out["user"][user_idx]
        artist_emb = out["artist"]
        scores = user_emb @ artist_emb.T

        # Mask out artists already in training set
        train_mask = data["user", "listens_to", "artist"].train_mask
        train_edges = data["user", "listens_to", "artist"].edge_index[:, train_mask]
        user_mask = train_edges[0] == user_idx
        train_artists = train_edges[1, user_mask]
        scores[train_artists] = -float("inf")

        top_scores, top_indices = torch.topk(scores, top_k)
    return top_indices.numpy(), top_scores.numpy()


def get_popularity_recommendations(data, user_idx, top_k=10):
    train_mask = data["user", "listens_to", "artist"].train_mask
    train_edges = data["user", "listens_to", "artist"].edge_index[:, train_mask]

    n_artists = data["artist"].num_nodes
    counts = torch.zeros(n_artists)
    for i in range(train_edges.shape[1]):
        counts[train_edges[1, i]] += 1

    # Exclude user's training items
    user_mask = train_edges[0] == user_idx
    user_artists = train_edges[1, user_mask]
    counts[user_artists] = -1

    top_indices = torch.argsort(counts, descending=True)[:top_k]
    return top_indices.numpy()


def get_artist_name(mappings, artist_idx):
    orig_id = mappings.get("idx_to_artist", {}).get(str(artist_idx), "")
    return mappings.get("artist_id_to_name", {}).get(str(orig_id), f"Artist {artist_idx}")


def get_tag_name(mappings, tag_idx):
    tag_id = mappings.get("idx_to_tag", {}).get(str(tag_idx), "")
    name = mappings.get("tag_id_to_name", {}).get(str(tag_id), "")
    return name if name else f"Tag {tag_id}"


def render_explanation_graph(paths, mappings):
    net = Network(height="400px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    added_nodes = set()

    for path_info in paths:
        path = path_info["path"]
        attn = path_info["attention"]

        for node_type, node_idx in path:
            node_id = f"{node_type}_{node_idx}"
            if node_id in added_nodes:
                continue
            added_nodes.add(node_id)

            if node_type == "user":
                net.add_node(node_id, label=f"User {node_idx}", color="#4CAF50", size=25)
            elif node_type == "artist":
                name = get_artist_name(mappings, node_idx)
                net.add_node(node_id, label=name, color="#2196F3", size=20)
            elif node_type == "tag":
                name = get_tag_name(mappings, node_idx)
                net.add_node(node_id, label=name, color="#FF9800", size=15)
            elif node_type == "era":
                net.add_node(node_id, label=f"Era {node_idx}", color="#9C27B0", size=15)

        # Add edges
        for i in range(len(path) - 1):
            src_type, src_idx = path[i]
            dst_type, dst_idx = path[i + 1]
            src_id = f"{src_type}_{src_idx}"
            dst_id = f"{dst_type}_{dst_idx}"
            net.add_edge(src_id, dst_id, value=attn, title=f"Attention: {attn:.4f}")

    return net


def main():
    st.set_page_config(page_title="KGAT Music Recommender", layout="wide")
    st.title("KGAT Music Recommender")
    st.caption("Knowledge Graph Attention Network for Explainable Artist Recommendation")

    model, data, mappings, cfg = load_model_and_data()

    # Sidebar: user selection
    n_users = data["user"].num_nodes
    user_idx = st.sidebar.number_input("Select User ID", min_value=0, max_value=n_users - 1, value=0)
    top_k = st.sidebar.slider("Number of recommendations", 5, 20, 10)

    # Main content: two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("KGAT Recommendations")
        rec_indices, rec_scores = get_recommendations(model, data, user_idx, top_k)

        for i, (idx, score) in enumerate(zip(rec_indices, rec_scores)):
            name = get_artist_name(mappings, int(idx))
            st.write(f"**{i+1}.** {name} (score: {score:.3f})")

    with col2:
        st.subheader("Popularity Baseline")
        pop_indices = get_popularity_recommendations(data, user_idx, top_k)

        for i, idx in enumerate(pop_indices):
            name = get_artist_name(mappings, int(idx))
            st.write(f"**{i+1}.** {name}")

    # Explanation section
    st.divider()
    st.subheader("Explanation: Why these recommendations?")

    if len(rec_indices) > 0:
        selected_artist = st.selectbox(
            "Select an artist to explain",
            options=rec_indices,
            format_func=lambda x: get_artist_name(mappings, int(x)),
        )

        if selected_artist is not None:
            paths = find_explanation_path(model, data, user_idx, int(selected_artist), top_k=5)

            if paths:
                st.write(f"**Top attention paths** (user {user_idx} → {get_artist_name(mappings, int(selected_artist))}):")
                for i, p in enumerate(paths):
                    path_str = " → ".join(
                        get_artist_name(mappings, idx) if ntype == "artist"
                        else get_tag_name(mappings, idx) if ntype == "tag"
                        else f"User {idx}" if ntype == "user"
                        else f"Era {idx}"
                        for ntype, idx in p["path"]
                    )
                    st.write(f"  {i+1}. {path_str} (attention: {p['attention']:.4f})")

                # Graph visualization
                net = render_explanation_graph(paths, mappings)
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                    html_path = Path(tmp.name)
                net.save_graph(str(html_path))
                with open(html_path) as f:
                    st.components.v1.html(f.read(), height=450)
                html_path.unlink(missing_ok=True)
            else:
                st.info("No explanation paths found for this recommendation.")


if __name__ == "__main__":
    main()
