"""KGAT model: Knowledge Graph Attention Network for track-level recommendation.

Architecture (Wang et al. KDD 2019):
  - Learnable embeddings per node type (user, track, artist, playlist)
  - N layers of HeteroConv wrapping GATConv (relation-aware attention)
  - Layer aggregation: CONCAT of [x^(0), x^(1), ..., x^(L)] per node, no ReLU
    between layers and no extra bias toward x^(0). Final per-node embedding
    has dim (L+1) * embed_dim.
  - Scoring: dot product of user/track final embeddings.
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv

EDGE_TYPES = [
    ("user", "liked", "track"),
    ("track", "rev_liked", "user"),
    ("track", "in_playlist", "playlist"),
    ("playlist", "rev_in_playlist", "track"),
    ("track", "performed_by", "artist"),
    ("artist", "rev_performed_by", "track"),
]


class KGAT(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_tracks: int,
        n_artists: int,
        n_playlists: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.out_dim = (n_layers + 1) * embed_dim

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.track_emb = nn.Embedding(n_tracks, embed_dim)
        self.artist_emb = nn.Embedding(n_artists, embed_dim)
        self.playlist_emb = nn.Embedding(n_playlists, embed_dim)

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            # Concrete in_channels (embed_dim, embed_dim) avoids GATConv's lazy
            # parameter init path. Lazy `(-1, -1)` materializes weights on the
            # first forward, and on MPS this triggers a Metal kernel recompile
            # whenever a sub-graph shape differs from the one used at init.
            # After Fix #7 every layer's input is exactly embed_dim wide
            # (concat=False keeps each layer's output at embed_dim too), so
            # the dims are known up front.
            conv_dict = {
                edge_type: GATConv(
                    (embed_dim, embed_dim), embed_dim, heads=n_heads, concat=False,
                    dropout=dropout, add_self_loops=False,
                )
                for edge_type in EDGE_TYPES
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.track_emb.weight)
        nn.init.xavier_uniform_(self.artist_emb.weight)
        nn.init.xavier_uniform_(self.playlist_emb.weight)

    def get_initial_embeddings(self, data: HeteroData) -> dict[str, torch.Tensor]:
        # Sub-graph batches from NeighborLoader carry `n_id` (global IDs of the
        # sampled nodes); we must look up embeddings for THOSE rows. Returning
        # `self.X_emb.weight` directly would feed the full-graph table into a
        # forward whose edge_index uses LOCAL sub-graph indices — wrong nodes
        # AND massive wasted compute (~600K rows vs ~15K). The full-graph path
        # (no `n_id`) keeps the .weight shortcut for completeness.
        emb_by_type = {
            "user": self.user_emb,
            "track": self.track_emb,
            "artist": self.artist_emb,
            "playlist": self.playlist_emb,
        }
        out: dict[str, torch.Tensor] = {}
        for nt, emb in emb_by_type.items():
            if hasattr(data[nt], "n_id"):
                out[nt] = emb(data[nt].n_id)
            else:
                out[nt] = emb.weight
        return out

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = self.get_initial_embeddings(data)
        edge_index_dict = data.edge_index_dict

        # Collect per-layer representations: layer 0 (initial) + each conv output.
        layer_outputs: dict[str, list[torch.Tensor]] = {k: [v] for k, v in x_dict.items()}

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            for k in layer_outputs:
                if k in x_dict:
                    layer_outputs[k].append(x_dict[k])

        # KGAT-paper aggregation: concat the L+1 layer reps per node.
        return {k: torch.cat(layers, dim=-1) for k, layers in layer_outputs.items()}

    def score(self, user_emb: torch.Tensor, track_emb: torch.Tensor) -> torch.Tensor:
        return (user_emb * track_emb).sum(dim=-1)

    def bpr_loss(
        self,
        user_emb: torch.Tensor,
        pos_track_emb: torch.Tensor,
        neg_track_emb: torch.Tensor,
    ) -> torch.Tensor:
        pos_scores = self.score(user_emb, pos_track_emb)
        neg_scores = self.score(user_emb, neg_track_emb)
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

