"""KGAT model: Knowledge Graph Attention Network for recommendation.

Architecture:
  - Learnable embeddings per node type (user, artist, tag, era)
  - N layers of HeteroConv wrapping GATConv (relation-aware attention)
  - Layer aggregation: sum of all layer outputs
  - Scoring: dot product of user/artist embeddings
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv


class KGAT(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_artists: int,
        n_tags: int,
        n_eras: int,
        embed_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers

        # Learnable embeddings
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.artist_emb = nn.Embedding(n_artists, embed_dim)
        self.tag_emb = nn.Embedding(n_tags, embed_dim)
        self.era_emb = nn.Embedding(n_eras, embed_dim)

        # Per-layer HeteroConv with GATConv per relation
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv_dict = {}
            for edge_type in [
                ("user", "listens_to", "artist"),
                ("artist", "rev_listens_to", "user"),
                ("artist", "tagged_with", "tag"),
                ("tag", "rev_tagged_with", "artist"),
                ("artist", "active_in_era", "era"),
                ("era", "rev_active_in_era", "artist"),
            ]:
                conv_dict[edge_type] = GATConv(
                    (-1, -1), embed_dim, heads=n_heads, concat=False,
                    dropout=dropout, add_self_loops=False,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.artist_emb.weight)
        nn.init.xavier_uniform_(self.tag_emb.weight)
        nn.init.xavier_uniform_(self.era_emb.weight)

    def get_initial_embeddings(self, data: HeteroData) -> dict[str, torch.Tensor]:
        return {
            "user": self.user_emb.weight[:data["user"].num_nodes],
            "artist": self.artist_emb.weight[:data["artist"].num_nodes],
            "tag": self.tag_emb.weight[:data["tag"].num_nodes],
            "era": self.era_emb.weight[:data["era"].num_nodes],
        }

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = self.get_initial_embeddings(data)
        edge_index_dict = data.edge_index_dict

        # Layer aggregation: sum embeddings from each layer
        out_dict = {k: v.clone() for k, v in x_dict.items()}

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
            for k in out_dict:
                if k in x_dict:
                    out_dict[k] = out_dict[k] + x_dict[k]

        return out_dict

    def score(self, user_emb: torch.Tensor, artist_emb: torch.Tensor) -> torch.Tensor:
        return (user_emb * artist_emb).sum(dim=-1)

    def bpr_loss(
        self,
        user_emb: torch.Tensor,
        pos_artist_emb: torch.Tensor,
        neg_artist_emb: torch.Tensor,
    ) -> torch.Tensor:
        pos_scores = self.score(user_emb, pos_artist_emb)
        neg_scores = self.score(user_emb, neg_artist_emb)
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
