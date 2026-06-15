"""KGAT model: Knowledge Graph Attention Network for track-level recommendation.

  - Learnable embeddings per node type (user, track, artist, playlist).
  - Per-relation projection `W_r ∈ ℝ^{embed_dim × kge_dim}` and relation
    vector `r ∈ R^{kge_dim}`. EDGE_TYPES indexes the 6 directed relations;
    forward and reverse get distinct ids.
  - L custom KGAT layers, each running TransR attention per-relation:
      pi(h, r, t) = (W_r e_t)^T tanh(W_r e_h + e_r)
    h = ego (PyG dst), t = neighbor (PyG src). The softmax denominator
    runs over ALL incoming edges to the ego, regardless of relation. The
    aggregated message is the un-projected neighbor embedding e_t.
  - GCN aggregator: e_h^(l) = LeakyReLU(W_gc^(l) · agg(h))
  - Per-layer ordering after the linear: message-dropout -> L2-normalize.
    L2-norm makes dot-product scoring scale-invariant across layers.
  - Layer aggregation: CONCAT of [x^(0), x^(1), ..., x^(L)] per node, with
    x^(0) un-normalized and x^(1..L) normalized. Final per-node embedding
    has dim (L+1) * embed_dim.
  - Phase I (CF): BPR over (u, t+, t-) triplets with dot-product score.
  - Phase II (KGE): TransR triplet loss softplus(S_pos - S_neg) with
      S = ||W_r h + r - W_r t||^2
    `W_r` and `relation_emb` are SHARED between the two phases — KGE loss
    directly shapes the attention weights used in Phase I.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter, softmax

EDGE_TYPES = [
    ("user", "liked", "track"),
    ("track", "rev_liked", "user"),
    ("track", "in_playlist", "playlist"),
    ("playlist", "rev_in_playlist", "track"),
    ("track", "performed_by", "artist"),
    ("artist", "rev_performed_by", "track"),
]

NODE_TYPES = ("user", "track", "artist", "playlist")


class KGAT(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_tracks: int,
        n_artists: int,
        n_playlists: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        mess_dropout: float = 0.1,
        kge_dim: int = 64,
        kge_reg: float = 1e-5,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.kge_dim = kge_dim
        self.kge_reg = kge_reg
        self.leaky_relu_slope = leaky_relu_slope
        self.out_dim = (n_layers + 1) * embed_dim

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.track_emb = nn.Embedding(n_tracks, embed_dim)
        self.artist_emb = nn.Embedding(n_artists, embed_dim)
        self.playlist_emb = nn.Embedding(n_playlists, embed_dim)

        self.n_relations = len(EDGE_TYPES)
        self.W_r = nn.Parameter(torch.empty(self.n_relations, embed_dim, kge_dim))
        self.relation_emb = nn.Embedding(self.n_relations, kge_dim)

        self.W_gc = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)])

        self.mess_dropout = nn.Dropout(mess_dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.track_emb.weight)
        nn.init.xavier_uniform_(self.artist_emb.weight)
        nn.init.xavier_uniform_(self.playlist_emb.weight)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        for lin in self.W_gc:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def get_initial_embeddings(self, data: HeteroData) -> dict[str, torch.Tensor]:
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

    def forward_one_layer(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        layer_idx: int,
    ) -> dict[str, torch.Tensor]:
        per_dst: dict[str, dict[str, list[torch.Tensor]]] = {
            nt: {"logits": [], "msgs": [], "dst": []} for nt in x_dict
        }

        for r_idx, et in enumerate(EDGE_TYPES):
            s_type, _, d_type = et
            ei = edge_index_dict.get(et)
            if ei is None or ei.numel() == 0:
                continue
            src_local, dst_local = ei[0], ei[1]

            x_src = x_dict[s_type][src_local] # [E_r, D] neighbor
            x_dst = x_dict[d_type][dst_local] # [E_r, D] ego
            Wr = self.W_r[r_idx] # [D, K]
            src_proj = x_src @ Wr # W_r · e_t
            dst_proj = x_dst @ Wr  # W_r · e_h
            r_e = self.relation_emb.weight[r_idx] # [K]

            # pi = (W_r e_t)^T · tanh(W_r e_h + e_r)
            logit = (src_proj * torch.tanh(dst_proj + r_e)).sum(-1)  # [E_r]

            per_dst[d_type]["logits"].append(logit)
            per_dst[d_type]["msgs"].append(x_src)
            per_dst[d_type]["dst"].append(dst_local)

        # softmax across all incoming relations terminating at each dst.
        out_dict: dict[str, torch.Tensor] = {}
        for nt, buf in per_dst.items():
            n_dst = x_dict[nt].size(0)
            if not buf["logits"]:
                # No incoming edges of any relation — pass through zero aggregation.
                out_dict[nt] = torch.zeros_like(x_dict[nt])
                continue
            all_logits = torch.cat(buf["logits"], dim=0)
            all_msgs = torch.cat(buf["msgs"], dim=0)
            all_dst = torch.cat(buf["dst"], dim=0)

            attn = softmax(all_logits, all_dst, num_nodes=n_dst)
            weighted = all_msgs * attn.unsqueeze(-1)
            out_dict[nt] = scatter(weighted, all_dst, dim=0, dim_size=n_dst, reduce="sum")

        # GCN aggregator: LeakyReLU(W_gc^(l) · agg)
        out_dict = {
            nt: F.leaky_relu(self.W_gc[layer_idx](v), negative_slope=self.leaky_relu_slope)
            for nt, v in out_dict.items()
        }
        return out_dict

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = self.get_initial_embeddings(data)
        edge_index_dict = data.edge_index_dict

        # Layer 0 (un-normalized) is the base of the concat.
        layer_outputs: dict[str, list[torch.Tensor]] = {k: [v] for k, v in x_dict.items()}

        for layer_idx in range(self.n_layers):
            x_dict = self.forward_one_layer(x_dict, edge_index_dict, layer_idx)
            x_dict = {k: self.mess_dropout(v) for k, v in x_dict.items()}
            x_dict = {k: F.normalize(v, p=2, dim=-1) for k, v in x_dict.items()}
            for k in layer_outputs:
                if k in x_dict:
                    layer_outputs[k].append(x_dict[k])

        return {k: torch.cat(layers, dim=-1) for k, layers in layer_outputs.items()}

    # Phase I: BPR
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

    # Phase II: KGE (TransR)
    def _emb_table(self, type_str: str) -> nn.Embedding:
        return {
            "user": self.user_emb,
            "track": self.track_emb,
            "artist": self.artist_emb,
            "playlist": self.playlist_emb,
        }[type_str]

    def kge_loss_one_relation(
        self,
        r_idx: int,
        h_global: torch.Tensor,
        t_pos_global: torch.Tensor,
        t_neg_global: torch.Tensor,
    ) -> torch.Tensor:
        s_type, _, d_type = EDGE_TYPES[r_idx]
        h = self._emb_table(s_type)(h_global) # [B, D]
        t_pos = self._emb_table(d_type)(t_pos_global) # [B, D]
        t_neg = self._emb_table(d_type)(t_neg_global) # [B, D]

        Wr = self.W_r[r_idx] # [D, K]
        r_e = self.relation_emb.weight[r_idx] # [K]

        h_proj = h @ Wr # [B, K]
        tp_proj = t_pos @ Wr
        tn_proj = t_neg @ Wr

        s_pos = ((h_proj + r_e - tp_proj) ** 2).sum(dim=-1)
        s_neg = ((h_proj + r_e - tn_proj) ** 2).sum(dim=-1)

        loss = F.softplus(s_pos - s_neg).mean()

        reg = (
            h_proj.pow(2).sum()
            + tp_proj.pow(2).sum()
            + tn_proj.pow(2).sum()
            + r_e.pow(2).sum()
        ) / max(1, h.size(0))
        return loss + self.kge_reg * reg

# Test for the model setup
if __name__ == "__main__":
    torch.manual_seed(0)
    data = HeteroData()
    data["user"].n_id = torch.tensor([0])
    data["track"].n_id = torch.tensor([0, 1])
    data["artist"].n_id = torch.tensor([0])
    data["playlist"].n_id = torch.tensor([0])
    data["user", "liked", "track"].edge_index = torch.tensor([[0, 0], [0, 1]])
    data["track", "rev_liked", "user"].edge_index = torch.tensor([[0, 1], [0, 0]])
    data["track", "performed_by", "artist"].edge_index = torch.tensor([[0, 1], [0, 0]])
    data["artist", "rev_performed_by", "track"].edge_index = torch.tensor([[0, 0], [0, 1]])
    data["track", "in_playlist", "playlist"].edge_index = torch.tensor([[0, 1], [0, 0]])
    data["playlist", "rev_in_playlist", "track"].edge_index = torch.tensor([[0, 0], [0, 1]])

    model = KGAT(n_users=1, n_tracks=2, n_artists=1, n_playlists=1, embed_dim=8, n_layers=2)
    out = model(data)
    print("Output shapes:", {k: tuple(v.shape) for k, v in out.items()})
    assert out["user"].shape == (1, model.out_dim)
    assert out["track"].shape == (2, model.out_dim)

    h = torch.tensor([0])
    tp = torch.tensor([0])
    tn = torch.tensor([1])
    kge = model.kge_loss_one_relation(r_idx=0, h_global=h, t_pos_global=tp, t_neg_global=tn)
    print(f"KGE loss (random init, single relation): {kge.item():.4f}  (expected ~0.69)")
