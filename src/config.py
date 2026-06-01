from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # Paths
    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def csv_path(self) -> Path:
        return self.project_root / "data" / "spotify_dataset.csv"

    # K-core filter thresholds
    min_user_likes: int = 5
    min_track_likers: int = 5
    min_playlist_size: int = 5

    # Model
    embed_dim: int = 64
    n_layers: int = 3
    n_heads: int = 4
    # Per-layer message dropout applied AFTER aggregation, BEFORE L2-normalize
    # (paper kgat_paper.py:289 then :292). Replaces the prior `dropout` field
    # which was GATConv internal attention dropout — semantically different
    # from the paper's "message dropout" and dead once GATConv is dropped in
    # Stage B.
    mess_dropout: float = 0.1
    # Slope for the GCN-aggregator non-linearity (paper default 0.2).
    leaky_relu_slope: float = 0.2
    # KGE-side knobs (used only by Stage B; declared here so config is stable
    # across both stages and Stage B doesn't churn this file again).
    kge_dim: int = 64
    kge_reg: float = 1e-5
    batch_size_kg: int = 2048

    # Training
    # 10× higher than paper's 1e-4 (deliberate divergence — our graph is
    # ~17M edges vs the paper's smaller benchmarks, and we want a tractable
    # wall-clock). Prior value 5e-3 was 50× the paper and overshot the BPR
    # minimum past epoch ~10 (val NDCG@10 0.487 → 0.45 collapse).
    lr: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 40
    batch_size: int = 1048
    # Per epoch we cap iteration at this many edges (LinkNeighborLoader reshuffles
    # each iter, so each cap is a fresh random subset of train edges). Decouples
    # per-epoch wall-clock from train-set size; 2M / 6.3M ≈ each edge gets ~3
    # gradient touches per epoch on average.
    edges_per_epoch: int = 1_000_000
    # NeighborLoader: one entry per layer (L=3). Required because full-graph
    # forward OOMs on MPS at our scale (~17M edges). Aggressive [3,3,3] fan-out
    # after [8,5,5] hung past batch 1 — likely sub-graph hub explosion or MPS
    # recompile on shape change between batches.
    num_neighbors: list[int] = field(default_factory=lambda: [3, 3, 3])

    # Evaluation
    top_k: list[int] = field(default_factory=lambda: [10, 20])
    eval_user_batch_size: int = 128       # for the user x track scoring chunks
    eval_inference_batch_size: int = 1024  # for the no-grad NeighborLoader embedding passes
    # Sampled metrics: per eval pass, sample this many users and score each
    # against (their eval positives) + (a shared pool of negative tracks).
    # Cuts eval wall-clock from ~12min to ~30s while keeping NDCG/Recall comparable
    # epoch-to-epoch for early stopping. Final eval should use larger numbers.
    n_eval_users: int = 500
    n_eval_negatives: int = 1000

    # Device
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ))
