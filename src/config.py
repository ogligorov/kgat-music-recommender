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
    # Per-layer message dropout applied AFTER aggregation, BEFORE L2-normalize.
    mess_dropout: float = 0.1
    leaky_relu_slope: float = 0.2
    # KGE-side knobs.
    kge_dim: int = 64
    kge_reg: float = 1e-5
    batch_size_kg: int = 2048

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 40
    batch_size: int = 1048
    # Per-epoch edge cap. LinkNeighborLoader reshuffles each iter, so each cap
    # is a fresh random subset of train edges.
    edges_per_epoch: int = 1_000_000
    # NeighborLoader fan-out, one entry per layer (L=3). Full-graph forward
    # OOMs at our scale (~17M edges).
    num_neighbors: list[int] = field(default_factory=lambda: [3, 3, 3])

    # Evaluation
    top_k: list[int] = field(default_factory=lambda: [10, 20])
    eval_user_batch_size: int = 128
    eval_inference_batch_size: int = 1024  # for the no-grad NeighborLoader embedding passes
    n_eval_users: int = 500
    n_eval_negatives: int = 1000

    # Device
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ))
