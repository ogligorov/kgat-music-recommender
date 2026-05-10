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

    # Dataset
    dataset_url: str = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"

    # Model
    embed_dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1

    # Training
    lr: float = 5e-3
    weight_decay: float = 1e-5
    n_epochs: int = 100

    # Evaluation
    top_k: list[int] = field(default_factory=lambda: [10, 20])

    # Device
    device: str = field(default_factory=lambda: "mps" if torch.backends.mps.is_available() else "cpu")
