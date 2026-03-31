import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class LSGANConfig:
    seed: int = 51
    n_epochs: int = 300
    batch_size: int = 64
    lr_g: float = 2e-4
    lr_d: float = 1e-4
    b1: float = 0.0
    b2: float = 0.999
    n_cpu: int = field(default_factory=lambda: max(2, (os.cpu_count() or 2)) * 2)
    latent_dim: int = 128
    img_size: int = 64
    channels: int = 3
    sample_interval: int = 500
    fixed_sample_size: int = 64

    real_label: float = 0.95
    fake_label: float = 0.0
    label_noise: float = 0.10
    grad_clip: float = 5.0

    data_root: Path = field(default_factory=lambda: Path("data/anime"))
    zip_name: Path = field(default_factory=lambda: Path("animefaces.zip"))
    data_url: str = "https://storage.googleapis.com/learning-datasets/Resources/anime-faces.zip"
    images_dir: Path = field(default_factory=lambda: Path("images/lsgan"))
    checkpoints_dir: Path = field(default_factory=lambda: Path("checkpoints"))

    @property
    def image_dir(self) -> Path:
        return self.data_root / "images"

    def set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_dirs(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def ensure_dirs(self) -> None:
        self.create_dirs()
