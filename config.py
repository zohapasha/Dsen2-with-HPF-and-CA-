"""Central configuration for the enhanced DSen2 replication project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class STACConfig:
    """Parameters for querying the Earth Search STAC catalog."""

    url: str = "https://earth-search.aws.element84.com/v1"
    collection: str = "sentinel-2-l2a"
    bbox: Tuple[float, float, float, float] = (12.0, 41.0, 12.6, 41.5)
    datetime_range: str = "2023-01-01/2023-12-31"
    max_cloud_cover: int = 20
    max_items: int = 32


@dataclass(frozen=True)
class ModelConfig:
    """Architectural hyperparameters for the enhanced network."""

    input_channels: int = 10
    output_channels: int = 6
    base_channels: int = 64
    num_residual_blocks: int = 8
    se_reduction: int = 16


@dataclass(frozen=True)
class PatchConfig:
    """Patch geometry used for the multi-resolution Sentinel-2 pipeline."""

    patch_size_10m: int = 128
    patch_size_20m: int = 64
    overlap_10m: int = 32
    overlap_20m: int = 16


@dataclass(frozen=True)
class TrainingConfig:
    """Training and evaluation defaults."""

    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 50
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_split: float = 0.2
    seed: int = 42
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))


GUIDE_BANDS_10M: Tuple[str, ...] = ("B02", "B03", "B04", "B08")
TARGET_BANDS_20M: Tuple[str, ...] = ("B05", "B06", "B07", "B8A", "B11", "B12")
ALL_BANDS: Tuple[str, ...] = GUIDE_BANDS_10M + TARGET_BANDS_20M
RGB_BANDS: Tuple[str, ...] = ("B04", "B03", "B02")

STAC = STACConfig()
MODEL = ModelConfig()
PATCH = PatchConfig()
TRAINING = TrainingConfig()
