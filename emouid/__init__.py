"""Public API for the manuscript-aligned Emo-UID implementation."""

from .config import EmoUIDConfig, LossWeights, TrainingConfig, dataset_config
from .model import EmoUID

__all__ = [
    "EmoUID",
    "EmoUIDConfig",
    "LossWeights",
    "TrainingConfig",
    "dataset_config",
]
