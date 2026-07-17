"""Public API for the manuscript-aligned Emo-UID implementation."""

from .config import EmoUIDConfig, LossWeights, dataset_config
from .model import EmoUID

__all__ = ["EmoUID", "EmoUIDConfig", "LossWeights", "dataset_config"]
