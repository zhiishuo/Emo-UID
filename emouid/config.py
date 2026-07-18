"""Configuration objects for the Emo-UID architecture."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple


@dataclass(frozen=True)
class LossWeights:
    """Top-level weights in Eq. (25) of the revised manuscript."""

    factorization: float = 1.0
    pgu: float = 1.0
    dc: float = 1.0


@dataclass(frozen=True)
class EmoUIDConfig:
    """Architecture configuration for the four evaluated benchmarks."""

    language_input_dim: int
    vision_input_dim: int
    acoustic_input_dim: int
    sentiment_anchors: Tuple[float, ...]

    model_dim: int = 64
    num_heads: int = 8
    shared_transformer_layers: int = 2
    private_transformer_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.1
    language_kernel_size: int = 5
    vision_kernel_size: int = 5
    acoustic_kernel_size: int = 5

    use_bert: bool = False
    bert_model_name: str = "bert-base-uncased"
    fine_tune_bert: bool = True

    prototypes_per_anchor: int = 2
    gram_temperature: float = 0.07
    ordinal_temperature: float = 0.2
    prototype_momentum: float = 0.9
    prototype_neighborhood_weight: float = 1.0
    cps_weight: float = 1.0

    gate_hidden_dim: int = 64
    regression_hidden_dim: int = 128
    max_sequence_length: int = 2048
    epsilon: float = 1e-8
    loss_weights: LossWeights = field(default_factory=LossWeights)

    def __post_init__(self) -> None:
        dimensions = (
            self.language_input_dim,
            self.vision_input_dim,
            self.acoustic_input_dim,
            self.model_dim,
            self.num_heads,
            self.feedforward_dim,
            self.prototypes_per_anchor,
            self.gate_hidden_dim,
            self.regression_hidden_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("All dimensions and counts must be positive.")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        if len(self.sentiment_anchors) < 2:
            raise ValueError("At least two ordered sentiment anchors are required.")
        if tuple(sorted(self.sentiment_anchors)) != self.sentiment_anchors:
            raise ValueError("sentiment_anchors must be strictly ordered.")
        if len(set(self.sentiment_anchors)) != len(self.sentiment_anchors):
            raise ValueError("sentiment_anchors must not contain duplicates.")
        for kernel in (
            self.language_kernel_size,
            self.vision_kernel_size,
            self.acoustic_kernel_size,
        ):
            if kernel <= 0 or kernel % 2 == 0:
                raise ValueError("Temporal kernel sizes must be positive odd integers.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.gram_temperature <= 0.0 or self.ordinal_temperature <= 0.0:
            raise ValueError("Temperatures must be positive.")
        if not 0.0 <= self.prototype_momentum < 1.0:
            raise ValueError("prototype_momentum must be in [0, 1).")
        if self.use_bert and not self.bert_model_name.strip():
            raise ValueError("bert_model_name must be non-empty when use_bert is enabled.")

    @property
    def input_dims(self) -> Dict[str, int]:
        return {
            "language": self.language_input_dim,
            "vision": self.vision_input_dim,
            "acoustic": self.acoustic_input_dim,
        }

    @property
    def kernel_sizes(self) -> Dict[str, int]:
        return {
            "language": self.language_kernel_size,
            "vision": self.vision_kernel_size,
            "acoustic": self.acoustic_kernel_size,
        }


_SEVEN_LEVEL_ANCHORS = tuple(float(value) for value in range(-3, 4))
_SIMS_ANCHORS = tuple(round(-1.0 + 0.2 * index, 1) for index in range(11))


_DATASET_PRESETS = {
    "mosi": dict(
        language_input_dim=768,
        vision_input_dim=20,
        acoustic_input_dim=5,
        sentiment_anchors=_SEVEN_LEVEL_ANCHORS,
        model_dim=50,
        num_heads=10,
        shared_transformer_layers=4,
        private_transformer_layers=4,
        feedforward_dim=200,
        language_kernel_size=5,
        vision_kernel_size=5,
        acoustic_kernel_size=5,
        use_bert=True,
    ),
    "mosei": dict(
        language_input_dim=768,
        vision_input_dim=35,
        acoustic_input_dim=74,
        sentiment_anchors=_SEVEN_LEVEL_ANCHORS,
        model_dim=30,
        num_heads=6,
        shared_transformer_layers=4,
        private_transformer_layers=4,
        feedforward_dim=120,
        language_kernel_size=5,
        vision_kernel_size=3,
        acoustic_kernel_size=1,
        use_bert=True,
    ),
    "chsims": dict(
        language_input_dim=768,
        vision_input_dim=709,
        acoustic_input_dim=33,
        sentiment_anchors=_SIMS_ANCHORS,
        model_dim=30,
        num_heads=6,
        shared_transformer_layers=4,
        private_transformer_layers=4,
        feedforward_dim=120,
    ),
    "chsimsv2": dict(
        language_input_dim=768,
        vision_input_dim=177,
        acoustic_input_dim=25,
        sentiment_anchors=_SIMS_ANCHORS,
        model_dim=30,
        num_heads=6,
        shared_transformer_layers=4,
        private_transformer_layers=4,
        feedforward_dim=120,
    ),
}


def dataset_config(dataset: str, **overrides: object) -> EmoUIDConfig:
    """Return a model preset for one of the four evaluated benchmarks.

    The aliases use only punctuation-insensitive dataset names. Keyword
    overrides are applied through :func:`dataclasses.replace`.
    """

    normalized = (
        dataset.lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .replace(".", "")
    )
    aliases = {
        "cmumosi": "mosi",
        "mosi": "mosi",
        "cmumosei": "mosei",
        "mosei": "mosei",
        "chsims": "chsims",
        "simsv1": "chsims",
        "chsimsv2": "chsimsv2",
        "chsimsv20": "chsimsv2",
        "simsv2": "chsimsv2",
    }
    key = aliases.get(normalized)
    if key is None:
        supported = ", ".join(sorted(_DATASET_PRESETS))
        raise KeyError(f"Unknown dataset '{dataset}'. Supported presets: {supported}.")
    config = EmoUIDConfig(**_DATASET_PRESETS[key])
    return replace(config, **overrides) if overrides else config
