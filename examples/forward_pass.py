"""Minimal architecture-only forward and backward pass."""

import torch

from emouid import EmoUID, EmoUIDConfig


def main() -> None:
    torch.manual_seed(7)
    config = EmoUIDConfig(
        language_input_dim=16,
        vision_input_dim=8,
        acoustic_input_dim=6,
        sentiment_anchors=(-1.0, 0.0, 1.0),
        model_dim=12,
        num_heads=3,
        shared_transformer_layers=1,
        private_transformer_layers=1,
        feedforward_dim=24,
        gate_hidden_dim=12,
        regression_hidden_dim=16,
        max_sequence_length=32,
    )
    model = EmoUID(config)
    output = model(
        language=torch.randn(3, 7, 16),
        vision=torch.randn(3, 5, 8),
        acoustic=torch.randn(3, 6, 6),
        labels=torch.tensor([-0.8, 0.1, 0.9]),
    )
    output["losses"]["total"].backward()

    print("prediction shape:", tuple(output["prediction"].shape))
    print("total loss:", round(output["losses"]["total"].item(), 6))
    print(
        "gate means:",
        {
            modality: round(gate.mean().item(), 4)
            for modality, gate in output["gates"].items()
        },
    )


if __name__ == "__main__":
    main()
