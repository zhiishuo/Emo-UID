from dataclasses import replace

import pytest
import torch

from emouid import EmoUID, EmoUIDConfig, dataset_config


@pytest.fixture
def config() -> EmoUIDConfig:
    return EmoUIDConfig(
        language_input_dim=8,
        vision_input_dim=6,
        acoustic_input_dim=4,
        sentiment_anchors=(-1.0, 0.0, 1.0),
        model_dim=12,
        num_heads=3,
        shared_transformer_layers=1,
        private_transformer_layers=1,
        feedforward_dim=24,
        dropout=0.0,
        prototypes_per_anchor=2,
        gate_hidden_dim=12,
        regression_hidden_dim=16,
        max_sequence_length=32,
    )


def make_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    return {
        "language": torch.randn(4, 7, 8),
        "vision": torch.randn(4, 5, 6),
        "acoustic": torch.randn(4, 6, 4),
        "labels": torch.tensor([-0.9, -0.2, 0.3, 0.8]),
    }


def test_forward_matches_architecture_contract(config: EmoUIDConfig) -> None:
    model = EmoUID(config)
    output = model(**make_batch())

    assert output["prediction"].shape == (4, 1)
    assert output["ordinal_target"].shape == (4, 3)
    assert torch.allclose(
        output["ordinal_target"].sum(dim=-1), torch.ones(4), atol=1e-6
    )
    for probabilities in output["auxiliary_probabilities"].values():
        assert probabilities.shape == (4, 3)
        assert torch.allclose(probabilities.sum(dim=-1), torch.ones(4), atol=1e-6)
    for gate in output["gates"].values():
        assert gate.shape == (4, 1)
        assert torch.all((gate > 0.0) & (gate < 1.0))
    assert set(output["losses"]) == {
        "task",
        "orthogonality",
        "reconstruction",
        "factorization",
        "gram",
        "sop",
        "pgu",
        "ordinal",
        "cps",
        "dc",
        "total",
    }
    assert all(torch.isfinite(value) for value in output["losses"].values())


def test_complete_objective_backpropagates_to_core_paths(config: EmoUIDConfig) -> None:
    model = EmoUID(config)
    output = model(**make_batch())
    output["losses"]["total"].backward()

    checked_parameters = {
        "shared factorization": model.factorization.shared_encoder.layers[0].weight,
        "private factorization": model.factorization.private_encoders[
            "language"
        ].layers[0].weight,
        "ordinal evidence": model.diversity_classification.private_heads[
            "vision"
        ].weight,
        "reliability gate": model.reliability_fusion.gates["acoustic"][0].weight,
        "regression": model.regression_head[-1].weight,
    }
    for name, parameter in checked_parameters.items():
        assert parameter.grad is not None, f"No gradient reached {name}."
        assert torch.isfinite(parameter.grad).all(), f"Invalid gradient in {name}."


def test_grouped_objective_matches_equation_25(config: EmoUIDConfig) -> None:
    model = EmoUID(config)
    losses = model(**make_batch())["losses"]
    weights = config.loss_weights
    expected = (
        losses["task"]
        + weights.factorization * losses["factorization"]
        + weights.pgu * losses["pgu"]
        + weights.dc * losses["dc"]
    )
    assert torch.allclose(
        losses["factorization"], losses["orthogonality"] + losses["reconstruction"]
    )
    assert torch.allclose(losses["pgu"], losses["gram"] + losses["sop"])
    assert torch.allclose(losses["dc"], losses["ordinal"] + config.cps_weight * losses["cps"])
    assert torch.allclose(losses["total"], expected)


def test_module_execution_order_matches_revised_method(config: EmoUIDConfig) -> None:
    model = EmoUID(config)
    order: list[str] = []
    hooks = [
        model.pgu.register_forward_hook(lambda *_: order.append("pgu")),
        model.shared_enhancer.register_forward_hook(
            lambda *_: order.append("shared_enhancement")
        ),
        model.private_enhancer.register_forward_hook(
            lambda *_: order.append("private_enhancement")
        ),
        model.diversity_classification.register_forward_hook(
            lambda *_: order.append("dc")
        ),
        model.reliability_fusion.register_forward_hook(
            lambda *_: order.append("gate")
        ),
    ]
    try:
        model(**make_batch(), update_prototypes=False)
    finally:
        for hook in hooks:
            hook.remove()

    assert order[0] == "pgu"
    assert order.count("shared_enhancement") == 3
    assert order.index("private_enhancement") < order.index("dc")
    assert order.index("dc") < order.index("gate")


def test_prototype_ema_updates_only_during_labeled_training(
    config: EmoUIDConfig,
) -> None:
    model = EmoUID(config)
    batch = make_batch()
    before = model.pgu.prototypes.clone()
    model.train()
    model(**batch)
    after_training = model.pgu.prototypes.clone()
    assert not torch.allclose(before, after_training)

    model.eval()
    with torch.no_grad():
        model(**batch)
    assert torch.allclose(after_training, model.pgu.prototypes)

    inference_batch = {key: value for key, value in batch.items() if key != "labels"}
    model.train()
    model(**inference_batch)
    assert torch.allclose(after_training, model.pgu.prototypes)


def test_unlabeled_inference_requires_no_training_target(config: EmoUIDConfig) -> None:
    model = EmoUID(config).eval()
    batch = make_batch()
    batch.pop("labels")
    with torch.no_grad():
        output = model(**batch)
    assert output["prediction"].shape == (4, 1)
    assert output["ordinal_target"] is None
    assert output["losses"] == {}


def test_masks_exclude_padding(config: EmoUIDConfig) -> None:
    model = EmoUID(replace(config, dropout=0.0)).eval()
    batch = make_batch()
    masks = {
        "language": torch.tensor([[1, 1, 1, 1, 0, 0, 0]] * 4, dtype=torch.bool),
        "vision": torch.tensor([[1, 1, 1, 0, 0]] * 4, dtype=torch.bool),
        "acoustic": torch.tensor([[1, 1, 1, 1, 0, 0]] * 4, dtype=torch.bool),
    }
    changed = {key: value.clone() for key, value in batch.items()}
    changed["language"][:, 4:] = 1000.0
    changed["vision"][:, 3:] = -1000.0
    changed["acoustic"][:, 4:] = 500.0

    with torch.no_grad():
        first = model(**batch, masks=masks, update_prototypes=False)["prediction"]
        second = model(**changed, masks=masks, update_prototypes=False)["prediction"]
    assert torch.allclose(first, second, atol=1e-6)


def test_dataset_presets_cover_revised_benchmarks() -> None:
    assert dataset_config("CMU-MOSI").input_dims == {
        "language": 768,
        "vision": 20,
        "acoustic": 5,
    }
    assert dataset_config("CH-SIMS v2.0").input_dims == {
        "language": 768,
        "vision": 177,
        "acoustic": 25,
    }
