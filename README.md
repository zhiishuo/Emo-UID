# Emo-UID

This repository provides the manuscript-aligned reference implementation of
**Emotion Unity in Diversity (Emo-UID)** for multimodal sentiment analysis.
The release is intentionally focused on the model architecture and its complete
training objective. Historical experiment runners, inactive knowledge-
distillation paths, datasets, checkpoints, and plotting utilities are excluded.

## Implemented architecture

The code follows the final method in the paper, in computational order:

1. shared-private feature factorization with orthogonality and reconstruction;
2. Prototype Gram Unity (PGU) on the pre-enhancement shared representations;
3. shared-stream and joint private-stream Transformer enhancement;
4. Diversity Classification (DC) with soft ordinal supervision and Confidence
   Product Suppression (CPS);
5. reliability-gated residual fusion; and
6. sentiment regression with the grouped objective

   `L_total = L_task + lambda_f L_fac + lambda_p L_PGU + lambda_d L_DC`.

The modality-specific DC heads are auxiliary ordinal evidence estimators. They
are supervised by the sample-level soft ordinal target and must not be
interpreted as classifiers trained with official unimodal sentiment labels.

There is no teacher network or knowledge-distillation objective in this code.

## Installation

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

## Minimal forward pass

Emo-UID accepts temporal feature sequences with shape `[batch, time, feature]`.
For CMU-MOSI and CMU-MOSEI, `language` denotes contextual BERT features. For
CH-SIMS and CH-SIMS v2.0, all three inputs denote the benchmark-provided
features.

```python
import torch

from emouid import EmoUID, dataset_config

config = dataset_config("mosi")
model = EmoUID(config)

batch = 2
output = model(
    language=torch.randn(batch, 12, 768),
    vision=torch.randn(batch, 12, 20),
    acoustic=torch.randn(batch, 12, 5),
    labels=torch.tensor([-0.8, 1.2]),
)

loss = output["losses"]["total"]
loss.backward()

print(output["prediction"].shape)  # [2, 1]
print({name: gate.shape for name, gate in output["gates"].items()})
```

Run the complete CPU example with:

```bash
python -m examples.forward_pass
```

## Dataset feature presets

`dataset_config` supplies the feature dimensions stated in the revised paper:

| Dataset | Language | Acoustic | Vision | Ordered anchors |
|---|---:|---:|---:|---|
| CMU-MOSI | 768 | 5 | 20 | -3 to 3 |
| CMU-MOSEI | 768 | 74 | 35 | -3 to 3 |
| CH-SIMS | 768 | 33 | 709 | -1 to 1 |
| CH-SIMS v2.0 | 768 | 25 | 177 | -1 to 1 |

The presets do not contain dataset paths or redistribute benchmark data.

## Paper-to-code map

See [ARCHITECTURE.md](ARCHITECTURE.md) for the equation-level mapping, tensor
semantics, and implementation invariants used to audit manuscript-code
consistency.

## Scope

This is a compact architectural release. Data acquisition and preprocessing
must follow the licenses and official pipelines of the four benchmarks. Fixed
discrepancy bins used for cross-model analysis are an evaluation protocol and
are intentionally separate from the model-internal reliability discrepancy.
