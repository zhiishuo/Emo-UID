# Emo-UID

This repository provides the manuscript-aligned reference implementation of
**Emotion Unity in Diversity (Emo-UID)** for multimodal sentiment analysis.
The release is intentionally focused on the model architecture and its complete
training objective. Historical experiment runners, inactive knowledge-
distillation paths, datasets, checkpoints, and plotting utilities are excluded.

## Implemented architecture

The code follows the final method in the paper, in computational order:

1. BERT language encoding for CMU-MOSI/CMU-MOSEI and modality-specific temporal
   front-ends;
2. shared-private feature factorization with orthogonality, squared-Frobenius
   reconstruction, and private-cycle consistency;
3. Prototype Gram Unity (PGU) on the pre-enhancement shared representations;
4. three independent common-stream DMD Transformer stacks, followed by six
   directed cross-modal Transformers and three modality-indexed memory
   Transformers for the private streams;
5. Diversity Classification (DC) with soft ordinal supervision and Confidence
   Product Suppression (CPS);
6. reliability-gated residual fusion; and
7. sentiment regression with the grouped objective

   `L_total = L_task + lambda_f L_fac + lambda_p L_PGU + lambda_d L_DC`.

The modality-specific DC heads are auxiliary ordinal evidence estimators. They
are supervised by the sample-level soft ordinal target and must not be
interpreted as classifiers trained with official unimodal sentiment labels.

There is no teacher network or knowledge-distillation objective in this code.
The Transformer enhancement topology follows DMD, while DMD-specific
distillation heads, flattened auxiliary predictors, and ensemble projections
are not part of Emo-UID.

## Installation

```bash
python -m pip install -e .
```

The English benchmarks instantiate BERT-base-uncased inside the model and
therefore require the optional language-model dependency:

```bash
python -m pip install -e '.[bert]'
```

## Dataset feature presets

`dataset_config` supplies the feature dimensions stated in the revised paper:

| Dataset | Language path | Acoustic | Vision | Ordered anchors |
|---|---|---:|---:|---|
| CMU-MOSI | fine-tuned BERT (768) | 5 | 20 | -3 to 3 |
| CMU-MOSEI | fine-tuned BERT (768) | 74 | 35 | -3 to 3 |
| CH-SIMS | provided features (768) | 33 | 709 | -1 to 1 |
| CH-SIMS v2.0 | provided features (768) | 25 | 177 | -1 to 1 |

The presets do not contain dataset paths or redistribute benchmark data.
`EmoUID.parameter_report()` separates the active language-encoder and Emo-UID
core parameters so that model size is not inferred from legacy experiment code.

## Released training configurations

Each dataset preset also exposes the optimization protocol through
`dataset_config(dataset).training`:

| Dataset | Batch | Max epochs | Early-stop patience | Weight decay |
|---|---:|---:|---:|---:|
| CMU-MOSI | 16 | 60 | 10 | 0.005 |
| CMU-MOSEI | 64 | 45 | 8 | 0.001 |
| CH-SIMS | 4 | 30 | 6 | 0.001 |
| CH-SIMS v2.0 | 4 | 30 | 6 | 0.001 |

All presets use Adam with learning rate `1e-4`, gradient clipping at `0.6`,
and a factor-`0.5` learning-rate reduction after five epochs without validation
improvement. Checkpoints are selected by validation weighted F1. The model
presets expose `K=2`, `theta=0.9`, `tau_o=0.2`, `lambda_CPS=1`, and grouped
objective weights `lambda_f=lambda_p=lambda_d=1`.

## Paper-to-code map

See [ARCHITECTURE.md](ARCHITECTURE.md) for the equation-level mapping, tensor
semantics, and implementation invariants used to audit manuscript-code
consistency.

## Scope

This is a compact architectural release. Data acquisition and preprocessing
must follow the licenses and official pipelines of the four benchmarks. Fixed
discrepancy bins used for cross-model analysis are an evaluation protocol and
are intentionally separate from the model-internal reliability discrepancy.
