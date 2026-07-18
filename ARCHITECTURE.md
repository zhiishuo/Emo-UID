# Architecture correspondence

This document records the correspondence between the revised manuscript and
the public implementation. It is intended as an audit map, not as an additional
method variant.

| Manuscript | Implementation | Role |
|---|---|---|
| Eq. (2) | `BertLanguageEncoder`, `TemporalProjector` | Dataset-specific language encoding and modality front-ends |
| Eqs. (3)-(4) | `SharedPrivateFactorization` | Shared/private decomposition, orthogonality, and reconstruction |
| Eqs. (5)-(7) | `PrototypeGramUnity._gram_volume_logits` | Symmetric tri-modal Gram-volume contrastive learning |
| Eqs. (8)-(11) | `soft_ordinal_target`, `PrototypeGramUnity` | Soft ordinal prototype anchoring and EMA updates |
| Eq. (12) | `losses["pgu"]` | `L_PGU = L_Gram + L_sop` |
| Eq. (13) | `SharedStreamTransformer`, `MultimodalPrivateTransformer` | Independent common-stream and joint private-stream enhancement |
| Eqs. (14)-(16) | `ConsensusPool`, `DiversityClassification` | Common ordinal decision coordinate and auxiliary distributions |
| Eqs. (17)-(19) | `DiversityClassification` | Pairwise surface, tri-modal volume, and CPS |
| Eqs. (20)-(23) | `ReliabilityGatedResidualFusion` | Entropy/JSD-conditioned scalar gates and residual fusion |
| Eqs. (24)-(25) | `EmoUID.forward` | MAE task loss and grouped training objective |

## Computational order

1. Project each modality and factorize it into shared and private streams.
2. Apply PGU to pooled shared representations before Transformer enhancement.
3. Independently enhance common streams and jointly enhance private streams.
4. Pool the enhanced streams and map them into one ordered sentiment coordinate.
5. Apply CPS to modality-indexed private distributions.
6. Estimate one reliability gate per private stream from its residual feature,
   consensus feature, entropy, and JSD from the consensus distribution.
7. Preserve the direct consensus path and add only gated private residuals.
8. Optimize the regression task and the three grouped auxiliary objectives.

## Reviewer-facing invariants

- **No knowledge distillation.** The architecture contains no teacher network,
  teacher logits, distillation objective, or teacher-student execution path.
- **No inactive BERT pooling head.** The English language path consumes
  token-level hidden states, so BERT is instantiated without its unused pooler.
- **No unimodal-label claim.** The three private heads are auxiliary ordinal
  evidence estimators supervised by the sample-level target. Their outputs are
  not official unimodal sentiment predictions.
- **Private streams remain modality-indexed.** They use separate private
  encoders and retain separate outputs after joint context modeling. A common
  ordinal target supplies a shared decision coordinate; it does not make the
  private representations identical.
- **Fusion is asymmetric.** Consensus is the direct path. Private features enter
  prediction only as modality-specific, reliability-weighted residuals.
- **Discrepancy measures are not conflated.** JSD in the gate is an internal,
  sample-specific model signal. Fixed DMD-derived bins for MOSI/MOSEI and
  annotation-derived bins for CH-SIMS/CH-SIMS v2.0 belong to the external
  cross-model evaluation protocol and are not implemented as model inputs.
- **Prototype updates are controlled.** The prototype bank is updated by EMA
  only in training mode and only when labels are available.

## Parameter accounting

`EmoUID.parameter_report()` reports total/trainable parameters and separates the
language encoder from the Emo-UID core. The report follows the instantiated
forward graph; legacy KD modules and historical experiment heads are neither
registered nor counted.

For the CMU-MOSI preset, BERT-base-uncased without its unused pooling head has
108,891,648 parameters and the complete Emo-UID core has 523,406 parameters,
giving 109,415,054 active trainable parameters in total. The EMA prototype bank
is a non-trainable buffer and is therefore not included in the parameter count.

## Tensor conventions

- English language input: `[batch, 3, time]` containing token ids, attention
  mask, and token-type ids
- Benchmark-provided modality feature: `[batch, time, input_dimension]`
- Valid-step mask: `[batch, time]`, where `True` denotes a valid step
- Shared/private temporal stream: `[batch, time, model_dimension]`
- Consensus/private pooled vector: `[batch, model_dimension]`
- Ordinal distribution: `[batch, number_of_anchors]`
- Reliability gate: `[batch, 1]`
- Sentiment prediction: `[batch, 1]`
