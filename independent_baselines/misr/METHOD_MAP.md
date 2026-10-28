# MISR Independent Implementation: Method Map

> Publication note (2026-10-28): the audit narrative below is historical.
> The equation/API map still describes the bundled core. The package now includes
> a BERT wrapper and training runner in `../train_full.py`; the old statements
> about missing integration are not current package status. See `../README.md`
> for terminal outcomes and clearly marked example configurations. No successful
> reproduction is claimed, and this core is not the full later diagnostic lineage.

## Scope and evidence boundary

This is a paper-grounded independent implementation, NOT an official-code
reproduction or verified algorithmic equivalent. CPU synthetic engineering tests
are not benchmark results. No real data, checkpoint, training runner, rebuttal,
environment, or original PDF was changed. No packages were installed. GPU/data
integration belongs to the parent. No benchmark metrics are reported here.

Primary source: Zhu et al., *Multimodal Invariant Sentiment Representation
Learning*, Findings ACL 2025, pp. 14743-14755. Read local PDF:
`LOCAL_PATH_OMITTED`.
Text inspected for Sections 3.1-3.5 and Appendix B/C/E; page 14747 was also
rendered and visually inspected (`paper_page5.png`).

Read-only context:
`LOCAL_PATH_OMITTED`.
The audit's October 22 code-access failure is historical, not newly rechecked.
The current task explicitly authorizes independent implementation despite that
earlier audit's official-resource boundary.

## Equation-to-code map

| Paper | Implementation in model.py | Choices / fidelity limits |
| --- | --- | --- |
| Section 3.2, p.14745, ME | `ModalityBranch.embedding`, `me`: projection plus TWO Transformer layers per modality | Hidden width 128; 4 heads, FFN 512, post-norm, dropout .1, sinusoidal positions are independent choices. Table 6 specifies FC width 128 but not complete Transformer settings. |
| Eq.1, p.14746, Gaussian posterior | `ve_mapping` + ReLU + TWO Transformer layers + separate `mean` and `scale` FC heads | Per-aligned-token mean [B,T,128], isotropic std [B,T,1]. `softplus(raw)+1e-4` ensures positive std. Text calls sigma variance, but Eqs.1/6 use sigma as standard deviation; equations take precedence here. Isotropy follows sigma-squared I literally; token granularity is not uniquely determined by paper. |
| Eq.2, uncertainty weights | `logits=1/std`, softmax over three modalities | Literal softmax of reciprocal std, NOT inverse variance or normalized precision. Missing tokens masked out. Stable softmax; no clipping or detached gradients. Very small scales can still saturate weights. |
| Eq.3, joint representation | `joint_tokens=sum(weights*mu)`, then masked mean to `joint` | Uses Gaussian means, not sampled z; pooling location unspecified in paper. Parent must use aligned inputs for this tokenwise implementation. |
| Eq.4, normalized alignment | `gaussian_kl(mu,std,0,1)` summed over modalities | Exact analytic Gaussian KL, sum latent coordinates, mean valid tokens within utterance then batch mean. |
| Eq.5, cross-modal alignment | `cma`: analytic KL for all SIX ordered distinct modality pairs | Both directions retained; no replacement by MSE/cosine. Valid-position intersection per pair. KL reductions unspecified by paper. |
| Eq.6, reparameterization | `z=mu+std*randn_like(mu)` | One Monte Carlo draw in training; gradients through mu and std. `sample=False` uses mu for deterministic evaluation, not exact expected reconstruction loss. |
| Eq.7, reconstruction | `vd_mapping` + ReLU + TWO Transformer layers + reconstruction FC; squared Euclidean residual | Reconstructs ME hidden h, not raw features; target is not detached. Sum latent coordinates, mean valid tokens/utterances, sum modalities. Reconstruction target granularity and reduction are unresolved. No substitution with a plain MLP decoder. |
| Eq.8, distribution constraint | `distribution_constraint=normalized_alignment+cross_modal_alignment+data_reconstruction` | All three components present and separately returned. No independent reweighting or hidden normalization. |
| Eq.9, constructed environments | `subsets` enumerates all 2^3=8 subsets of text/audio/vision; `environments=u+sum(H_m)` | Empty subset is u. Each pooled H_m is ME output, not posterior mu. Sum fusion chosen because f_i is unspecified. Shared LayerNorm follows Figure 2; exact placement uncertain. Environments are feature combinations of the same batch, not label strata or random corruption. |
| Eq.10, invariance statement | Shared `phi` and `predictor` across all environments | Eq.10 is a desideratum, not guaranteed by this implementation. Phi is an independently chosen two-FC ReLU map after shared LayerNorm. Nothing enforces exact conditional independence. |
| Eq.11, invariant constraint | Batch mean L1 per environment, then population variance over 8 risks | REx-style risk variance, NOT IRMv1 gradient penalty or variance of representations. Printed summation inside Phi is ambiguous; interpret each environment separately, consistent with explanatory prose. No fabricated conflict labels. |
| Eq.12, task prediction | Shared two-FC ReLU `predictor`; `sentiment_prediction=risks[0]` | Use u-only environment for main prediction/SP, consistent with Appendix E Table 8 N1. Paper does not precisely specify SP aggregation across environments. Other environments still receive gradients through IC. |
| Eq.13, combined loss | `total=1*SP+0.4*DC+1*IC` | MOSI Table 6 coefficients. Total exists only when regression labels supplied. No rescaling added to make a smoke pass. |

## Interface and supervision

```python
from misr.model import MISR, MISRConfig
model = MISR(MISRConfig()).cpu()
output = model(language, audio, vision, masks=masks, labels=labels)
output['losses']['total'].backward()
```

- Inputs: contextual floating language `[B,T,768]`, audio `[B,T,5]`, vision
  `[B,T,20]`. Default hidden width 128; arbitrary positive T, initially MOSI T=50.
- All inputs must share device/dtype and **aligned semantic positions**. Equal
  lengths alone do not establish alignment. This implementation cannot silently
  ingest raw unaligned streams; parent must verify alignment before a pilot.
- Optional masks: dictionary `language`, `audio`, `vision`, each boolean `[B,T]`,
  True means valid. Omitted masks mean all positions valid. Padding may contain
  NaN and is sanitized; valid NaN/Inf is rejected. Fully missing modalities or
  no pairwise valid-position overlap in an utterance are rejected explicitly.
- Labels: multimodal continuous regression targets `[B]` or `[B,1]`, normally
  MOSI range [-3,3]. No extra unimodal labels, pseudo-labels, conflict annotations,
  or environment labels are required by the implemented MOSI objective.
- CHERMA requires seven-class cross-entropy per Section 3.5 and is NOT supported
  by this regression-only implementation. Do not feed categorical labels here.
  SIMS/MOSEI dimensions and coefficients require explicit separate configuration.
- Returns `prediction [B]`, named `losses`, eight `environment_predictions`,
  `environment_risks` when labeled, `environment_names`, `fusion_weights`,
  `joint_representation`, and `invariant_representation` for u-only inference.
- Without labels only distribution losses are computed; there is no synthetic
  task loss, IC, or total. Eval defaults to deterministic mu reconstruction.

The historical external loader used `acoustic` as the audio batch/mask key.
The public `../train_full.py` wrapper instead reads `audio`, maps masks to the
core names, and converts `text_bert` tokens into contextual BERT embeddings.
It does not depend on that external loader.

## BERT and training-protocol boundary

Starting 768 language features assume BERT-base-compatible contextual output,
not proof of the paper's main-table backbone. Table 6 lists 768 AND 1024; Table 7
compares base and large. This package neither downloads nor embeds BERT. It
preserves language-input gradients so the parent can fine-tune a wrapper.
Precomputed frozen features must be labeled a different protocol.

Paper-reported MOSI and MOSEI setup: Adam, batch 32, 100 epochs, BERT LR 5e-6,
other LR 1e-4, sequence length 50, FC 128, coefficients 1/.4/1. Section 4.2
explicitly reports fixed seed 1111 and averaging three experiments; the per-run
seed policy remains unspecified. Main-table backbone, checkpoint selection,
preprocessing and exact evaluator remain unresolved. MOSEI audio/vision widths
are 74/35, not MOSI's 5/20. These are source settings, not completed runs.

See `FULL_TRAIN_PROTOCOL.json` and `FULL_TRAIN_PROTOCOL.md` for the full-source
audit. No source-backed reduction of large KL/reconstruction scale was found;
coordinate sums, six directed KL pairs and lambda_dc=.4 remain unchanged.
The KL numerical expression was stabilized without changing its mathematics or
public API. No other architecture change was justified by the inspected source.

## Tests and bounded pilot decision

From `independent_baselines/`, using Python 3.11 / torch 2.10.0:

```sh
python3 -B -m unittest discover -s misr -p test_model.py -v
```

All tests are synthetic CPU engineering checks. They cover complete/default T50
backward, finite gradients in every parameter and input modality, analytic KL
against torch.distributions, Eqs.8/11/12/13 arithmetic, eight environment names,
masked weight normalization, NaN-padding invariance, invalid-input rejection,
deterministic unlabeled eval, stochastic reconstruction, and nonzero IC gradients.
The follow-up audit adds extreme-scale KL stability, coordinate-sum preservation,
all six directed pairs, squared reconstruction/direct target gradient, and both
768/1024 MOSEI contextual-input interface tests. These are not BERT training tests.
These checks do not measure scientific performance or prove invariance.

**Implementation ready for a bounded parent-owned real-data training pilot,
conditional on aligned-data and BERT-wrapper verification. Not ready for a
faithful-reproduction claim or benchmark comparison.** Suggested gate: one
verified training batch with component/gradient logging, then a short
train/validation-only pilot under a frozen independent-implementation config.
Do not use held-out test outcomes to choose the unresolved architecture choices.

Important adverse possibilities to retain: DC may dominate SP because latent
dimensions and six directed KL terms are summed; reconstruction targets may
co-adapt/collapse; reciprocal-std weighting may saturate; IC can be small when
all environments are equally poor, and can cancel with symmetric residuals;
sum fusion plus LayerNorm may weaken environment differences. Passing finite
CPU tests resolves none of these scientific concerns. Record component scales
and retain failed/partial pilot outcomes instead of silently changing objectives.
