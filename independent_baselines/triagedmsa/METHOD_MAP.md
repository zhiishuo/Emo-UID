# TriagedMSA independent implementation

## Status and evidence boundary

Executable paper-grounded independent implementation, NOT official code and NOT
an equivalent reproduction of the published results. Nine synthetic CPU tests
pass on installed Python 3.11 / torch 2.10.0. No benchmark, data integration,
training run, GPU work, or rebuttal edit occurred. CPU backward tests are not
accuracy evidence. The model core is ready for parent integration into a bounded
real-data pilot, conditional on the train-only lifecycle below. It is not a
standalone training pipeline.

Primary source: Luo et al., TriagedMSA, IEEE TAFFC 16(3), 1557-1569,
DOI 10.1109/TAFFC.2024.3524789.

- PDF: `LOCAL_PATH_OMITTED`
- Verified SHA256: `d5d6282460d6b705a3d21bb336ee6c19afa1cf8dd8870e2d01c24b5894ba787b`
- Audit read: `LOCAL_PATH_OMITTED`
- Primary methods pp.1560-1564 extracted directly; pp.1562-1563 visually checked
  in `evidence/primary-06.png` and `evidence/primary-07.png`.

## Equation-to-code mapping

| Paper | Code | Implementation and unresolved choice |
| --- | --- | --- |
| Eq.1-4, p.1560-1561 | `Encoder` | Contextual language projection + self-attention + ReLU FF. A/V packed bidirectional LSTM + sinusoidal PE + self-attention + residual/LN + FF. One layer of each; exact depth and FF dimensions unspecified. BERT is external, not replaced with a trainable embedding or pseudo tokens. |
| Eq.6-8 | `SDT` | Three full pairwise time-time bmm matrices, each flattened into learned linear+tanh, concatenated then linear+tanh to scalar. Explicit fixed `max_length`, zero-padded similarities. Flattening and maximum length are independent dimension choices, not specified by the paper. No replacement with pooled cosine similarities. |
| SDT route, p.1561 | `TriagedMSA.forward` | Hard predicted score <0 selects SSA; >=0 selects SCA. Both branches evaluated, selected via `where`. No gold/pseudo-label routing and no straight-through/soft routing surrogate. SDT learns from its own MAE, not from the nondifferentiable branch decision. |
| Eq.9-11, Fig.4 | `SSA` | Three modality projections split into H semantic heads each; all 3H query heads attend to all 3H key/value heads, including self. Each directed pair retains its temporal cross-attention output; target-head results concatenate within each query head, then query heads concatenate. MLP softmax assigns one weight per query head per timestep; weight broadcasts over that head's features. A final FF projects the weighted concatenation to common hidden width. |
| Eq.12-14 | `SCA` | Two separate learned text projections query audio/video key-values; add corresponding nontext residual; concatenate and FF to hidden width. Literal text-query interpretation of Eq.12/13 requires aligned time indices. |
| Eq.15-16 | `APD.set_references` | Exact class means from a caller-provided full train snapshot of pooled routed fused representations. Label >=0 is positive, <0 negative. Buffers store references, counts, version. Reject empty class, invalid split, eval mode, or nonfinite input before mutation. |
| Eq.17 | `APD.forward` | Each encoded modality queries the full routed fused sequence, linear projection, BatchNorm across valid tokens only. Then masked temporal mean. Keeping a sequence of fused keys avoids collapsing attention to a single key and losing query dependence. |
| Eq.18 | `APD.infer_targets` | Squared Euclidean distance of pooled mapped unimodal features to fused references; epsilon 1e-8. Uses mapped features despite equation dropping the prime, following surrounding prose. Algorithm 1's generic Norm could mean nonsquared distance; squared form follows Eq.18. |
| Eq.19 | `APD.forward` | Sum of three L1 consistency losses, each averaged over feature width and valid tokens per sample, then batch. Both pre/post mapping representations receive gradients. Scalar reduction over feature/time dimensions is unspecified in the paper. |
| Eq.20 / Algorithm 1 | `APD.infer_targets` | Strict sign-product <0 (Algorithm 1), NOT Eq.20's <1. Same-sign weak polarities must not become disagreement merely because product <1. Exact zeros count as agreement, including text zero versus opposite A/V, as the literal algorithm does. This known corner case is retained. |
| Eq.21 / Algorithm 1 | `APD.smooth_targets` | Follow printed Eq.21 literally: first epoch zero; later ((n-1)*previous + raw)/(n+1). Algorithm 1 prints (n-1)/2*(n+1) on previous, potentially growing without bound. Neither that expression nor an invented convex EMA is used. Eq.21 coefficients sum to n/(n+1), NOT one; its shrinkage is intentionally retained. |
| Eq.5,22,23 | named `losses` | sentiment MAE + triage MAE + consistency, all unit weights. Eq.5 prints negative MAE; use positive MAE in accord with prose/minimization. No unimodal sentiment regression loss is invented. |

SSA ambiguities are substantial: the paper does not settle head width, exact
pairwise Q/K/V parameter sharing, feature versus head softmax axis, temporal
pooling, or final dimension matching between SSA/SCA. Here Q/K/V maps are shared
across the directed pairs after modality-specific head projection, and weighting
is per query semantic head. The prose mentions self-attention for contributions,
whereas Eq.11/Fig.4 show linear-ReLU-linear-softmax; we implement the displayed
MLP rather than add an unspecified second attention block. All pair interactions
remain present; this is not ordinary concatenation followed by standard MHA.

## API and defaults

`TriagedMSA(language_dim=768, audio_dim=5, vision_dim=20, hidden=256,
heads=2, dropout=0.5, max_length=50)` starts with requested MOSI dimensions.
Hidden/heads/dropout follow MOSI Table I; max length 50 is an explicit engineering
choice, not a verified paper setting. Parent must choose it from the feature
protocol before model construction. MOSEI can use configurable input dimensions,
heads=4 and dropout=0.7 per Table I, but has not been data-tested. No package install.

Inputs: language/audio/vision float tensors `[B,T,D]`, optional bool `masks`
`[B,T]` or tuple of three masks (True=valid), optional regression `labels [B]`,
and optional cached APD `relationship_targets [B]` in [-1,1]. Shared aligned T is
required, with right padding and at least one valid token per sample/modality.
Unequal valid lengths are accepted, but semantic alignment is a parent obligation.
No interpolation of unaligned features is silently performed. Masked NaN padding
is sanitized; nonfinite valid features are rejected. All-missing modalities are
unsupported. APD BatchNorm requires at least two valid tokens per modality during
training; a B=1,T=1 training batch raises rather than changing the algorithm.

Returns `prediction [B]`, `triage_score`, `disagreement`, `ssa_gates`, pooled
`fused [B,H]`, `shared [B,3,H]`, `apd_raw_targets` and `apd_polarity` (None before
both references exist), plus named `losses`: consistency always; sentiment when
labels supplied; triage when cached targets supplied; total only when both supplied.
Training with labels but no relationship targets raises. Evaluation labels may
compute sentiment loss, but never influence prediction, APD inference or state.
No clipping or evaluation metrics are built into the model.

## Required parent-owned lifecycle (train only)

1. Bind immutable train IDs and split membership. Initialize each train ID's
   relationship target to zero for epoch 1. Train with original multimodal labels
   and the current cached targets; never use manufactured unimodal labels.
2. After an epoch, run a frozen `eval()`/no-grad pass over the TRAIN split only to
   collect pooled fused features by ID. Do not run validation/test data in train
   mode: BatchNorm statistics are train state too. Snapshot passes freeze BN.
3. Restore `train()` mode, call `apd.set_references(train_fused, train_labels,
   source_split='train')` once with complete train coverage. Then use a second
   frozen train-only pass with those references to obtain raw APD targets.
4. For epoch 2 targets apply `smooth_targets(zeros, raw, epoch=2)`; subsequently
   apply the same helper with each ID's prior cached target and next epoch index.
   This one-epoch-lag schedule is an independent leakage-safe interpretation of
   the paper's ambiguous within-epoch update timing. Do not update per minibatch
   or mix feature snapshots from changing model weights without disclosure.
5. Save state_dict (includes APD refs, counts, version, BN stats), model config,
   optimizer, train ID -> target cache, cache epoch, RNG state and split hashes.
   Restore them together. The model cannot enforce identity alignment or detect
   test samples falsely tagged as train. The parent must test those invariants.
6. Validation/test use `eval()` with fixed state, never call reference setters or
   update target histories from these splits. Metrics use labels only outside the
   predictive path. Checkpoint selection must not use the test set.

For CH-SIMS.v2 the paper uses real unimodal labels to derive agreement targets
and omits Lc. This module's total is deliberately the MOSI/MOSEI APD objective;
it is NOT a turnkey CH-SIMS.v2 implementation. Feeding synthetic/oracle unimodal
labels is not part of this implementation or an acceptable MOSI/MOSEI shortcut.

## Exact verification

From `LOCAL_PATH_OMITTED`:

```sh
python3 -B -m unittest discover -s experiments/emouid_independent_baselines_20261026/triagedmsa -p test_model.py -v
```

Observed: 9 tests, OK, 0.253 seconds on CPU. The installed torch import emits an
existing pynvml deprecation warning; no environment changes were made.

Tests cover forced SSA/SCA backward with finite nonzero gradients in selected
branch/each encoder/SDT/APD/regressor; default 256-hidden MOSI backward; evaluation
state immutability and label independence; no implicit train-forward reference
updates; train-split/reference class checks and exact means; weak-polarity <0
threshold and literal Eq.21 arithmetic; full state roundtrip; masked NaN and
added-padding invariance; shape/mask/supervision/single-token BN failures.

## Pilot admission and limitations

Core engineering admission: PASS. Real-data pilot admission: CONDITIONAL on the
parent integrating and testing the lifecycle above plus actual preprocessing,
alignment, BERT provenance/freeze policy, split IDs, checkpoint selection and
metrics. No accuracy, convergence, equivalence, speed benchmark or competitive
ranking is claimed. Dead-route imbalance, head gate saturation, APD label collapse,
class-empty snapshots, BN behavior and sensitivity to the chosen Eq.21 recurrence
are unresolved real-training risks; log these rather than silently modifying them.
Both branches are computed even when one is selected, and SSA scales quadratically
in head count and sequence length. SDT's fixed positional similarity flattening
also makes max_length a checkpoint architecture choice. These are bounded design
choices, not evidence that the original implementation used them.
