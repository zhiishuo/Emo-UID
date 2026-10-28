# MISR Full-Training Protocol Audit

> Historical source/protocol audit, not a final runnable configuration or current
> run-status report. See `../README.md` for the terminal unsuccessful reproduction
> outcome and `../configs/` for explicitly illustrative public configurations.

## Status and provenance

This is an independent paper-grounded implementation, not verified official-code
equivalence. No training, GPU launch, benchmark evaluation or rebuttal integration
was performed in this audit. The parent owns the trainer and eventual measured
results. All changes are confined to this MISR directory. The JSON companion
separates paper facts, existing choices, unresolved choices and parent requirements.

Primary source: Zhu et al., *Multimodal Invariant Sentiment Representation
Learning*, Findings ACL 2025, pp. 14743-14755, local PDF:
`LOCAL_PATH_OMITTED`.
SHA-256: `1d89f08fdda31e3c018611ed01b62dc92f2f64f4014ac6a774b71a75074d471d`.
Read Sections 3.1-3.5, 4.2 and Appendices A-E. Equation 7 was checked visually
in the saved page image: the norm is **squared**, not an unsquared L2 norm.
Official code was not inspected; historical code-access failures were not retested.

## Original settings

Source: Section 4.2 (p.14748), Tables 5-6 (p.14754).

| Setting | MOSI | MOSEI |
| --- | --- | --- |
| Text dimensions | 768 and 1024 | 768 and 1024 |
| Audio / visual dimensions | 5 / 20 | 74 / 35 |
| Train / validation / test | 1284 / 229 / 686 | 16326 / 1871 / 4659 |
| Optimizer | Adam | Adam |
| Batch size / epochs | 32 / 100 | 32 / 100 |
| BERT / other learning rate | 5e-6 / 1e-4 | 5e-6 / 1e-4 |
| Sequence length T / FC width | 50 / 128 | 50 / 128 |
| SP / DC / IC coefficients | 1 / 0.4 / 1 | 1 / 0.4 / 1 |
| Task loss / sentiment range | L1 / [-3,3] | L1 / [-3,3] |

The paper reports PyTorch, hardware named "NVIDIA RTX A40", fixed random seed
1111, and averaging three experiments. The three per-run seeds and how they
differ despite the fixed seed statement are not supplied. Do not invent a
three-seed list and attribute it to the paper. A parent-declared seed policy is
needed for a new multi-run comparison.

Section 4.2 reports Acc-2 and F1 with and without neutral labels (first:
negative/non-negative including zero; second: negative/positive), Acc-7, MAE,
and correlation. Exact F1 averaging and Acc-7 rounding/clipping are unspecified.
Reported split counts do not establish identity of local split IDs/features.

## Does the enormous DC scale justify reducing it?

**No source-supported reduction was found. Keep the objective unchanged.**

Equations 4-5 specify Gaussian KL and sums over modalities and distinct modality
pairs. For a D-dimensional Gaussian, KL sums D coordinate contributions; a
coordinate mean divides that divergence by D. The current implementation retains
all six ordered pairs. Equation 7 specifies squared reconstruction norm. It is
not per-coordinate mean squared error. At D=128, replacing both coordinate sums
with means reduces all three DC components by 128, effectively turning the
reported coefficient 0.4 into 0.003125 relative to the existing objective.
Averaging the six CMA terms independently changes the relative NA/CMA/DR weights.
Neither change is justified merely because training looks better.

The current token/batch averaging is an independent choice: equal weight per
utterance, averaging valid tokens within each utterance, then batch mean. DR
sums modalities, although Eq.7 suppresses that aggregation. The paper does not
settle these implementation details. Thus the present reduction is a transparent
equation-guided choice, not a claim of exact official reduction recovery.

Synthetic CPU diagnostic: seed 1111, default width 128, random [2,50,D] inputs,
eval mode, deterministic reconstruction, labels [-0.5,0.5]: NA=216.0111,
CMA=2626.3218, DR=542.5582, DC=3384.8911, SP=0.4944, IC=0.00000812,
total=1354.4508. This is **not a dataset result or training curve**. Initial
LayerNorm-scale features make a squared residual sum grow with width; KL also
grows with width and can grow sharply with small target standard deviations.
Six directed pairs compound that effect. These facts explain possible scale,
but do not prove successful optimization or diagnose the parent's actual batch.

Before training, log raw and weighted components, component gradient norms,
posterior standard deviations and fusion-weight saturation. Large values must
be retained as adverse evidence. If a normalized alternative is explored later,
label it as a separate variant, retain this baseline, and do not select it on
test performance. No clipping, target detachment or loss rescaling was added.

## BERT setup

The existing module takes contextual embeddings; it does not contain BERT.
Section 4.2 and Table 6 give a separate BERT learning rate, supporting trainable
BERT, not frozen cached features as the default reproduction interpretation.
Exact trainable layers are not specified. Table 6 lists 768 **and** 1024, and
Appendix C/Table 7 discusses base and large. Neither identifies the backbone
behind the main Table 1 results. Base is not established just because it matches
the current default input width.

The trainer must declare checkpoint/revision, tokenizer/casing, trainable layers,
output layer, special-token handling and subword alignment. Preserve autograd and
put BERT parameters in the 5e-6 group, other parameters in the 1e-4 group. Verify
a BERT gradient and actual update. For base use language_dim=768; for large use
1024. Frozen cached features require a distinct protocol label. Equal sequence
length is not proof that acoustic/visual positions match BERT tokens.

## Remaining architecture and training choices

No equation-level architecture error was established from the inspected source.
Do not present unspecified details as recovered facts. Two Transformer layers
in ME, VE and VD, mean/std heads, reciprocal-std softmax, Gaussian KL,
reparameterization, reconstruction, environment risk variance and L1 prediction
are implemented. The following remain independent choices or ambiguous:

- Four heads, FFN=4D, dropout=.1, sinusoidal positions and post-norm.
- Tokenwise isotropic posterior, softplus std plus 1e-4; prose calls sigma
  variance while Eqs.1/6 use standard deviation. Equations guide the code.
- Mean pooling, summed environment fusion, two-FC Phi, LayerNorm after fusion.
  Figure 2 suggests a normalization stage but does not fix the exact placement.
- Eight modality subsets, u-only SP and inference (consistent with Table 8 N1),
  population variance of eight batch risks; Eq.11 has an ambiguous inner sum.
- One reconstruction sample while training; deterministic mean reconstruction
  at evaluation is not the expected stochastic reconstruction objective. Select
  a checkpoint by a declared predictive validation criterion, not by pretending
  this deterministic DC is an unbiased estimate of training DC.
- Adam betas/epsilon, weight decay, scheduler/warmup, clipping, precision,
  shuffling/drop_last, early stopping and checkpoint selection are unreported.
- Feature source/version, alignment, normalization, split IDs and evaluator
  equivalence need parent verification. Do not equate a dimension match with
  matching the paper's preprocessing.

## Code change and integration contract

`gaussian_kl` now computes log differences and standardized ratios before
squaring. This is algebraically the same KL, but avoids artificial 0/0 and
inf/inf for equal tiny/large finite standard deviations. It does not clamp true
large divergences or change any loss reduction. All module names, parameter
shapes, state-dict keys, config defaults, call signatures and output keys remain
unchanged. Genuine out-of-range divergences can still overflow.

MOSI: `MISRConfig(language_dim=768, audio_dim=5, vision_dim=20)`.
MOSEI: `MISRConfig(language_dim=768, audio_dim=74, vision_dim=35)`.
These explicitly use the independent base-compatible default, not a verified
main-table backbone. Map parent `acoustic` mask/batch key to `audio`; supply
aligned contextual tensors and multimodal regression labels only.

Five added tests guard extreme-scale KL stability, coordinate-summed KL,
squared reconstruction including its direct target gradient, and all six
directed KL pairs, plus MOSEI base/large contextual interfaces and gradients.
All 15 synthetic CPU tests passed. No benchmark claims are
licensed by those tests. The unresolved choices must be frozen and recorded by
the parent before full training and disclosed with any eventual rebuttal results.
