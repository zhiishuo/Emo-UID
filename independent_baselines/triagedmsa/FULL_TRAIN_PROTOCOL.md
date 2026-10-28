# Full Training Protocol: TriagedMSA

> Historical source/protocol audit, not a final runnable configuration or current
> run-status report. See `../README.md` for failed validation diagnostics and
> `../configs/` for explicitly illustrative public configurations.

Source: Luo et al., DOI 10.1109/TAFFC.2024.3524789. Table I and IV.A,
printed p.1564 (PDF p.8), visually verified in `evidence/protocol-page-08.png`.
The JSON companion separates source facts, unreported settings and runner choices.
No training or numerical benchmark claim accompanies these files.

| Published setting | MOSI | MOSEI |
| --- | --- | --- |
| Learning rate | 1e-4 | 1e-4 |
| Batch size | 128 | 128 |
| Hidden size | 256 | 256 |
| Attention heads | 2 | 4 |
| Dropout | 0.5 | 0.7 |
| Total epochs | 50 | 50 |
| Early stop (printed value) | 20 | 20 |

IV.A reports validation-set grid search over LR {1e-5,5e-5,1e-4,5e-4},
heads {1,2,4,8}, dropout {0.1,0.2,0.5}, hidden {512,256,128}. MOSEI's
final dropout 0.7 lies outside that prose grid: prefer the final Table I value
and disclose the inconsistency. The implementation requires even hidden widths
divisible by heads, which all listed hidden/head combinations satisfy.

## Selection and Unreported Details

Validation is explicitly used for hyperparameter search. The paper does not
specify optimizer, scheduler, weight decay, gradient clipping, checkpoint metric,
tie-breaking, exact early-stop semantics, seeds or number of repeated runs.
BERT is specified in III.D, but checkpoint and freeze/fine-tune policy are not.
Exact sequence truncation, feature processing and alignment are not supplied.
IV.D lists Acc2, F1, MAE and Pearson correlation; F1 averaging and treatment of
zero-valued labels are not settled by that section. Do not silently import
another method's conventions as published TriagedMSA settings.

For an executable independent protocol, recommend explicitly declared Adam
(betas 0.9/0.999, eps 1e-8, weight decay 0), no scheduler, minimum validation
sentiment MAE checkpoint, earliest tie, patience 20 validation evaluations.
These are independent choices, not recovered original settings. Use the printed
50-epoch cap for the paper-settings run. The requested 100-epoch ceiling is a
separate extension; label it accordingly. Validation-only search must finish
before fixing the configuration for five seeds. Freeze exact seeds, feature
provenance, split hashes, BERT policy and metric definitions before test scoring.

## Generic Runner Integration

Preferred string-ID API (model forward API unchanged):

```python
from .lifecycle import APDCache, refresh_apd_cache

cache = APDCache.initialize(split.ids)  # canonical full train order, epoch 1
# Training batch: relationship_targets=cache.for_ids(batch.ids, device=device)
# After training epoch cache.epoch, with external BERT set to eval:
cache = refresh_apd_cache(
    core, batches, forward, cache=cache, train_labels=full_train_labels,
    completed_epoch=cache.epoch, source_split='train',
)
# batches() yields (stable_sample_ids, payload), exactly all train IDs once.
# forward(core, payload) returns core output with fused and shared tensors.
```

The helper makes two complete eval/no-grad passes with fixed weights, installs
class references between passes, computes fresh raw targets from second-pass
shared representations and applies printed Eq.21. It supports reordered batches
and restores every core submodule's prior mode. It rejects duplicate, missing
and foreign IDs. A failed second pass restores old references and version.
Callbacks must not perform optimization, change modes/state, or use stochastic
augmentation. External BERT is outside the core: parent must freeze and restore
its mode, and keep its weights unchanged throughout both passes. Full labels
must match canonical train IDs; neither a split tag nor the helper can prove
the provenance of data falsely tagged as train.

Optional integer-index adapter: `refresh_apd(core, forward_batches, train_ids,
previous_targets, next_epoch, source_split='train')` returns CPU targets in
train_ids order. Each callback invocation yields `(CPU LongTensor indices,
output_dict, CPU float labels)`. It checks label consistency between passes.
Use `refresh_apd_cache` for actual string sample IDs; do not coerce them to integers.

Persist `cache.state_dict()` with model, optimizer, RNG, configuration and split
hashes; restore with `APDCache.from_state_dict()`. Reuse the returned cache, not
the old one. Epoch validation does not replace the runner's checkpoint/commit
discipline. Cache tensors must be treated as immutable by the caller.

## Fidelity Audit

Re-read primary III.D-I and IV.A-D against model.py. No additional justified
core-equation change was identified; the model API and parameters remain intact.
Existing deliberate interpretations remain: positive MAE despite Eq.5's printed
minus, Algorithm 1 strict sign product <0 despite Eq.20's <1, squared distances
from Eq.18, literal Eq.21 shrinkage, full semantic-head pair attention in SSA.
The new lifecycle closes the previously missing full-train refresh integration,
not the unresolved SSA dimension/weight-sharing, pooling, SDT flattening or BERT
provenance ambiguities. Its one-epoch-lag timing is an independent interpretation,
not evidence of the original implementation's exact update schedule.

CPU-only verification from `independent_baselines/`:
`python3 -B -m unittest discover -s triagedmsa -p 'test_*.py' -v`.
These tests establish engineering behavior only.
