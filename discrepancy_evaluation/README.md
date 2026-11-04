# Discrepancy protocol toolkit v1.0.0

New reusable protocol tooling, **not recovery of historical Fig. 3 sample
memberships**. No training, plotting, network access, or third-party dependencies.
Python 3.9+; run the scripts together from any directory.

## Historical evidence boundary

The parent-managed public repository contains 294 author-supplied aggregate rows
under `discrepancy_evaluation/data`. See its `DATA_NOTES.md` for the aggregate
release's provenance and limitations. That file and those data are outside this
toolkit's scope and are not generated or independently verified here.

The supplied Fig. 3 CSV is aggregate-only. Historical exact bin boundaries and
per-sample memberships are missing from the supplied material. Aggregate values
cannot recover sample IDs, boundaries, labels, or predictions. This toolkit does
not invent any of them, and its tests are not experimental evidence. Do not pass
aggregate rows to the sample-level interface or describe new groups as historical
Fig. 3 reproduction. There is deliberately no default scientific configuration.

## Inputs

`build_groups.py` requires a UTF-8 CSV with this exact ordered header:

```text
dataset,sample_id,text,audio,visual
```

Each modality column is a finite scalar on the same meaningful scale. The caller
must establish that comparability; the tool does not normalize or infer scales.
IDs are opaque strings (leading zeros preserved), unique as `(dataset,sample_id)`.
Missing fields, duplicate IDs, malformed rows, unknown datasets, nonfinite values,
and range overflow reject the entire build before output publication. Invalid rows
are not silently excluded. The CLI prints a diagnostic and exits with code 2.

Supply a JSON object with exactly `source_type`, `source_description`, `datasets`.
Every dataset must have explicit `edges` and `closure`; dataset names must exactly
match those in the input. Edges are finite, nonnegative, strictly increasing JSON
numbers. No quantiles, inferred historical boundaries, or implicit defaults exist.

The following is **synthetic test configuration only, not a recommended scientific
protocol or historical Fig. 3 definition**:

```json
{
  "source_type": "official_labels",
  "source_description": "SYNTHETIC TEST FIXTURE ONLY",
  "datasets": {
    "synthetic": {"edges": [0, 1, 2, 3, 4], "closure": "left"}
  }
}
```

For real use, replace every example value with user-supplied exact boundaries and
provenance. `source_type` must be `official_labels` or
`frozen_reference_predictions`. For labels, identify the dataset/annotation release
and scale in `source_description`. For frozen references, identify the frozen
model/checkpoint, inference protocol, and scale. These are caller declarations,
not provenance certified by the tool. Freeze reference scores and configuration
before comparing evaluated models; never derive groups from their errors.

For each row, discrepancy is `max(text,audio,visual) - min(text,audio,visual)`.
Groups are zero-based interval indices within each dataset. `left` means `[lo,hi)`
for **every** interval; `right` means `(lo,hi]` for **every** interval. There is no
special inclusion of the final/first endpoint and no epsilon or rounding. Python
binary64 arithmetic is used; exactness here means explicitly supplied boundaries,
not decimal-exact arithmetic. Choose and document the appropriate scale/edges.

## Build and evaluate

The following commands refer to caller-provided files, not bundled research data:

```sh
python3 build_groups.py --input samples.csv --config protocol.json --output groups_release
python3 evaluate_groups.py --groups-dir groups_release --predictions predictions.csv --output evaluation_release
```

Output directories must not already exist, and their parents must exist. Use a
distinct destination per invocation; concurrent writes to one destination are not
supported. Validation happens before writing; a sibling temporary directory is
renamed into place after files are complete. Neither command overwrites releases.

Build outputs:

- `groups.csv`: original values, discrepancy, and group for included samples,
  sorted by dataset and sample ID.
- `exclusions.csv`: every valid out-of-range sample, its values/discrepancy, and
  `reason=outside_configured_intervals`, including excluded outer endpoints.
- `config.json`: byte-for-byte copy of the supplied configuration.
- `manifest.json`: schema/tool version, source type, counts, SHA-256 hashes of
  input bytes, configuration, output files, and implementation files. Preserve the
  original input separately for a complete audit; it is hashed but not copied.

Predictions require this exact ordered header:

```text
dataset,sample_id,target,prediction
```

`target` is the explicitly supplied task ground truth, not automatically a modality
label or their mean. Supply one model/run per prediction CSV. Evaluation verifies
release file hashes, recomputes ranges/group assignments, and joins by both IDs,
never row order. Coverage must match the included group IDs exactly: missing,
extra, excluded, or duplicate prediction IDs are errors. Targets/predictions must
be finite and nonmissing. For model comparisons, keep both groups and targets
fixed and independently verify target provenance.

Evaluation emits `metrics.csv` with dataset, group, lower/upper edges, closure,
sample count, and MAE (`sum(abs(prediction-target))/n`). Empty bins have `n=0` and
`mae=NA`, never zero. A wholly excluded release is auditable but evaluation fails
because it has no included samples. The evaluation `manifest.json` hashes its
metrics, input prediction CSV, source release manifest, and implementation. Hashes
detect accidental modification, not authenticity against an adversary who can
rewrite manifests. No statistical significance or scientific claim is inferred.

## Verification

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_protocol
```

All fixtures are generated under temporary directories inside this toolkit and
removed after tests. They are explicitly synthetic test fixtures only, including
the nominal source-type declarations. Tests cover both interval closures,
endpoints, range computation, exclusions, schema validation, strict ID coverage,
invalid numbers, overflow, duplicates, multi-dataset IDs, MAE, empty bins,
deterministic releases, hash tampering, overwrite prevention, and both CLIs.

Publication is a separate parent-managed action. This toolkit does not push or
publish anything and contains no recovered historical sample-level results.
