# Four-dataset discrepancy evaluation: released results

This release provides author-supplied aggregate CSV values used in the revised
Fig. 3, and reusable construction/evaluation tooling. It does not include the
historical figure's sample-ID membership maps or verified interval edges.
Publishing aggregate values does not establish independent reproduction of the
underlying experiments. The authors confirm that methods within each panel used
the same groups; the aggregate CSV cannot independently verify that statement.

## Measurement

For each sample, discrepancy is max(t,a,v) - min(t,a,v).

| Dataset | Values used for t,a,v | Released CSV |
| --- | --- | --- |
| CH-SIMS | Official unimodal sentiment annotations | data/ch_sims_bin_results.csv |
| CH-SIMS v2.0 | Official unimodal sentiment annotations | data/ch_sims_v2_bin_results.csv |
| CMU-MOSI | Modality predictions from a fixed DMD reference | data/mosi_bin_results.csv |
| CMU-MOSEI | Modality predictions from a fixed DMD reference | data/mosei_bin_results.csv |

The English reference is fixed across evaluated models because official unimodal
sentiment annotations are unavailable. Each evaluated model's predictions must
not redefine its own evaluation groups. DMD head selection, checkpoint, feature
processing, exact edges and closure rules must be recorded when constructing
groups. We do not infer these missing historical settings from plotted ticks.
This plotting protocol is separate from percentile-based Low/High-20% tests.

## CSV interpretation

`protocol_overview.csv` summarizes the four dataset definitions and displayed
sample coverage without model-performance columns. Counts are summed from the
author-supplied bins, not independently recovered sample-ID manifests or complete
benchmark test-set sizes.

`discrepancy_coordinate` is the plotting coordinate, not an interval boundary.
`sample_count` is the author-supplied count in the displayed group.
`mae` is the mean absolute error reported for that group.
`highest_displayed_bin.csv` selects the largest plotting coordinate per dataset
for MISR, TriagedMSA and Emo-UID. It is not a top-20% or full-test-set comparison.
MISR and TriagedMSA are independent implementations, not the original authors'
published scores. The curves have no supplied per-bin training-seed uncertainty.

All 294 numerical rows preserve the supplied values. Display model aliases are
normalized in aggregate_manifest.json; no sample IDs or predictions are created.
The legacy source's Acc2 column is not released because its binary protocol is
not specified. File hashes are provided to identify this result snapshot.

## Reproduction status

The CSV supports replotting the reported aggregate results. Recomputing the
historical figure from samples additionally requires the exact reference exports,
sample memberships, selection rules and per-model predictions. Those artifacts
are not supplied by this release. Newly generated groups must be identified as a
new protocol instance, not silently presented as the original Fig. 3 groups.

No raw video, audio, text transcripts or original dataset archives are distributed.
Obtain benchmark data from its original providers under their terms.
