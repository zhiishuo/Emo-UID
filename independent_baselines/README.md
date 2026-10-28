# Independent MISR and TriagedMSA implementations

These are our paper-guided implementations, not official author code or verified reproductions of the published performance. They are released to document the implementation attempts discussed in the Emo-UID response.

## Scope and outcome

- MISR: five-seed MOSI and MOSEI evaluations completed, but published performance was not recovered. MAE (mean +/- sample standard deviation) was 1.064044 +/- 0.344179 on MOSI and 0.548746 +/- 0.003606 on MOSEI. All five seeds, including unfavorable runs, are retained in the aggregate.
- TriagedMSA: validation diagnostics did not recover effective learning. No formal five-seed test comparison is claimed.
- These outcomes concern these independent implementations, not the effectiveness of the original methods. No matched superiority claim for Emo-UID is supported by this release.

The model cores, APD lifecycle, training/evaluation runner, engineering tests, source-to-equation maps and protocol notes are included. METHOD_MAP and FULL_TRAIN_PROTOCOL files are historical implementation/protocol records; their original execution-status fields are not current run summaries. This README states the terminal outcome. The bundled cores are the initial independent implementation snapshot; subsequent optimization diagnostics are not represented as an official training protocol or as a successful reproduction. The summary above is an experiment record, not a guarantee that these default settings reproduce it.

## Sources

- Zhu et al., *Multimodal Invariant Sentiment Representation Learning*, Findings of ACL 2025, pp. 14743-14755.
- *TriagedMSA: Triaging Sentimental Disagreement in Multimodal Sentiment Analysis*, DOI: 10.1109/TAFFC.2024.3524789.

Consult each model's METHOD_MAP.md and FULL_TRAIN_PROTOCOL.json for reported settings, independent choices and unresolved ambiguities. In particular, uncertainty about architecture details, feature pipelines and loss reductions prevents an exact-equivalence claim. The runner supports exploratory consistency scheduling; this is not a confirmed TriagedMSA paper setting.

## Running

Use Python 3.11. `requirements.txt` records the versions used for publication CPU
contract checks, not a lockfile for the historical GPU experiments. Install a
PyTorch build compatible with your platform and CUDA driver. From this directory:

```bash
python -m venv zzs-venv-baselines
source zzs-venv-baselines/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s misr -p 'test_*.py'
python -m unittest discover -s triagedmsa -p 'test_*.py'
python -m unittest discover -s . -p 'test_training_contract.py'
```

These tests use synthetic CPU inputs, including a randomly initialized tiny BERT;
they do not download models, start benchmark training, or verify published scores.
Run the three discovery commands separately: recursive discovery alone can miss
the MISR folder, which is not a Python package.

### Example configurations (not historical final configs)

`configs/{misr,triagedmsa}_{mosi,mosei}.example.json` supplies four illustrative
starting points. Each has `_example_only: true`; the runner refuses to execute
it until that flag is removed after review. Copy to an ignored `*.local.json`,
replace the data path/hash and local BERT directory, and declare your protocol.
Relative paths resolve from the working directory, not the config file's directory.
No example is claimed to regenerate the five-seed summary above.

Paper-listed values are taken from the historical protocol records. Seed 1111
is only an illustrative single seed (not a recovered five-seed list). Validation
MAE selection, patience 20 for MISR, Adam for TriagedMSA, full BERT fine-tuning,
TriagedMSA BERT LR 1e-4, zero weight decay and no clipping are independent choices.
The examples do not enable the exploratory consistency schedule. Audio/vision
dimensions and TriagedMSA `max_length` are inferred from the train features;
do not repeat those keys in `model_kwargs`. Set `language_dim` to the local
BERT hidden size; 768 is the example choice, not verified main-table provenance.

The full runner requires an NVIDIA CUDA GPU, uses logical `cuda:0`, and caps its
allocator at 40% of device memory. It has no CPU training mode; CPU tests are
separate. These are manual commands for a reviewed local config, not commands
executed during the publication audit:

```bash
python train_full.py --config configs/misr_mosi.local.json --run-dir runs/zzs-misr-mosi-seed1111
# Resume an interrupted run with identical config/source/data:
python train_full.py --config configs/misr_mosi.local.json --run-dir runs/zzs-misr-mosi-seed1111 --resume
# Only after validation-only selection and all protocol choices are frozen:
python train_full.py --config configs/misr_mosi.local.json --run-dir runs/zzs-misr-mosi-seed1111 --evaluate-only
```

Training writes `best.pt`, `latest.pt`, validation predictions, history and receipts.
Deferred evaluation loads the validation-selected checkpoint and writes test
predictions/metrics. These outputs can contain local paths and sample IDs; do not
publish them without separate review. `--resume` is for interruption recovery,
not changing a completed run's epoch budget. TriagedMSA stopped checkpoints lack
a next-epoch APD cache and are rejected. Source hashes intentionally prevent
resuming/evaluating older checkpoints with this edited publication runner.

### Data and dependency contract

Obtain aligned benchmark features and a BERT checkpoint separately. No tokenizer
or feature extractor is provided. The checkpoint must be available locally to
Transformers `AutoModel`, accept BERT token/type/mask arguments and expose
`last_hidden_state`; all its parameters are trainable. Required pickle structure:

- Top-level `train`, `valid`, `test` dictionaries (plain `.pkl` or gzip `.gz`).
- `text_bert`: integer `[N,3,T]`, channels token IDs, 0/1 attention mask, type IDs.
- `audio`, `vision`: float `[N,T,D]`, matching text positions, finite on valid tokens.
- `regression_labels`: finite continuous `[N]` or `[N,1]` sentiment labels.
- `id` or `ids`: unique stable sample IDs, disjoint across splits. Without IDs,
  split-prefixed row IDs are generated; these cannot detect cross-split duplication.

Masks must be nonempty and right-padded. MOSI usually uses A/V widths 5/20;
MOSEI uses 74/35. Check special-token/subword alignment yourself: matching shapes
does not establish semantic alignment. The runner checks the exact data SHA-256
and structural constraints, not feature provenance. It unpickles the complete
file even during training, but only constructs/uses test batches in deferred
evaluation. Never use test outcomes to choose a configuration. Only load trusted
pickles and checkpoints; loading them can execute code.

`audit_data.py` reports structural checks and split identities. The older
`run_real_data_smoke.py` is specifically a CUDA/MOSEI two-batch admission tool,
not a generic trainer or APD lifecycle test; its fixed receipt wording is historical.
Neither was run against real data in this publication audit. All Python imports
used by the full runner are included here or supplied by `requirements.txt`.

No datasets, original-paper PDFs, correspondence, private server paths, credentials, checkpoints or individual test transcripts are distributed here. Engineering tests check implementation contracts, not benchmark reproduction success.
