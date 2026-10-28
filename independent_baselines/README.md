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

Use Python 3.11 with PyTorch, NumPy, scikit-learn and transformers installed. From this directory:

```bash
python -m unittest discover -s misr -p 'test_*.py'
python -m unittest discover -s triagedmsa -p 'test_*.py'
python train_full.py --config /path/to/config.json --run-dir /path/to/new_run
```

Training requires separately obtained aligned benchmark features and a locally available BERT checkpoint. The runner reads `text_bert`, `audio`, `vision`, and `regression_labels`; it checks the configured data SHA-256 and validates alignment. Inspect `train_full.py` for configuration fields and evaluation behavior. Evaluation uses validation-selected checkpoints; test data must not be used for configuration selection. Only load trusted pickle datasets.

No datasets, original-paper PDFs, correspondence, private server paths, credentials, checkpoints or individual test transcripts are distributed here. Engineering tests check implementation contracts, not benchmark reproduction success.
