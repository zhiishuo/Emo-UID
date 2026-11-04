"""Evaluate exactly one prediction per included (dataset, sample_id)."""

import argparse
import json
import math
from pathlib import Path

from build_groups import GROUP_FIELDS
from protocol import VERSION, bin_for, config_from_bytes, csv_bytes, digest, finite, json_bytes, publish, rows

PREDICTION_FIELDS = ["dataset", "sample_id", "target", "prediction"]


def evaluate(release, predictions, output):
    release = Path(release)
    manifest_data = (release / "manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported release schema")
    hashes = manifest.get("files_sha256", {})
    if set(hashes) != {"groups.csv", "exclusions.csv", "config.json"}:
        raise ValueError("invalid release file manifest")
    files = {name: (release / name).read_bytes() for name in hashes}
    for name, data in files.items():
        if digest(data) != hashes[name]:
            raise ValueError(f"release hash mismatch: {name}")
    config = config_from_bytes(files["config.json"])
    groups = rows(files["groups.csv"], GROUP_FIELDS)
    prediction_data = Path(predictions).read_bytes()
    predicted = rows(prediction_data, PREDICTION_FIELDS)
    key = lambda row: (row["dataset"], row["sample_id"])
    lookup = {key(row): row for row in predicted}
    expected = {key(row) for row in groups}
    if set(lookup) != expected:
        raise ValueError(f"strict coverage failed: missing={len(expected - set(lookup))}, extra={len(set(lookup) - expected)}")
    errors = {(name, str(i)): [] for name, spec in config["datasets"].items()
              for i in range(len(spec["edges"]) - 1)}
    for row in groups:
        spec = config["datasets"].get(row["dataset"])
        values = [finite(row[k], k) for k in ("text", "audio", "visual")]
        discrepancy = finite(max(values) - min(values), "computed discrepancy")
        if spec is None or finite(row["discrepancy"], "discrepancy") != discrepancy or bin_for(discrepancy, spec) != row["group"]:
            raise ValueError("inconsistent group record")
        pred = lookup[key(row)]
        error = finite(abs(finite(pred["prediction"], "prediction") - finite(pred["target"], "target")), "absolute error")
        errors[(row["dataset"], row["group"])].append(error)
    metrics = []
    for (dataset, group), values in sorted(errors.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        spec = config["datasets"][dataset]
        i = int(group)
        mae = finite(math.fsum(v / len(values) for v in values), "MAE") if values else "NA"
        metrics.append(dict(dataset=dataset, group=group, lower=spec["edges"][i],
                            upper=spec["edges"][i + 1], closure=spec["closure"],
                            n=len(values), mae=mae))
    result = csv_bytes(["dataset", "group", "lower", "upper", "closure", "n", "mae"], metrics)
    evaluation_manifest = {
        "schema_version": 1, "tool_version": VERSION, "n": len(groups),
        "release_manifest_sha256": digest(manifest_data),
        "predictions_sha256": digest(prediction_data), "metrics_sha256": digest(result),
        "code_sha256": {name: digest(Path(__file__).with_name(name).read_bytes())
                        for name in ("evaluate_groups.py", "build_groups.py", "protocol.py")},
    }
    publish(output, {"metrics.csv": result, "manifest.json": json_bytes(evaluation_manifest)})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-dir", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        metrics = evaluate(args.groups_dir, args.predictions, args.output)
    except (ValueError, OSError, TypeError, KeyError, OverflowError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"evaluated {sum(row['n'] for row in metrics)} samples")


if __name__ == "__main__":
    main()
