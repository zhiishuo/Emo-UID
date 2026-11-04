"""Build a new, explicitly configured discrepancy grouping release."""

import argparse
from pathlib import Path

from protocol import VERSION, bin_for, config_from_bytes, csv_bytes, digest, finite, json_bytes, publish, rows

INPUT_FIELDS = ["dataset", "sample_id", "text", "audio", "visual"]
GROUP_FIELDS = INPUT_FIELDS + ["discrepancy", "group"]
EXCLUSION_FIELDS = INPUT_FIELDS + ["discrepancy", "reason"]


def build(source, config_path, output):
    source_data = Path(source).read_bytes()
    config_data = Path(config_path).read_bytes()
    config = config_from_bytes(config_data)
    samples = rows(source_data, INPUT_FIELDS)
    if {r["dataset"] for r in samples} != set(config["datasets"]):
        raise ValueError("input dataset names must exactly match configured datasets")
    groups, excluded = [], []
    for row in sorted(samples, key=lambda r: (r["dataset"], r["sample_id"])):
        values = [finite(row[k], f"{row['dataset']}/{row['sample_id']}/{k}") for k in INPUT_FIELDS[2:]]
        discrepancy = finite(max(values) - min(values), "computed discrepancy")
        group = bin_for(discrepancy, config["datasets"][row["dataset"]])
        record = dict(row, discrepancy=repr(discrepancy))
        if group is None:
            excluded.append(dict(record, reason="outside_configured_intervals"))
        else:
            groups.append(dict(record, group=group))
    files = {
        "groups.csv": csv_bytes(GROUP_FIELDS, groups),
        "exclusions.csv": csv_bytes(EXCLUSION_FIELDS, excluded),
        "config.json": config_data,
    }
    manifest = {
        "schema_version": 1, "tool_version": VERSION,
        "purpose": "new_reusable_protocol_not_historic_fig3_recovery",
        "source_type": config["source_type"],
        "source_sha256": digest(source_data),
        "config_sha256": digest(config_data),
        "counts": {"input": len(samples), "included": len(groups), "excluded": len(excluded)},
        "files_sha256": {name: digest(data) for name, data in files.items()},
        "code_sha256": {name: digest(Path(__file__).with_name(name).read_bytes())
                        for name in ("build_groups.py", "protocol.py")},
    }
    files["manifest.json"] = json_bytes(manifest)
    publish(output, files)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = build(args.input, args.config, args.output)
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(result["counts"])


if __name__ == "__main__":
    main()
