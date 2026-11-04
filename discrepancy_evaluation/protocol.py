"""Shared strict parsing and reproducibility primitives (stdlib only)."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path

VERSION = "1.0.0"
SOURCE_TYPES = {"official_labels", "frozen_reference_predictions"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite(value, context):
    if isinstance(value, bool) or value is None or str(value).strip() == "":
        raise ValueError(f"{context}: missing or invalid number")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{context}: invalid number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context}: nonfinite number")
    return result


def rows(data, fields):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    if reader.fieldnames != fields:
        raise ValueError(f"CSV header must be exactly {','.join(fields)}")
    result, seen = [], set()
    for line, row in enumerate(reader, 2):
        if None in row or any(v is None or not v.strip() for v in row.values()):
            raise ValueError(f"CSV line {line}: missing or malformed fields")
        key = (row["dataset"], row["sample_id"])
        if any(v != v.strip() for v in key):
            raise ValueError(f"CSV line {line}: IDs must not have surrounding whitespace")
        if key in seen:
            raise ValueError(f"duplicate ID {key!r}")
        seen.add(key)
        result.append(row)
    if not result:
        raise ValueError("CSV contains no samples")
    return result


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def config_from_bytes(data):
    config = json.loads(data, object_pairs_hook=unique_object)
    if not isinstance(config, dict) or set(config) != {"source_type", "source_description", "datasets"}:
        raise ValueError("config requires exactly source_type, source_description, datasets")
    if config["source_type"] not in SOURCE_TYPES:
        raise ValueError("unsupported source_type")
    if not isinstance(config["source_description"], str) or not config["source_description"].strip():
        raise ValueError("source_description must identify label release or frozen reference")
    datasets = config["datasets"]
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("datasets must be a nonempty object")
    for name, spec in datasets.items():
        if not name.strip() or name != name.strip():
            raise ValueError("invalid dataset name")
        if not isinstance(spec, dict) or set(spec) != {"edges", "closure"}:
            raise ValueError(f"{name}: explicit edges and closure required")
        if spec["closure"] not in ("left", "right"):
            raise ValueError("closure must be left or right")
        edges = spec["edges"]
        if not isinstance(edges, list) or len(edges) < 2:
            raise ValueError("at least two explicit numeric edges required")
        if any(type(x) not in (int, float) for x in edges):
            raise ValueError("edges must be JSON numbers")
        values = [finite(x, "edge") for x in edges]
        if values[0] < 0 or any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError("edges must be nonnegative and strictly increasing")
    return config


def bin_for(value, spec):
    for i, (lo, hi) in enumerate(zip(spec["edges"], spec["edges"][1:])):
        if (lo <= value < hi if spec["closure"] == "left" else lo < value <= hi):
            return str(i)
    return None


def csv_bytes(fields, records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue().encode("utf-8")


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def publish(directory, files):
    # Never overwrite a previous release or leave a partially published directory.
    import os
    import shutil
    import tempfile

    directory = Path(directory)
    if directory.exists():
        raise ValueError(f"output directory already exists: {directory}")
    stage = Path(tempfile.mkdtemp(prefix=".zzs-protocol-", dir=directory.parent))
    try:
        for name, data in files.items():
            (stage / name).write_bytes(data)
        if directory.exists():
            raise ValueError("output directory appeared during publication")
        os.rename(stage, directory)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
