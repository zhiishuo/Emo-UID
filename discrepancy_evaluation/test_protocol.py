"""SYNTHETIC TEST FIXTURES ONLY. No historical samples or scientific results."""

import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from build_groups import build
from evaluate_groups import evaluate
from protocol import config_from_bytes, digest

ROOT = Path(__file__).resolve().parent


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        # Keep all test-generated files inside the assigned toolkit directory.
        self.temp = tempfile.TemporaryDirectory(prefix="zzs-synthetic-test-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "synthetic.csv"
        self.config = self.root / "synthetic_config.json"
        self.predictions = self.root / "synthetic_predictions.csv"
        self.release = self.root / "groups"
        self.output = self.root / "evaluation"
        self.cfg = {"source_type": "official_labels", "source_description": "SYNTHETIC TEST FIXTURE ONLY",
                    "datasets": {"synthetic": {"edges": [0, 1, 2, 3, 4], "closure": "left"}}}
        self.save_config()
        self.source.write_text("dataset,sample_id,text,audio,visual\nsynthetic,a,0,0,0\nsynthetic,b,-0.5,0.5,0\nsynthetic,c,0,4,2\n", encoding="utf-8")
        self.predictions.write_text("dataset,sample_id,target,prediction\nsynthetic,b,1,3\nsynthetic,a,0,1\n", encoding="utf-8")

    def save_config(self):
        self.config.write_text(json.dumps(self.cfg), encoding="utf-8")

    def build(self):
        return build(self.source, self.config, self.release)

    def read_csv(self, name):
        return list(csv.DictReader(io.StringIO((self.release / name).read_text())))

    def test_left_boundaries_range_and_exclusions(self):
        manifest = self.build()
        groups = self.read_csv("groups.csv")
        self.assertEqual([(r["sample_id"], r["group"]) for r in groups], [("a", "0"), ("b", "1")])
        self.assertEqual(groups[1]["discrepancy"], "1.0")
        self.assertEqual(self.read_csv("exclusions.csv")[0]["reason"], "outside_configured_intervals")
        self.assertEqual(manifest["counts"], {"input": 3, "included": 2, "excluded": 1})

    def test_right_boundaries(self):
        self.cfg["datasets"]["synthetic"]["closure"] = "right"
        self.save_config()
        self.build()
        self.assertEqual([(r["sample_id"], r["group"]) for r in self.read_csv("groups.csv")], [("b", "0"), ("c", "3")])
        self.assertEqual(self.read_csv("exclusions.csv")[0]["sample_id"], "a")

    def test_frozen_reference_source(self):
        self.cfg["source_type"] = "frozen_reference_predictions"
        self.save_config()
        self.assertEqual(self.build()["source_type"], "frozen_reference_predictions")

    def test_hashes_and_repeatability(self):
        manifest = self.build()
        self.assertEqual(manifest["source_sha256"], digest(self.source.read_bytes()))
        self.assertEqual(manifest["config_sha256"], digest(self.config.read_bytes()))
        for name, checksum in manifest["files_sha256"].items():
            self.assertEqual(digest((self.release / name).read_bytes()), checksum)
        second = self.root / "second"
        build(self.source, self.config, second)
        for file in self.release.iterdir():
            self.assertEqual(file.read_bytes(), (second / file.name).read_bytes())

    def test_duplicate_ids_rejected(self):
        self.source.write_text(self.source.read_text() + "synthetic,a,1,1,1\n")
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            self.build()
        self.assertFalse(self.release.exists())

    def test_invalid_numbers_rejected(self):
        for value in ("", "NaN", "inf", "-inf", "bad", "1e400"):
            with self.subTest(value=value):
                self.source.write_text(f"dataset,sample_id,text,audio,visual\nsynthetic,a,{value},0,1\n")
                with self.assertRaises(ValueError):
                    self.build()
                self.assertFalse(self.release.exists())

    def test_overflow_range_rejected(self):
        self.source.write_text("dataset,sample_id,text,audio,visual\nsynthetic,a,-1e308,1e308,0\n")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.build()

    def test_missing_malformed_headers_and_fields(self):
        for content in ("dataset,sample_id,text,audio\nsynthetic,a,0,1\n",
                        "dataset,sample_id,text,audio,visual\nsynthetic,a,0,1\n",
                        "dataset,sample_id,text,audio,visual\nsynthetic,a,0,1,2,3\n",
                        "dataset,sample_id,text,audio,visual\nsynthetic,,0,1,2\n",
                        "dataset,sample_id,text,audio,visual\n"):
            with self.subTest(content=content):
                self.source.write_text(content)
                with self.assertRaises(ValueError):
                    self.build()

    def test_dataset_mismatch(self):
        self.cfg["datasets"]["absent"] = {"edges": [0, 1], "closure": "left"}
        self.save_config()
        with self.assertRaisesRegex(ValueError, "dataset names"):
            self.build()

    def test_exact_config_required(self):
        for field in ("source_type", "source_description", "datasets"):
            config = dict(self.cfg)
            del config[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                config_from_bytes(json.dumps(config).encode())
        for spec in ({"closure": "left"}, {"edges": [0, 1]},
                     {"edges": [0, 1], "closure": "both"}):
            self.cfg["datasets"]["synthetic"] = spec
            with self.assertRaises(ValueError):
                config_from_bytes(json.dumps(self.cfg).encode())

    def test_invalid_edges(self):
        for edges in ([0], [0, 0, 1], [2, 1], [-1, 1], [0, float("inf")],
                      [0, float("nan")], [0, "1"], [False, 1]):
            with self.subTest(edges=edges):
                self.cfg["datasets"]["synthetic"]["edges"] = edges
                self.save_config()
                with self.assertRaises(ValueError):
                    self.build()

    def test_duplicate_config_keys(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            config_from_bytes(b'{"source_type":"official_labels","source_type":"official_labels"}')

    def test_strict_source_type(self):
        self.cfg["source_type"] = "inferred"
        self.save_config()
        with self.assertRaisesRegex(ValueError, "source_type"):
            self.build()

    def test_id_join_mae_and_empty_bins(self):
        self.build()
        result = evaluate(self.release, self.predictions, self.output)
        self.assertEqual([(r["n"], r["mae"]) for r in result], [(1, 1.0), (1, 2.0), (0, "NA"), (0, "NA")])
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["metrics_sha256"], digest((self.output / "metrics.csv").read_bytes()))
        self.assertEqual(manifest["predictions_sha256"], digest(self.predictions.read_bytes()))

    def test_mean_with_multiple_samples(self):
        self.cfg["datasets"]["synthetic"]["edges"] = [0, 3]
        self.save_config()
        self.build()
        self.assertEqual(evaluate(self.release, self.predictions, self.output)[0]["mae"], 1.5)

    def test_missing_extra_excluded_prediction_coverage(self):
        self.build()
        for records in ("synthetic,a,0,1\n", "synthetic,a,0,1\nsynthetic,b,1,3\nsynthetic,c,0,1\n",
                        "synthetic,a,0,1\nsynthetic,b,1,3\nother,x,0,0\n"):
            self.predictions.write_text("dataset,sample_id,target,prediction\n" + records)
            with self.assertRaisesRegex(ValueError, "strict coverage"):
                evaluate(self.release, self.predictions, self.output)
            self.assertFalse(self.output.exists())

    def test_duplicate_predictions(self):
        self.build()
        self.predictions.write_text(self.predictions.read_text() + "synthetic,a,0,0\n")
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            evaluate(self.release, self.predictions, self.output)

    def test_invalid_predictions_and_targets(self):
        self.build()
        for target, prediction in (("NaN", "1"), ("0", "inf"), ("", "1"), ("-1e308", "1e308")):
            self.predictions.write_text(f"dataset,sample_id,target,prediction\nsynthetic,a,{target},{prediction}\nsynthetic,b,0,0\n")
            with self.assertRaises(ValueError):
                evaluate(self.release, self.predictions, self.output)

    def test_tampering_detected(self):
        self.build()
        path = self.release / "groups.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            evaluate(self.release, self.predictions, self.output)

    def test_no_overwrite(self):
        self.build()
        original = (self.release / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.build()
        self.assertEqual((self.release / "manifest.json").read_bytes(), original)

    def test_all_excluded_has_audit_but_no_evaluation(self):
        self.cfg["datasets"]["synthetic"]["edges"] = [5, 6]
        self.save_config()
        self.assertEqual(self.build()["counts"]["excluded"], 3)
        with self.assertRaisesRegex(ValueError, "no samples"):
            evaluate(self.release, self.predictions, self.output)

    def test_same_sample_id_in_different_datasets(self):
        self.cfg["datasets"]["second"] = {"edges": [0, 2], "closure": "left"}
        self.save_config()
        self.source.write_text(self.source.read_text() + "second,a,0,0,0\n")
        self.predictions.write_text(self.predictions.read_text() + "second,a,2,2\n")
        self.build()
        result = evaluate(self.release, self.predictions, self.output)
        self.assertEqual(sum(r["n"] for r in result), 3)

    def test_cli_smoke_and_required_config(self):
        built = subprocess.run([sys.executable, str(ROOT / "build_groups.py"), "--input", str(self.source),
                                "--config", str(self.config), "--output", str(self.release)], capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stderr)
        evaluated = subprocess.run([sys.executable, str(ROOT / "evaluate_groups.py"), "--groups-dir", str(self.release),
                                    "--predictions", str(self.predictions), "--output", str(self.output)], capture_output=True, text=True)
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        missing = subprocess.run([sys.executable, str(ROOT / "build_groups.py")], capture_output=True)
        self.assertEqual(missing.returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
