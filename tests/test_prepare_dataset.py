"""Check balancing, geometric grouping and immutable dataset provenance."""

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(ROOT)] + sys.path):
    spec = importlib.util.spec_from_file_location("prepare_dataset", ROOT / "prepare_dataset.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def frame(self, xyz, yaw=None):
        data = pd.DataFrame(xyz, columns=["x", "y", "z"])
        data["roll"] = 0.0
        data["pitch"] = 0.0
        data["yaw"] = yaw if yaw is not None else 0.0
        return data

    def test_euler_wrap_is_a_near_rotation(self):
        frame = self.frame([[0, 0, 0], [0, 0, 0]], [np.pi - 0.001, -np.pi + 0.001])
        groups, pairs = prepare.pose_groups(frame)
        self.assertEqual(groups[0], groups[1])
        self.assertAlmostEqual(pairs.angle_rad.iloc[0], 0.002, places=8)

    def test_both_position_and_rotation_must_be_close(self):
        frame = self.frame([[0, 0, 0], [0.002, 0, 0], [0, 0, 0]], [0, 0, 0.1])
        groups, pairs = prepare.pose_groups(frame)
        self.assertEqual(len(set(groups)), 3)
        self.assertTrue(pairs.empty)

    def test_transitive_neighbors_share_a_group(self):
        frame = self.frame([[0, 0, 0], [0.00075, 0, 0], [0.0015, 0, 0]])
        groups, pairs = prepare.pose_groups(frame)
        self.assertEqual(len(set(groups)), 1)
        self.assertEqual(len(pairs), 2)

    def test_balancing_is_reproducible_without_replacement(self):
        frame = pd.DataFrame({"success": [0] * 15 + [1] * 10})
        a = prepare.balanced_indices(frame, 8, 42)
        b = prepare.balanced_indices(frame, 8, 42)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(set(a)), 16)
        self.assertEqual(frame.iloc[a].success.value_counts().to_dict(), {0: 8, 1: 8})

    def test_not_enough_real_samples_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Not enough"):
            prepare.balanced_indices(pd.DataFrame({"success": [0, 1]}), 2, 42)

    def test_changed_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "batch.json").write_text("{}")
            (root / "cohort.json").write_text(json.dumps({
                "runs": [{"run_id": "example", "manifest": "batch.json",
                          "manifest_sha256": "incorrect"}]}))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                prepare.load_cohort(root / "cohort.json", root)

    def test_actual_cohort_selection_preserves_exact_sources_and_raw_files(self):
        manifest = ROOT / "output/dev_pr2_cube_v1_manifest.json"
        records, hashes, _ = prepare.load_cohort(manifest)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "balanced"
            data, provenance, metadata = prepare.prepare_dataset(manifest, output)
            self.assertEqual(data.shape, (240, 7))
            self.assertEqual(list(data.columns), prepare.COLUMNS)
            self.assertEqual(data.success.value_counts().to_dict(), {0: 120, 1: 120})
            self.assertFalse(provenance.duplicated(["run_id", "attempt_id"]).any())
            expected = records.iloc[provenance.source_cohort_row.to_numpy() - 1]
            np.testing.assert_allclose(data.to_numpy(), expected[prepare.COLUMNS].to_numpy())
            self.assertEqual(provenance.run_id.tolist(), expected.run_id.tolist())
            self.assertEqual(provenance.attempt_id.tolist(), expected.attempt_id.tolist())
            for path, digest in hashes.items():
                self.assertEqual(prepare.sha256(path), digest)
            for name, digest in metadata["output_hashes"].items():
                self.assertEqual(prepare.sha256(output / name), digest)
            with self.assertRaises(FileExistsError):
                prepare.prepare_dataset(manifest, output)
            with patch.object(sys, "path", [str(ROOT)] + sys.path):
                X, y = prepare.inspect_dataset(output / "development.csv")
            self.assertEqual(X.shape, (240, 6))
            self.assertEqual(y.shape, (240,))


if __name__ == "__main__":
    unittest.main()
