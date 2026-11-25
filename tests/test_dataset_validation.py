"""Validate the CSV contract and audit the current Stage 1 collection."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inspect_dataset", ROOT / "inspect_dataset.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
COLUMNS = module.FEATURE_COLUMNS + ["success"]
DATA = ROOT / "output" / "pr2_cube_demo"


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "fixture.csv"
        self.frame = pd.DataFrame([
            [0.01, 0.02, 0.33, 0.0, 1.0, 0.0, 0],
            [0.02, 0.03, 0.32, 0.1, 0.9, 0.2, 1],
        ], columns=COLUMNS)

    def inspect(self, frame):
        frame.to_csv(self.path, index=False)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            X, y = module.inspect_dataset(self.path)
        return X, y, output.getvalue()

    def test_valid_data_preserves_features_labels_and_row_alignment(self):
        X, y, _ = self.inspect(self.frame)
        pd.testing.assert_frame_equal(X, self.frame[module.FEATURE_COLUMNS])
        pd.testing.assert_series_equal(y, self.frame["success"])

    def test_missing_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected columns"):
            self.inspect(self.frame.drop(columns="yaw"))

    def test_wrong_column_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected columns"):
            self.inspect(self.frame[list(reversed(COLUMNS))])

    def test_extra_input_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected columns"):
            self.inspect(self.frame.assign(post_lift_height=0.3))

    def test_empty_table_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no completed trials"):
            self.inspect(self.frame.iloc[:0])

    def test_missing_value_is_rejected(self):
        self.frame.loc[0, "x"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing values"):
            self.inspect(self.frame)

    def test_text_feature_is_rejected(self):
        self.frame["x"] = ["invalid", "0.02"]
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            self.inspect(self.frame)

    def test_nonfinite_value_is_rejected(self):
        for value in [np.inf, -np.inf]:
            with self.subTest(value=value):
                self.frame.loc[0, "x"] = value
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    self.inspect(self.frame)

    def test_invalid_labels_are_rejected(self):
        for value in [0.5, -1, 2]:
            with self.subTest(value=value):
                frame = self.frame.copy()
                frame["success"] = frame.success.astype(float)
                frame.loc[0, "success"] = value
                with self.assertRaisesRegex(ValueError, "either 0 or 1"):
                    self.inspect(frame)

    def test_single_class_is_reported_without_fabricating_labels(self):
        self.frame["success"] = 0
        _, y, output = self.inspect(self.frame)
        self.assertIn("Only one class", output)
        self.assertEqual(y.tolist(), [0, 0])

    def test_imbalance_is_reported_without_dropping_rows(self):
        frame = pd.concat([self.frame, self.frame.iloc[:1]], ignore_index=True)
        _, y, output = self.inspect(frame)
        self.assertIn("imbalanced", output)
        self.assertEqual(len(y), 3)


class CurrentCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = [DATA / name for name in ["grasps.csv", "attempts.csv", "config.json"]]
        cls.before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.files}
        cls.grasps = pd.read_csv(cls.files[0])
        cls.audit = pd.read_csv(cls.files[1])
        cls.config = json.loads(cls.files[2].read_text())

    @classmethod
    def tearDownClass(cls):
        for path in cls.files:
            if hashlib.sha256(path.read_bytes()).hexdigest() != cls.before[path]:
                raise AssertionError(f"Source artifact changed during validation: {path}")

    def test_current_training_table_is_valid(self):
        with contextlib.redirect_stdout(io.StringIO()):
            X, y = module.inspect_dataset(self.files[0])
        self.assertEqual(len(X), len(y))
        self.assertEqual(set(y), {0, 1})

    def test_completed_audit_rows_match_training_rows_in_order(self):
        completed = self.audit[self.audit.run_status == "completed"]
        self.assertEqual(len(completed), len(self.grasps))
        np.testing.assert_allclose(
            completed[COLUMNS].to_numpy(dtype=float),
            self.grasps.to_numpy(dtype=float), rtol=1e-12, atol=1e-12)
        self.assertTrue(self.audit.attempt_id.is_unique)
        self.assertTrue(self.audit.attempt_id.is_monotonic_increasing)
        self.assertEqual(self.audit.gripper.unique().tolist(), ["PR2"])
        self.assertEqual(self.audit.object.unique().tolist(), ["cube_small"])

    def test_rejected_or_error_attempts_have_no_training_label(self):
        self.assertTrue(self.audit.run_status.isin(
            ["completed", "rejected_initial_collision", "error"]).all())
        excluded = self.audit[self.audit.run_status != "completed"]
        self.assertTrue(excluded.success.isna().all())

    def test_labels_agree_with_saved_diagnostics(self):
        completed = self.audit[self.audit.run_status == "completed"]
        successes = completed[completed.success == 1]
        failures = completed[completed.success == 0]
        self.assertFalse(successes.empty)
        self.assertTrue(successes.failure_reason.isna().all())
        self.assertTrue((successes.stable_hold_seconds > 3).all())
        self.assertTrue((successes.stable_hold_seconds >= self.config["hold_seconds"] - 1e-9).all())
        self.assertTrue((successes.min_hold_height - successes.initial_box_height
                         >= self.config["min_lift"] - 1e-9).all())
        self.assertTrue((successes.max_relative_slip <= self.config["slip_tolerance"]).all())
        self.assertTrue((successes.max_relative_rotation
                         <= self.config["rotation_tolerance"]).all())
        self.assertTrue(successes.touched.all())
        self.assertTrue(failures.failure_reason.notna().all())
        for _, row in failures.iterrows():
            reason = row.failure_reason
            if reason == "insufficient_lift":
                self.assertLess(row.min_hold_height - row.initial_box_height,
                                self.config["min_lift"])
            elif reason == "excessive_slip":
                self.assertTrue(row.max_relative_slip > self.config["slip_tolerance"]
                                or row.max_relative_rotation > self.config["rotation_tolerance"])
            else:
                self.assertIn(reason, ["floor_contact", "lost_gripper_contact"])
        # Contact/floor histories are not exported, so those two failure reasons
        # cannot be independently reconstructed from the CSV alone.


if __name__ == "__main__":
    unittest.main()
