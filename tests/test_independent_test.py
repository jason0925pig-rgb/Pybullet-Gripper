"""Check prediction timing, mandatory execution and test/development separation."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(ROOT)] + sys.path):
    spec = importlib.util.spec_from_file_location("run_independent_test", ROOT / "run_independent_test.py")
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)


class IndependentTestChecks(unittest.TestCase):
    def trial(self, callback):
        hand = independent.PR2Hand(gui=False, seed=42)
        events = []
        try:
            hand.connect()
            pos, (yaw, roll, pitch) = hand.sample_pose()
            ori = independent.p.getQuaternionFromEuler([roll, pitch, yaw])
            hand.reset_trial(pos, ori)
            real_grasp = hand.grasp

            def grasp():
                events.append("close")
                return real_grasp()

            def observe(features):
                self.assertNotIn("close", events)
                self.assertEqual(list(features), independent.FEATURE_COLUMNS)
                events.append("prediction")

            with patch.object(hand, "grasp", side_effect=grasp):
                record = hand.run_trial(pos, ori, observe if callback else None)
            return record, events
        finally:
            hand.disconnect()

    def test_callback_precedes_closure_and_preserves_original_physics(self):
        plain, _ = self.trial(False)
        observed, events = self.trial(True)
        self.assertEqual(events[0], "prediction")
        self.assertIn("close", events)
        for name, value in plain.items():
            if isinstance(value, (float, int)):
                self.assertAlmostEqual(value, observed[name])
            else:
                self.assertEqual(value, observed[name])

    def test_predicted_failures_are_executed_and_logged_before_outcomes(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(Pipeline, "fit", side_effect=AssertionError("No fitting allowed")), \
             patch.object(Pipeline, "predict", side_effect=lambda X: np.zeros(len(X), dtype=int)):
            records, summary = independent.run_test(Path(directory) / "test", seed=7777)
            self.assertEqual(len(records), 10)
            self.assertTrue(records.prediction.eq(0).all())
            self.assertFalse(summary["fit_performed"])
            events = [json.loads(line) for line in (Path(directory) / "test/events.jsonl").read_text().splitlines()]
            self.assertEqual(len(events), 20)
            for before, after in zip(events[::2], events[1::2]):
                self.assertEqual(before["event"], "prediction_before_closure")
                self.assertEqual(after["event"], "outcome_after_lift")
                self.assertEqual(before["attempt_id"], after["attempt_id"])
                self.assertNotIn("success", before)
            self.assertEqual(summary["exact_measured_feature_overlap_with_development_cohort"], 0)

    def test_development_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "new test seed"):
                independent.run_test(Path(directory) / "test", seed=1001)
            self.assertFalse((Path(directory) / "test").exists())


if __name__ == "__main__":
    unittest.main()
