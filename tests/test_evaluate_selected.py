"""Check prediction-only evaluation of the frozen CV-selected artifact."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(ROOT)] + sys.path):
    spec = importlib.util.spec_from_file_location("evaluate_selected", ROOT / "evaluate_selected.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)


class SelectedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = ROOT / "output/dev_pr2_cube_v1_balanced240"
        self.baseline = ROOT / "output/baseline_pr2_cube_v1_seed42"
        self.search = ROOT / "output/cv_pr2_cube_v1_seed42"
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "evaluation"

    def test_no_fit_exact_validation_membership_and_correct_metrics(self):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(
                Pipeline, "fit", side_effect=AssertionError("Fitting is forbidden during evaluation")):
            scores, predictions = evaluation.evaluate_selected(
                self.dataset, self.baseline, self.search, self.output)
        split = pd.read_csv(self.baseline / "split_provenance.csv")
        expected = split.loc[split.split == "validation"].dataset_row_number.tolist()
        self.assertEqual(predictions.dataset_row_number.tolist(), expected)
        self.assertEqual(len(predictions), 48)
        recomputed = evaluation.evaluate(predictions.actual_success, predictions.predicted_success)
        self.assertEqual(scores, recomputed)
        errors = pd.read_csv(self.output / "errors.csv")
        self.assertEqual(len(errors), scores["fp"] + scores["fn"])
        self.assertTrue(errors.outcome.isin(["FP", "FN"]).all())
        metadata = json.loads((self.output / "evaluation_manifest.json").read_text())
        self.assertFalse(metadata["fit_performed"])
        self.assertFalse(metadata["hyperparameters_changed"])
        for path, digest in metadata["source_hashes"].items():
            self.assertEqual(evaluation.sha256(path), digest)

    def test_tampered_model_is_rejected_before_prediction(self):
        copied = Path(self.directory.name) / "search"
        shutil.copytree(self.search, copied)
        with (copied / "selected_training_model.joblib").open("ab") as model:
            model.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            evaluation.evaluate_selected(self.dataset, self.baseline, copied, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_evaluation_cannot_be_overwritten(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("original")
        with self.assertRaises(FileExistsError):
            evaluation.evaluate_selected(self.dataset, self.baseline, self.search, self.output)
        self.assertEqual(marker.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
