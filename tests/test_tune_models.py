"""Check frozen holdout exclusion and fold-local model/preprocessing fits."""

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
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(ROOT)] + sys.path):
    spec = importlib.util.spec_from_file_location("tune_models", ROOT / "tune_models.py")
    tuning = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuning)


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(np.arange(240).reshape(40, 6), columns=tuning.FEATURE_COLUMNS)
        self.y = pd.Series([0, 1] * 20)
        self.provenance = pd.DataFrame({"dataset_row_number": np.arange(1, 41),
                                        "pose_group_id": [str(i) for i in range(40)]})

    def test_same_folds_and_one_out_of_fold_prediction_per_candidate_and_row(self):
        scores, summary, assignments, predictions = tuning.compare_candidates(
            self.X, self.y, self.provenance)
        self.assertEqual(len(scores), 30)
        self.assertEqual(len(summary), 6)
        self.assertTrue(assignments.cv_validation_fold.isin(range(1, 6)).all())
        self.assertEqual(len(predictions), 240)
        self.assertFalse(predictions.duplicated(["candidate", "dataset_row_number"]).any())
        self.assertEqual(predictions.groupby("dataset_row_number").cv_validation_fold.nunique().max(), 1)
        for _, row in summary.iterrows():
            values = scores.loc[scores.candidate == row.candidate, "accuracy"]
            self.assertAlmostEqual(row.mean_accuracy, values.mean())
            self.assertAlmostEqual(row.std_accuracy, values.std(ddof=0))
        self.assertTrue(summary.mean_accuracy.is_monotonic_decreasing)

    def test_group_leakage_is_rejected(self):
        self.provenance["pose_group_id"] = "same_group"
        with self.assertRaisesRegex(ValueError, "group-aware splitter"):
            tuning.compare_candidates(self.X, self.y, self.provenance)

    def test_tree_only_search_preserves_folds_and_scores(self):
        full_scores, _, full_folds, _ = tuning.compare_candidates(
            self.X, self.y, self.provenance)
        scores, summary, folds, predictions = tuning.compare_candidates(
            self.X, self.y, self.provenance, model_family="decision_tree")
        pd.testing.assert_frame_equal(folds, full_folds)
        expected = full_scores.loc[full_scores.candidate.str.startswith("tree_")].reset_index(drop=True)
        pd.testing.assert_frame_equal(scores, expected)
        self.assertEqual(len(summary), 3)
        self.assertEqual(len(predictions), len(self.X) * 3)
        self.assertEqual(set(summary.candidate), {"tree_depth_2", "tree_depth_3", "tree_depth_4"})

    def test_actual_tree_search_saves_tree_and_excludes_holdout(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "tree_cv"
            tuning.run_search(ROOT / "output/dev_pr2_cube_v1_balanced240",
                              ROOT / "output/baseline_pr2_cube_v1_seed42",
                              output, model_family="decision_tree")
            metadata = json.loads((output / "search_manifest.json").read_text())
            self.assertEqual(metadata["model_family"], "decision_tree")
            self.assertEqual(metadata["training_rows"], 192)
            self.assertEqual(metadata["excluded_outer_validation_rows"], 48)
            self.assertFalse(metadata["outer_validation_evaluated"])
            self.assertEqual(len(pd.read_csv(output / "cv_predictions.csv")), 192 * 3)
            bundle = tuning.joblib.load(output / "selected_training_model.joblib")
            self.assertIsInstance(bundle["estimator"], tuning.DecisionTreeClassifier)
            self.assertEqual(bundle["candidate"], metadata["selected_candidate"])

    def test_unknown_family_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown model family"):
            tuning.candidates(model_family="unknown")

    def test_forest_search_uses_agreed_parameters_and_identical_folds(self):
        models = tuning.candidates(model_family="random_forest")
        self.assertEqual(list(models), [f"forest_depth_{depth}" for depth in [2, 3, 4]])
        for depth, model in zip([2, 3, 4], models.values()):
            self.assertEqual(model.max_depth, depth)
            self.assertEqual(model.n_estimators, 100)
            self.assertEqual(model.min_samples_leaf, 5)
            self.assertEqual(model.random_state, 42)
        _, _, reference_folds, _ = tuning.compare_candidates(
            self.X, self.y, self.provenance, model_family="decision_tree")
        scores, summary, folds, predictions = tuning.compare_candidates(
            self.X, self.y, self.provenance, model_family="random_forest")
        pd.testing.assert_frame_equal(folds, reference_folds)
        self.assertEqual(len(scores), 15)
        self.assertEqual(len(summary), 3)
        self.assertEqual(len(predictions), len(self.X) * 3)
        self.assertFalse(predictions.duplicated(["candidate", "dataset_row_number"]).any())
        self.assertTrue(summary.mean_accuracy.is_monotonic_decreasing)

    def test_actual_forest_search_and_evaluation_do_not_train_on_holdout(self):
        from evaluate_selected import evaluate_selected

        dataset = ROOT / "output/dev_pr2_cube_v1_balanced240"
        baseline = ROOT / "output/baseline_pr2_cube_v1_seed42"
        split = pd.read_csv(baseline / "split_provenance.csv")
        training_ids = set(split.loc[split.split == "training", "dataset_row_number"])
        holdout_ids = set(split.loc[split.split == "validation", "dataset_row_number"])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "forest_cv"
            tuning.run_search(dataset, baseline, output, model_family="random_forest")
            predictions = pd.read_csv(output / "cv_predictions.csv")
            self.assertEqual(set(predictions.dataset_row_number), training_ids)
            self.assertFalse(set(predictions.dataset_row_number) & holdout_ids)
            self.assertEqual(len(predictions), 192 * 3)
            metadata = json.loads((output / "search_manifest.json").read_text())
            self.assertEqual(metadata["model_family"], "random_forest")
            self.assertFalse(metadata["outer_validation_evaluated"])
            self.assertFalse(metadata["independent_test_performed"])
            bundle = tuning.joblib.load(output / "selected_training_model.joblib")
            self.assertIsInstance(bundle["estimator"], tuning.RandomForestClassifier)
            self.assertEqual(len(bundle["estimator"].estimators_), 100)
            with patch.object(tuning.RandomForestClassifier, "fit",
                              side_effect=AssertionError("Evaluation must not fit")):
                _, evaluated = evaluate_selected(dataset, baseline, output,
                                                Path(directory) / "validation")
            self.assertEqual(set(evaluated.dataset_row_number), holdout_ids)
            self.assertEqual(len(evaluated), 48)

    def test_insufficient_class_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least five"):
            tuning.compare_candidates(self.X.iloc[:8], self.y.iloc[:8], self.provenance.iloc[:8])

    def test_every_scaler_is_fit_only_on_its_fold_training_rows(self):
        fitted = []

        def capture(prototype):
            estimator = clone(prototype)
            fitted.append(estimator)
            return estimator

        with patch.object(tuning, "clone", side_effect=capture):
            tuning.compare_candidates(self.X, self.y, self.provenance)
        splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(self.X, self.y))
        pipelines = [model for model in fitted if hasattr(model, "named_steps")]
        self.assertEqual(len(pipelines), 15)
        for i, model in enumerate(pipelines):
            indices = splits[i % 5][0]
            scaler = model.named_steps["scaler"]
            self.assertEqual(scaler.n_samples_seen_, len(indices))
            np.testing.assert_allclose(scaler.mean_, self.X.iloc[indices].mean().to_numpy())

    def test_actual_search_excludes_outer_validation_and_preserves_inputs(self):
        dataset = ROOT / "output/dev_pr2_cube_v1_balanced240"
        baseline = ROOT / "output/baseline_pr2_cube_v1_seed42"
        split = pd.read_csv(baseline / "split_provenance.csv")
        training_ids = set(split.loc[split.split == "training", "dataset_row_number"])
        holdout_ids = set(split.loc[split.split == "validation", "dataset_row_number"])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "cv"
            tuning.run_search(dataset, baseline, output)
            predictions = pd.read_csv(output / "cv_predictions.csv")
            self.assertEqual(set(predictions.dataset_row_number), training_ids)
            self.assertFalse(set(predictions.dataset_row_number) & holdout_ids)
            self.assertEqual(len(predictions), 192 * 6)
            metadata = json.loads((output / "search_manifest.json").read_text())
            self.assertFalse(metadata["outer_validation_evaluated"])
            self.assertFalse(metadata["independent_test_performed"])
            for path, digest in metadata["source_hashes"].items():
                self.assertEqual(tuning.sha256(path), digest)
            bundle = tuning.joblib.load(output / "selected_training_model.joblib")
            estimator = bundle["estimator"]
            if hasattr(estimator, "named_steps"):
                self.assertEqual(estimator.named_steps["scaler"].n_samples_seen_, 192)
            with self.assertRaises(FileExistsError):
                tuning.run_search(dataset, baseline, output)


if __name__ == "__main__":
    unittest.main()
