"""Check baseline splitting, preprocessing isolation, metrics and persistence."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(ROOT)] + sys.path):
    spec = importlib.util.spec_from_file_location("train_baseline", ROOT / "train_baseline.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)


class BaselineTests(unittest.TestCase):
    def test_split_is_reproducible_stratified_and_disjoint(self):
        y = pd.Series([0] * 120 + [1] * 120)
        provenance = pd.DataFrame({"pose_group_id": [str(i) for i in range(240)]})
        a, b = training.make_split(y, provenance)
        again_a, again_b = training.make_split(y, provenance)
        np.testing.assert_array_equal(a, again_a)
        np.testing.assert_array_equal(b, again_b)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(len(a), 192)
        self.assertEqual(len(b), 48)
        self.assertEqual(y.iloc[a].value_counts().to_dict(), {0: 96, 1: 96})
        self.assertEqual(y.iloc[b].value_counts().to_dict(), {0: 24, 1: 24})

    def test_group_leakage_is_rejected(self):
        y = pd.Series([0] * 120 + [1] * 120)
        provenance = pd.DataFrame({"pose_group_id": ["same_group"] * 240})
        with self.assertRaisesRegex(ValueError, "group-aware split"):
            training.make_split(y, provenance)

    def test_confusion_matrix_uses_success_as_positive(self):
        result = training.evaluate([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual([result[k] for k in ["tn", "fp", "fn", "tp"]], [1, 1, 1, 1])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["success_precision"], 0.5)
        self.assertEqual(result["success_recall"], 0.5)

    def test_constant_failure_has_zero_success_recall(self):
        result = training.evaluate([0, 0, 1, 1], [0, 0, 0, 0])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["success_recall"], 0)

    def test_scaler_fits_training_only_and_prediction_does_not_refit(self):
        X = pd.DataFrame(np.arange(48).reshape(8, 6), columns=training.FEATURE_COLUMNS)
        y = pd.Series([0, 1] * 4)
        model = training.make_models()["logistic_regression"]
        model.fit(X, y)
        mean = model.named_steps["scaler"].mean_.copy()
        np.testing.assert_allclose(mean, X.mean().to_numpy())
        model.predict(X + 10000)
        np.testing.assert_array_equal(mean, model.named_steps["scaler"].mean_)
        self.assertEqual(model.named_steps["scaler"].n_samples_seen_, 8)

    def test_actual_experiment_artifacts_match_split_metrics_and_saved_models(self):
        dataset = ROOT / "output/dev_pr2_cube_v1_balanced240"
        X, y, _, _, hashes = training.load_development(dataset)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "experiment"
            metrics = training.run_experiment(dataset, output)
            split = pd.read_csv(output / "split_provenance.csv")
            predictions = pd.read_csv(output / "validation_predictions.csv")
            metadata = json.loads((output / "experiment.json").read_text())
            self.assertEqual(split.split.value_counts().to_dict(), {"training": 192, "validation": 48})
            train_groups = set(split.loc[split.split == "training", "pose_group_id"])
            val_groups = set(split.loc[split.split == "validation", "pose_group_id"])
            self.assertFalse(train_groups & val_groups)
            indices = predictions.dataset_row_number.to_numpy() - 1
            np.testing.assert_array_equal(predictions.actual_success, y.iloc[indices])
            for _, row in metrics.iterrows():
                name = row.model
                bundle = joblib.load(output / "models" / f"{name}.joblib")
                actual = bundle["estimator"].predict(X.iloc[indices])
                np.testing.assert_array_equal(actual, predictions[f"{name}_prediction"])
                result = training.evaluate(y.iloc[indices], actual)
                for key, value in result.items():
                    self.assertAlmostEqual(row[key], value)
            self.assertEqual(metrics.loc[metrics.model == "dummy", "accuracy"].iloc[0], 0.5)
            for path, digest in hashes.items():
                self.assertEqual(training.sha256(path), digest)
            self.assertFalse(metadata["independent_test_performed"])
            with self.assertRaises(FileExistsError):
                training.run_experiment(dataset, output)


if __name__ == "__main__":
    unittest.main()
