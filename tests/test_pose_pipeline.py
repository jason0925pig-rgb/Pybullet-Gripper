"""Verify the six-input saved pose pipeline and prediction-only evaluation."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_pose_features import encode_angles
from evaluate_selected import evaluate_selected
import train_pose_pipeline as training


class PosePipelineTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(np.linspace(-3, 3, 240).reshape(40, 6), columns=training.FEATURE_COLUMNS)
        self.y = pd.Series([0, 1] * 20)

    def test_pipeline_matches_manual_encoding_and_preserves_inputs(self):
        original = self.X.copy(deep=True)
        model = training.make_pose_pipeline().fit(self.X, self.y)
        manual = clone(training.fixed_models()["forest_depth_2"]).fit(encode_angles(self.X), self.y)
        np.testing.assert_array_equal(model.predict(self.X), manual.predict(encode_angles(self.X)))
        np.testing.assert_allclose(model.predict_proba(self.X), manual.predict_proba(encode_angles(self.X)), rtol=0, atol=0)
        self.assertEqual(model.named_steps["classifier"].n_features_in_, 9)
        self.assertEqual(list(model.feature_names_in_), training.FEATURE_COLUMNS)
        pd.testing.assert_frame_equal(original, self.X)

    def test_prediction_rejects_wrong_order_and_double_encoding(self):
        model = training.make_pose_pipeline().fit(self.X, self.y)
        for X in [self.X[list(reversed(self.X.columns))], encode_angles(self.X)]:
            with self.assertRaisesRegex(ValueError, "six pose columns"):
                model.predict(X)

    def test_saved_pipeline_loads_in_a_fresh_python_process(self):
        model = training.make_pose_pipeline().fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            joblib.dump(model, folder / "model.joblib")
            self.X.to_csv(folder / "poses.csv", index=False)
            code = (
                "import json, sys, joblib, pandas as pd; "
                "from pathlib import Path; "
                "p = Path(sys.argv[1]); "
                "m = joblib.load(p / 'model.joblib'); "
                "x = pd.read_csv(p / 'poses.csv'); "
                "print(json.dumps(m.predict(x).tolist()))"
            )
            completed = subprocess.run([sys.executable, "-c", code, str(folder)], cwd=ROOT,
                                       check=True, capture_output=True, text=True)
            np.testing.assert_array_equal(json.loads(completed.stdout), model.predict(self.X))

    def test_actual_refit_excludes_holdout_and_evaluation_never_fits(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "model"
            training.train_pipeline(output)
            split = pd.read_csv(ROOT / "output/baseline_pr2_cube_v1_seed42/split_provenance.csv")
            provenance = pd.read_csv(output / "training_provenance.csv")
            expected_training = split.loc[split.split.eq("training"), "dataset_row_number"].tolist()
            expected_validation = split.loc[split.split.eq("validation"), "dataset_row_number"].tolist()
            self.assertEqual(provenance.dataset_row_number.tolist(), expected_training)
            metadata = json.loads((output / "search_manifest.json").read_text())
            self.assertFalse(metadata["outer_validation_evaluated"])
            self.assertTrue(metadata["save_reload_predictions_verified"])
            with patch.object(Pipeline, "fit", side_effect=AssertionError("Evaluation must not fit")), patch.object(
                    RandomForestClassifier, "fit", side_effect=AssertionError("Evaluation must not fit")):
                scores, predictions = evaluate_selected(
                    ROOT / "output/dev_pr2_cube_v1_balanced240",
                    ROOT / "output/baseline_pr2_cube_v1_seed42", output,
                    Path(directory) / "validation")
            self.assertEqual(predictions.dataset_row_number.tolist(), expected_validation)
            self.assertEqual(len(predictions), 48)
            self.assertEqual(int(predictions.correct.sum()), scores["tn"] + scores["tp"])
            for path, digest in metadata["source_hashes"].items():
                self.assertEqual(training.sha256(path), digest)


if __name__ == "__main__":
    unittest.main()
