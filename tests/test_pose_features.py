"""Verify periodic encoding and paired training-only feature experiments."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compare_pose_features as features
import diagnose_models as diagnostics


class PoseFeatureTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(np.linspace(-3, 3, 240).reshape(40, 6), columns=features.FEATURE_COLUMNS)

    def test_periodicity_translations_index_and_source_preserved(self):
        original = self.X.copy(deep=True)
        encoded = features.encode_angles(self.X)
        shifted = self.X.copy()
        shifted[["roll", "pitch", "yaw"]] += 2 * np.pi
        np.testing.assert_allclose(encoded, features.encode_angles(shifted), atol=1e-14)
        pd.testing.assert_frame_equal(encoded[["x", "y", "z"]], self.X[["x", "y", "z"]])
        pd.testing.assert_frame_equal(original, self.X)
        self.assertEqual(encoded.shape, (40, 9))
        self.assertEqual(list(encoded.columns), ["x", "y", "z", "roll_sin", "roll_cos",
                                                "pitch_sin", "pitch_cos", "yaw_sin", "yaw_cos"])
        for angle in ["roll", "pitch", "yaw"]:
            np.testing.assert_allclose(encoded[f"{angle}_sin"] ** 2 + encoded[f"{angle}_cos"] ** 2, 1)

    def test_near_boundary_angles_are_near_in_encoding(self):
        X = self.X.iloc[:2].copy()
        X.loc[:, "yaw"] = [-np.pi + 0.001, np.pi - 0.001]
        encoded = features.encode_angles(X)
        distance = np.linalg.norm(encoded.iloc[0][["yaw_sin", "yaw_cos"]]
                                  - encoded.iloc[1][["yaw_sin", "yaw_cos"]])
        self.assertLess(distance, 0.0021)

    def test_schema_and_nonfinite_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "six pose columns"):
            features.encode_angles(self.X[list(reversed(self.X.columns))])
        self.X.loc[0, "roll"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            features.encode_angles(self.X)

    def test_encoded_scaler_is_fitted_inside_each_fold(self):
        encoded = features.encode_angles(self.X)
        y = pd.Series([0, 1] * 20)
        provenance = pd.DataFrame({"dataset_row_number": np.arange(1, 41),
                                    "pose_group_id": [f"group_{i}" for i in range(40)]})
        fitted = []
        original_clone = diagnostics.clone

        def capture(prototype):
            model = original_clone(prototype)
            fitted.append(model)
            return model

        with patch.object(diagnostics, "clone", side_effect=capture):
            _, _, _, predictions = diagnostics.repeat_cv(encoded, y, provenance, repeats=1)
        pipelines = [model for model in fitted if hasattr(model, "named_steps")]
        self.assertEqual(len(pipelines), 5)
        for fold, model in enumerate(pipelines, 1):
            validation = predictions.loc[predictions.candidate.eq("logistic_C_10")
                                         & predictions.fold.eq(fold), "dataset_row_number"]
            train = encoded.loc[~encoded.index.isin(validation.to_numpy() - 1)]
            scaler = model.named_steps["scaler"]
            self.assertEqual(scaler.n_samples_seen_, len(train))
            np.testing.assert_allclose(scaler.mean_, train.mean(), atol=1e-14)

    def test_pairing_requires_identical_keys_and_reports_differences(self):
        scores = pd.DataFrame([
            {"candidate": "test", "repeat": 1, "fold": 1, "representation": "raw_euler", "accuracy": 0.5},
            {"candidate": "test", "repeat": 1, "fold": 1, "representation": "sincos", "accuracy": 0.75},
        ])
        _, summary, repeated = features.paired_comparison(scores)
        self.assertEqual(summary.iloc[0].mean_accuracy_delta, 0.25)
        self.assertEqual(summary.iloc[0].improved_folds, 1)
        self.assertEqual(repeated.iloc[0].mean_accuracy_delta, 0.25)
        scores.loc[1, "fold"] = 2
        with self.assertRaisesRegex(ValueError, "identical model/fold keys"):
            features.paired_comparison(scores)


if __name__ == "__main__":
    unittest.main()
