"""Verify nested training subsets, held-out fold isolation and fold-local fits."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import plot_learning_curves as curves


class LearningCurveTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(np.arange(240).reshape(40, 6), columns=["x", "y", "z", "roll", "pitch", "yaw"])
        self.y = pd.Series([0, 1] * 20)
        self.provenance = pd.DataFrame({"dataset_row_number": np.arange(1, 41),
                                        "pose_group_id": [f"group_{i}" for i in range(40)]})

    def test_nested_subsets_identical_model_inputs_no_fold_leakage_and_local_scaling(self):
        fitted, fit_indices = [], []
        original_clone = curves.clone

        def capture(prototype):
            model = original_clone(prototype)
            original_fit = model.fit

            def fit(X, y):
                fit_indices.append(set(X.index + 1))
                fitted.append(model)
                return original_fit(X, y)

            model.fit = fit
            return model

        with patch.object(curves, "clone", side_effect=capture):
            scores, summary, membership = curves.learning_curve_scores(
                self.X, self.y, self.provenance, repeats=1, sizes=(8, 16, None))
        self.assertEqual(len(scores), 45)
        self.assertEqual(len(summary), 9)
        self.assertEqual(len(fitted), 45)
        for fold in range(1, 6):
            previous = set()
            fold_rows = membership.loc[membership.fold.eq(fold)]
            validation_sets = []
            for level, size in enumerate([8, 16, 32], 1):
                rows = fold_rows.loc[fold_rows.size_level.eq(level)]
                train = rows.loc[rows.role.eq("training")]
                validation = rows.loc[rows.role.eq("validation")]
                ids = set(train.dataset_row_number)
                self.assertEqual(len(train), size)
                self.assertEqual(int(train.actual_success.sum()), size // 2)
                self.assertFalse(ids & set(validation.dataset_row_number))
                self.assertTrue(previous.issubset(ids))
                previous = ids
                validation_sets.append(set(validation.dataset_row_number))
                position = ((fold - 1) * 3 + level - 1) * 3
                for offset in range(3):
                    self.assertEqual(fit_indices[position + offset], ids)
                scaler = fitted[position].named_steps["scaler"]
                self.assertEqual(scaler.n_samples_seen_, size)
                np.testing.assert_allclose(scaler.mean_, self.X.iloc[sorted(i - 1 for i in ids)].mean())
            self.assertTrue(all(ids == validation_sets[0] for ids in validation_sets))
        for _, row in summary.iterrows():
            values = scores.loc[scores.candidate.eq(row.candidate) & scores.size_level.eq(row.size_level)]
            self.assertAlmostEqual(row.mean_cv_accuracy, values.accuracy.mean())
            self.assertAlmostEqual(row.std_cv_accuracy, values.accuracy.std(ddof=0))

    def test_pose_group_leakage_is_rejected(self):
        self.provenance["pose_group_id"] = "shared"
        with self.assertRaisesRegex(ValueError, "Pose groups cross"):
            curves.learning_curve_scores(self.X, self.y, self.provenance, sizes=(8, None))

    def test_invalid_or_oversized_subsets_are_rejected(self):
        for sizes in [(16, 8, None), (8, 8, None), (8, 16), (32, None)]:
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                curves.learning_curve_scores(self.X, self.y, self.provenance, sizes=sizes)


if __name__ == "__main__":
    unittest.main()
