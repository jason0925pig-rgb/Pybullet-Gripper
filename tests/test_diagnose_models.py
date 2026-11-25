"""Check training-only diagnostics, fixed estimators and repeated OOF coverage."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import diagnose_models as diagnostics


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(np.arange(240).reshape(40, 6), columns=diagnostics.FEATURE_COLUMNS)
        self.y = pd.Series([0, 1] * 20)
        self.provenance = pd.DataFrame({"dataset_row_number": np.arange(1, 41),
                                        "pose_group_id": [f"group_{i}" for i in range(40)]})

    def test_repeated_coverage_shared_folds_fixed_seed_and_metrics(self):
        fitted = []
        original_clone = diagnostics.clone

        def capture(prototype):
            model = original_clone(prototype)
            fitted.append(model)
            return model

        with patch.object(diagnostics, "clone", side_effect=capture):
            scores, repeated, summary, predictions = diagnostics.repeat_cv(
                self.X, self.y, self.provenance, repeats=2)
        self.assertEqual(len(fitted), 30)
        self.assertEqual(len(scores), 30)
        self.assertEqual(len(repeated), 6)
        self.assertEqual(len(summary), 3)
        self.assertEqual(len(predictions), 40 * 2 * 3)
        self.assertFalse(predictions.duplicated(["candidate", "repeat", "dataset_row_number"]).any())
        self.assertTrue(predictions.groupby(["repeat", "dataset_row_number"]).fold.nunique().eq(1).all())
        for model in fitted:
            estimator = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
            self.assertEqual(estimator.random_state, 42)
        for _, row in scores.iterrows():
            subset = predictions.loc[predictions.candidate.eq(row.candidate)
                                     & predictions["repeat"].eq(row["repeat"])
                                     & predictions.fold.eq(row.fold)]
            self.assertAlmostEqual(row.accuracy, subset.prediction.eq(subset.actual_success).mean())

    def test_group_leakage_is_rejected(self):
        self.provenance["pose_group_id"] = "shared"
        with self.assertRaisesRegex(ValueError, "Pose groups cross"):
            diagnostics.repeat_cv(self.X, self.y, self.provenance)

    def test_existing_oof_diagnostics_exclude_all_outer_validation_rows(self):
        searches = {"logistic_C_10": ROOT / "output/cv_pr2_cube_v1_seed42",
                    "tree_depth_3": ROOT / "output/cv_tree_pr2_cube_v1_seed42",
                    "forest_depth_2": ROOT / "output/cv_forest_pr2_cube_v1_seed42"}
        X, y, provenance, wide, _, _ = diagnostics.load_training(
            ROOT / "output/dev_pr2_cube_v1_balanced240",
            ROOT / "output/baseline_pr2_cube_v1_seed42", searches)
        split = pd.read_csv(ROOT / "output/baseline_pr2_cube_v1_seed42/split_provenance.csv")
        training_ids = set(split.loc[split.split.eq("training"), "dataset_row_number"])
        holdout_ids = set(split.loc[split.split.eq("validation"), "dataset_row_number"])
        self.assertEqual(len(X), 192)
        self.assertEqual(len(y), 192)
        self.assertEqual(set(wide.dataset_row_number), training_ids)
        self.assertFalse(set(wide.dataset_row_number) & holdout_ids)
        self.assertEqual(set(provenance.dataset_row_number), training_ids)
        columns = [f"{name}_wrong" for name in searches]
        np.testing.assert_array_equal(wide.wrong_models, wide[columns].sum(axis=1))
        batch, features = diagnostics.error_tables(wide)
        self.assertTrue(batch.groupby("candidate").rows.sum().eq(192).all())
        self.assertTrue(features.groupby(["candidate", "feature"]).rows.sum().eq(192).all())

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "rows.csv"
            source.write_text("original\n")
            digest = diagnostics.sha256(source)
            (folder / "manifest.json").write_text(json.dumps({
                "source_hashes": {str(source): digest}, "output_hashes": {}}))
            source.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "Artifact hash mismatch"):
                diagnostics.checked_manifest(folder, "manifest.json", {})


if __name__ == "__main__":
    unittest.main()
