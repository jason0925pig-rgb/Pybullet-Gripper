"""Evaluate the frozen CV-selected model on the existing outer validation rows."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from artifact_paths import artifact_key
from train_baseline import ROOT, FEATURE_COLUMNS, evaluate, load_development, sha256


def evaluate_selected(dataset_dir, baseline_dir, search_dir, output_dir):
    """Predict only: do not fit, tune or modify any existing model/data artifact."""
    X, y, provenance, dataset_manifest, fingerprints = load_development(dataset_dir)
    dataset_fingerprints = fingerprints.copy()
    baseline, search = Path(baseline_dir), Path(search_dir)
    search_manifest_path = search / "search_manifest.json"
    search_manifest = json.loads(search_manifest_path.read_text())
    fingerprints[artifact_key(search_manifest_path)] = sha256(search_manifest_path)
    search_sources = {artifact_key(ROOT / path): digest
                      for path, digest in search_manifest["source_hashes"].items()}
    for path, digest in search_sources.items():
        if sha256(path) != digest:
            raise ValueError(f"Frozen search source changed: {path}")
        fingerprints[path] = digest
    for name, digest in search_manifest["output_hashes"].items():
        path = search / name
        if sha256(path) != digest:
            raise ValueError(f"Search artifact hash mismatch: {path}")
        fingerprints[artifact_key(path)] = digest
    for path, digest in dataset_fingerprints.items():
        if search_sources.get(path) != digest:
            raise ValueError("Evaluation dataset differs from the CV source dataset.")
    baseline_manifest_path = baseline / "experiment.json"
    if search_sources.get(artifact_key(baseline_manifest_path)) != sha256(
            baseline_manifest_path):
        raise ValueError("Evaluation baseline differs from the CV source baseline.")
    baseline_manifest = json.loads(baseline_manifest_path.read_text())
    for name in ["split_provenance.csv", "metrics.csv", "validation_predictions.csv"]:
        path = baseline / name
        digest = baseline_manifest["output_hashes"][name]
        if sha256(path) != digest:
            raise ValueError(f"Baseline artifact hash mismatch: {path}")
        fingerprints[artifact_key(path)] = digest
    split = pd.read_csv(baseline / "split_provenance.csv")
    pd.testing.assert_frame_equal(split[provenance.columns], provenance)
    np.testing.assert_array_equal(split.success, y)
    if not split.split.isin(["training", "validation"]).all():
        raise ValueError("Unknown split assignment.")
    indices = np.flatnonzero(split.split.to_numpy() == "validation")
    train_indices = np.flatnonzero(split.split.to_numpy() == "training")
    if len(indices) != search_manifest["excluded_outer_validation_rows"]:
        raise ValueError("Validation count differs from the frozen experiment.")
    if len(train_indices) != search_manifest["training_rows"]:
        raise ValueError("Training count differs from the frozen experiment.")
    if set(provenance.iloc[indices].pose_group_id) & set(provenance.iloc[train_indices].pose_group_id):
        raise ValueError("Pose groups cross the outer split.")
    # Load only the trusted local artifact produced by the earlier training script.
    bundle = joblib.load(search / "selected_training_model.joblib")
    if (bundle["feature_columns"] != FEATURE_COLUMNS
            or bundle["pairing"] != dataset_manifest["pairing"]
            or bundle["candidate"] != search_manifest["selected_candidate"]
            or bundle["role"] != "CV_selected_refit_on_frozen_training_only"):
        raise ValueError("Model bundle contract differs from the frozen search.")
    model = bundle["estimator"]
    predicted = model.predict(X.iloc[indices])
    probability = model.predict_proba(X.iloc[indices])[:, int(np.flatnonzero(model.classes_ == 1)[0])]
    scores = evaluate(y.iloc[indices], predicted)
    predictions = provenance.iloc[indices].reset_index(drop=True).copy()
    predictions["actual_success"] = y.iloc[indices].to_numpy()
    predictions["predicted_success"] = predicted
    predictions["success_probability"] = probability
    predictions["correct"] = predictions.actual_success == predictions.predicted_success
    predictions["outcome"] = np.select(
        [(predictions.actual_success == 0) & (predictions.predicted_success == 0),
         (predictions.actual_success == 0) & (predictions.predicted_success == 1),
         (predictions.actual_success == 1) & (predictions.predicted_success == 0)],
        ["TN", "FP", "FN"], default="TP")
    comparisons = pd.read_csv(baseline / "metrics.csv").drop(columns="training_accuracy")
    comparisons = pd.concat([comparisons, pd.DataFrame([
        {"model": bundle["candidate"], **scores}])], ignore_index=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([{"model": bundle["candidate"], **scores}]).to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    predictions.loc[~predictions.correct].to_csv(output / "errors.csv", index=False)
    comparisons.to_csv(output / "baseline_comparison.csv", index=False)
    if any(sha256(path) != digest for path, digest in fingerprints.items()):
        raise ValueError("An input changed during evaluation.")
    metadata = {"role": "existing_outer_validation_development_evaluation",
                "candidate": bundle["candidate"], "validation_rows": len(indices),
                "correct_predictions": int(predictions.correct.sum()), "metrics": scores,
                "fit_performed": False, "hyperparameters_changed": False,
                "independent_simulation_test_performed": False,
                "warning": "outer validation was previously viewed during baseline development",
                "source_hashes": fingerprints, "script_sha256": sha256(__file__),
                "output_hashes": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "evaluation_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(comparisons.to_string(index=False))
    print(f"\nFrozen candidate {bundle['candidate']}: {int(predictions.correct.sum())}/{len(indices)} correct.")
    print("No fitting, tuning or independent simulation testing was performed.")
    return scores, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "output/dev_pr2_cube_v1_balanced240")
    parser.add_argument("--baseline", type=Path, default=ROOT / "output/baseline_pr2_cube_v1_seed42")
    parser.add_argument("--search", type=Path, default=ROOT / "output/cv_pr2_cube_v1_seed42")
    parser.add_argument("--output", type=Path, default=ROOT / "output/selected_validation_pr2_cube_v1")
    args = parser.parse_args()
    evaluate_selected(args.dataset, args.baseline, args.search, args.output)
