"""Compare raw Euler poses with sine/cosine features on training-only folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_models import load_training, repeat_cv
from train_baseline import FEATURE_COLUMNS, ROOT, sha256


def encode_angles(X):
    """Keep translations and replace each radian Euler angle with sin and cos."""
    if list(X.columns) != FEATURE_COLUMNS:
        raise ValueError("Expected six pose columns in the recorded feature order.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Pose features must be finite numeric values.")
    result = X[["x", "y", "z"]].copy()
    for angle in ["roll", "pitch", "yaw"]:
        result[f"{angle}_sin"] = np.sin(X[angle])
        result[f"{angle}_cos"] = np.cos(X[angle])
    return result


def paired_comparison(scores):
    """Pair folds before computing descriptive accuracy differences."""
    keys = ["candidate", "repeat", "fold"]
    raw = scores.loc[scores.representation.eq("raw_euler"), keys + ["accuracy"]]
    encoded = scores.loc[scores.representation.eq("sincos"), keys + ["accuracy"]]
    paired = raw.merge(encoded, on=keys, suffixes=("_raw", "_sincos"), validate="one_to_one")
    if len(paired) != len(raw) or len(paired) != len(encoded):
        raise ValueError("Representations must cover identical model/fold keys.")
    paired["accuracy_delta"] = paired.accuracy_sincos - paired.accuracy_raw
    summary = paired.groupby("candidate", as_index=False).agg(
        raw_mean_accuracy=("accuracy_raw", "mean"), sincos_mean_accuracy=("accuracy_sincos", "mean"),
        mean_accuracy_delta=("accuracy_delta", "mean"),
        improved_folds=("accuracy_delta", lambda x: int((x > 1e-12).sum())),
        tied_folds=("accuracy_delta", lambda x: int((abs(x) <= 1e-12).sum())),
        worse_folds=("accuracy_delta", lambda x: int((x < -1e-12).sum())))
    repeated = paired.groupby(["candidate", "repeat"], as_index=False).agg(
        raw_mean_accuracy=("accuracy_raw", "mean"), sincos_mean_accuracy=("accuracy_sincos", "mean"),
        mean_accuracy_delta=("accuracy_delta", "mean"))
    return paired, summary, repeated


def run_feature_experiment(output_dir):
    searches = {"logistic_C_10": ROOT / "output/cv_pr2_cube_v1_seed42",
                "tree_depth_3": ROOT / "output/cv_tree_pr2_cube_v1_seed42",
                "forest_depth_2": ROOT / "output/cv_forest_pr2_cube_v1_seed42"}
    X, y, provenance, _, manifest, hashes = load_training(
        ROOT / "output/dev_pr2_cube_v1_balanced240",
        ROOT / "output/baseline_pr2_cube_v1_seed42", searches)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    encoded = encode_angles(X)
    score_frames, prediction_frames, summary_frames = [], [], []
    for name, features in [("raw_euler", X), ("sincos", encoded)]:
        scores, _, summary, predictions = repeat_cv(features, y, provenance)
        for frame in [scores, summary, predictions]:
            frame["representation"] = name
        score_frames.append(scores)
        summary_frames.append(summary)
        prediction_frames.append(predictions)
    scores = pd.concat(score_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    paired, comparison, repeated = paired_comparison(scores)
    keys = ["candidate", "repeat", "dataset_row_number"]
    raw_predictions = predictions.loc[predictions.representation.eq("raw_euler")]
    encoded_predictions = predictions.loc[predictions.representation.eq("sincos")]
    pd.testing.assert_frame_equal(
        raw_predictions[keys + ["fold", "actual_success"]].reset_index(drop=True),
        encoded_predictions[keys + ["fold", "actual_success"]].reset_index(drop=True))
    scores.to_csv(output / "fold_scores.csv", index=False)
    summaries.to_csv(output / "representation_summary.csv", index=False)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    paired.to_csv(output / "paired_fold_differences.csv", index=False)
    comparison.to_csv(output / "feature_comparison.csv", index=False)
    repeated.to_csv(output / "paired_repeat_differences.csv", index=False)
    pd.concat([provenance, encoded], axis=1).to_csv(output / "encoded_training_features.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(comparison))
    ax.bar(positions - 0.18, comparison.raw_mean_accuracy * 100, width=0.36, label="Raw Euler (6 features)")
    ax.bar(positions + 0.18, comparison.sincos_mean_accuracy * 100, width=0.36, label="Sin/cos (9 features)")
    ax.set_xticks(positions, comparison.candidate)
    ax.set(ylabel="Mean CV accuracy (%)", ylim=(0, 100), title="Feature representation: same rows, folds and model settings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "feature_comparison.png", dpi=150)
    plt.close(fig)
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError("An input artifact changed during feature comparison.")
    metadata = {"role": "training_only_feature_ablation", "training_rows": len(X),
                "pairing": manifest["pairing"], "folds": 5, "repeats": 5, "fits": len(scores),
                "raw_columns": FEATURE_COLUMNS, "encoded_columns": list(encoded.columns),
                "angle_unit": "radians", "outer_validation_evaluated": False,
                "independent_test_performed": False, "hyperparameter_search_performed": False,
                "deployment_model_replaced": False, "new_model_saved": False,
                "transform": "stateless per-row sin/cos; translations unchanged; no learned statistics",
                "warning": "Previously selected fixed hyperparameters on reused training data; descriptive paired differences are not independent significance tests. Sin/cos does not remove Euler singularities. Forest max_features='sqrt' allows 2 of 6 raw features versus 3 of 9 encoded features per split; encoding also changes model capacity and regularization geometry.",
                "source_hashes": hashes,
                "code_hashes": {name: sha256(ROOT / name) for name in
                                ["compare_pose_features.py", "diagnose_models.py", "tune_models.py", "train_baseline.py"]},
                "output_hashes": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "feature_experiment_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(comparison.to_string(index=False))
    print("\nPaired repeat differences:")
    print(repeated.to_string(index=False))
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/feature_ablation_pr2_cube_v1")
    args = parser.parse_args()
    run_feature_experiment(args.output)
