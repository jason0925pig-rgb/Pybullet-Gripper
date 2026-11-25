"""Analyze training-only OOF errors and repeat CV for three fixed classifiers."""

import argparse
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from artifact_paths import artifact_key
from prepare_dataset import pose_groups
from train_baseline import FEATURE_COLUMNS, ROOT, evaluate, load_development, sha256
from tune_models import candidates


DEVELOPMENT_DATASET = ROOT / "output/dev_pr2_cube_v1_balanced240"
BASELINE_RUN = ROOT / "output/baseline_pr2_cube_v1_seed42"
FROZEN_SEARCHES = {
    "logistic_C_10": ROOT / "output/cv_pr2_cube_v1_seed42",
    "tree_depth_3": ROOT / "output/cv_tree_pr2_cube_v1_seed42",
    "forest_depth_2": ROOT / "output/cv_forest_pr2_cube_v1_seed42",
}


def fixed_models():
    """Freeze model parameters, including estimator seeds, across all repeats."""
    original = candidates(42)
    return {"logistic_C_10": original["logistic_C_10"],
            "tree_depth_3": original["tree_depth_3"],
            "forest_depth_2": candidates(42, "random_forest")["forest_depth_2"]}


def checked_manifest(folder: Union[str, Path], filename: str, fingerprints: dict) -> dict:
    """Verify manifest artifacts and add their hashes to the supplied dict."""
    path = Path(folder) / filename
    metadata = json.loads(path.read_text())
    expected = dict(metadata["source_hashes"])
    expected.update({artifact_key(Path(folder) / name): digest
                     for name, digest in metadata["output_hashes"].items()})
    for name, digest in expected.items():
        if sha256(name) != digest:
            raise ValueError(f"Artifact hash mismatch: {name}")
    fingerprints.update(expected)
    fingerprints[artifact_key(path)] = sha256(path)
    return metadata


def load_training(dataset, baseline, searches):
    """Verify historical OOF predictions and extract only frozen training rows."""
    X, y, provenance, manifest, hashes = load_development(dataset)
    checked_manifest(baseline, "experiment.json", hashes)
    split = pd.read_csv(Path(baseline) / "split_provenance.csv")
    pd.testing.assert_frame_equal(split[provenance.columns], provenance)
    np.testing.assert_array_equal(split.success, y)
    if not split.split.isin(["training", "validation"]).all():
        raise ValueError("Unknown outer split assignment.")
    indices = np.flatnonzero(split.split.eq("training"))
    training = provenance.iloc[indices].reset_index(drop=True)
    holdout = provenance.loc[split.split.eq("validation")]
    if set(training.pose_group_id) & set(holdout.pose_group_id):
        raise ValueError("Pose groups cross the outer split.")
    train_X = X.iloc[indices].reset_index(drop=True)
    train_y = y.iloc[indices].reset_index(drop=True)
    wide = pd.concat([training, train_X], axis=1).assign(actual_success=train_y)
    reference_folds = None
    for name, folder in searches.items():
        metadata = checked_manifest(folder, "search_manifest.json", hashes)
        if metadata["selected_candidate"] != name:
            raise ValueError("Saved selection differs from the fixed diagnostic configuration.")
        rows = pd.read_csv(Path(folder) / "cv_predictions.csv")
        rows = rows.loc[rows.candidate.eq(name)].sort_values("dataset_row_number")
        if rows.dataset_row_number.duplicated().any():
            raise ValueError("Repeated OOF prediction for one candidate and row.")
        pd.testing.assert_frame_equal(rows[training.columns].reset_index(drop=True), training)
        np.testing.assert_array_equal(rows.actual_success, train_y)
        if not rows.prediction.isin([0, 1]).all():
            raise ValueError("OOF predictions must be binary.")
        folds = rows.cv_validation_fold.to_numpy()
        if reference_folds is None:
            reference_folds = folds
        np.testing.assert_array_equal(folds, reference_folds)
        wide[f"{name}_prediction"] = rows.prediction.to_numpy()
        wide[f"{name}_wrong"] = rows.prediction.to_numpy() != train_y.to_numpy()
    wide["cv_validation_fold"] = reference_folds
    wide["wrong_models"] = wide[[f"{name}_wrong" for name in searches]].sum(axis=1)
    return train_X, train_y, training, wide, manifest, hashes


def repeat_cv(X, y, provenance, repeats=5, seed=42):
    """Use identical repeated folds for all fixed models; never refit on holdout."""
    if repeats < 1 or set(y) != {0, 1} or y.value_counts().min() < 5:
        raise ValueError("Repeated CV requires positive repeats and five rows per class.")
    models = fixed_models()
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    score_rows, prediction_rows = [], []
    for number, (train, validation) in enumerate(splitter.split(X, y)):
        repeat, fold = number // 5 + 1, number % 5 + 1
        if set(provenance.iloc[train].pose_group_id) & set(provenance.iloc[validation].pose_group_id):
            raise ValueError("Pose groups cross repeated CV folds.")
        for name, prototype in models.items():
            model = clone(prototype)
            model.fit(X.iloc[train], y.iloc[train])
            predicted = model.predict(X.iloc[validation])
            score_rows.append({"candidate": name, "repeat": repeat, "fold": fold,
                               "training_rows": len(train), "validation_rows": len(validation),
                               "training_accuracy": float(np.mean(model.predict(X.iloc[train]) == y.iloc[train])),
                               **evaluate(y.iloc[validation], predicted)})
            rows = provenance.iloc[validation].copy()
            rows["candidate"], rows["repeat"], rows["fold"] = name, repeat, fold
            rows["actual_success"], rows["prediction"] = y.iloc[validation].to_numpy(), predicted
            prediction_rows.append(rows)
    scores = pd.DataFrame(score_rows)
    per_repeat = scores.groupby(["candidate", "repeat"], as_index=False).agg(
        mean_accuracy=("accuracy", "mean"), mean_training_accuracy=("training_accuracy", "mean"))
    per_repeat["rank"] = per_repeat.groupby("repeat").mean_accuracy.rank(ascending=False, method="min")
    summary = scores.groupby("candidate", as_index=False).agg(
        mean_accuracy=("accuracy", "mean"), std_fold_accuracy=("accuracy", lambda s: s.std(ddof=0)),
        mean_training_accuracy=("training_accuracy", "mean"),
        mean_success_precision=("success_precision", "mean"),
        mean_success_recall=("success_recall", "mean"))
    summary = summary.merge(per_repeat.groupby("candidate", as_index=False).agg(
        first_place_repeats=("rank", lambda s: int(s.eq(1).sum()))))
    return scores, per_repeat, summary.sort_values("mean_accuracy", ascending=False), pd.concat(prediction_rows)


def error_tables(wide):
    """Produce descriptive counts; bins are not rules or feature selection."""
    batch_rows, feature_rows = [], []
    for name in fixed_models():
        for run, subset in wide.groupby("run_id"):
            batch_rows.append({"candidate": name, "run_id": run, "rows": len(subset),
                               "successes": int(subset.actual_success.sum()),
                               "errors": int(subset[f"{name}_wrong"].sum()),
                               "accuracy": float(1 - subset[f"{name}_wrong"].mean())})
        for feature in FEATURE_COLUMNS:
            bins = pd.qcut(wide[feature], 4, duplicates="drop")
            for interval, subset in wide.groupby(bins, observed=True):
                feature_rows.append({"candidate": name, "feature": feature,
                                     "interval": str(interval), "rows": len(subset),
                                     "errors": int(subset[f"{name}_wrong"].sum()),
                                     "error_rate": float(subset[f"{name}_wrong"].mean()),
                                     "success_rate": float(subset.actual_success.mean())})
    return pd.DataFrame(batch_rows), pd.DataFrame(feature_rows)


def run_diagnostics(output_dir: Union[str, Path], repeats: int = 5) -> pd.DataFrame:
    """Write diagnostics for the frozen training split to a fresh directory."""
    X, y, provenance, wide, manifest, hashes = load_training(
        DEVELOPMENT_DATASET, BASELINE_RUN, FROZEN_SEARCHES
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    wide.to_csv(output / "original_oof_error_analysis.csv", index=False)
    wide.loc[wide.wrong_models.eq(3)].to_csv(output / "all_models_wrong.csv", index=False)
    batches, features = error_tables(wide)
    batches.to_csv(output / "batch_errors.csv", index=False)
    features.to_csv(output / "feature_bin_errors.csv", index=False)
    _, pairs = pose_groups(X)
    pairs["dataset_row_a"] = [int(provenance.iloc[i - 1].dataset_row_number) for i in pairs.source_row_a]
    pairs["dataset_row_b"] = [int(provenance.iloc[i - 1].dataset_row_number) for i in pairs.source_row_b]
    pairs["opposite_labels"] = [bool(y.iloc[a - 1] != y.iloc[b - 1])
                                 for a, b in zip(pairs.source_row_a, pairs.source_row_b)]
    pairs.to_csv(output / "near_training_poses.csv", index=False)
    scores, repeated, summary, predictions = repeat_cv(X, y, provenance, repeats)
    scores.to_csv(output / "repeated_fold_scores.csv", index=False)
    repeated.to_csv(output / "repeat_scores.csv", index=False)
    summary.to_csv(output / "repeated_summary.csv", index=False)
    predictions.to_csv(output / "repeated_oof_predictions.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, subset in repeated.groupby("candidate"):
        ax.plot(subset["repeat"], 100 * subset.mean_accuracy, marker="o", label=name)
    ax.set(xlabel="CV repeat", ylabel="Mean five-fold accuracy (%)",
           title="Fixed classifiers on the original 192 training rows")
    ax.set_xticks(range(1, repeats + 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "repeated_accuracy.png", dpi=150)
    plt.close(fig)
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError("An input artifact changed during diagnosis.")
    metadata = {"role": "training_only_exploratory_diagnostics", "training_rows": len(X),
                "outer_validation_evaluated": False, "independent_test_performed": False,
                "hyperparameter_search_performed": False, "repeats": repeats, "folds": 5,
                "splitter_seed": 42, "estimator_seed": 42, "pairing": manifest["pairing"],
                "warning": "Configurations were selected previously on these same training rows; repeated CV is not an unbiased independent evaluation. Repeats share observations; fold standard deviations are not confidence intervals.",
                "bin_warning": "Feature quartiles are training-only descriptive bins; correlated pose features and small subgroups prevent causal conclusions.",
                "model_parameters": {name: {key: str(value) for key, value in model.get_params().items()}
                                     for name, model in fixed_models().items()},
                "error_overlap_counts": {str(k): int(v) for k, v in wide.wrong_models.value_counts().items()},
                "near_pose_pairs": len(pairs), "near_pose_position_tolerance_m": 0.001,
                "near_pose_angle_tolerance_rad": float(np.pi / 180),
                "source_hashes": hashes, "script_sha256": sha256(__file__),
                "output_hashes": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "diagnostic_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("\nPer-repeat scores:")
    print(repeated.to_string(index=False))
    print(f"\nOriginal OOF rows misclassified by all models: {int(wide.wrong_models.eq(3).sum())}/{len(wide)}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/diagnostics_pr2_cube_v1")
    args = parser.parse_args()
    run_diagnostics(args.output)
