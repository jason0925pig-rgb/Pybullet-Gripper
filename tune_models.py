"""Compare fixed model configurations by five-fold CV on frozen training rows only."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from artifact_paths import artifact_key
from train_baseline import ROOT, FEATURE_COLUMNS, evaluate, load_development, sha256


def candidates(seed=42, model_family="all"):
    """Keep the small search space agreed before observing CV scores."""
    if model_family == "random_forest":
        return {
            f"forest_depth_{depth}": RandomForestClassifier(
                n_estimators=100, max_depth=depth, min_samples_leaf=5,
                max_features="sqrt", bootstrap=True, random_state=seed, n_jobs=1)
            for depth in [2, 3, 4]
        }
    result = {}
    for value in [0.1, 1.0, 10.0]:
        result[f"logistic_C_{value:g}"] = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=value, max_iter=1000, random_state=seed)),
        ])
    for depth in [2, 3, 4]:
        result[f"tree_depth_{depth}"] = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=5, random_state=seed)
    if model_family == "all":
        return result
    if model_family == "decision_tree":
        return {name: model for name, model in result.items() if name.startswith("tree_")}
    raise ValueError(f"Unknown model family: {model_family}")


def compare_candidates(X, y, provenance, seed=42, model_family="all"):
    """Accept training data only; clone and fit preprocessing inside each fold."""
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    if y.value_counts().min() < 5 or set(y) != {0, 1}:
        raise ValueError("Five-fold classification requires at least five rows per class.")
    splits = list(splitter.split(X, y))
    assignment = provenance.reset_index(drop=True).copy()
    assignment["cv_validation_fold"] = 0
    for fold, (train, validation) in enumerate(splits, 1):
        if set(assignment.iloc[train].pose_group_id) & set(assignment.iloc[validation].pose_group_id):
            raise ValueError("Pose groups cross CV folds; a group-aware splitter is required.")
        assignment.loc[validation, "cv_validation_fold"] = fold
    score_rows, prediction_rows = [], []
    models = candidates(seed, model_family)
    for name, prototype in models.items():
        for fold, (train, validation) in enumerate(splits, 1):
            model = clone(prototype)
            model.fit(X.iloc[train], y.iloc[train])
            prediction = model.predict(X.iloc[validation])
            train_accuracy = float(np.mean(model.predict(X.iloc[train]) == y.iloc[train].to_numpy()))
            score_rows.append({"candidate": name, "fold": fold,
                               "training_rows": len(train), "cv_validation_rows": len(validation),
                               "training_accuracy": train_accuracy,
                               **evaluate(y.iloc[validation], prediction)})
            rows = assignment.iloc[validation].copy()
            rows["candidate"] = name
            rows["actual_success"] = y.iloc[validation].to_numpy()
            rows["prediction"] = prediction
            prediction_rows.append(rows)
    scores = pd.DataFrame(score_rows)
    summary_rows = []
    for order, name in enumerate(models):
        subset = scores.loc[scores.candidate == name]
        summary_rows.append({"candidate": name, "declaration_order": order,
                             "mean_accuracy": float(subset.accuracy.mean()),
                             "std_accuracy": float(subset.accuracy.std(ddof=0)),
                             "mean_training_accuracy": float(subset.training_accuracy.mean()),
                             "mean_success_precision": float(subset.success_precision.mean()),
                             "mean_success_recall": float(subset.success_recall.mean()),
                             "mean_success_f1": float(subset.success_f1.mean())})
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_accuracy", "declaration_order"], ascending=[False, True]).reset_index(drop=True)
    return scores, summary, assignment, pd.concat(prediction_rows, ignore_index=True)


def run_search(dataset_dir, baseline_dir, output_dir, seed=42, model_family="all"):
    """Preserve the outer validation set; refit the CV choice on training rows only."""
    X, y, provenance, data_manifest, fingerprints = load_development(dataset_dir)
    baseline = Path(baseline_dir)
    baseline_manifest = baseline / "experiment.json"
    experiment = json.loads(baseline_manifest.read_text())
    split_path = baseline / "split_provenance.csv"
    if sha256(split_path) != experiment["output_hashes"]["split_provenance.csv"]:
        raise ValueError("The frozen baseline split has changed.")
    for name, digest in experiment["source_hashes"].items():
        if sha256(name) != digest or fingerprints.get(artifact_key(ROOT / name)) != digest:
            raise ValueError("The baseline was produced from different source artifacts.")
    split = pd.read_csv(split_path)
    if len(split) != len(provenance) or not split.split.isin(["training", "validation"]).all():
        raise ValueError("The outer split does not cover the prepared dataset.")
    pd.testing.assert_frame_equal(split[provenance.columns], provenance)
    np.testing.assert_array_equal(split.success, y)
    train_indices = np.flatnonzero(split.split.to_numpy() == "training")
    held_out_indices = np.flatnonzero(split.split.to_numpy() == "validation")
    if not len(train_indices) or not len(held_out_indices):
        raise ValueError("Both frozen partitions must be non-empty.")
    training_provenance = provenance.iloc[train_indices].reset_index(drop=True)
    if set(training_provenance.pose_group_id) & set(provenance.iloc[held_out_indices].pose_group_id):
        raise ValueError("Pose groups cross the frozen outer split.")
    # Never evaluate or predict the held-out rows during this search.
    train_X = X.iloc[train_indices].reset_index(drop=True)
    train_y = y.iloc[train_indices].reset_index(drop=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    scores, summary, assignment, predictions = compare_candidates(
        train_X, train_y, training_provenance, seed, model_family)
    best_name = summary.iloc[0].candidate
    selected = clone(candidates(seed, model_family)[best_name])
    selected.fit(train_X, train_y)
    bundle = {"estimator": selected, "candidate": best_name,
              "feature_columns": FEATURE_COLUMNS, "pairing": data_manifest["pairing"],
              "feature_contract": "measured gripper-in-object base COM pose before closure; meters/radians",
              "role": "CV_selected_refit_on_frozen_training_only"}
    joblib.dump(bundle, output / "selected_training_model.joblib")
    reloaded = joblib.load(output / "selected_training_model.joblib")
    np.testing.assert_array_equal(selected.predict(train_X), reloaded["estimator"].predict(train_X))
    scores.to_csv(output / "fold_scores.csv", index=False)
    summary.to_csv(output / "cv_summary.csv", index=False)
    assignment.to_csv(output / "fold_provenance.csv", index=False)
    predictions.to_csv(output / "cv_predictions.csv", index=False)
    fingerprints[artifact_key(baseline_manifest)] = sha256(baseline_manifest)
    fingerprints[artifact_key(split_path)] = experiment["output_hashes"]["split_provenance.csv"]
    if any(sha256(path) != digest for path, digest in fingerprints.items()):
        raise ValueError("An input artifact changed during search.")
    metadata = {"search_seed": seed, "folds": 5, "training_rows": len(train_X),
                "excluded_outer_validation_rows": len(held_out_indices),
                "outer_validation_evaluated": False, "independent_test_performed": False,
                "selection_rule": "highest mean CV accuracy; exact ties use fixed declaration order",
                "model_family": model_family,
                "candidate_order": list(candidates(seed, model_family)), "selected_candidate": best_name,
                "selected_mean_cv_accuracy": float(summary.iloc[0].mean_accuracy),
                "std_definition": "population standard deviation of five fold accuracies, ddof=0",
                "warning": "selection CV score is not an unbiased final performance estimate",
                "refit_scope": "original training partition only; not all 240 rows",
                "save_reload_predictions_verified": True, "pairing": data_manifest["pairing"],
                "source_hashes": fingerprints, "script_sha256": sha256(__file__),
                "packages": experiment["packages"],
                "output_hashes": {str(p.relative_to(output)): sha256(p)
                                  for p in output.iterdir() if p.is_file()}}
    (output / "search_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"\nCV-selected candidate: {best_name}; refit on {len(train_X)} training rows only.")
    print(f"Outer validation rows excluded from search: {len(held_out_indices)}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "output/dev_pr2_cube_v1_balanced240")
    parser.add_argument("--baseline", type=Path, default=ROOT / "output/baseline_pr2_cube_v1_seed42")
    parser.add_argument("--output", type=Path, default=ROOT / "output/cv_pr2_cube_v1_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--family", choices=["all", "decision_tree", "random_forest"],
                        default="all", help="'all' preserves the original six-model comparison")
    args = parser.parse_args()
    run_search(args.dataset, args.baseline, args.output, args.seed, args.family)
