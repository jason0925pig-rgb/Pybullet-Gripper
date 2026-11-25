"""Run a fixed, reproducible offline grasp-classification baseline experiment."""

import argparse
import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from artifact_paths import ROOT, artifact_key, sha256
from inspect_dataset import FEATURE_COLUMNS, inspect_dataset


def load_development(directory):
    """Read the frozen development artifact and check its provenance contract."""
    directory = Path(directory)
    manifest_path = directory / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["role"] != "development_only":
        raise ValueError("This entry expects a development-only dataset.")
    fingerprints = {artifact_key(manifest_path): sha256(manifest_path)}
    for name, digest in manifest["output_hashes"].items():
        path = directory / name
        if sha256(path) != digest:
            raise ValueError(f"Dataset artifact hash mismatch: {path}")
        fingerprints[artifact_key(path)] = digest
    for name, digest in manifest["source_hashes"].items():
        if sha256(name) != digest:
            raise ValueError(f"Raw source hash mismatch: {name}")
        fingerprints[artifact_key(ROOT / name)] = digest
    with contextlib.redirect_stdout(io.StringIO()):
        X, y = inspect_dataset(directory / "development.csv")
    provenance = pd.read_csv(directory / "provenance.csv")
    if len(provenance) != len(X) or len(X) != manifest["selected_rows"]:
        raise ValueError("Dataset/provenance row counts disagree.")
    if not np.array_equal(provenance.dataset_row_number, np.arange(1, len(X) + 1)):
        raise ValueError("Provenance is not in dataset row order.")
    if provenance[["run_id", "attempt_id", "pose_group_id"]].isna().any().any():
        raise ValueError("Provenance keys or pose groups are missing.")
    if provenance.duplicated(["run_id", "attempt_id"]).any():
        raise ValueError("Repeated source trials must be resolved before splitting.")
    if set(y) != {0, 1}:
        raise ValueError("Both classes are required for this experiment.")
    return X, y, provenance, manifest, fingerprints


def make_split(y, provenance, seed=42):
    """Use one stratified 80/20 split; reject pose-group leakage explicitly."""
    train, validation = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
    train_groups = set(provenance.iloc[train].pose_group_id)
    validation_groups = set(provenance.iloc[validation].pose_group_id)
    if train_groups & validation_groups:
        raise ValueError("Pose groups cross the split. A group-aware split is required.")
    return train, validation


def make_models(seed=42):
    """Freeze the three pre-agreed first-round configurations."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
        ]),
        "decision_tree": DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=5, random_state=seed),
    }


def evaluate(y_true, predictions):
    """Report binary outcomes with class 1 as successful grasp."""
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", pos_label=1, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, predictions)),
            "success_precision": float(precision), "success_recall": float(recall),
            "success_f1": float(f1), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)}


def run_experiment(dataset_dir, output_dir, seed=42):
    X, y, provenance, data_manifest, fingerprints = load_development(dataset_dir)
    train, validation = make_split(y, provenance, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    split = provenance.copy()
    split["split"] = "training"
    split.loc[validation, "split"] = "validation"
    split["success"] = y
    split.to_csv(output / "split_provenance.csv", index=False)
    predictions = provenance.iloc[validation].reset_index(drop=True).copy()
    predictions["actual_success"] = y.iloc[validation].to_numpy()
    rows = []
    parameters = {}
    for name, model in make_models(seed).items():
        model.fit(X.iloc[train], y.iloc[train])
        train_prediction = model.predict(X.iloc[train])
        prediction = model.predict(X.iloc[validation])
        rows.append({"model": name, "training_accuracy": float(accuracy_score(
            y.iloc[train], train_prediction)), **evaluate(y.iloc[validation], prediction)})
        predictions[f"{name}_prediction"] = prediction
        index = int(np.flatnonzero(model.classes_ == 1)[0])
        predictions[f"{name}_success_probability"] = model.predict_proba(X.iloc[validation])[:, index]
        bundle = {"estimator": model, "feature_columns": FEATURE_COLUMNS,
                  "pairing": data_manifest["pairing"], "feature_contract":
                  "measured gripper-in-object base COM pose before closure; meters/radians",
                  "role": "development_baseline_not_final_planner"}
        model_path = output / "models" / f"{name}.joblib"
        joblib.dump(bundle, model_path)
        # Load only the artifact just produced by this process, never an untrusted pickle.
        reloaded = joblib.load(model_path)
        np.testing.assert_array_equal(
            reloaded["estimator"].predict(X.iloc[validation]), prediction)
        parameters[name] = model.get_params(deep=False) if name != "logistic_regression" else {
            "scaler": "StandardScaler", "classifier": model.named_steps["classifier"].get_params()}
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    if any(sha256(name) != digest for name, digest in fingerprints.items()):
        raise ValueError("An input artifact changed during training.")
    metadata = {"stage": "first_baseline_comparison", "split_seed": seed,
                "training_rows": len(train), "validation_rows": len(validation),
                "training_class_counts": {str(k): int(v) for k, v in y.iloc[train].value_counts().items()},
                "validation_class_counts": {str(k): int(v) for k, v in y.iloc[validation].value_counts().items()},
                "pose_group_overlap": 0, "feature_columns": FEATURE_COLUMNS,
                "pairing": data_manifest["pairing"], "model_parameters": parameters,
                "source_hashes": fingerprints, "script_sha256": sha256(__file__),
                "python": platform.python_version(),
                "packages": {n: importlib.metadata.version(n) for n in ["scikit-learn", "numpy", "pandas", "joblib"]},
                "independent_test_performed": False, "hyperparameter_search_performed": False,
                "final_model_selected": False, "save_reload_predictions_verified": True,
                "output_hashes": {str(p.relative_to(output)): sha256(p)
                                  for p in output.rglob("*") if p.is_file()}}
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metrics.to_string(index=False))
    print(f"\nTraining rows: {len(train)}; validation rows: {len(validation)}; group overlap: 0")
    print(f"Saved experiment to {output.resolve()}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "output/dev_pr2_cube_v1_balanced240")
    parser.add_argument("--output", type=Path, default=ROOT / "output/baseline_pr2_cube_v1_seed42")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment(args.dataset, args.output, args.seed)
