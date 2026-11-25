"""Freeze and save the training-CV-selected sine/cosine forest pipeline."""

import argparse
import json
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from compare_pose_features import encode_angles
from diagnose_models import (
    BASELINE_RUN,
    DEVELOPMENT_DATASET,
    FROZEN_SEARCHES,
    checked_manifest,
    fixed_models,
    load_training,
)
from train_baseline import FEATURE_COLUMNS, ROOT, sha256


CANDIDATE = "sincos_forest_depth_2"


def make_pose_pipeline() -> Pipeline:
    """Accept six raw pose columns and transform them inside the saved model."""
    return Pipeline([
        ("pose_encoding", FunctionTransformer(encode_angles, validate=False)),
        ("classifier", clone(fixed_models()["forest_depth_2"])),
    ])


def train_pipeline(output_dir: Union[str, Path]) -> Pipeline:
    """Refit the frozen representation/model choice on the original training rows."""
    X, y, provenance, _, dataset, hashes = load_training(
        DEVELOPMENT_DATASET, BASELINE_RUN, FROZEN_SEARCHES
    )
    experiment_dir = ROOT / "output/feature_ablation_pr2_cube_v1"
    checked_manifest(experiment_dir, "feature_experiment_manifest.json", hashes)
    summary = pd.read_csv(experiment_dir / "representation_summary.csv")
    selected = summary.sort_values("mean_accuracy", ascending=False).iloc[0]
    if selected.representation != "sincos" or selected.candidate != "forest_depth_2":
        raise ValueError("Recorded feature experiment does not select the frozen pipeline.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    training_rows = len(X)
    validation_rows = dataset["selected_rows"] - training_rows
    protocol = {
        "selected_candidate": CANDIDATE,
        "selection_basis": "highest mean repeated-CV accuracy across the six recorded representation/model combinations",
        "selected_mean_cv_accuracy": float(selected.mean_accuracy),
        "training_rows": training_rows,
        "excluded_outer_validation_rows": validation_rows,
        "outer_validation_evaluated": False, "independent_test_performed": False,
        "input_columns": FEATURE_COLUMNS, "internal_columns": list(encode_angles(X).columns),
        "classifier_parameters": fixed_models()["forest_depth_2"].get_params(),
        "refit_scope": f"original {training_rows} training rows only",
        "warning": "Feature/model selection reuses development data; CV score is not an independent final performance estimate.",
    }
    # Write the frozen protocol before fitting or any later outer evaluation.
    (output / "frozen_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    model = make_pose_pipeline()
    model.fit(X, y)
    bundle = {
        "estimator": model, "candidate": CANDIDATE, "feature_columns": FEATURE_COLUMNS,
        "pairing": dataset["pairing"], "representation": "sincos_inside_pipeline",
        "feature_contract": "measured gripper-in-object base COM pose before closure; meters/radians",
        "role": "CV_selected_refit_on_frozen_training_only",
    }
    model_path = output / "selected_training_model.joblib"
    joblib.dump(bundle, model_path)
    reloaded = joblib.load(model_path)["estimator"]
    np.testing.assert_array_equal(model.predict(X), reloaded.predict(X))
    np.testing.assert_allclose(model.predict_proba(X), reloaded.predict_proba(X), rtol=0, atol=0)
    # Independently check equivalence to the earlier manually encoded experiment.
    manual = clone(fixed_models()["forest_depth_2"]).fit(encode_angles(X), y)
    np.testing.assert_array_equal(model.predict(X), manual.predict(encode_angles(X)))
    np.testing.assert_allclose(model.predict_proba(X), manual.predict_proba(encode_angles(X)), rtol=0, atol=0)
    provenance.to_csv(output / "training_provenance.csv", index=False)
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError("An input artifact changed during pipeline training.")
    metadata = {**protocol, "pairing": dataset["pairing"], "source_hashes": hashes,
                "save_reload_predictions_verified": True,
                "manual_encoding_equivalence_verified": True,
                "script_sha256": sha256(__file__),
                "dependency_code_hashes": {name: sha256(ROOT / name) for name in
                    ["compare_pose_features.py", "diagnose_models.py", "tune_models.py", "train_baseline.py"]},
                "output_hashes": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "search_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved frozen pipeline: {CANDIDATE}")
    print(f"Training rows: {training_rows}, outer validation rows excluded: {validation_rows}")
    print("Six raw pose inputs -> nine sine/cosine features -> fixed random forest")
    print("Save/reload and manual-encoding prediction equivalence verified.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/sincos_forest_pr2_cube_v1_seed42")
    args = parser.parse_args()
    train_pipeline(args.output)
