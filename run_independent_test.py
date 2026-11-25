"""Predict at the measured pre-closure pose, then execute every valid test grasp."""

import argparse
import json
import runpy
from pathlib import Path
from typing import Iterable, Union

import joblib
import numpy as np
import pandas as pd
import pybullet as p

from artifact_paths import artifact_key
from prepare_dataset import load_cohort
from train_baseline import FEATURE_COLUMNS, ROOT, evaluate, sha256

COLLECTOR_PATH = ROOT / "version1_gripper"
DEFAULT_SEARCH_DIR = ROOT / "output/cv_pr2_cube_v1_seed42"
DEVELOPMENT_SEEDS = {42, 1001, 1002, 1003, 1004}
PR2Hand = runpy.run_path(str(COLLECTOR_PATH))["PR2Hand"]


def load_frozen_model(search_dir: Union[str, Path]) -> tuple[dict, dict[str, str]]:
    """Load a local fitted model after verifying its artifacts and pose contract."""
    search_dir = Path(search_dir).resolve()
    manifest_path = search_dir / "search_manifest.json"
    frozen = json.loads(manifest_path.read_text())
    model_path = search_dir / "selected_training_model.joblib"
    if sha256(model_path) != frozen["output_hashes"][model_path.name]:
        raise ValueError("The frozen model hash has changed.")
    for path, digest in frozen["source_hashes"].items():
        if sha256(path) != digest:
            raise ValueError(f"A frozen development source has changed: {path}")
    # Only load the trusted local model created by the earlier CV experiment.
    bundle = joblib.load(model_path)
    if bundle["feature_columns"] != FEATURE_COLUMNS or bundle["pairing"] != "PR2/cube_small":
        raise ValueError("The model does not match the measured-pose test contract.")
    if bundle["candidate"] != frozen["selected_candidate"]:
        raise ValueError("The candidate differs from the frozen search.")
    model = bundle["estimator"]
    try:
        labels = set(model.classes_)
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("The frozen model must predict labels 0 and 1 and their probabilities.") from exc
    if (
        labels != {0, 1}
        or not callable(getattr(model, "predict", None))
        or not callable(getattr(model, "predict_proba", None))
    ):
        raise ValueError("The frozen model must predict labels 0 and 1 and their probabilities.")
    hashes = {artifact_key(manifest_path): sha256(manifest_path),
              artifact_key(model_path): frozen["output_hashes"][model_path.name]}
    # The serialized pipeline calls this module at inference time.
    name = "compare_pose_features.py"
    digest = frozen.get("dependency_code_hashes", {}).get(name)
    if digest is not None and bundle.get("representation") == "sincos_inside_pipeline":
        path = ROOT / name
        if sha256(path) != digest:
            raise ValueError(f"Frozen model code has changed: {path}")
        hashes[artifact_key(path)] = digest
    return bundle, hashes


def _predict_models(
    features: dict[str, float],
    models: dict,
    columns: dict[str, tuple[str, str]],
) -> dict[str, Union[int, float]]:
    """Predict one measured pose and keep the output column order stable."""
    pose = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    if not np.isfinite(pose.to_numpy()).all():
        raise ValueError("Measured test features are not finite.")
    predictions = {}
    for name, model in models.items():
        prediction_column, probability_column = columns[name]
        success_index = int(np.flatnonzero(model.classes_ == 1)[0])
        predictions[prediction_column] = int(model.predict(pose)[0])
        predictions[probability_column] = float(model.predict_proba(pose)[0, success_index])
    return predictions


def run_test(
    output_dir: Union[str, Path],
    samples: int = 10,
    seed: int = 9001,
    max_attempts: int = 200,
    search_dir: Union[str, Path] = DEFAULT_SEARCH_DIR,
    *,
    comparison_dirs: Iterable[Union[str, Path]] = (),
) -> tuple[pd.DataFrame, dict]:
    """Execute sampled grasps and return paired predictions and test metrics."""
    if samples < 10 or max_attempts < samples:
        raise ValueError("At least ten valid tests and an adequate attempt limit are required.")
    if seed in DEVELOPMENT_SEEDS:
        raise ValueError("Use a new test seed, not a pilot or development seed.")
    search_dir = Path(search_dir).resolve()
    bundle, model_hashes = load_frozen_model(search_dir)
    primary = bundle["candidate"]
    models = {primary: bundle["estimator"]}
    for directory in comparison_dirs:
        extra, hashes = load_frozen_model(directory)
        name = extra["candidate"]
        if name in models:
            raise ValueError(f"Repeated test candidate: {name}")
        models[name] = extra["estimator"]
        model_hashes.update(hashes)
    columns = {name: ("prediction", "success_probability") if name == primary else
               (f"{name}_prediction", f"{name}_success_probability") for name in models}
    source, source_hashes, _ = load_cohort(ROOT / "output/dev_pr2_cube_v1_manifest.json")
    reference_config = json.loads((ROOT / "output/dev_pr2_cube_v1_b01_seed1001/config.json").read_text())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"role": "independent_new_pose_simulation_test", "seed": seed,
                "requested_samples": samples, "max_attempts": max_attempts,
                "prediction_time": "after approach settling, before closure",
                "execute_predicted_failures": True, "candidate": bundle["candidate"],
                "model_sha256": model_hashes[artifact_key(Path(search_dir) / "selected_training_model.joblib")],
                "search_manifest_sha256": model_hashes[artifact_key(Path(search_dir) / "search_manifest.json")],
                "candidates": list(models), "prediction_columns": columns,
                "frozen_model_artifact_hashes": model_hashes,
                "collector_sha256": sha256(COLLECTOR_PATH),
                "training_reference_config": reference_config,
                "config_note": "same physical/sampling/measurement/label defaults; new seed and count"}
    (output / "test_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records, attempts = [], []
    hand = PR2Hand(gui=False, seed=seed)
    with (output / "events.jsonl").open("w") as events:
        def log(event):
            events.write(json.dumps(event) + "\n")
            events.flush()

        try:
            hand.connect()
            for attempt_id in range(1, max_attempts + 1):
                pos, (yaw, roll, pitch) = hand.sample_pose()
                ori = p.getQuaternionFromEuler([roll, pitch, yaw])
                audit = {"attempt_id": attempt_id, "planned_approach_x": float(pos[0]),
                         "planned_approach_y": float(pos[1]), "planned_approach_z": float(pos[2]),
                         "planned_roll": float(roll), "planned_pitch": float(pitch),
                         "planned_yaw": float(yaw)}
                pending = {}

                def predict(features: dict[str, float]) -> None:
                    predictions = _predict_models(features, models, columns)
                    pending.update(features)
                    pending.update(predictions)
                    # This line is flushed before run_trial issues its first close command.
                    log({"event": "prediction_before_closure", "attempt_id": attempt_id, **pending})

                try:
                    hand.reset_trial(pos, ori)
                    if hand.is_overlapping():
                        audit["run_status"] = "rejected_initial_collision"
                    else:
                        result = hand.run_trial(pos, ori, on_pose_measured=predict)
                        np.testing.assert_allclose([pending[k] for k in FEATURE_COLUMNS],
                                                   [result[k] for k in FEATURE_COLUMNS])
                        record = {"attempt_id": attempt_id, **result}
                        for model_columns in columns.values():
                            record.update({key: pending[key] for key in model_columns})
                        record["correct"] = record["prediction"] == record["success"]
                        records.append(record)
                        audit.update(record, run_status="completed")
                        log({"event": "outcome_after_lift", "attempt_id": attempt_id,
                             "success": result["success"], "prediction": pending["prediction"]})
                        print(f"Test {len(records)}/{samples}: predicted={record['prediction']} "
                              f"actual={record['success']} correct={record['correct']}", flush=True)
                except (p.error, ValueError, RuntimeError) as exc:
                    audit.update(pending, run_status="error", error=str(exc))
                attempts.append(audit)
                if len(records) >= samples:
                    break
        finally:
            hand.disconnect()
            pd.DataFrame(records).to_csv(output / "test_predictions.csv", index=False)
            pd.DataFrame(attempts).to_csv(output / "attempts.csv", index=False)
    complete = len(records) == samples
    results = pd.DataFrame(records)
    metrics = evaluate(results.success, results.prediction) if records else None
    overlap = (len(results.merge(source[FEATURE_COLUMNS], on=FEATURE_COLUMNS)) if records else 0)
    for path, digest in source_hashes.items():
        if sha256(path) != digest:
            raise ValueError("A development source changed during testing.")
    for path, digest in model_hashes.items():
        if sha256(path) != digest:
            raise ValueError(f"A frozen model artifact changed during testing: {path}")
    summary = {"completed_requested_test": complete, "valid_test_attempts": len(records),
               "candidate_status_counts": pd.DataFrame(attempts).run_status.value_counts().to_dict(),
               "prediction_metrics": metrics,
               "metrics_by_candidate": {name: evaluate(results.success, results[prediction_column])
                                        for name, (prediction_column, _) in columns.items()} if records else {},
               "physical_grasp_success_rate": float(results.success.mean()) if records else None,
               "exact_measured_feature_overlap_with_development_cohort": overlap,
               "fit_performed": False, "threshold_or_parameters_changed": False,
               "warning": "small test set; only PR2/cube and measured pre-closure prediction",
               "source_hashes": source_hashes,
               "output_hashes": {f.name: sha256(f) for f in output.iterdir() if f.is_file()}}
    (output / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["prediction_metrics"], indent=2))
    if not complete:
        raise RuntimeError("Attempt limit reached before the requested test count.")
    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/independent_pr2_cube_v1_seed9001")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--search", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument("--compare", type=Path, action="append", default=[])
    args = parser.parse_args()
    run_test(args.output, args.samples, args.seed, search_dir=args.search, comparison_dirs=args.compare)
