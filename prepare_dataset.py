"""Build a traceable balanced development dataset without changing raw trials."""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from artifact_paths import ROOT, artifact_key, sha256
from inspect_dataset import FEATURE_COLUMNS, inspect_dataset


COLUMNS = FEATURE_COLUMNS + ["success"]


def load_cohort(manifest_path, root=ROOT):
    """Verify source hashes, pairing, configuration and row-level provenance."""
    root = Path(root)
    manifest_path = Path(manifest_path)
    cohort = json.loads(manifest_path.read_text())
    fingerprints = {artifact_key(manifest_path): sha256(manifest_path)}
    frames = []
    reference_config = None
    collector_hash = None
    seen_runs = set()
    for item in cohort["runs"]:
        if item["run_id"] in seen_runs:
            raise ValueError("The cohort contains a repeated run ID.")
        seen_runs.add(item["run_id"])
        path = root / item["manifest"]
        if sha256(path) != item["manifest_sha256"]:
            raise ValueError(f"Batch manifest hash mismatch: {path}")
        fingerprints[artifact_key(path)] = sha256(path)
        batch = json.loads(path.read_text())
        if batch["run_id"] != item["run_id"] or batch["pairing"] != cohort["pairing"]:
            raise ValueError("Run ID or pairing differs from the cohort manifest.")
        for name, digest in batch["hashes"].items():
            source = path.parent / name
            if sha256(source) != digest:
                raise ValueError(f"Source artifact hash mismatch: {source}")
            fingerprints[artifact_key(source)] = digest
        config = json.loads((path.parent / "config.json").read_text())
        comparable = {k: v for k, v in config.items()
                      if k not in {"seed", "n_samples", "max_attempts"}}
        if reference_config is None:
            reference_config = comparable
            collector_hash = batch["collector_sha256"]
        if comparable != reference_config or collector_hash != batch["collector_sha256"]:
            raise ValueError("Source runs use incompatible experiment configurations.")
        with contextlib.redirect_stdout(io.StringIO()):
            X, y = inspect_dataset(path.parent / "grasps.csv")
        data = X.assign(success=y)
        audit = pd.read_csv(path.parent / "attempts.csv")
        completed = audit.loc[audit.run_status == "completed"].reset_index(drop=True)
        provenance = pd.read_csv(path.parent / "provenance.csv")
        if len(data) != batch["completed_trials"] or len(completed) != len(data):
            raise ValueError("Source row counts disagree.")
        if not np.allclose(data.to_numpy(), completed[COLUMNS].to_numpy(dtype=float),
                           rtol=1e-12, atol=1e-12):
            raise ValueError("Training and audit rows disagree.")
        if len(provenance) != len(data):
            raise ValueError("Provenance row count differs from the training table.")
        if not provenance.run_id.eq(item["run_id"]).all():
            raise ValueError("Provenance has an incorrect run ID.")
        if not np.array_equal(provenance.attempt_id, completed.attempt_id):
            raise ValueError("Provenance attempt IDs do not match completed audit rows.")
        if not np.array_equal(provenance.training_row_number, np.arange(1, len(data) + 1)):
            raise ValueError("Provenance row order differs from the training table.")
        frames.append(pd.concat([data.reset_index(drop=True), provenance], axis=1))
    if not frames:
        raise ValueError("No source runs were supplied.")
    records = pd.concat(frames, ignore_index=True)
    if len(records) != cohort["completed_trials"]:
        raise ValueError("Cohort total differs from source records.")
    counts = {str(k): int(v) for k, v in records.success.value_counts().items()}
    if counts != cohort["class_counts"]:
        raise ValueError("Cohort class counts differ from source records.")
    if records.duplicated(["run_id", "attempt_id"]).any():
        raise ValueError("Source keys are not unique.")
    return records, fingerprints, cohort["pairing"]


def pose_groups(records, position_tolerance=0.001, angle_tolerance=np.pi / 180):
    """Group nearby poses using translation and quaternion angular distance.

    Connected components keep transitive neighbors on the same side of a later split.
    Thresholds are configurable diagnostics, not guarantees of physical independence.
    """
    if not np.isfinite([position_tolerance, angle_tolerance]).all() or min(
            position_tolerance, angle_tolerance) < 0:
        raise ValueError("Pose tolerances must be finite and non-negative.")
    xyz = records[["x", "y", "z"]].to_numpy(dtype=float)
    roll, pitch, yaw = (records[["roll", "pitch", "yaw"]].to_numpy(dtype=float) / 2).T
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    quats = np.column_stack([sr * cp * cy - cr * sp * sy,
                             cr * sp * cy + sr * cp * sy,
                             cr * cp * sy - sr * sp * cy,
                             cr * cp * cy + sr * sp * sy])
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    pairs = []
    for i in range(len(records)):
        for j in range(i):
            distance = float(np.linalg.norm(xyz[i] - xyz[j]))
            if distance > position_tolerance:
                continue
            angle = float(2 * np.arccos(np.clip(abs(np.dot(quats[i], quats[j])), 0, 1)))
            if angle <= angle_tolerance + 1e-12:
                parents[find(i)] = find(j)
                pairs.append({"source_row_a": j + 1, "source_row_b": i + 1,
                              "distance_m": distance, "angle_rad": angle})
    names = {}
    groups = []
    for i in range(len(records)):
        representative = find(i)
        if representative not in names:
            names[representative] = f"pose_group_{len(names) + 1:04d}"
        groups.append(names[representative])
    return groups, pd.DataFrame(pairs, columns=[
        "source_row_a", "source_row_b", "distance_m", "angle_rad"])


def balanced_indices(records, per_class, seed):
    """Select real rows without replacement, then shuffle their output order."""
    if per_class < 1:
        raise ValueError("per_class must be positive.")
    rng = np.random.default_rng(seed)
    selected = []
    for label in [0, 1]:
        candidates = np.flatnonzero(records.success.to_numpy() == label)
        if len(candidates) < per_class:
            raise ValueError(f"Not enough class {label} samples: {len(candidates)} < {per_class}")
        selected.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    return rng.permutation(selected)


def prepare_dataset(manifest_path, output_dir, per_class=120, seed=20261124,
                    position_tolerance=0.001, angle_tolerance=np.pi / 180, root=ROOT):
    records, fingerprints, pairing = load_cohort(manifest_path, root)
    records["source_cohort_row"] = np.arange(1, len(records) + 1)
    groups, pairs = pose_groups(records, position_tolerance, angle_tolerance)
    records["pose_group_id"] = groups
    indices = balanced_indices(records, per_class, seed)
    chosen = records.iloc[indices].reset_index(drop=True)
    dataset = chosen[COLUMNS].copy()
    provenance = chosen[["run_id", "attempt_id", "training_row_number",
                         "source_cohort_row", "pose_group_id"]].copy()
    provenance.insert(0, "dataset_row_number", np.arange(1, len(chosen) + 1))
    if any(sha256(path) != digest for path, digest in fingerprints.items()):
        raise ValueError("A source artifact changed during preparation.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    dataset.to_csv(output / "development.csv", index=False)
    provenance.to_csv(output / "provenance.csv", index=False)
    records[["source_cohort_row", "run_id", "attempt_id", "pose_group_id", "success"]].to_csv(
        output / "source_inventory.csv", index=False)
    pairs.to_csv(output / "near_pose_pairs.csv", index=False)
    sizes = records.groupby("pose_group_id").size()
    conflicts = records.groupby("pose_group_id").success.nunique()
    metadata = {"pairing": pairing, "role": "development_only", "selection_seed": seed,
                "per_class": per_class, "source_rows": len(records), "selected_rows": len(dataset),
                "class_counts": {str(k): int(v) for k, v in dataset.success.value_counts().items()},
                "source_exact_feature_duplicates": int(records.duplicated(FEATURE_COLUMNS).sum()),
                "position_tolerance_m": position_tolerance, "angle_tolerance_rad": angle_tolerance,
                "near_pose_pairs": len(pairs), "pose_groups": len(sizes),
                "multirow_groups": int((sizes > 1).sum()),
                "mixed_label_groups": int((conflicts > 1).sum()),
                "grouping_scope": "entire source cohort before selection",
                "future_split_rule": "keep pose_group_id together; do not use provenance as features",
                "source_hashes": fingerprints,
                "output_hashes": {name: sha256(output / name) for name in [
                    "development.csv", "provenance.csv", "source_inventory.csv", "near_pose_pairs.csv"]},
                "model_trained": False, "independent_test_created": False}
    (output / "dataset_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k not in {
        "source_hashes", "output_hashes"}}, indent=2))
    return dataset, provenance, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "output/dev_pr2_cube_v1_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output/dev_pr2_cube_v1_balanced240")
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--position-tolerance", type=float, default=0.001)
    parser.add_argument("--angle-tolerance-deg", type=float, default=1.0)
    args = parser.parse_args()
    prepare_dataset(args.manifest, args.output, args.per_class, args.seed,
                    args.position_tolerance, np.deg2rad(args.angle_tolerance_deg))
