"""Measure fixed-classifier learning curves using frozen training rows only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from diagnose_models import fixed_models, load_training
from train_baseline import ROOT, evaluate, sha256


def learning_curve_scores(X, y, provenance, repeats=5, sizes=(32, 64, 96, 128, None)):
    """Use nested stratified subsets and shared validation folds for all models."""
    if repeats < 1 or set(y) != {0, 1} or y.value_counts().min() < 5:
        raise ValueError("Learning curves require positive repeats and five rows per class.")
    if not sizes or sizes[-1] is not None or any(
            not isinstance(n, int) or n < 2 for n in sizes[:-1]):
        raise ValueError("Sizes must be positive integer counts followed by None for full folds.")
    if list(sizes[:-1]) != sorted(set(sizes[:-1])):
        raise ValueError("Training sizes must be strictly increasing.")
    models = fixed_models()
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=42)
    scores, membership = [], []
    for number, (train, validation) in enumerate(splitter.split(X, y)):
        if set(provenance.iloc[train].pose_group_id) & set(provenance.iloc[validation].pose_group_id):
            raise ValueError("Pose groups cross learning-curve folds.")
        if any(size >= len(train) for size in sizes[:-1]):
            raise ValueError("Requested subset must be smaller than every full fold.")
        rng = np.random.default_rng(20261124 + number)
        class_indices = [rng.permutation(train[y.iloc[train].to_numpy() == label]) for label in [0, 1]]
        previous = set()
        for level, requested in enumerate(sizes, 1):
            if requested is None:
                subset = train
            else:
                count0 = int(np.rint(requested * len(class_indices[0]) / len(train)))
                count0 = max(1, min(count0, len(class_indices[0]), requested - 1))
                count1 = requested - count0
                if count1 > len(class_indices[1]):
                    raise ValueError("Not enough samples for the requested stratified subset.")
                subset = np.concatenate([class_indices[0][:count0], class_indices[1][:count1]])
            if not previous.issubset(set(subset)):
                raise ValueError("Stratified training subsets must be nested.")
            previous = set(subset)
            repeat, fold = number // 5 + 1, number % 5 + 1
            for role, indices in [("training", subset), ("validation", validation)]:
                rows = provenance.iloc[indices].copy()
                rows["repeat"], rows["fold"], rows["size_level"], rows["role"] = repeat, fold, level, role
                rows["actual_success"] = y.iloc[indices].to_numpy()
                membership.append(rows)
            for name, prototype in models.items():
                model = clone(prototype)
                model.fit(X.iloc[subset], y.iloc[subset])
                scores.append({"candidate": name, "repeat": repeat, "fold": fold,
                               "size_level": level, "requested_size": requested,
                               "training_rows": len(subset), "validation_rows": len(validation),
                               "training_successes": int(y.iloc[subset].sum()),
                               "training_accuracy": float(np.mean(model.predict(X.iloc[subset]) == y.iloc[subset])),
                               **evaluate(y.iloc[validation], model.predict(X.iloc[validation]))})
    scores = pd.DataFrame(scores)
    summary = scores.groupby(["candidate", "size_level"], as_index=False).agg(
        mean_training_rows=("training_rows", "mean"), min_training_rows=("training_rows", "min"),
        max_training_rows=("training_rows", "max"), mean_training_accuracy=("training_accuracy", "mean"),
        mean_cv_accuracy=("accuracy", "mean"), std_cv_accuracy=("accuracy", lambda s: s.std(ddof=0)))
    return scores, summary, pd.concat(membership, ignore_index=True)


def run_learning_curves(output_dir):
    searches = {"logistic_C_10": ROOT / "output/cv_pr2_cube_v1_seed42",
                "tree_depth_3": ROOT / "output/cv_tree_pr2_cube_v1_seed42",
                "forest_depth_2": ROOT / "output/cv_forest_pr2_cube_v1_seed42"}
    X, y, provenance, _, manifest, hashes = load_training(
        ROOT / "output/dev_pr2_cube_v1_balanced240",
        ROOT / "output/baseline_pr2_cube_v1_seed42", searches)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    scores, summary, membership = learning_curve_scores(X, y, provenance)
    scores.to_csv(output / "fold_scores.csv", index=False)
    summary.to_csv(output / "curve_summary.csv", index=False)
    membership.to_csv(output / "subset_membership.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (name, subset) in zip(axes, summary.groupby("candidate")):
        ax.plot(subset.mean_training_rows, 100 * subset.mean_training_accuracy, marker="o", label="Training")
        ax.plot(subset.mean_training_rows, 100 * subset.mean_cv_accuracy, marker="o", label="CV validation")
        ax.set(title=name, xlabel="Training rows per fold", ylim=(40, 100))
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Mean accuracy (%)")
    fig.suptitle("Learning curves: fixed parameters, 5 folds x 5 repeats, training partition only")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError("An input changed during learning-curve evaluation.")
    metadata = {"role": "training_only_learning_curve", "training_partition_rows": len(X),
                "outer_validation_evaluated": False, "independent_test_performed": False,
                "hyperparameters_changed": False, "new_model_saved": False,
                "folds": 5, "repeats": 5, "fits": len(scores), "splitter_seed": 42,
                "subset_seed_rule": "20261124 + zero-based split index",
                "estimator_seed": 42, "subset_sizes": [32, 64, 96, 128, "full_fold"],
                "subset_strategy": "nested class-stratified prefixes with rounded class proportions",
                "pairing": manifest["pairing"],
                "warning": "Previously selected configurations on reused training data. Correlated folds are not independent experiments; fold SD is not a confidence interval. More data may help but gains beyond observed sizes are not established.",
                "source_hashes": hashes,
                "code_hashes": {name: sha256(ROOT / name) for name in
                                ["plot_learning_curves.py", "diagnose_models.py", "tune_models.py", "train_baseline.py"]},
                "output_hashes": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "learning_curve_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/learning_curves_pr2_cube_v1")
    args = parser.parse_args()
    run_learning_curves(args.output)
