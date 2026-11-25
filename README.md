# PyBullet Gripper — demo_0

This branch runs a PR2 gripper and cube simulation from the command line.
It records the measured pose before closing the gripper and labels each
attempt from the simulated lift. Classifiers learn to predict that outcome
from the pose. The included data and models cover this one gripper/object
pair. `main.py` opens the PyQt mock interface.

**See the result first.** The [30-grasp comparison](output/independent_comparison_pr2_cube_v1_seed9002/test_predictions.csv)
shows what each saved classifier predicted and what happened in the simulator.
The [experiment report](docs/ML_EXPERIMENT_REPORT.md) explains the scores and
how to reproduce them.

## Quick start

Run these commands from the repository directory with your Python environment
active. The tested environment uses Python 3.9, PyBullet 3.2.5, NumPy 2.0.2,
pandas 2.3.1, scikit-learn 1.6.1 and joblib 1.5.1. Plotting uses Matplotlib
3.9.4 and the optional GUI uses PyQt6.

**Inspect the data used for the ML experiments.**

```bash
python inspect_dataset.py output/dev_pr2_cube_v1_balanced240/development.csv
```

Expect 240 rows, six pose features, and 120 samples per class. Label `0`
means failure and `1` means success.

**Run a small simulation.** This collects five valid attempts and writes
`grasps.csv` (poses and labels) and `attempts.csv` (including rejected
candidates) to the chosen directory. Pick a fresh directory for each run.
See the [simulation and pose data guide](ML_DATA_COLLECTION.md) for details.

```bash
python version1_gripper --direct --samples 5 --seed 42 --output output/quick_start_recheck
```

## Where to work

| Task | Entry point |
| --- | --- |
| Sample and execute PR2 grasps | [version1_gripper](version1_gripper) |
| Verify and balance recorded poses | [prepare_dataset.py](prepare_dataset.py) |
| Make the frozen split and compare classifiers | [train_baseline.py](train_baseline.py), [tune_models.py](tune_models.py) |
| Explore errors and training set size | [diagnose_models.py](diagnose_models.py), [plot_learning_curves.py](plot_learning_curves.py) |
| Compare pose features and save the selected pipeline | [compare_pose_features.py](compare_pose_features.py), [train_pose_pipeline.py](train_pose_pipeline.py) |
| Score saved models on validation rows or new grasps | [evaluate_selected.py](evaluate_selected.py), [run_independent_test.py](run_independent_test.py) |

Run the automated checks with `python -m unittest discover -s tests`. The
[coursework brief](OOP%20-%20Final%20CW%20Description2026.pdf) describes the full assignment.

The default dataset, split and model search directories shared by the
diagnostics, learning curves and pipeline training are in `diagnose_models.py`.
When editing pose encoding, train and save a new pipeline. The existing
sine/cosine bundle references `compare_pose_features.encode_angles` and the
independent runner verifies that module against the saved hash.
