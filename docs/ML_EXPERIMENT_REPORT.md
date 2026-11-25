# PR2/cube simulation and classification report

## Scope

This is the single-pair command-line demo for PR2 and `cube_small`. The full
coursework still requires coverage of at least two grippers and two objects,
plus integration of the user training and prediction workflow into the GUI.

## Stages 1–2 · Simulation and data

The sampler generates an approach pose around the cube. The simulator settles
at the intended grasp and measures the gripper base relative to the object
before closing the fingers. The six features are x, y, z in meters and roll,
pitch, yaw in radians. The binary label comes from the simulated lift and
stable hold. See the [collection guide](../ML_DATA_COLLECTION.md).

The development cohort contains four batches of 100 valid trials (seeds
1001–1004), with 130 successes and 270 failures. Initial collisions are recorded
separately. The balanced dataset contains 120 samples per class with exact
source-row provenance. The frozen split has 192 training and 48 outer validation
rows. The five-fold search uses only the training partition. The original
pose-group check found no pairs within 1 mm and 1 degree in the cohort.

[Raw batch index](../output/dev_pr2_cube_v1_manifest.json) ·
[Balanced development data](../output/dev_pr2_cube_v1_balanced240/development.csv) ·
[Frozen split](../output/baseline_pr2_cube_v1_seed42/split_provenance.csv)

## Stage 3 · Development experiments

| Configuration | Training-only repeated CV (5 x 5) | Existing outer validation |
| --- | ---: | ---: |
| Raw logistic regression, C=10 | 68.83% | 68.75% (33/48) |
| Raw decision tree, depth 3, leaf minimum 5 | 67.29% | 60.42% (29/48) |
| Raw random forest, 100 trees, depth 2, leaf minimum 5 | 72.58% | 58.33% (28/48) |
| Sine/cosine random forest, same forest settings | 75.60% | 60.42% (29/48) |

The original out-of-fold analysis found 29 rows all three raw classifiers
misclassified. Learning curves improved from 32 to 64 training rows per fold
and showed no consistent rise at the largest sizes tested. Converting each
angle to sine and cosine improved repeated CV for all three models. The
frozen sine/cosine forest passed six raw pose values through a saved pipeline
and corrected two errors while introducing one new error in outer validation
relative to the raw forest. CV scores and the previously viewed outer
validation set are development evidence.

[Feature comparison](../output/feature_ablation_pr2_cube_v1/feature_comparison.csv) ·
[Saved pipeline settings](../output/sincos_forest_pr2_cube_v1_seed42/frozen_protocol.json) ·
[Pipeline outer predictions](../output/sincos_forest_validation_pr2_cube_v1/validation_predictions.csv)

## Stage 4 · Independent simulation

The initial logistic regression run (seed 9001) scored 8/10. The later paired
test (seed 9002) used 30 valid grasps from the same sampling strategy. Three
saved models predicted each measured pose before closure and were scored
against the same outcome. Six of 36 sampled candidates were rejected by the
initial collision filter. Twelve of the 30 tested grasps succeeded physically.

| Configuration | Correct | Accuracy | True positives | False positives |
| --- | ---: | ---: | ---: | ---: |
| Raw logistic regression, C=10 | 20/30 | 66.67% | 9 | 7 |
| Raw random forest, depth 2 | 18/30 | 60.00% | 7 | 7 |
| Sine/cosine random forest, depth 2 | 19/30 | 63.33% | 7 | 6 |

The 30 measured poses had no exact matches in the development cohort or the
earlier test. The models differ by one or two correct predictions in this batch.
These results cover one gripper/object pairing.

[Paired predictions](../output/independent_comparison_pr2_cube_v1_seed9002/test_predictions.csv) ·
[Paired metrics and audit](../output/independent_comparison_pr2_cube_v1_seed9002/test_summary.json)

## Reproduce or replay

Run from the repository root in the environment listed in the README. These
commands use fresh output directories. The seed-9002 command replays the
published test. Use a new seed for a new independent sample.

```bash
python diagnose_models.py --output output/diagnostics_pr2_cube_v1_recheck
python plot_learning_curves.py --output output/learning_curves_pr2_cube_v1_recheck
python compare_pose_features.py --output output/feature_ablation_pr2_cube_v1_recheck
python train_pose_pipeline.py --output output/sincos_forest_pr2_cube_v1_seed42_recheck
python evaluate_selected.py --search output/sincos_forest_pr2_cube_v1_seed42_recheck --output output/sincos_forest_validation_pr2_cube_v1_recheck
python run_independent_test.py --samples 30 --seed 9002 \
  --compare output/cv_forest_pr2_cube_v1_seed42 \
  --compare output/sincos_forest_pr2_cube_v1_seed42 \
  --output output/independent_comparison_pr2_cube_v1_seed9002_recheck
```

Artifact manifests use paths relative to the repository for checked-in inputs.
References to migrated manifests use refreshed manifest hashes. Original
measurement tables, predictions, model binaries and historical code hashes
retain their original contents.
