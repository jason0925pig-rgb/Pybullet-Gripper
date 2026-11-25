# PR2 grasp data collection

The collector preserves the original hemisphere approach sampling, PR2 finger
commands, inward approach and vertical lift. It does not change `main.py` or the
GUI mock data generator. Run it with a Python environment containing NumPy,
pandas and PyBullet; pandas is already required by the coursework.

## Run

From this repository:

```bash
python version1_gripper --direct --samples 20 --seed 42 --output output/pr2_cube_demo_recheck
```

Omit `--direct` to inspect motions in the GUI. A GUI run deliberately sleeps
between steps; DIRECT runs use the same physics timestep without that sleep.
`--max-attempts` bounds all generated candidates, including rejected/error attempts
(default: 20 times the requested number of valid trials). An incomplete collection
prints a warning and still saves its valid records. Use a different output directory
for each run: files in a reused directory are overwritten, not appended.

## Execution and feature measurement

1. Reset the cube and gripper poses, velocities and finger states; command fingers open.
2. Reject initial candidates too close to the cube or touching the floor.
3. Step for one simulated second to prepare the scene.
4. Approach the original fixed final radius, then settle for 0.25 simulated seconds.
5. Read both actual base poses **before finger closure**, at the same simulation step.
6. Compute `inverse(object_pose) * gripper_pose`; convert its orientation to Euler angles.
7. Close for 1.5 seconds, lift over 480 steps, and evaluate the post-lift hold.

Features use PyBullet base center-of-mass frames, not a fingertip or grasp-center
frame. Positions are in meters; Euler angles are roll, pitch, yaw in radians.
A change to the reference frame or measurement time requires matching changes in
training and inference. Planned approach coordinates and final grasp targets are
world-frame commands, not measured classifier inputs.

The sampling distribution is uniform in the original upper hemisphere shell.
The final target radius is always R_MIN; varying the approach radius does not vary
the final grasp radius. Rejection changes the distribution of accepted trials.

## Binary ground truth

The old accumulated-contact labels 0/0.5/1 are replaced by a continuous post-lift
check. A valid trial succeeds only if all conditions hold from the end of lifting
through a 3.1-second observation window:

- Cube center is at least 0.10 m above its settled pre-approach height.
- The cube is not in contact with the floor.
- The gripper and cube remain in contact.
- Cube translation relative to the gripper changes by at most 0.02 m.
- Cube rotation relative to the gripper changes by at most 0.35 radians.

Slip is measured relative to the pose at lift completion. Any observed violation
ends the evaluation with success=0 and a reason. Otherwise success=1. These
thresholds are configurable constructor parameters and are engineering choices,
not coursework-specified tolerances. The minimum lift and contact conditions
reject clear failures; bounded relative motion is an operational stability proxy,
not a force-closure proof. Inspect successful, failed and borderline trials in GUI
before relying on large-scale labels. One-step contact interruptions count as failure.

Runtime errors are separate from physical failures and have no training label.
The same label evaluator must be used during independent test experiments.

## Output contract

- `grasps.csv`: valid trials only, columns `x,y,z,roll,pitch,yaw,success`.
  The six pose values are measured before closure; success is binary and last.
- `attempts.csv`: all finished candidate attempts, including rejection and errors.
  Contains attempt ID, gripper/object identifiers, planned approach pose, final
  grasp target for completed trials, measured features, outcome and diagnostics.
  Filtering to `run_status == "completed"` preserves the order of `grasps.csv`.
- `config.json`: seed, timestep, sampling limits, label thresholds, reference frame,
  units and measurement time.

`run_demo()` returns the same seven-column pandas DataFrame that it exports.
`run_trial()` assumes connection, reset and initial collision screening have already
been performed by its caller. `evaluate_hold()` replaces the old `get_label()`:
calling it advances the simulation while evaluating a new post-lift window.

The collector does not fabricate balanced labels or discard failures to force 1:1.
Retain raw records, then prepare a balanced development dataset separately; do not
rebalance an independent test set by cherry-picking outcomes. This script supports
only the original PR2/cube pair and does not yet implement training or inference.
The formal test-time feature availability question (planned versus measured pose)
needs to be agreed with the team/course staff before evaluating a deployed planner.

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests cover stable hold, floor support, insufficient lift, loss of contact, relative
slip and later dropping using controlled observations; a real DIRECT world checks
relative transforms in a rotated reference frame, bounded rejection, and error
records/connection cleanup. Run the collector itself for an end-to-end physics check.
