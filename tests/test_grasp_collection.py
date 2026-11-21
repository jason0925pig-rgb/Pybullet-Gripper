"""Focused checks for measurement, labelling, and collection failure paths."""

import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
PR2Hand = runpy.run_path(str(ROOT / "version1_gripper"))["PR2Hand"]


class HoldTests(unittest.TestCase):
    def evaluate(self, *, height=0.3, floor=False, contact=True, slip=False,
                 drop_after_step=False):
        hand = PR2Hand(gui=False)
        steps = [0]

        def step():
            steps[0] += 1

        def pose(*args, **kwargs):
            z = 0.025 if drop_after_step and steps[0] else height
            return (0, 0, z), (0, 0, 0, 1)

        def contacts(body_a, body_b, **kwargs):
            return [object()] if (floor if body_b == hand.plane_id else contact) else []

        hand.box_id, hand.gripper_id, hand.plane_id = 1, 2, 3
        reference = ((0, 0, 0.1), (0, 0, 0, 1))
        shifted = ((0.05, 0, 0.1), (0, 0, 0, 1))
        with patch.object(hand, "step_sim", side_effect=step), \
             patch.object(hand, "grasp"), \
             patch.object(hand, "relative_pose", side_effect=lambda *args:
                          shifted if slip and steps[0] else reference), \
             patch.object(p, "getBasePositionAndOrientation", side_effect=pose), \
             patch.object(p, "getContactPoints", side_effect=contacts):
            return hand.evaluate_hold(0.025)

    def test_stable_hold_exceeds_three_seconds(self):
        result = self.evaluate()
        self.assertEqual(result["success"], 1)
        self.assertGreater(result["stable_hold_seconds"], 3)

    def test_touching_without_lift_is_failure(self):
        result = self.evaluate(height=0.025)
        self.assertEqual(result["success"], 0)
        self.assertEqual(result["failure_reason"], "insufficient_lift")

    def test_floor_support_is_failure(self):
        self.assertEqual(self.evaluate(floor=True)["failure_reason"], "floor_contact")

    def test_lost_contact_is_failure(self):
        self.assertEqual(self.evaluate(contact=False)["failure_reason"], "lost_gripper_contact")

    def test_slip_is_failure(self):
        self.assertEqual(self.evaluate(slip=True)["failure_reason"], "excessive_slip")

    def test_later_drop_cannot_retain_success(self):
        self.assertEqual(self.evaluate(drop_after_step=True)["success"], 0)


class CollectionTests(unittest.TestCase):
    def test_relative_pose_in_rotated_frame(self):
        hand = PR2Hand(gui=False)
        hand.client_id = p.connect(p.DIRECT)
        try:
            reference = p.createMultiBody(
                basePosition=[1, 2, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]),
                physicsClientId=hand.client_id)
            body = p.createMultiBody(
                basePosition=[1, 3, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]),
                physicsClientId=hand.client_id)
            pos, ori = hand.relative_pose(body, reference)
            np.testing.assert_allclose(pos, [1, 0, 0], atol=1e-6)
            np.testing.assert_allclose(p.getEulerFromQuaternion(ori), [0, 0, 0], atol=1e-6)
        finally:
            hand.disconnect()

    def test_rejection_limit_and_empty_dataset(self):
        hand = PR2Hand(gui=False, seed=1)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(hand, "is_overlapping", return_value=True):
                result = hand.run_demo(1, directory, max_attempts=2)
            self.assertTrue(result.empty)
            self.assertIsNone(hand.client_id)
            audit = pd.read_csv(Path(directory) / "attempts.csv")
            self.assertEqual(len(audit), 2)
            self.assertTrue((audit.run_status == "rejected_initial_collision").all())
            self.assertEqual(list(pd.read_csv(Path(directory) / "grasps.csv").columns),
                             ["x", "y", "z", "roll", "pitch", "yaw", "success"])

    def test_simulation_errors_do_not_become_negative_samples(self):
        hand = PR2Hand(gui=False, seed=1)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(hand, "is_overlapping", return_value=False), \
                 patch.object(hand, "run_trial", side_effect=RuntimeError("Injected failure")):
                result = hand.run_demo(1, directory, max_attempts=1)
            self.assertTrue(result.empty)
            self.assertIsNone(hand.client_id)
            audit = pd.read_csv(Path(directory) / "attempts.csv")
            self.assertEqual(audit.run_status.iloc[0], "error")
            self.assertNotIn("success", audit.columns)


if __name__ == "__main__":
    unittest.main()
