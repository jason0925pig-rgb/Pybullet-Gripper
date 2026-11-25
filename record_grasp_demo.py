"""Record the existing collector in DIRECT mode without opening a GUI."""

import argparse
import runpy
import time
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
PR2Hand = runpy.run_path(str(ROOT / "version1_gripper"))["PR2Hand"]


class RecordedHand(PR2Hand):
    """Add observation-only rendering to the unchanged physics stepping."""

    def __init__(self, output_dir, seed):
        super().__init__(gui=False, seed=seed)
        self.output_dir = Path(output_dir)
        self.trial_number = 0
        self.recording = False
        self.frames = []
        self.steps = 0

    def capture_frame(self):
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.26], distance=0.95,
            yaw=45, pitch=-20, roll=0, upAxisIndex=2)
        projection = p.computeProjectionMatrixFOV(50, 640 / 400, 0.01, 5)
        pixels = p.getCameraImage(
            640, 400, viewMatrix=view, projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self.client_id)[2]
        frame = Image.fromarray(np.asarray(pixels, dtype=np.uint8)[:, :, :3])
        draw = ImageDraw.Draw(frame)
        elapsed = self.steps * self.dt
        if self.steps <= 240:
            stage = "Prepare"
        elif self.steps <= 440:
            stage = "Approach"
        elif self.steps <= 500:
            stage = "Settle"
        elif self.steps <= 860:
            stage = "Close"
        elif self.steps <= 1340:
            stage = "Lift"
        else:
            stage = "Post-lift hold"
        height = p.getBasePositionAndOrientation(
            self.box_id, physicsClientId=self.client_id)[0][2]
        draw.rectangle([0, 0, 640, 48], fill="white")
        draw.text((10, 8), f"Trial {self.trial_number} | {stage} | Simulated time: {elapsed:.2f} s",
                  fill="black")
        draw.text((10, 27), f"Cube center height: {height:.3f} m | Orange cube", fill="black")
        self.frames.append(frame)

    def step_sim(self):
        super().step_sim()
        if self.recording:
            self.steps += 1
            if self.steps % 24 == 0:
                self.capture_frame()

    def run_trial(self, pos, ori):
        self.trial_number += 1
        self.frames = []
        self.steps = 0
        p.changeVisualShape(self.box_id, -1, rgbaColor=[1, 0.45, 0, 1],
                            physicsClientId=self.client_id)
        self.recording = True
        try:
            self.capture_frame()
            record = super().run_trial(pos, ori)
            self.capture_frame()
        finally:
            self.recording = False
        reason = record["failure_reason"] or "stable_hold"
        last = self.frames[-1].copy()
        draw = ImageDraw.Draw(last)
        draw.rectangle([0, 350, 640, 400], fill="white")
        draw.text((10, 365), f"success={record['success']} | {reason}", fill="black")
        self.frames.append(last)
        durations = [100] * len(self.frames)
        durations[-1] = 1500
        path = self.output_dir / f"trial_{self.trial_number:02d}.gif"
        self.frames[0].save(path, save_all=True, append_images=self.frames[1:],
                            duration=durations, loop=0)
        print(f"Recorded {path}", flush=True)
        return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "recorded_seed42")
    args = parser.parse_args()
    start = time.perf_counter()
    hand = RecordedHand(args.output, args.seed)
    hand.run_demo(args.samples, args.output)
    print(f"Recording finished in {time.perf_counter() - start:.2f} wall-clock seconds.")
