"""Primary instruction + settled pose -> native trajectory pipeline (no robot I/O)."""
from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from common.plan import frames
from common.provenance import digest, write_json
from generator.conditioned import load_generator, origins_array, sample_batch
from generator.plans import recipe_to_plan
from inference.defaults import default_config, generator_checkpoint
from planner.state import state_messages


@dataclass
class MotionResult:
    """Unanchored trajectory and provenance. The caller owns guarded playback."""
    trajectory: np.ndarray
    receipt: dict

    def save(self, out):
        out = Path(out)
        receipt_path = out.with_suffix(".json")
        if out.suffix != ".npz":
            raise ValueError("Use an .npz output filename")
        if out.exists() or receipt_path.exists():
            raise FileExistsError("Preserve existing samples; choose a fresh output")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("xb") as stream:
            np.savez_compressed(stream, trajectory=self.trajectory,
                                origin=origins_array([self.receipt["origin"]])[0])
        write_json(receipt_path, {**self.receipt, "trajectory_sha256": digest(out)})
        return receipt_path


class MotionPipeline:
    """Retained 4B RL planner + nominal expansion + v4 residual-500.

    Models are loaded lazily once. Each call requires a complete settled pose.
    Outputs still require application-owned anchoring and safety admission.
    """
    def __init__(self, device="auto"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self._planner = self._generator = None
        self._lock = threading.Lock()

    @property
    def config(self):
        """A copy of the pinned configuration; modifying it cannot change loading."""
        return default_config()

    def generate(self, prompt, *, current_pose, seed=123):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty instruction is required")
        state_messages(prompt, current_pose)
        origin = dict(current_pose)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        with self._lock:
            if self._planner is None:
                from planner.retained import RetainedPlanner
                self._planner = RetainedPlanner(self.device)
            if self._generator is None:
                self._generator = load_generator(generator_checkpoint(), self.device)
            idea, recipe, completion = self._planner.plan(prompt, origin)
            net, stats, metadata = self._generator
            settings = self.config["generator_settings"]
            plan = recipe_to_plan(recipe, seed, settings["mode"])
            if len(frames(plan)) > net.maxlen:
                raise ValueError("Recipe exceeds the checkpoint's maximum duration")
            trajectory = sample_batch(net, stats, [plan], [origin], [seed],
                steps=settings["steps"], cfg=settings["cfg"], lowpass_hz=settings["lowpass_hz"])[0]
        receipt = dict(version="state-aware-motion-sample-v1", pipeline=default_config(),
            prompt=prompt, idea=idea, recipe=recipe, completion=completion, origin=origin, seed=seed,
            metadata=metadata, shape=list(trajectory.shape), **self.config["output"],
            physical_robot_tested=False)
        return MotionResult(trajectory, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    pose = parser.add_mutually_exclusive_group(required=True)
    pose.add_argument("--origin", help="JSON object with all nine named settled-pose fields")
    pose.add_argument("--origin-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    origin = json.loads(args.origin if args.origin is not None else args.origin_file.read_text())
    if args.out.suffix != ".npz":
        parser.error("--out must be an .npz filename")
    if args.out.exists() or args.out.with_suffix(".json").exists():
        parser.error("output already exists; choose a fresh filename")
    result = MotionPipeline(args.device).generate(args.prompt, current_pose=origin, seed=args.seed)
    receipt = result.save(args.out)
    print(json.dumps({"pipeline": result.receipt["pipeline"]["id"], "shape": list(result.trajectory.shape),
                         "anchored_to_origin": False, "receipt": str(receipt)}, indent=2))


if __name__ == "__main__":
    main()
