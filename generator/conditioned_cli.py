"""Sample a recipe with explicit settled-pose conditioning; writes numerical NPZ.

This command does not anchor motion to the pose, project it through IK, apply
controller checks, or connect to a robot. The trajectory retains native units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.provenance import digest, write_json
from generator.conditioned import load_generator, origins_array, sample_batch
from generator.plans import recipe_to_plan


def sample_recipe(checkpoint, recipe, origin, out, *, seed=123, mode="nominal",
                  steps=8, cfg=1., lowpass_hz=4., device="cpu"):
    """Write one native (T,9) sample and a hash-bound settings receipt."""
    from common.plan import frames
    from planner.state import state_messages

    state_messages("", origin)  # Same finite numeric nine-field pose validation as the planner.
    state = origins_array([origin])
    out = Path(out)
    receipt = out.with_suffix(".json")
    if out.suffix != ".npz":
        raise ValueError("Use an .npz output filename")
    if out.exists() or receipt.exists():
        raise FileExistsError("Preserve existing numerical samples; choose a fresh output")
    net, stats, metadata = load_generator(checkpoint, device)
    plan = recipe_to_plan(recipe, seed, mode)
    if len(frames(plan)) > net.maxlen:
        raise ValueError("Recipe exceeds the checkpoint's maximum duration")
    trajectory = sample_batch(net, stats, [plan], [origin], [seed],
                              steps=steps, cfg=cfg, lowpass_hz=lowpass_hz)[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, trajectory=trajectory, origin=state[0])
    temporary.replace(out)
    result = dict(version="state-conditioned-recipe-sample-v1", checkpoint_sha256=digest(checkpoint),
        trajectory_sha256=digest(out), metadata=metadata, recipe=recipe, origin=origin, seed=seed,
        settings=dict(mode=mode, steps=steps, cfg=cfg, lowpass_hz=lowpass_hz),
        shape=list(trajectory.shape), fps=25, units="xyz metres; roll/pitch/yaw, signed antennas and body radians",
        anchored_to_origin=False, physical_robot_tested=False)
    write_json(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recipe", required=True)
    pose = parser.add_mutually_exclusive_group(required=True)
    pose.add_argument("--origin", help="JSON object with all nine named settled-pose fields")
    pose.add_argument("--origin-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mode", choices=("nominal", "randomized"), default="nominal")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=1.)
    parser.add_argument("--lowpass-hz", type=float, default=4.)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    origin = json.loads(args.origin if args.origin is not None else args.origin_file.read_text())
    result = sample_recipe(args.checkpoint, args.recipe, origin, args.out,
        seed=args.seed, mode=args.mode, steps=args.steps, cfg=args.cfg, lowpass_hz=args.lowpass_hz, device=args.device)
    print(json.dumps(dict(shape=result["shape"], checkpoint_sha256=result["checkpoint_sha256"],
                         anchored_to_origin=False, receipt=str(args.out.with_suffix(".json"))), indent=2))


if __name__ == "__main__":
    main()
