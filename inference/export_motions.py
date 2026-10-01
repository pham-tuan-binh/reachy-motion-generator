"""Every dataset row -> one reachable 25 Hz trajectory (for visualisation datasets such as LeRobot).

  python -m inference.export_motions --dataset distill_data/dataset.jsonl --out build/motions.pkl [--steps 100]
Uses the serving settings (2 Hz / 0.25 s recipe expansion, CFG 1.5), batched on the GPU.
"""
import argparse
import json
import pickle
import time

import numpy as np

from common.motion import to_move, traj
from common.reach import Reach
from generator.sample import generate_batch, load
from planner.dsl import variants


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="distill_data/dataset.jsonl"); ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="checkpoints/generator.pt"); ap.add_argument("--steps", type=int, default=100); ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.dataset)]
    net, stats, dev = load(a.ckpt, "cuda"); R = Reach(); out = []; t0 = time.time()
    rows.sort(key=lambda r: len(r["recipe"]))                    # similar lengths per batch = less padding
    for i in range(0, len(rows), a.batch):
        B = rows[i:i + a.batch]
        plans = [variants(r["recipe"], 1, seed=0, fc=2.0, kdt=0.25)[0] for r in B]
        for r, A in zip(B, generate_batch(net, stats, plans, dev, seeds=[0] * len(B), steps=a.steps)):
            m, frac = R.project(to_move(A, r["prompt"]))
            out.append(dict(r, traj=traj(m).astype(np.float32), projected=round(frac, 4)))
        if (i // a.batch) % 20 == 0: print(f"  {len(out)}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    out.sort(key=lambda r: r["id"])
    pickle.dump(out, open(a.out, "wb"))
    print(f"{len(out)} trajectories, {sum(len(r['traj']) for r in out)} frames -> {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
