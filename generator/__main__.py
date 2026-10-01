"""Generator CLI.

  python -m generator train   --out checkpoints/generator.pt [--steps 5000] [--no-held-out]
  python -m generator sample  --plans plans.jsonl --out motions/ [--seeds 2 --cfg 1.5]
  python -m generator extract --moves some_moves/ --out plans.jsonl     # plans of existing motion
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser(prog="generator", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--out", default="checkpoints/generator.pt")
    t.add_argument("--steps", type=int, default=5000); t.add_argument("--bs", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4); t.add_argument("--no-dances", action="store_true")
    t.add_argument("--no-held-out", action="store_true", help="train on every clip (for a final model)")
    s = sub.add_parser("sample"); s.add_argument("--plans", required=True); s.add_argument("--out", required=True)
    s.add_argument("--ckpt", default="checkpoints/generator.pt"); s.add_argument("--seeds", type=int, default=1)
    s.add_argument("--cfg", type=float, default=1.5); s.add_argument("--steps", type=int, default=8)
    e = sub.add_parser("extract"); e.add_argument("--moves", required=True); e.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        from generator.train import train
        from common.data import HELD_OUT
        train(out=a.out, steps=a.steps, bs=a.bs, lr=a.lr, dances=not a.no_dances, held_out=[] if a.no_held_out else HELD_OUT)
    elif a.cmd == "sample":
        from generator.sample import read_plans, sample_plans
        n = sample_plans(read_plans(a.plans), a.ckpt, a.out, seeds=a.seeds, cfg=a.cfg, steps=a.steps)
        print(f"{n} motions -> {a.out}")
    else:
        from common import plan as PL
        from common.motion import load_move, traj
        with open(a.out, "w") as fh:
            for p in sorted(glob.glob(os.path.join(a.moves, "*.json"))):
                m = load_move(p)
                fh.write(json.dumps(dict(name=os.path.basename(p)[:-5], prompt=m.get("description", ""), **PL.extract(traj(m)))) + "\n")
        print(f"plans -> {a.out}")


if __name__ == "__main__":
    main()
