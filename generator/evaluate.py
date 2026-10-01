"""Generator check on the 12 held-out real clips, from their TRUE plans:

  identification  is each generated motion closest to its own real clip among the 12? (chance 8.3%, mean rank 6.5)
  speed           95th-percentile and peak head-pitch / ear speeds, generated vs real (too slow = sluggish,
                  too fast = jittery)

  python -m generator.evaluate [--ckpt checkpoints/generator.pt] [--seeds 3]
"""
import argparse

import numpy as np

from common import plan as PL
from common.data import EMOTIONS, HELD_OUT, caption, library
from common.motion import FPS, traj

NT = 64


class HeldOut:
    """The 12 held-out emotions. ``rank(A, name)``: where clip ``name`` ranks among the 12 by its distance to the
    trajectory A (time-shift-tolerant RMS over resampled, z-scored 9-DoF trajectories); 1 = identified."""
    def __init__(self):
        lib = dict(library(EMOTIONS))
        allA = np.concatenate([traj(m) for m in lib.values()]); self.MU, self.SD = allA.mean(0), allA.std(0) + 1e-6
        self.real = {h: traj(lib[h]) for h in HELD_OUT}
        self.prompts = {h: caption(h, lib[h]) for h in HELD_OUT}
        self.feats = {h: self.feat(A) for h, A in self.real.items()}

    def feat(self, A):
        u = np.linspace(0, 1, NT); v = np.linspace(0, 1, len(A))
        return (np.stack([np.interp(u, v, A[:, j]) for j in range(9)], -1) - self.MU) / self.SD

    @staticmethod
    def dist(a, b, shift=4):
        return min(float(np.sqrt(((a[max(0, s):NT + min(0, s)] - b[max(0, -s):NT - max(0, s)]) ** 2).mean()))
                   for s in range(-shift, shift + 1))

    def rank(self, A, name):
        f = self.feat(A); d = {k: self.dist(f, v) for k, v in self.feats.items()}
        return sorted(d, key=d.get).index(name) + 1


def speeds(A):
    pitch = np.abs(np.diff(np.degrees(A[:, 4]))) * FPS; ear = np.abs(np.diff(np.degrees(A[:, 6:8]), axis=0)).max(1) * FPS
    return np.percentile(pitch, 95), pitch.max(), np.percentile(ear, 95), ear.max()


def main():
    from generator.sample import CKPT, generate, load
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=CKPT); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100)
    a = ap.parse_args()
    net, stats, dev = load(a.ckpt); H = HeldOut(); ranks, sg, sr = [], [], []
    for h, A in H.real.items():
        pl = PL.extract(A); sr.append(speeds(A))
        for s in range(a.seeds):
            G = generate(net, stats, pl, dev, seed=s, steps=a.steps); sg.append(speeds(G)); ranks.append(H.rank(G, h))
    r = np.array(ranks); g, q = np.mean(sg, 0), np.mean(sr, 0)
    print(f"  identification from true plans: top-1 {100*(r==1).mean():.0f}%  mean rank {r.mean():.2f}  (n={len(r)}, chance 8.3%)")
    print(f"  head pitch speed p95 / peak: generated {g[0]:.0f} / {g[1]:.0f} deg/s   real {q[0]:.0f} / {q[1]:.0f}")
    print(f"  ear speed        p95 / peak: generated {g[2]:.0f} / {g[3]:.0f} deg/s   real {q[2]:.0f} / {q[3]:.0f}")


if __name__ == "__main__":
    main()
