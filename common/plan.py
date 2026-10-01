"""Motion plan: the interface between the planner (text -> plan) and the generator (plan -> motion).

One keyframe every ``KDT`` = 0.5 s with 8 channels:

=========  ==================================================================
earR/earL  antenna droop in degrees (0 = straight up, ~150 = fully drooped)
pitch      head pitch, deg (+ = head lowered)
roll       head roll, deg
yaw        head yaw, deg
z          head height, mm (+ = up)
body       body yaw, deg
energy     RMS (deg) of the fast (> 1 Hz) detail riding on top of the posture
=========  ==================================================================

The posture channels are low-passed below 1 Hz, so a plan says *where the body is and how lively it
is*, not the individual wiggles. Any motion clip can be turned into its plan (``extract``), which is
what lets the generator train on motion alone.
"""
import numpy as np
from scipy.signal import butter, filtfilt

from .motion import FPS

KDT = 0.5
CH = ["earR", "earL", "pitch", "roll", "yaw", "z", "body", "energy"]


def lowpass(x, fc, fps=FPS, order=2):
    if len(x) < 16:
        return x.copy()
    b, a = butter(order, fc / (fps / 2))
    return filtfilt(b, a, x, axis=0, padlen=min(len(x) - 1, 9))


def posture(A):
    """(T, 9) trajectory -> (T, 7) posture channels in plan units."""
    d = np.degrees
    return np.stack([-d(A[:, 6]), d(A[:, 7]), d(A[:, 4]), d(A[:, 3]), d(A[:, 5]), 1000 * A[:, 2], d(A[:, 8])], -1)


def extract(A, fc=1.0, kdt=KDT):
    """(T, 9) trajectory -> plan dict ``{"duration", "keys": [{t, earR, ..., energy}]}``."""
    P = posture(A); slow = lowpass(P, fc)
    fast = P[:, :5] - slow[:, :5]
    T = len(A); dur = T / FPS; keys = []
    w = max(KDT, kdt)            # energy window stays >= 0.5 s so it keeps meaning "fast detail around here"
    for t in np.arange(0, dur + 1e-9, kdt):
        i = min(T - 1, int(round(t * FPS)))
        lo, hi = max(0, int((t - w / 2) * FPS)), min(T, int((t + w / 2) * FPS) + 1)
        e = float(np.sqrt((fast[lo:hi] ** 2).mean())) if hi > lo else 0.0
        keys.append(dict(t=round(float(t), 2), **{c: round(float(slow[i, k]), 1) for k, c in enumerate(CH[:7])},
                         energy=round(e, 1)))
    return dict(duration=round(dur, 2), keys=keys)


def frames(plan, T=None):
    """Plan -> (T, 8) per-frame conditioning, linearly interpolated between keyframes.
    Channels missing from a key hold the previous key's value (0 before the first)."""
    T = T or max(2, int(round(plan["duration"] * FPS)))
    last = {c: 0.0 for c in CH}; rows = []
    for k in sorted(plan["keys"], key=lambda k: k["t"]):
        for c in CH:
            if c in k:
                last[c] = float(k[c])
        rows.append([k["t"]] + [last[c] for c in CH])
    R = np.array(rows); u = np.arange(T) / FPS
    return np.stack([np.interp(u, R[:, 0], R[:, 1 + j]) for j in range(len(CH))], -1)
