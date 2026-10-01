"""Motion recipes: a tiny keyframe language the LLM writes, expanded into randomised plans.

A recipe is a sequence of segments separated by ``|``::

    go D k=v ...           cosine-ease to the target over D seconds
    hold D [E=v]           stay in the current pose (optionally changing energy)
    osc D ch amp per [k=v] sinusoid on channel ``ch`` (amplitude ``amp``, period ``per`` s) around the pose

Channels: ``e`` (both ears) ``eR`` ``eL`` ``p`` (pitch) ``r`` (roll) ``y`` (yaw) ``z`` (height)
``b`` (body yaw); ``E`` is the energy (RMS of fast detail, deg). Units are those of ``common.plan``.
Every recipe starts from ``NEUTRAL``.

Example (sobbing)::

    go 1 e=150 p=22 z=-16 E=5 | osc 3 z 4 .9 E=6 | hold 1 E=4
"""
import numpy as np

from common.motion import FPS
from common.plan import CH, KDT, lowpass

KEYMAP = {"eR": 0, "eL": 1, "p": 2, "r": 3, "y": 4, "z": 5, "b": 6, "E": 7}
NEUTRAL = np.array([15., 15., 0., 0., 0., 3., 0., 0.5])
LIMITS = {"e": (-25, 175), "eR": (-25, 175), "eL": (-25, 175), "p": (-30, 30), "r": (-25, 25), "y": (-50, 50),
          "z": (-25, 25), "b": (-60, 60), "E": (0, 12)}


class RecipeError(ValueError):
    pass


def _kv(toks, tgt, amp):
    for tk in toks:
        if "=" not in tk:
            raise RecipeError(f"expected key=value, got {tk!r}")
        k, v = tk.split("=", 1)
        if k not in LIMITS:
            raise RecipeError(f"unknown channel {k!r} (use e eR eL p r y z b E)")
        try:
            v = float(v)
        except ValueError:
            raise RecipeError(f"bad number in {tk!r}")
        lo, hi = LIMITS[k]
        if not lo <= v <= hi:
            raise RecipeError(f"{k}={v} outside [{lo}, {hi}]")
        if k in ("p", "r", "y", "z", "b"):
            v *= amp
        if k == "e": tgt[0] = tgt[1] = v
        else: tgt[KEYMAP[k]] = v


def expand(recipe, rng=None, amp=1.0, tsc=1.0):
    """Recipe -> (T, 8) per-frame [earR earL pitch roll yaw z body energy] (unfiltered).
    ``amp`` scales the head/body targets, ``tsc`` the tempo; ``rng`` adds ±15% per-segment timing jitter."""
    rng = rng or np.random.default_rng(0)
    cur = NEUTRAL.copy(); out = [cur.copy()]
    segs = [s.split() for s in recipe.split("|") if s.strip()]
    if not segs:
        raise RecipeError("empty recipe")
    for tok in segs:
        cmd = tok[0]
        if cmd not in ("go", "hold", "osc") or len(tok) < 2:
            raise RecipeError(f"bad segment {' '.join(tok)!r}")
        try:
            d = float(tok[1])
        except ValueError:
            raise RecipeError(f"bad duration in {' '.join(tok)!r}")
        if not 0.05 <= d <= 10:
            raise RecipeError(f"duration {d} outside [0.05, 10] s")
        d *= tsc * rng.uniform(.85, 1.15); n = max(1, int(round(d * FPS)))
        if cmd in ("go", "hold"):
            tgt = cur.copy(); _kv(tok[2:], tgt, amp)
            if cmd == "hold": tgt[:7] = cur[:7]
            w = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)
            seg = cur + w[:, None] * (tgt - cur)
        else:
            if len(tok) < 5:
                raise RecipeError(f"osc needs: osc D ch amp period, got {' '.join(tok)!r}")
            ch = tok[2]
            if ch not in ("e", "eR", "eL", "p", "r", "y", "z", "b"):
                raise RecipeError(f"osc on unknown channel {ch!r}")
            if float(tok[4]) < 0.3:     # validate what was written, before jitter
                raise RecipeError("osc period below 0.3 s: fast shaking belongs in E (energy), not osc")
            a, per = float(tok[3]) * amp * rng.uniform(.8, 1.2), float(tok[4]) * rng.uniform(.85, 1.15)
            tgt = cur.copy(); _kv(tok[5:], tgt, amp)
            u = np.arange(1, n + 1) / FPS
            s = np.sin(2 * np.pi * u / per) * np.minimum(1, np.minimum(u, u[-1] - u + 1 / FPS) / .3)
            seg = np.repeat(cur[None], n, 0); seg[:, 7] = np.linspace(cur[7], tgt[7], n)
            for j in ([0, 1] if ch == "e" else [KEYMAP[ch]]):
                seg[:, j] += a * s
        out += list(seg); cur = seg[-1].copy()
    F = np.array(out)
    if len(F) / FPS > 30:
        raise RecipeError(f"recipe lasts {len(F)/FPS:.1f} s; keep it under 30 s")
    return F


def to_plan(F, rng=None, mirror=False, ear_jitter=6.0, fc=1.0, kdt=KDT):
    """Frames -> plan: posture low-passed at ``fc`` Hz, keys every ``kdt`` s. The default (1 Hz, 0.5 s) matches
    ``common.plan.extract``, i.e. the generator's training plans, but it smears a 0.12 s snap into a ~0.7 s ramp."""
    rng = rng or np.random.default_rng(0); F = F.copy()
    F[:, 0] += rng.normal(0, ear_jitter); F[:, 1] += rng.normal(0, ear_jitter)   # real clips are never symmetric
    if mirror:
        F[:, [3, 4, 6]] *= -1; F[:, [0, 1]] = F[:, [1, 0]]
    S = lowpass(F[:, :7], fc) if fc else F[:, :7]; T = len(F); keys = []
    for t in np.arange(0, T / FPS + 1e-9, kdt):
        i = min(T - 1, int(round(t * FPS)))
        keys.append(dict(t=round(float(t), 2), **{c: round(float(S[i, k]), 1) for k, c in enumerate(CH[:7])},
                         energy=round(float(max(0, F[i, 7])), 1)))
    return dict(duration=round(T / FPS, 2), keys=keys)


def variants(recipe, n, seed=0, fc=1.0, kdt=KDT):
    """n randomised plans from one recipe: amplitude x0.75-1.25, tempo x0.8-1.25, odd variants mirrored."""
    rng = np.random.default_rng(seed); out = []
    for v in range(n):
        F = expand(recipe, rng, amp=rng.uniform(.75, 1.25), tsc=rng.uniform(.8, 1.25))
        out.append(to_plan(F, rng, mirror=bool(v % 2), fc=fc, kdt=kdt))
    return out


def check(recipe):
    """None if the recipe is valid, else the error message."""
    try:
        expand(recipe); return None
    except (RecipeError, ValueError, IndexError) as e:
        return str(e)
