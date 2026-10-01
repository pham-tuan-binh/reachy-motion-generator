"""Probe suite: out-of-distribution prompts with automatic PHYSICAL checks on the expanded plan.

Loss and averaged plan metrics did not catch the failure that mattered ("sneezing" snapping the head up).
Each probe asks one concrete question about the motion ("does the release move the head down?"), each
prompt is sampled several times, and the score is the pass rate. sft.py leak-filters every probe prompt (and the
OOD-core concepts, by keyword) out of training.
"""
import numpy as np

from planner.dsl import expand

EAR, P, R, Y, Z, B, E = 0, 2, 3, 4, 5, 6, 7


def _F(recipe):
    F = expand(recipe, np.random.default_rng(0)); F[:, 0] = F[:, :2].mean(1); return F


def _osc_cycles(x, amp):
    """count half-cycles of an oscillation of at least `amp` around its running mean."""
    k = max(3, len(x) // 10); m = np.convolve(x, np.ones(k) / k, "same"); d = x - m
    s = np.sign(np.where(np.abs(d) > amp / 2, d, 0)); s = s[s != 0]
    return int((np.diff(s) != 0).sum()) if len(s) else 0


def sneeze(F):          # release (energy peak) moves the head DOWN relative to just before it
    i = int(np.argmax(F[:, E])); pre = F[max(0, i - 10), P]
    return F[max(0, i - 2):i + 8, P].max() - pre > 5


def startle(F):         # a fast event that moves the head up/back or taller, ears up
    i = int(np.argmax(F[:, E])); w = slice(i, i + 13)
    return (F[w, Z].max() - F[max(0, i - 3), Z] > 4 or F[w, P].min() - F[max(0, i - 3), P] < -4) and F[w, EAR].min() < 20


PROBES = [
    ("sneezing. You build up and then release a sudden sharp sneeze.", sneeze, "release moves head DOWN"),
    ("a big sneeze is coming. Ah... ah... choo!", sneeze, "release moves head DOWN"),
    ("startled. A sudden loud noise just made you jump.", startle, "jolt up/back, ears up"),
    ("bowing deeply. You thank the audience for coming.", lambda F: F[:, P].max() >= 15 and F[-1, P] <= 8, "pitch >= 15 then back up"),
    ("nodding yes. You agree with everything enthusiastically.", lambda F: _osc_cycles(F[:, P], 5) >= 4, ">= 2 pitch oscillations"),
    ("shaking your head no. You refuse firmly.", lambda F: _osc_cycles(F[:, Y], 8) >= 4, ">= 2 yaw oscillations"),
    ("looking up at the stars. The night sky is beautiful.", lambda F: (F[:, P] < -8).mean() >= 0.4, "head up >= 40% of the time"),
    ("cowering in fear. Something huge looms over you.", lambda F: F[:, Z].min() <= -8 and F[:, EAR].max() >= 100, "sinks low, ears drooped"),
    ("heartbroken. You have just received news that devastated you.", lambda F: ((F[:, EAR] >= 110) & (F[:, P] >= 10)).mean() >= 0.25, "ears down + head down, sustained"),
    ("ecstatic. You are overjoyed and can barely contain yourself.", lambda F: F[:, Z].max() >= 8 and F[:, EAR].min() <= 10 and F[:, E].max() >= 5, "tall, ears up, high energy"),
    ("sleepy toddler. You are fighting to stay awake and keep nodding off.",
     lambda F: F[:, EAR].max() >= 100 and any(F[i, P] >= 12 and F[i:, P].min() <= F[i, P] - 8 for i in range(0, len(F), 5)),
     "droops AND recovers at least once"),
    ("drunk. You are unsteady and your movements are loose and uncoordinated.", lambda F: np.ptp(F[:, R]) >= 15 and len(F) / 25 >= 5, "big roll wobble, >= 5 s"),
    ("a cat stalking prey. You crouch low, freeze, and creep forward slowly.",
     lambda F: F[:, Z].min() <= -6 and F[:, EAR].min() <= 20 and (F[:, E] <= 0.6).sum() >= 25, "low, ears up, >= 1 s still"),
    ("jumping for joy. You just got the best news ever.", lambda F: np.ptp(F[:, Z]) >= 14 and F[:, E].max() >= 5, "big height change, high energy"),
    ("yawning widely. You are so sleepy.", lambda F: F[:, P].min() <= -10 and (F[np.argmin(F[:, P]):, P].max() >= 4 or F[np.argmin(F[:, P]):, EAR].max() >= 80), "head back, then droop"),
    ("checking both ways. You look left and right before crossing.", lambda F: F[:, Y].max() >= 15 and F[:, Y].min() <= -15, "looks both ways"),
]


# probes whose concept appears in NO training data (eval prompts + blocklisted concepts): the generalisation score.
# The rest (nod, bow, look up, cower, jump, yawn, look both ways) have close relatives in training: skill checks.
OOD_CORE = {"sneezing", "a big sneeze is coming", "startled", "heartbroken", "ecstatic", "sleepy toddler", "drunk", "a cat stalking prey"}


def split(per):
    core = [v for p, v in per.items() if p.split(".")[0] in OOD_CORE]; skill = [v for p, v in per.items() if p.split(".")[0] not in OOD_CORE]
    return float(np.mean(core)), float(np.mean(skill))


def score(recipes_per_prompt):
    """{prompt: [recipe or None, ...]} -> (overall pass rate, {prompt: pass rate})."""
    per = {}
    for prompt, fn, _ in PROBES:
        rs = recipes_per_prompt.get(prompt, [])
        ok = []
        for r in rs:
            try: ok.append(bool(fn(_F(r))) if r else False)
            except Exception: ok.append(False)
        per[prompt] = float(np.mean(ok)) if ok else 0.0
    return float(np.mean(list(per.values()))), per
