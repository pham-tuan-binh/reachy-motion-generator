"""Figures for the technical report, computed from the real recipe code, plan code and generator.

  python docs/make_figures.py            # -> docs/figures/*.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import plan as PL
from common.motion import FPS
from generator.sample import load
from planner.dsl import expand, to_plan

OUT = os.path.join(os.path.dirname(__file__), "figures"); os.makedirs(OUT, exist_ok=True)
RECIPE = ("go .5 e=20 E=.5 | go 1 p=-10 z=8 e=30 E=2 | go .15 p=18 z=-8 e=90 E=8 | go .3 p=6 z=0 E=3 | "
          "go .15 p=18 z=-8 e=90 E=8 | go .8 p=2 z=2 e=30 E=1.5 | hold .6")          # "a big cough" (teacher data)
PROMPT = "a big cough. Something tickles your throat."
C = dict(ear="#d62728", pitch="#1f77b4", z="#2ca02c", energy="#9467bd", roll="#ff7f0e", yaw="#8c564b")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140, bbox_inches="tight"); plt.close(fig); print("  ", name)


# ---------------------------------------------------------------- 1. pipeline overview
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.4)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 3.4)
    boxes = [(0.1, "text prompt", '"a big cough.\nSomething tickles\nyour throat."', "#eeeeee"),
             (2.35, "planner (LLM)", "writes a recipe:\ngo 1 p=-10 z=8 |\ngo .15 p=18 z=-8 E=8 | ...", "#fde2c8"),
             (4.6, "expand", "recipe -> plan\n8 channels,\n1 key / 0.5 s", "#fde2c8"),
             (6.85, "generator", "flow-matching\ntransformer\nnoise -> motion", "#cfe3f7"),
             (9.1, "reachability", "IK check, pull\nunreachable poses\nback (SDK IK)", "#d6f0d6"),
             (11.35, "move file", "9 DoF @ 25 Hz\nplay on the robot\nor in MuJoCo", "#eeeeee")]
    for x, title, body, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.35), 1.6, 2.5, boxstyle="round,pad=0.05", fc=col, ec="#555"))
        ax.text(x + 0.8, 2.55, title, ha="center", va="center", weight="bold")
        ax.text(x + 0.8, 1.45, body, ha="center", va="center", fontsize=8.5, family="monospace" if "go" in body else None)
    for x, _, _, _ in boxes[:-1]:
        ax.add_patch(FancyArrowPatch((x + 1.72, 1.6), (x + 2.2, 1.6), arrowstyle="-|>", mutation_scale=16, color="#333"))
    ax.text(3.9, 0.02, "knows WHAT to express\n(world knowledge)", ha="center", va="top", color="#b35900", fontsize=9)
    ax.text(7.65, 0.02, "knows HOW Reachy moves\n(9 min of real motion, never sees text)", ha="center", va="top", color="#1f5c99", fontsize=9)
    ax.set_ylim(-0.6, 3.4)
    save(fig, "1_pipeline.png")


# ---------------------------------------------------------------- 2. recipe -> plan
def fig_recipe_to_plan():
    F = expand(RECIPE, np.random.default_rng(0)); plan = to_plan(F, np.random.default_rng(0), ear_jitter=0)
    t = np.arange(len(F)) / FPS; P = PL.frames(plan, len(F)); kt = [k["t"] for k in plan["keys"]]
    segs = [s.strip() for s in RECIPE.split("|")]; bounds = [0.0]
    for s in segs: bounds.append(bounds[-1] + float(s.split()[1]))
    fig, axs = plt.subplots(4, 1, figsize=(12, 7.5), sharex=True)
    rows = [("ear droop (deg)\n0 = up", 0, "ear", "earR"), ("head pitch (deg)\n+ = down", 2, "pitch", "pitch"),
            ("head height (mm)", 5, "z", "z"), ("energy\n(fast detail)", 7, "energy", "energy")]
    for ax, (lab, j, c, key) in zip(axs, rows):
        for i in range(len(segs)):
            ax.axvspan(bounds[i], bounds[i + 1], color=["#f4f4f4", "#e8e8e8"][i % 2], zorder=0)
        ax.plot(t, F[:, j], color=c and C[c], lw=1, alpha=.45, label="recipe expanded (25 Hz)")
        ax.plot(t, P[:, j], color=C[c], lw=2.2, label="plan (1 Hz smoothed, interpolated)")
        ax.scatter(kt, [k[key] for k in plan["keys"]], color=C[c], s=28, zorder=5, label="plan keyframes (every 0.5 s)")
        ax.set_ylabel(lab, fontsize=9)
    for i, s in enumerate(segs):
        axs[0].text((bounds[i] + bounds[i + 1]) / 2, axs[0].get_ylim()[1], s.split()[0] + " " + s.split()[1], ha="center", va="bottom", fontsize=7.5, family="monospace")
    axs[2].annotate("build-up: head rises\nand tilts back", (1.3, 7), (2.6, 7.5), fontsize=8.5, arrowprops=dict(arrowstyle="->"))
    axs[1].annotate("release: two fast\ndownward snaps", (1.6, 12), (2.9, 14), fontsize=8.5, arrowprops=dict(arrowstyle="->"))
    axs[3].annotate("fast shaking lives in\nENERGY, not in the curves", (1.65, 7), (2.9, 6), fontsize=8.5, arrowprops=dict(arrowstyle="->"))
    axs[0].legend(loc="lower right", fontsize=8); axs[-1].set_xlabel("time (s)")
    fig.suptitle(f'Recipe -> plan   ("{PROMPT}")\n{RECIPE}', fontsize=9.5, family="monospace", y=1.02)
    save(fig, "2_recipe_to_plan.png")
    return plan


# ---------------------------------------------------------------- generator internals
class Tap:
    """Record attention weights of every block during a forward pass."""
    def __init__(self, net):
        self.maps = []
        for b in net.blocks:
            orig = b.att.forward
            def fwd(q, k, v, key_padding_mask=None, need_weights=False, _o=orig, **kw):
                out, w = _o(q, k, v, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=True)
                self.maps.append(w.detach().cpu().numpy()[0]); return out, None
            b.att.forward = fwd


@torch.no_grad()
def sample_with_snapshots(net, stats, plan, dev, seed=0, steps=100, cfg=1.5, snaps=(0, 10, 30, 60, 100), tap=None, tap_step=70):
    MU, SD, PMU, PSD = (np.array(stats[k]) for k in ("MU", "SD", "PMU", "PSD"))
    P = PL.frames(plan); T = len(P); Q = torch.tensor((P - PMU) / PSD, dtype=torch.float32, device=dev)[None]
    x = torch.randn(1, T, 9, generator=torch.Generator().manual_seed(seed)).to(dev); pad = torch.zeros(1, T, dtype=torch.bool, device=dev)
    one, zero = torch.ones(1, 1, 1, device=dev), torch.zeros(1, 1, 1, device=dev); out = {}; tokens = None
    for k in range(steps + 1):
        if k in snaps: out[k] = x[0].cpu().numpy() * SD + MU
        if k == steps: break
        t = torch.full((1,), 1.0 - k / steps, device=dev)
        if k == tap_step:
            tokens = (x[0].cpu().numpy(), Q[0].cpu().numpy())
            if tap: tap.maps.clear()
        v = net(x, t, Q, one, pad)
        if cfg != 1.0: v = net(x, t, Q, zero, pad) + cfg * (v - net(x, t, Q, zero, pad))
        if k == tap_step and tap: tap.keep = [m.copy() for m in tap.maps[:len(net.blocks)]]
        x = x - v / steps
    return out, tokens, P


def fig_tokens(tokens, P):
    xn, qn = tokens; T = len(xn); t = np.arange(T) / FPS
    M = np.concatenate([xn.T, qn.T, np.ones((1, T))], 0)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, extent=[0, T / FPS, 18, 0])
    names = ["x", "y", "z", "roll", "pitch", "yaw", "ant R", "ant L", "body"] + [f"plan {c}" for c in PL.CH] + ["has_plan"]
    ax.set_yticks(np.arange(18) + .5); ax.set_yticklabels(names, fontsize=8)
    ax.axhline(9, color="k", lw=2); ax.axhline(17, color="k", lw=2)
    ax.text(T / FPS + .05, 4.5, "noisy motion x_t\n(what the model\nis cleaning up)", va="center", fontsize=9)
    ax.text(T / FPS + .05, 13, "the plan at\nthis frame\n(the sketch)", va="center", fontsize=9)
    f0 = int(1.6 * FPS)
    ax.add_patch(plt.Rectangle((f0 / FPS, 0), 1 / FPS, 18, fill=False, ec="gold", lw=2.5))
    ax.annotate("ONE COLUMN = ONE TOKEN = one frame (1/25 s)\n18 numbers -> Linear -> 384-dim vector (+ position)",
                ((f0 + .5) / FPS, 18), (f0 / FPS + .35, 21.2), fontsize=9, arrowprops=dict(arrowstyle="->", color="goldenrod"), annotation_clip=False)
    ax.set_xlabel("time (s)", labelpad=36); ax.set_title(f"Generator input: {T} frames -> {T} tokens (continuous vectors, no codebook). Colour = z-scored value.", fontsize=10)
    save(fig, "3_tokens.png")


def fig_denoise(snaps, P):
    T = len(P); t = np.arange(T) / FPS; ks = sorted(snaps)
    fig, axs = plt.subplots(2, len(ks), figsize=(14, 4.6), sharex=True, sharey="row")
    for i, k in enumerate(ks):
        A = snaps[k]; tt = 1 - k / 100
        axs[0, i].plot(t, P[:, 2], color="#999", lw=2, ls="--", label="plan pitch")
        axs[0, i].plot(t, np.degrees(A[:, 4]), color=C["pitch"], lw=1.4, label="motion pitch")
        axs[1, i].plot(t, -np.degrees(A[:, 6]), color=C["ear"], lw=1.4, label="right ear droop")
        axs[1, i].plot(t, P[:, 0], color="#999", lw=2, ls="--", label="plan ear")
        axs[0, i].set_title(f"step {k}/100\nt = {tt:.2f}" + ("  (pure noise)" if k == 0 else "  (final)" if k == 100 else ""), fontsize=9)
        axs[1, i].set_xlabel("time (s)")
    axs[0, 0].set_ylabel("head pitch (deg)"); axs[1, 0].set_ylabel("ear droop (deg)")
    axs[0, 0].set_ylim(-45, 45); axs[1, 0].set_ylim(-120, 200)
    axs[0, -1].legend(fontsize=7.5, loc="lower right"); axs[1, -1].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Sampling: start from random noise, take 100 small steps along the model's predicted velocity; the plan pulls the noise into motion", fontsize=10)
    save(fig, "4_denoise.png")


def fig_attention(maps, P):
    T = len(P); ext = [0, T / FPS, T / FPS, 0]; b = min(3, len(maps) - 1)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    for ax, j in zip(axs[:2], [0, len(maps) - 1]):
        ax.imshow(maps[j], cmap="magma", extent=ext, vmax=np.percentile(maps[j], 99.5))
        ax.set_title(f"block {j + 1}: who attends to whom\n(row = query frame, column = key frame)", fontsize=9)
        ax.set_xlabel("attended frame (s)"); ax.set_ylabel("query frame (s)")
    snap = int(1.6 * FPS)
    for j, col in [(0, "#aaa"), (b, "#555"), (len(maps) - 1, "k")]:
        axs[2].plot(np.arange(T) / FPS, maps[j][snap], color=col, lw=1.6, label=f"block {j + 1}")
    ax2 = axs[2].twinx(); ax2.plot(np.arange(T) / FPS, P[:, 2], color=C["pitch"], ls="--", lw=1.5, alpha=.7); ax2.set_ylabel("plan pitch (deg)", color=C["pitch"])
    axs[2].axvline(snap / FPS, color="gold", lw=2); axs[2].set_title("where the frame at the first snap (1.6 s) looks", fontsize=9)
    axs[2].set_xlabel("time (s)"); axs[2].set_ylabel("attention weight"); axs[2].legend(fontsize=8)
    fig.suptitle("Self-attention over frame tokens: every frame can look at the whole clip (plan AND noisy motion). No cross-attention.", fontsize=10)
    save(fig, "5_attention.png")


@torch.no_grad()
def fig_guidance(net, stats, plan, dev):
    fig, ax = plt.subplots(figsize=(11, 3.6)); P = PL.frames(plan); t = np.arange(len(P)) / FPS
    ax.plot(t, P[:, 2], color="#999", lw=2.5, ls="--", label="plan pitch")
    for cfg, col in [(0.0, "#bbbbbb"), (1.0, "#7fb2e5"), (1.5, "#1f77b4"), (3.0, "#08306b")]:
        s, _, _ = sample_with_snapshots(net, stats, plan, dev, seed=3, cfg=cfg, snaps=(100,))
        ax.plot(t, np.degrees(s[100][:, 4]), color=col, lw=1.6, label=f"cfg {cfg}" + ("  (no plan: generic motion)" if cfg == 0 else "  (default)" if cfg == 1.5 else ""))
    ax.set_xlabel("time (s)"); ax.set_ylabel("head pitch (deg)"); ax.legend(fontsize=8, ncol=2)
    ax.set_title("Guidance: same noise, different strength of the plan. v = v_no_plan + cfg * (v_plan - v_no_plan)", fontsize=10)
    save(fig, "6_guidance.png")


@torch.no_grad()
def fig_duration(net, stats, dev):
    fig, axs = plt.subplots(3, 1, figsize=(11, 5.6), sharex=True)
    for ax, tsc in zip(axs, [0.8, 1.0, 1.25]):
        F = expand(RECIPE, np.random.default_rng(0), tsc=tsc); pl = to_plan(F, np.random.default_rng(0), ear_jitter=0)
        s, _, P = sample_with_snapshots(net, stats, pl, dev, seed=1, snaps=(0, 100)); t = np.arange(len(P)) / FPS
        ax.plot(t, P[:, 2], color="#999", ls="--", lw=2); ax.plot(t, np.degrees(s[100][:, 4]), color=C["pitch"], lw=1.5)
        ax.axvline(len(P) / FPS, color="k", lw=1)
        ax.text(len(P) / FPS + .05, 0, f"plan duration {pl['duration']:.2f} s\n-> noise of {len(P)} frames\n-> motion of {len(s[100])} frames", fontsize=8.5, va="center")
        ax.set_ylabel(f"tempo x{tsc}\npitch (deg)", fontsize=9)
    axs[-1].set_xlabel("time (s)")
    fig.suptitle("Duration is set by the plan: the noise tensor has T = duration x 25 frames and the model returns one frame per input frame", fontsize=10)
    save(fig, "7_duration.png")


def fig_sneeze():
    """Why the first students failed: the same prompt, teacher vs student recipes (head pitch)."""
    teacher_keys = None
    p = os.environ.get("TEACHER_OOD_PLANS")
    if p and os.path.exists(p):
        teacher_keys = [x for x in json.load(open(p)) if x["name"] == "sneezing"][0]
    students = {"student v1 (4B, no context)": "go .6 e=20 p=4 E=1 | hold .8 E=3 | go .15 e=-15 p=-12 z=12 E=9 | hold .3 E=4 | go .8 e=20 p=4 z=0 E=1",
                "student v2 (8B, no context)": "go .6 e=20 p=4 E=1 | osc 1.5 p 4 .5 E=2 | go .15 p=-10 z=10 e=100 E=9 | hold .6 E=3 | go 1 p=4 z=0 e=30 E=1"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, ch, lab in [(axs[0], 2, "head pitch (deg, + = down)"), (axs[1], 5, "head height (mm)")]:
        if teacher_keys:
            P = PL.frames(teacher_keys); ax.plot(np.arange(len(P)) / FPS, P[:, ch], color="k", lw=2.5, label="teacher")
        for (n, r), col in zip(students.items(), ["#e377c2", "#d62728"]):
            P = PL.frames(to_plan(expand(r, np.random.default_rng(0)), np.random.default_rng(0), ear_jitter=0))
            ax.plot(np.arange(len(P)) / FPS, P[:, ch], color=col, lw=1.8, label=n)
        ax.axhline(0, color="#ccc", lw=1); ax.set_xlabel("time (s)"); ax.set_title(lab, fontsize=10)
    axs[0].legend(fontsize=8)
    axs[0].annotate("teacher: 'ah... ah...' head tilts back,\nthen 'CHOO' snap DOWN", (1.95, 19), (2.25, 14), fontsize=8, arrowprops=dict(arrowstyle="->"))
    axs[0].annotate("students: head goes UP\n(reads as a startle)", (1.8, -8), (2.6, -15), fontsize=8, arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728")
    fig.subplots_adjust(top=0.82); fig.suptitle('"sneezing": the prompt no student saw. Posture right, direction of the release wrong.', fontsize=10)
    save(fig, "8_sneeze_failure.png")


if __name__ == "__main__":
    fig_pipeline(); plan = fig_recipe_to_plan()
    net, stats, dev = load(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "generator.pt"), "cpu")
    tap = Tap(net)
    snaps, tokens, P = sample_with_snapshots(net, stats, plan, dev, tap=tap)
    fig_tokens(tokens, P); fig_denoise(snaps, P); fig_attention(tap.keep, P)
    for b in net.blocks: b.att.forward = type(b.att).forward.__get__(b.att)       # untap
    fig_guidance(net, stats, plan, dev); fig_duration(net, stats, dev); fig_sneeze()
