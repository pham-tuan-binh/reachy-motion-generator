"""Plans -> reachable Reachy Mini moves."""
import json
import os

import numpy as np
import torch
from scipy.signal import butter, filtfilt

from common import plan as PL
from common.motion import FPS, to_move
from common.reach import Reach
from generator.model import MotionGenerator, device

CKPT = "checkpoints/generator.pt"
HUB = "binhpham/reachy-mini-motion-planner-4b"     # every published bundle carries the same generator.pt


def load(ckpt=CKPT, dev=None):
    """Checkpoint -> (net, stats, device). Falls back to the Hugging Face copy when ``ckpt`` does not exist locally."""
    if not os.path.exists(ckpt):
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(HUB, "generator.pt")
    dev = dev or device(); ck = torch.load(ckpt, map_location=dev, weights_only=False)
    net = MotionGenerator().to(dev); net.load_state_dict(ck["sd"]); net.eval()
    return net, ck["stats"], dev


@torch.no_grad()
def generate_batch(net, stats, plans, dev, seeds=None, steps=8, cfg=1.5, lowpass_hz=4.0):
    """Plans -> (T_i, 9) trajectories. Euler integration of the flow from noise (t=1) to data (t=0) with
    classifier-free guidance ``cfg`` on the plan, then a 4 Hz low-pass (the robot's useful bandwidth).
    All plans (padded to the longest, masked) and both guidance branches share one forward per step.
    8 steps are 1.3 deg RMS from 100 steps, well below the 6 deg between two noise seeds."""
    MU, SD, PMU, PSD = (np.array(stats[k]) for k in ("MU", "SD", "PMU", "PSD"))
    Ps = [PL.frames(pl)[:net.maxlen] for pl in plans]; Ts = [len(P) for P in Ps]; B, T = len(Ps), max(Ts)
    Q = torch.zeros(B, T, 8, device=dev); pad = torch.ones(B, T, dtype=torch.bool, device=dev)
    x = torch.zeros(B, T, 9, device=dev)
    for i, P in enumerate(Ps):
        Q[i, :Ts[i]] = torch.tensor((P - PMU) / PSD, dtype=torch.float32, device=dev); pad[i, :Ts[i]] = False
        g = torch.Generator(device="cpu").manual_seed(int(seeds[i]) if seeds is not None else i)
        x[i, :Ts[i]] = torch.randn(Ts[i], 9, generator=g).to(dev)
    guided = cfg != 1.0
    has = torch.ones(B, 1, 1, device=dev)
    if guided:
        has, Q, pad = torch.cat([has, torch.zeros_like(has)]), torch.cat([Q, Q]), torch.cat([pad, pad])
    for k in range(steps):
        t = torch.full((len(has),), 1.0 - k / steps, device=dev)
        v = net(torch.cat([x, x]) if guided else x, t, Q, has, pad)
        if guided:
            vc, vu = v[:B], v[B:]; v = vu + cfg * (vc - vu)
        x = x - v / steps
    out = []
    b, a = butter(4, lowpass_hz / (FPS / 2))
    for i in range(B):
        A = x[i, :Ts[i]].cpu().numpy() * SD + MU
        if Ts[i] > 20 and lowpass_hz: A = filtfilt(b, a, A, axis=0, padlen=min(Ts[i] - 1, 15))
        out.append(A)
    return out


def generate(net, stats, plan, dev, seed=0, **kw):
    """One plan -> (T, 9) trajectory."""
    return generate_batch(net, stats, [plan], dev, seeds=[seed], **kw)[0]


def sample_plans(plans, ckpt, outdir, seeds=1, cfg=1.5, steps=8):
    net, stats, dev = load(ckpt); R = Reach(); os.makedirs(outdir, exist_ok=True); n = 0
    for s in range(seeds):
        for i, pl in enumerate(plans):
            A = generate(net, stats, pl, dev, seed=s * 100003 + i, steps=steps, cfg=cfg)
            m, frac = R.project(to_move(A, pl.get("prompt", pl["name"])))
            name = pl["name"] + (f"__seed{s}" if seeds > 1 else "")
            json.dump(m, open(os.path.join(outdir, name + ".json"), "w")); n += 1
            print(f"  {name:48s} {len(A)/FPS:5.1f}s  projected {100*frac:4.1f}% of frames", flush=True)
    return n


def read_plans(path):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path) if l.strip()]
    d = json.load(open(path)); return d if isinstance(d, list) else [d]
