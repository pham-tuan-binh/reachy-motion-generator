"""Train the plan -> motion generator on Pollen's emotions + dances libraries.

Loss = masked flow-matching MSE + a velocity loss on the implied clean estimate
x0_hat = x_t - t * v (frame-to-frame differences, weighted by 1 - t). Without the velocity term the
model under-shoots fast motion (antennas and head were ~2x too slow).
Plan dropout (10%) trains the unconditional branch for classifier-free guidance.
The dataset is tiny (~9 min), so the model overfits after a few thousand steps: we keep the checkpoint
with the best held-out loss.
"""
import os
import time

import numpy as np
import torch

from common.data import DANCES, EMOTIONS, HELD_OUT, library
from generator.data import bucketize, fit_stats, samples_from_moves
from generator.model import MotionGenerator, device


def loss_fn(net, x0, Q, M, plan_drop=0.1, vel_w=1.0):
    B = x0.shape[0]; dev = x0.device
    has = (torch.rand(B, 1, 1, device=dev) >= plan_drop).float()
    t = torch.sigmoid(torch.randn(B, device=dev) - 0.4); t_ = t.view(-1, 1, 1)
    x1 = torch.randn_like(x0); xt = t_ * x1 + (1 - t_) * x0
    v = net(xt, t, Q, has, M == 0); w = M.unsqueeze(-1)
    loss = (((v - (x1 - x0)) ** 2) * w).sum() / (w.sum() * x0.shape[-1])
    if vel_w == 0:
        return loss
    x0h = xt - t_ * v
    dh, dd = x0h[:, 1:] - x0h[:, :-1], x0[:, 1:] - x0[:, :-1]
    wv = (M[:, 1:] * M[:, :-1]).unsqueeze(-1) * (1 - t_)
    return loss + vel_w * ((dh - dd) ** 2 * wv).sum() / ((dd ** 2 * wv).sum() + 1e-6)


def train(out="checkpoints/generator.pt", steps=5000, bs=16, lr=3e-4, eval_every=250, seed=0,
          held_out=HELD_OUT, dances=True):
    dev = device(); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    moves = library(EMOTIONS) + (library(DANCES) if dances else [])
    tr = [(n, m) for n, m in moves if n not in held_out]; va = [(n, m) for n, m in moves if n in held_out]
    S = samples_from_moves(tr); stats = fit_stats(S)
    buckets = bucketize(S, stats, dev); vb = bucketize(samples_from_moves(va), stats, dev) if va else []
    net = MotionGenerator().to(dev)
    print(f"[generator] {len(tr)} train clips -> {len(S)} augmented | {len(va)} held-out | "
          f"{sum(p.numel() for p in net.parameters())/1e6:.1f}M params | {dev}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=min(0.3, max(0.05, 2 / steps)))
    probs = np.array([len(b["X"]) for b in buckets], float); probs /= probs.sum()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True); best = (float("inf"), 0); hist = []; t0 = time.time()
    for step in range(1, steps + 1):
        b = buckets[rng.choice(len(buckets), p=probs)]
        idx = torch.as_tensor(rng.integers(0, len(b["X"]), min(bs, len(b["X"]))), device=dev)
        loss = loss_fn(net, b["X"][idx], b["Q"][idx], b["M"][idx])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step(); hist.append(loss.item())
        if step % eval_every == 0 or step == steps:
            vl = float("nan")
            if vb:
                net.eval(); torch.manual_seed(0)
                with torch.no_grad():
                    vl = float(np.mean([loss_fn(net, v["X"], v["Q"], v["M"], plan_drop=0.0, vel_w=0.0).item() for v in vb for _ in range(4)]))
                net.train()
            if not vb or vl < best[0]:
                best = (vl, step)
                torch.save(dict(sd=net.state_dict(), stats=stats, step=step, held_out=list(held_out)), out)
            print(f"[generator] step {step:5d}/{steps} train {np.mean(hist[-eval_every:]):.4f} held-out {vl:.4f} "
                  f"{(time.time()-t0)/step:.3f}s/it", flush=True)
    print(f"[generator] kept step {best[1]} (held-out {best[0]:.4f}) -> {out}")
