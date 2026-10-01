"""The motion generator: a small flow-matching transformer (21.8M parameters by default).

Input per frame: the noisy 9-DoF motion x_t (normalised), the plan channels interpolated to that
frame (normalised) times a ``has_plan`` flag, and the flag itself. The plan is concatenated to every
frame rather than cross-attended, so the model cannot ignore it. The flow time ``t`` enters through
AdaLN (zero-initialised, so each block starts as identity). The output is the velocity
v = x1 - x0 (noise minus data).
"""
import math

import torch
import torch.nn as nn

N_DOF, N_PLAN = 9, 8


def device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def timestep_embedding(t, dim=256):
    h = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(h, device=t.device) / h)
    a = t[:, None] * f[None] * 1000.0
    return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False); self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, pad):
        a1, b1, g1, a2, b2, g2 = self.ada(c).unsqueeze(1).chunk(6, -1)
        h = self.n1(x) * (1 + b1) + a1
        x = x + g1 * self.att(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        h = self.n2(x) * (1 + b2) + a2
        return x + g2 * self.mlp(h)


class MotionGenerator(nn.Module):
    def __init__(self, d=384, heads=6, layers=8, maxlen=720):
        super().__init__()
        self.maxlen = maxlen
        self.inp = nn.Linear(N_DOF + N_PLAN + 1, d)
        self.pos = nn.Parameter(torch.randn(1, maxlen, d) * 0.02)
        self.temb = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.nf = nn.LayerNorm(d); self.out = nn.Linear(d, N_DOF)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, t, plan, has, pad):
        """x (B,T,9) noisy motion; t (B,) flow time (1 = noise); plan (B,T,8); has (B,1,1); pad (B,T) True = padding."""
        cond = torch.cat([plan * has, has.expand(-1, x.shape[1], 1)], -1)
        h = self.inp(torch.cat([x, cond], -1)) + self.pos[:, :x.shape[1]]
        c = self.temb(timestep_embedding(t))
        for b in self.blocks:
            h = b(h, c, pad)
        return self.out(self.nf(h))
