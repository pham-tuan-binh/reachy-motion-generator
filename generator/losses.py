"""Masked v4 flow losses and an optional external clearance-table surrogate.

No calibration table is distributed here. The Euler-offset approximation omits
exact SO(3), inverse kinematics, lag and controller checks; it is not admission.
"""
import math

import numpy as np
import torch

from common.provenance import digest
from generator.conditioned import RESIDUAL_BOUNDS, decode_clean

DEFAULT_WEIGHTS = dict(flow=1., fidelity=.25, plan=.05, endpoint=.25, velocity=.05, safety=.01)


class ClearanceSurrogate(torch.nn.Module):
    """Piecewise multilinear table with separate penalties outside its domain."""
    def __init__(self, path):
        super().__init__()
        self.path, self.sha256 = str(path), digest(path)
        with np.load(path) as data:
            names = ("rel_yaw", "pitch", "roll", "z_mm", "x_mm")
            axes = [np.asarray(data[name], dtype=np.float32) for name in names]
            table = np.asarray(data["gap_decimm"], dtype=np.float32) / 10.
        if (any(axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
                or not np.all(np.diff(axis) > 0) or not np.allclose(np.diff(axis), axis[1] - axis[0])
                for axis in axes) or table.shape != tuple(len(axis) for axis in axes)
                or not np.isfinite(table).all()):
            raise ValueError("Clearance table must have five finite uniform increasing axes and matching gap data")
        self.register_buffer("table", torch.from_numpy(table.reshape(-1)))
        self.register_buffer("lo", torch.tensor([axis[0] for axis in axes]))
        self.register_buffer("hi", torch.tensor([axis[-1] for axis in axes]))
        self.register_buffer("step", torch.tensor([axis[1] - axis[0] for axis in axes]))
        self.register_buffer("counts", torch.tensor(table.shape, dtype=torch.long))
        strides = [math.prod(table.shape[i + 1:]) for i in range(5)]
        self.register_buffer("strides", torch.tensor(strides, dtype=torch.long))
        self.register_buffer("corners", torch.tensor([[corner >> dim & 1 for dim in range(5)] for corner in range(32)]))

    def gap(self, world):
        """world[...,9] follows ORIGIN_FIELDS in degrees/metres."""
        roll, pitch, yaw, body = world[..., :4].unbind(-1)
        x, y, z = world[..., 6:].unbind(-1)
        angle = body * (math.pi / 180)
        xb = (x * angle.cos() + y * angle.sin()) * 1000
        yb = (-x * angle.sin() + y * angle.cos()) * 1000
        rel = yaw - body
        point = torch.stack([rel.abs(), pitch, torch.where(rel >= 0, roll, -roll),
                             torch.minimum(z * 1000, torch.zeros_like(z)), xb], -1)
        domain = (torch.relu(self.lo - point).square() + torch.relu(point - self.hi).square()).sum(-1)
        domain = domain + torch.relu(yb.abs() - 12.).square()
        u = ((point - self.lo) / self.step).clamp_min(0)
        u = torch.minimum(u, self.counts.to(u.dtype) - 1)
        index = torch.minimum(u.floor().long(), self.counts - 2)
        fraction = u - index
        offsets = ((index.unsqueeze(-2) + self.corners) * self.strides).sum(-1)
        weights = torch.where(self.corners.bool(), fraction.unsqueeze(-2), 1. - fraction.unsqueeze(-2)).prod(-1)
        gap = (self.table[offsets] * weights).sum(-1) - .2 * yb.abs()
        return gap, domain

    def forward(self, trajectory, origin, mask):
        # Absolute pose approximation intentionally differs from replay's SO(3)
        # composition when the generated first frame is not near neutral.
        relative = trajectory - trajectory[:, :1]
        world = origin[:, None].expand(-1, trajectory.shape[1], -1).clone()
        world[..., :3] = world[..., :3] + relative[..., 3:6] * (180 / math.pi)
        world[..., 3] = world[..., 3] + relative[..., 8] * (180 / math.pi)
        world[..., 6:] = world[..., 6:] + relative[..., :3]
        gap, domain = self.gap(world)
        penalty = torch.relu(3. - gap).square() + domain + torch.relu(world[..., 1] - 17.).square()
        return (penalty * mask).sum() / mask.sum().clamp_min(1)


def masked_mean(value, mask):
    expanded = mask.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1)


def objective(net, batch, stats, safety, *, weights=None, randomness=None):
    """All terms mask padding. Explicit randomness supports meaningful parity tests."""
    weights = DEFAULT_WEIGHTS if weights is None else weights
    x, q, backbone, mask, origin = (batch[key] for key in ("X", "Q", "B", "M", "origin"))
    mu, sd, pmu, psd = (x.new_tensor(stats[key]) for key in ("MU", "SD", "PMU", "PSD"))
    if net.kind == "residual":
        bounds = x.new_tensor(RESIDUAL_BOUNDS)
        target = ((x - backbone) / bounds).clamp(-1, 1)
    else:
        target = (x - mu) / sd
    plan = (q - pmu) / psd
    # Padded data never enter attention keys or any objective denominator.
    target = target * mask[..., None]
    plan = plan * mask[..., None]
    if randomness is None:
        t = torch.sigmoid(torch.randn(len(x), device=x.device) - .4)
        noise = torch.randn_like(target)
        has = (torch.rand(len(x), 1, 1, device=x.device) >= .1).to(x.dtype)
    else:
        t, noise, has = randomness
    time_weight = t[:, None, None]
    noisy = time_weight * noise + (1 - time_weight) * target
    velocity = net(noisy, t, plan, has, ~mask, origin)
    clean = noisy - time_weight * velocity
    predicted = decode_clean(net, clean, backbone, stats, mask)
    clean_target = decode_clean(net, target, backbone, stats, mask)
    # Residual's attainable clipped target is explicit; fidelity is to original
    # X for full models and bounded attainable motion for residual models.
    norm_error = (predicted - clean_target) / sd
    valid = mask[..., None]
    flow = masked_mean((velocity - (noise - target)).square(), valid)
    fidelity = masked_mean(norm_error.square(), valid)
    plan_loss = masked_mean(((predicted - backbone) / sd).square(), valid)
    lengths = mask.sum(1).long()
    rows = torch.arange(len(x), device=x.device)
    ends = lengths - 1
    endpoint = (((predicted[rows, ends] - clean_target[rows, ends]) / sd).square().mean()
                + ((predicted[:, 0] - clean_target[:, 0]) / sd).square().mean()) / 2
    deltas = ((predicted[:, 1:] - predicted[:, :-1]) - (clean_target[:, 1:] - clean_target[:, :-1])) / sd
    velocity_loss = masked_mean(deltas.square(), (mask[:, 1:] & mask[:, :-1])[..., None])
    safety_loss = safety(predicted, origin, mask) if safety is not None else x.new_zeros(())
    terms = dict(flow=flow, fidelity=fidelity, plan=plan_loss, endpoint=endpoint, velocity=velocity_loss, safety=safety_loss)
    total = sum(terms[name] * weights[name] for name in DEFAULT_WEIGHTS)
    return total, terms
