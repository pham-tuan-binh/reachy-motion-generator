"""State-conditioned flow generators and bounded detail, with no robot I/O.

Original checkpoints retain the original sampler exactly. V4 residual samples
are bounded around the deterministic plan *after* filtering, and have zero
detail at both endpoints. This is an architectural bound, not a safety proof.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

VERSION = "state-conditioned-generator-v4"
ORIGIN_FIELDS = ("roll", "pitch", "yaw", "body", "right", "left", "x", "y", "z")
ORIGIN_SCALE = (20., 20., 45., 45., 90., 90., .01, .01, .01)
# Native trajectory order: xyz metres, xyz Euler radians, signed ears, body.
RESIDUAL_BOUNDS = (.0005, .0005, .0005, *([math.pi / 180] * 3),
                   5 * math.pi / 180, 5 * math.pi / 180, math.pi / 180)
CONFIGS = {"full": dict(d=384, heads=6, layers=8, maxlen=720),
           "residual": dict(d=384, heads=6, layers=8, maxlen=720),
           "large": dict(d=512, heads=8, layers=12, maxlen=720)}


def origins_array(origins):
    if not origins or any(set(origin) != set(ORIGIN_FIELDS) for origin in origins):
        raise ValueError("Each origin must contain exactly the nine named pose fields")
    values = np.asarray([[origin[key] for key in ORIGIN_FIELDS] for origin in origins], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite origin")
    return values


class ConditionedGenerator(nn.Module):
    def __init__(self, config, kind="full"):
        super().__init__()
        from generator.model import MotionGenerator
        self.config, self.kind = dict(config), kind
        self.model = MotionGenerator(**config)
        self.maxlen = self.model.maxlen
        old = self.model.inp
        expanded = nn.Linear(old.in_features + len(ORIGIN_FIELDS), old.out_features)
        with torch.no_grad():
            expanded.weight[:, :old.in_features].copy_(old.weight)
            expanded.weight[:, old.in_features:].zero_()
            expanded.bias.copy_(old.bias)
        self.model.inp = expanded
        self.register_buffer("origin_scale", torch.tensor(ORIGIN_SCALE), persistent=False)

    def forward(self, x, t, plan, has, pad, origin):
        from generator.model import timestep_embedding
        if origin.shape != (len(x), len(ORIGIN_FIELDS)):
            raise ValueError("Origin batch shape differs from motion batch")
        m = self.model
        cond = torch.cat([plan * has, has.expand(-1, x.shape[1], 1)], -1)
        state = (origin / self.origin_scale).unsqueeze(1).expand(-1, x.shape[1], -1)
        h = m.inp(torch.cat([x, cond, state], -1)) + m.pos[:, :x.shape[1]]
        c = m.temb(timestep_embedding(t))
        for block in m.blocks:
            h = block(h, c, pad)
        return m.out(m.nf(h))

    def warm_start(self, state):
        """Exact original weights plus zero state columns; no missing keys allowed."""
        state = {key: value.clone() for key, value in state.items()}
        original = state["inp.weight"]
        expected = self.model.inp.weight.shape
        if original.shape != (expected[0], expected[1] - len(ORIGIN_FIELDS)):
            raise ValueError("Original input shape does not match requested architecture")
        state["inp.weight"] = torch.cat([original, original.new_zeros((len(original), len(ORIGIN_FIELDS)))], dim=1)
        self.model.load_state_dict(state, strict=True)


def create_model(kind, original_checkpoint=None, config=None):
    if kind not in CONFIGS:
        raise ValueError("Unknown generator kind")
    model = ConditionedGenerator(config or CONFIGS[kind], kind)
    if kind == "full" and original_checkpoint is not None:
        checkpoint = torch.load(original_checkpoint, map_location="cpu", weights_only=False)
        model.warm_start(checkpoint["sd"])
    return model


def checkpoint_payload(net, stats, **metadata):
    return dict(version=VERSION, kind=net.kind, config=net.config, sd=net.state_dict(), stats=stats,
                residual_bounds=list(RESIDUAL_BOUNDS), origin_fields=list(ORIGIN_FIELDS),
                origin_scale=list(ORIGIN_SCALE), **metadata)


def load_generator(checkpoint, device="cpu"):
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if saved.get("version") == VERSION:
        if (saved["origin_fields"] != list(ORIGIN_FIELDS) or saved["origin_scale"] != list(ORIGIN_SCALE)
                or saved["residual_bounds"] != list(RESIDUAL_BOUNDS)):
            raise ValueError("Checkpoint state or residual contract differs")
        net = create_model(saved["kind"], config=saved["config"])
        net.load_state_dict(saved["sd"], strict=True)
        metadata = {key: saved[key] for key in ("version", "kind", "config")}
    elif "version" in saved:
        raise ValueError("Unsupported generator checkpoint version")
    else:
        from generator.model import MotionGenerator
        net = MotionGenerator()
        net.load_state_dict(saved["sd"], strict=True)
        metadata = dict(version="original", kind="original", state_conditioned=False)
    net.to(device).eval()
    net._generator_v4_metadata = metadata
    return net, saved["stats"], {**metadata, "device": str(device)}


def plan_native(frames):
    """Torch conversion preserving gradients; energy has no direct pose inverse."""
    out = frames.new_zeros((*frames.shape[:-1], 9))
    out[..., 2] = frames[..., 5] / 1000
    out[..., 3:6] = frames[..., [3, 2, 4]] * (math.pi / 180)
    out[..., 6] = -frames[..., 0] * (math.pi / 180)
    out[..., 7] = frames[..., 1] * (math.pi / 180)
    out[..., 8] = frames[..., 6] * (math.pi / 180)
    return out


def endpoint_gate(mask):
    """400ms linear ramps, exactly zero on each unpadded endpoint."""
    lengths = mask.sum(1).long()
    index = torch.arange(mask.shape[1], device=mask.device)[None]
    gate = torch.minimum(index / 10., (lengths[:, None] - 1 - index) / 10.).clamp(0, 1)
    return gate.unsqueeze(-1) * mask.unsqueeze(-1)


def decode_clean(net, clean, backbone, stats, mask):
    if net.kind == "residual":
        bounds = clean.new_tensor(RESIDUAL_BOUNDS)
        return backbone + clean.clamp(-1, 1) * bounds * endpoint_gate(mask)
    mu, sd = (clean.new_tensor(stats[key]) for key in ("MU", "SD"))
    return clean * sd + mu


@torch.no_grad()
def sample_batch(net, stats, plans, origins, seeds, steps=8, cfg=1.5, lowpass_hz=4.0):
    from scipy.signal import butter, filtfilt

    from common import plan as pl
    from generator.sample import generate_batch

    if (not plans or len(plans) != len(origins) or len(plans) != len(seeds)
            or type(steps) is not int or steps < 1 or not math.isfinite(cfg)
            or not 0 <= lowpass_hz < 12.5):
        raise ValueError("Invalid generator sample settings")
    state = origins_array(origins)
    device = next(net.parameters()).device
    if not isinstance(net, ConditionedGenerator) and lowpass_hz:
        return generate_batch(net, stats, plans, device, seeds=seeds, steps=steps, cfg=cfg, lowpass_hz=lowpass_hz)
    frames = [pl.frames(plan)[:net.maxlen] for plan in plans]
    lengths = [len(value) for value in frames]
    if min(lengths) < 2:
        raise ValueError("Plan is too short")
    batch, duration = len(plans), max(lengths)
    q = torch.zeros(batch, duration, 8, device=device)
    x = torch.zeros(batch, duration, 9, device=device)
    mask = torch.zeros(batch, duration, dtype=torch.bool, device=device)
    backbone = torch.zeros_like(x)
    pmu, psd = (np.asarray(stats[key]) for key in ("PMU", "PSD"))
    for index, value in enumerate(frames):
        length = lengths[index]
        q[index, :length] = torch.as_tensor((value - pmu) / psd, dtype=torch.float32, device=device)
        mask[index, :length] = True
        backbone[index, :length] = plan_native(torch.as_tensor(value, dtype=torch.float32, device=device))
        generator = torch.Generator(device="cpu").manual_seed(int(seeds[index]))
        x[index, :length] = torch.randn(length, 9, generator=generator).to(device)
    origin = torch.as_tensor(state, device=device)
    has = torch.ones(batch, 1, 1, device=device)
    guided = cfg != 1.
    model_q, model_pad, model_origin = q, ~mask, origin
    if guided:
        has = torch.cat([has, torch.zeros_like(has)])
        model_q, model_pad, model_origin = torch.cat([q, q]), torch.cat([~mask, ~mask]), torch.cat([origin, origin])
    for step in range(steps):
        t = torch.full((len(has),), 1. - step / steps, device=device)
        inputs = (torch.cat([x, x]) if guided else x, t, model_q, has, model_pad)
        velocity = net(*inputs, model_origin) if isinstance(net, ConditionedGenerator) else net(*inputs)
        if guided:
            conditional, unconditional = velocity[:batch], velocity[batch:]
            velocity = unconditional + cfg * (conditional - unconditional)
        x = x - velocity / steps
    output = []
    filt = butter(4, lowpass_hz / 12.5) if lowpass_hz else None
    residual = isinstance(net, ConditionedGenerator) and net.kind == "residual"
    if residual:
        raw = x.cpu().numpy()
        base = backbone.cpu().numpy()
        gates = endpoint_gate(mask).cpu().numpy()
    else:
        mu, sd = (np.asarray(stats[key]) for key in ("MU", "SD"))
        raw = x.cpu().numpy() * sd + mu
    for index, length in enumerate(lengths):
        value = raw[index, :length]
        if filt is not None and length > 20:
            value = filtfilt(*filt, value, axis=0, padlen=min(length - 1, 15))
        if residual:
            value = base[index, :length] + np.clip(value, -1., 1.) * np.asarray(RESIDUAL_BOUNDS) * gates[index, :length]
        if not np.isfinite(value).all():
            raise ValueError("Generator returned nonfinite motion")
        output.append(value)
    return output
