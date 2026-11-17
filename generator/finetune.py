"""Train the portable state-conditioned v4 models on a frozen NPZ curriculum.

Run ``python -m generator.finetune --help``. Clearance supervision is disabled
unless an explicitly supplied, separately calibrated table is provided.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from common.provenance import digest, write_json
from generator.conditioned import checkpoint_payload, create_model
from generator.curriculum import (
    load_training_data,
    make_batch,
    residual_target_diagnostics,
)
from generator.losses import DEFAULT_WEIGHTS, ClearanceSurrogate, objective

VERSION = "generator-training-v4-portable"
ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("generator/conditioned.py", "generator/curriculum.py", "generator/losses.py",
                "generator/finetune.py", "generator/model.py", "generator/sample.py",
                "common/plan.py", "common/provenance.py", "common/assets/kinematics_data.json")


def atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    """Train into a new directory; bind checkpoints to immutable input receipts."""
    if (args.steps < 1 or args.batch_size < 2 or args.save_every < 1 or args.lr <= 0
            or args.max_seconds < 1 or not 0 <= args.real_fraction <= 1):
        raise ValueError("Invalid positive training count/rate or real fraction")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("Use a fresh training directory; preserve existing evidence")
    examples, data_hashes = load_training_data(args.data, require_mixture=0 < args.real_fraction < 1)
    original_path = Path(args.original_checkpoint)
    original = torch.load(original_path, map_location="cpu", weights_only=False)
    stats = original["stats"]
    net = create_model(args.kind, original_path, config=getattr(args, "config", None)).to(args.device).train()
    safety = ClearanceSurrogate(args.clearance_table).to(args.device) if args.clearance_table else None
    weights = dict(DEFAULT_WEIGHTS)
    if safety is None:
        weights["safety"] = 0.
    real_count = (min(args.batch_size - 1, max(1, round(args.batch_size * args.real_fraction)))
                  if 0 < args.real_fraction < 1 else int(args.real_fraction * args.batch_size))
    # Validate source availability before creating the output directory.
    make_batch(examples, args.batch_size, np.random.default_rng(args.seed), "cpu", args.real_fraction)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=.01)
    out.mkdir(parents=True)
    manifest = dict(version=VERSION, checkpoint_version="state-conditioned-generator-v4", kind=args.kind,
        steps=args.steps, batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed, device=args.device,
        real_fraction_requested=args.real_fraction, real_examples_per_batch=real_count,
        real_fraction_actual=real_count / args.batch_size, parameters=sum(p.numel() for p in net.parameters()),
        config=net.config, losses=weights, original_generator_sha256=digest(original_path), data_sha256=data_hashes,
        source_sha256={name: digest(ROOT / name) for name in SOURCE_FILES},
        clearance_table_sha256=safety.sha256 if safety else None,
        clearance_supervision="external_table" if safety else "disabled",
        initialization="original_warm_start_zero_state_columns" if args.kind == "full" else "from_scratch",
        safety_surrogate=("Euler-offset anchoring and piecewise multilinear supplied clearance table; excludes exact SO(3), IK and lag"
                          if safety else "Disabled: no external clearance table supplied; differs from the calibrated pilot objective"),
        residual_target="clipped attainable residual within fixed per-channel bounds" if args.kind == "residual" else None,
        residual_target_diagnostics=residual_target_diagnostics(examples) if args.kind == "residual" else None,
        started_at=datetime.now(timezone.utc).isoformat())
    write_json(out / "manifest.json", manifest)
    started = time.monotonic()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with (out / "training.jsonl").open("w") as stream:
        for step in range(1, args.steps + 1):
            batch = make_batch(examples, args.batch_size, rng, args.device, args.real_fraction)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                loss, terms = objective(net, batch, stats, safety, weights=weights)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite generator training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite generator gradient")
            opt.step()
            row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm),
                seconds=time.monotonic() - started, **{key: float(value.detach()) for key, value in terms.items()})
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if step == 1 or step % 25 == 0:
                print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                atomic_save(out / f"checkpoint-{step}.pt", checkpoint_payload(
                    net, stats, step=step, training_manifest_sha256=digest(out / "manifest.json")))
            if time.monotonic() - started > args.max_seconds:
                atomic_save(out / f"checkpoint-time-limit-{step}.pt", checkpoint_payload(net, stats, step=step))
                raise TimeoutError("Generator training time budget reached")
    for relative, expected in data_hashes.items():
        if digest(Path(args.data) / relative) != expected:
            raise ValueError("Training data changed during optimization")
    if digest(original_path) != manifest["original_generator_sha256"]:
        raise ValueError("Original generator changed during optimization")
    for name, expected in manifest["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Training source or asset changed during optimization")
    if safety is not None and digest(args.clearance_table) != safety.sha256:
        raise ValueError("External clearance table changed during optimization")
    final = dict(completed=True, steps=args.steps, seconds=time.monotonic() - started,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
        checkpoint_sha256={path.name: digest(path) for path in sorted(out.glob("checkpoint-*.pt"))})
    write_json(out / "training.json", final)
    print(json.dumps(final), flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "original-checkpoint", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--kind", choices=("full", "residual", "large"), required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=481729)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--real-fraction", type=float, default=.2)
    parser.add_argument("--clearance-table", type=Path)
    parser.add_argument("--model-config", type=Path, help="Optional JSON architecture override, e.g. a small smoke model")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.lr = args.lr if args.lr is not None else 1e-4 if args.kind == "full" else 3e-4
    args.config = json.loads(args.model_config.read_text()) if args.model_config else None
    train(args)


if __name__ == "__main__":
    main()
