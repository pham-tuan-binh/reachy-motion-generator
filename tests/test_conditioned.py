"""Portable numerical generator invariants; no robot, network or Loopy imports."""
from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pytest
import torch

from common.provenance import digest
from generator.conditioned import (
    ORIGIN_FIELDS,
    RESIDUAL_BOUNDS,
    checkpoint_payload,
    create_model,
    load_generator,
    origins_array,
    sample_batch,
)
from generator.curriculum import load_training_data, make_batch
from generator.losses import objective
from generator.model import MotionGenerator

TINY = {"d": 16, "heads": 2, "layers": 1, "maxlen": 64}


@pytest.fixture(autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def origin():
    return {"roll": 0., "pitch": -4., "yaw": 0., "body": 0., "right": -10., "left": 10., "x": 0., "y": 0., "z": 0.}


def stats():
    return {"MU": [0.] * 9, "SD": [.01, .01, .01, .3, .3, .3, 1., 1., .5], "PMU": [0.] * 8, "PSD": [30.] * 8}


def plan():
    return {"duration": 1.2, "keys": [
        {"t": 0., "pitch": 0., "roll": 0., "yaw": 0., "body": 0., "earR": 15., "earL": 15., "z": 3., "energy": 0.},
        {"t": 1.16, "pitch": 8., "roll": 0., "yaw": 0., "body": 0., "earR": 15., "earL": 15., "z": 3., "energy": 0.}]}


def native_from_channels(channels):
    """Independent convention: metres, Euler radians, signed antenna radians."""
    q = np.asarray(channels, dtype=np.float32)
    output = np.zeros((len(q), 9), dtype=np.float32)
    output[:, 2] = q[:, 5] / 1000
    output[:, 3:6] = q[:, [3, 2, 4]] * np.float32(np.pi / 180)
    output[:, 6] = -q[:, 0] * np.float32(np.pi / 180)
    output[:, 7] = q[:, 1] * np.float32(np.pi / 180)
    output[:, 8] = q[:, 6] * np.float32(np.pi / 180)
    return output


def batch():
    q = np.zeros((30, 8), dtype=np.float32)
    q[:, :2] = 15.
    q[:, 5] = 3.
    q[:, 2] = np.linspace(0, 6, 30)
    native = native_from_channels(q)
    return {"X": torch.tensor(np.stack([native, native])), "Q": torch.tensor(np.stack([q, q])),
                "B": torch.tensor(np.stack([native, native])),
                "M": torch.tensor([[True] * 30, [True] * 20 + [False] * 10]),
                "origin": torch.tensor(origins_array([origin(), origin()]))}


def test_zero_state_extension_preserves_original_nontrivial_flow():
    torch.manual_seed(73)
    original = MotionGenerator(**TINY).eval()
    with torch.no_grad():
        original.out.weight.normal_(0, .03)
        # Nonzero block gates make attention and MLP behavior part of parity.
        original.blocks[0].ada[1].bias.normal_(0, .05)
    conditioned = create_model("full", config=TINY).eval()
    conditioned.warm_start(original.state_dict())
    x, q = torch.randn(2, 24, 9), torch.randn(2, 24, 8)
    t, has, pad = torch.rand(2), torch.ones(2, 1, 1), torch.zeros(2, 24, dtype=torch.bool)
    state = torch.tensor(origins_array([origin(), {**origin(), "pitch": 7.}]))
    torch.testing.assert_close(conditioned(x, t, q, has, pad, state), original(x, t, q, has, pad),
                               rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(conditioned.model.inp.weight[:, -len(ORIGIN_FIELDS):]) == 0


@pytest.mark.parametrize("kind", ["full", "residual", "large"])
def test_each_kind_checkpoints_roundtrip_and_samples_are_deterministic(tmp_path, kind):
    model = create_model(kind, config=TINY).eval()
    checkpoint = tmp_path / f"{kind}.pt"
    torch.save(checkpoint_payload(model, stats(), step=8), checkpoint)
    restored, norm, metadata = load_generator(checkpoint, "cpu")
    assert metadata["kind"] == kind
    actual = sample_batch(model, stats(), [plan()], [origin()], [481], steps=2, cfg=1.5)[0]
    reloaded = sample_batch(restored, norm, [plan()], [origin()], [481], steps=2, cfg=1.5)[0]
    np.testing.assert_array_equal(actual, reloaded)
    assert actual.shape == (30, 9) and np.isfinite(actual).all()


@pytest.mark.parametrize("lowpass_hz", [0., 4., 8.])
def test_residual_bound_and_zero_detail_endpoints_hold_after_filter(lowpass_hz):
    from common.plan import frames
    model = create_model("residual", config=TINY).eval()
    baseline = native_from_channels(frames(plan()))
    values = sample_batch(model, stats(), [plan()], [origin()], [271], steps=2, cfg=1., lowpass_hz=lowpass_hz)[0]
    detail = values - baseline
    assert np.all(np.abs(detail) <= np.array(RESIDUAL_BOUNDS) + 1e-8)
    np.testing.assert_array_equal(detail[[0, -1]], np.zeros((2, 9)))


@pytest.mark.parametrize("kind", ["full", "residual", "large"])
def test_objective_masks_padding_and_produces_nonzero_gradient(kind):
    model = create_model(kind, config=TINY)
    values = batch()
    randomness = (torch.tensor([.25, .5]), torch.randn_like(values["X"]), torch.ones(2, 1, 1))
    loss, terms = objective(model, values, stats(), None, randomness=randomness)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.model.out.weight.grad.abs().sum() > 0
    changed = copy.deepcopy(values)
    for key in ("X", "Q", "B"):
        changed[key][1, 20:] = 10000.
    again, altered_terms = objective(model, changed, stats(), None, randomness=randomness)
    torch.testing.assert_close(again, loss.detach())
    for key in terms:
        torch.testing.assert_close(altered_terms[key], terms[key].detach())


def write_dataset(folder, *, mixed=True):
    folder.mkdir()
    values = batch()
    rows = []
    for index, source in enumerate(("real", "synthetic") if mixed else ("synthetic",)):
        path = folder / f"fixture-{index}.npz"
        np.savez(path, **{key: values[key][0].numpy() for key in ("X", "Q", "B")})
        rows.append({"id": f"fixture-{index}", "group_id": f"fixture-{index}", "split": "train", "source": source,
                         "file": path.name, "sha256": digest(path), "origin": origin()})
    index = folder / "index-train.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"files": {p.name: digest(p) for p in folder.iterdir()}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return rows


def test_loader_rejects_changed_npz_and_index(tmp_path):
    data = tmp_path / "data"
    rows = write_dataset(data)
    examples, hashes = load_training_data(data)
    assert len(examples) == 2 and "manifest.json" in hashes
    with (data / rows[0]["file"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="content hash"):
        load_training_data(data)
    index = data / "index-train.jsonl"
    index.write_text(index.read_text() + "\n")
    with pytest.raises(ValueError, match="index differs"):
        load_training_data(data)


@pytest.mark.parametrize("invalid", ["test", "../outside.npz"])
def test_loader_rejects_test_leakage_and_path_escape(tmp_path, invalid):
    record = {"id": "fixture", "group_id": "fixture", "source": "synthetic", "origin": origin(), "split": "train", "file": "unused.npz"}
    record["split" if invalid == "test" else "file"] = invalid
    (tmp_path / "index-train.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="only labeled training|escapes"):
        load_training_data(tmp_path)


def test_real_fraction_is_an_actual_batch_count_and_missing_sources_reject(tmp_path):
    data = tmp_path / "data"
    write_dataset(data)
    examples, _ = load_training_data(data)
    for row in examples:
        row["X"] = np.full_like(row["X"], 1. if row["source"] == "real" else 0.)
    values = make_batch(examples, 16, np.random.default_rng(7), "cpu", real_fraction=.2)
    assert int((values["X"][:, 0, 0] == 1.).sum()) == 3
    with pytest.raises(ValueError, match="missing source"):
        make_batch(examples[:1], 16, np.random.default_rng(7), "cpu", real_fraction=.2)


@pytest.mark.parametrize("kind", ["full", "residual", "large"])
def test_two_optimizer_steps_save_reload_and_sample_without_local_assets(tmp_path, kind):
    from generator.finetune import train
    data, checkpoint, out = tmp_path / "data", tmp_path / "original.pt", tmp_path / "trained"
    write_dataset(data)
    original = MotionGenerator(**TINY)
    torch.save({"sd": original.state_dict(), "stats": stats()}, checkpoint)
    args = SimpleNamespace(data=data, original_checkpoint=checkpoint, out=out, kind=kind, steps=2,
                           batch_size=2, lr=1e-4, seed=27, save_every=1, max_seconds=30, device="cpu",
                           real_fraction=.2, clearance_table=None, config=TINY)
    result = train(args)
    assert result["completed"] and result["steps"] == 2
    assert set(result["checkpoint_sha256"]) == {"checkpoint-1.pt", "checkpoint-2.pt"}
    model, norm, _ = load_generator(out / "checkpoint-2.pt", "cpu")
    assert model.model.out.weight.abs().sum() > 0
    sample = sample_batch(model, norm, [plan()], [origin()], [177], steps=2)[0]
    assert np.isfinite(sample).all()
    with pytest.raises(FileExistsError):
        train(args)


def test_actual_original_checkpoint_retains_pinned_sampler_outputs():
    """Opt-in cross-version check on real weights, not a recreated toy sampler."""
    checkpoint = os.environ.get("MOTION_ORIGINAL_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set MOTION_ORIGINAL_CHECKPOINT for actual-weight sampling parity")
    from generator.sample import generate_batch, load
    original, norm, device = load(checkpoint, "cpu")
    ported, values, metadata = load_generator(checkpoint, "cpu")
    assert metadata["kind"] == "original"
    expected = generate_batch(original, norm, [plan()], device, seeds=[73], steps=2, cfg=1.5)
    actual = sample_batch(ported, values, [plan()], [origin()], [73], steps=2, cfg=1.5)
    np.testing.assert_array_equal(actual[0], expected[0])
