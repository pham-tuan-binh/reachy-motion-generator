"""Hash-verified frozen NPZ curriculum loading and explicit source mixtures."""
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from common.provenance import digest, read_rows
from generator.conditioned import RESIDUAL_BOUNDS, origins_array


def extract_archive(data):
    """Verify the published archive against its manifests before extracting it.

    Downloads belong in ``data``. Existing arrays are preserved; use a fresh
    directory for a fresh extraction. Only regular ``arrays/*.npz`` are allowed.
    """
    data = Path(data).resolve()
    if (data / "arrays").exists():
        raise FileExistsError("Preserve existing arrays; extract into a fresh dataset directory")
    expected = json.loads((data / "trajectory-hashes.json").read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    if not expected:
        raise ValueError("Dataset archive has no declared trajectories")
    for name, value in expected.items():
        path = PurePosixPath(name)
        if (len(path.parts) != 2 or path.parts[0] != "arrays" or path.suffix != ".npz"
                or manifest["files"].get(name) != value):
            raise ValueError("Invalid archive trajectory path or manifest hash")
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=data) as temporary:
        temporary = Path(temporary)
        (temporary / "arrays").mkdir()
        with tarfile.open(data / "trajectories.tar.gz", "r:gz") as archive:
            members = archive.getmembers()
            if len(members) != len(expected) or {member.name for member in members} != set(expected):
                raise ValueError("Archive membership differs from declared trajectories")
            for member in members:
                if not member.isfile():
                    raise ValueError("Only regular NPZ archive members are supported")
                content = archive.extractfile(member).read()
                if hashlib.sha256(content).hexdigest() != expected[member.name]:
                    raise ValueError("Archive member differs from declared content hash")
                (temporary / member.name).write_bytes(content)
        shutil.move(str(temporary / "arrays"), str(data / "arrays"))
    return len(expected)

def load_training_data(data, require_mixture=True):
    data = Path(data).resolve()
    index = data / "index-train.jsonl"
    records = read_rows(index)
    if not records or any(row["split"] != "train" or row["source"] not in ("real", "synthetic") for row in records):
        raise ValueError("Training index must contain only labeled training records")
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate training IDs")
    arrays, hashes = [], {"index-train.jsonl": digest(index)}
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None:
        if manifest["files"].get("index-train.jsonl") != hashes["index-train.jsonl"]:
            raise ValueError("Training index differs from frozen dataset manifest")
        hashes["manifest.json"] = digest(manifest_path)
    for record in records:
        path = (data / record["file"]).resolve()
        if not path.is_relative_to(data):
            raise ValueError("Training array escapes dataset root")
        relative = str(path.relative_to(data))
        actual = digest(path)
        if record.get("sha256") is not None and record["sha256"] != actual:
            raise ValueError("Training array differs from indexed content hash")
        if manifest is not None and manifest["files"].get(relative) != actual:
            raise ValueError("Training array differs from frozen dataset manifest")
        hashes[relative] = actual
        with np.load(path, allow_pickle=False) as content:
            values = {key: np.array(content[key], dtype=np.float32) for key in ("X", "Q", "B")}
        length = len(values["X"])
        if (not 2 <= length <= 720 or values["X"].shape != (length, 9)
                or values["Q"].shape != (length, 8) or values["B"].shape != (length, 9)
                or not all(np.isfinite(value).all() for value in values.values())):
            raise ValueError("Invalid aligned training arrays")
        values["origin"] = origins_array([record["origin"]])[0]
        values["source"] = record["source"]
        arrays.append(values)
    if require_mixture and not {"real", "synthetic"} <= {row["source"] for row in records}:
        raise ValueError("The preregistered mixture requires both real and synthetic training data")
    return arrays, hashes


def residual_target_diagnostics(examples):
    """Describe exactly how much target detail the fixed architecture discards."""
    bounds = np.asarray(RESIDUAL_BOUNDS)
    output = {}
    for source in ("real", "synthetic"):
        rows = [row for row in examples if row["source"] == source]
        if not rows:
            continue
        values = np.concatenate([(row["X"] - row["B"]) / bounds for row in rows])
        output[source] = dict(frames=len(values), coordinates=values.size,
            fraction_coordinates_outside_bounds=float(np.mean(np.abs(values) > 1)),
            fraction_frames_outside_any_bound=float(np.mean(np.any(np.abs(values) > 1, axis=1))),
            max_normalized_residual=float(np.abs(values).max()))
    output["endpoint_ramp"] = "Clipped latent target is additionally multiplied by 400ms endpoint ramps during decoding"
    return output


def make_batch(examples, batch_size, rng, device, real_fraction=.2):
    real = [index for index, row in enumerate(examples) if row["source"] == "real"]
    synthetic = [index for index, row in enumerate(examples) if row["source"] == "synthetic"]
    if batch_size < 1 or not 0 <= real_fraction <= 1:
        raise ValueError("Invalid batch size or real fraction")
    count_real = (min(batch_size - 1, max(1, round(batch_size * real_fraction)))
                  if 0 < real_fraction < 1 else int(real_fraction * batch_size))
    if (count_real and not real) or (batch_size > count_real and not synthetic):
        raise ValueError("The requested mixture needs missing source examples")
    indices = list(rng.choice(real, count_real, replace=True)) + list(rng.choice(synthetic, batch_size - count_real, replace=True))
    rng.shuffle(indices)
    chosen = [examples[index] for index in indices]
    length = max(len(row["X"]) for row in chosen)
    values = {key: np.zeros((batch_size, length, size), dtype=np.float32) for key, size in (("X", 9), ("Q", 8), ("B", 9))}
    values["M"] = np.zeros((batch_size, length), dtype=bool)
    values["origin"] = np.stack([row["origin"] for row in chosen])
    for index, row in enumerate(chosen):
        size = len(row["X"])
        for key in ("X", "Q", "B"):
            values[key][index, :size] = row[key]
        values["M"][index, :size] = True
    return {key: torch.from_numpy(value).to(device) for key, value in values.items()}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify and extract a published frozen NPZ curriculum")
    parser.add_argument("--extract", type=Path, required=True, help="Downloaded dataset directory")
    args = parser.parse_args()
    print(json.dumps(dict(extracted=extract_archive(args.extract))))
