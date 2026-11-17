"""Immutable model selection shared by the package and Hugging Face cards."""
import json
from importlib.resources import files
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from common.provenance import digest


def default_config():
    """Return a fresh copy so callers cannot mutate the package defaults."""
    return json.loads(files("inference").joinpath("default_pipeline.json").read_text())


def checked_file(path, expected):
    path = Path(path)
    if digest(path) != expected:
        raise ValueError(f"Checkpoint hash differs from the pinned default: {path.name}")
    return path


def generator_checkpoint():
    spec = default_config()["generator"]
    path = hf_hub_download(spec["repo_id"], spec["filename"], revision=spec["revision"])
    return checked_file(path, spec["sha256"])


def planner_checkpoints():
    config = default_config()
    base, adapter = config["base"], config["adapter"]
    root = snapshot_download(base["repo_id"], revision=base["revision"],
                             allow_patterns=[base["subfolder"] + "/*"])
    adapter_root = Path(snapshot_download(adapter["repo_id"], revision=adapter["revision"],
                                         allow_patterns=list(adapter["files"])))
    for name, expected in adapter["files"].items():
        checked_file(adapter_root / name, expected)
    return Path(root) / base["subfolder"], adapter_root
