"""Immutable model selection shared by the package and Hugging Face cards."""
import hashlib
import json
from importlib.resources import files
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from common.provenance import digest


def default_config(planner_mode="intent"):
    """Return a fresh copy so callers cannot mutate the package defaults."""
    names = {"intent": "default_pipeline.json", "recipe": "recipe_pipeline.json"}
    if planner_mode not in names:
        raise ValueError("planner_mode must be 'intent' or 'recipe'")
    return json.loads(files("inference").joinpath(names[planner_mode]).read_text())


def checked_file(path, expected):
    path = Path(path)
    if digest(path) != expected:
        raise ValueError(f"Checkpoint hash differs from the pinned default: {path.name}")
    return path


def generator_checkpoint(config=None):
    if config is None:
        config = default_config()
    spec = config["generator"]
    path = hf_hub_download(spec["repo_id"], spec["filename"], revision=spec["revision"])
    return checked_file(path, spec["sha256"])


def intent_asset(name, config):
    """Load the package copy of a frozen input, verified against the v8 release."""
    expected = config["intent_assets"]["files"][name]
    data = files("planner").joinpath(name).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Intent asset hash differs from the pinned release: {name}")
    return json.loads(data)


def planner_checkpoints(config=None):
    if config is None:
        config = default_config()
    base, adapter = config["base"], config["adapter"]
    root = snapshot_download(base["repo_id"], revision=base["revision"],
                             allow_patterns=[base["subfolder"] + "/*"])
    adapter_root = Path(snapshot_download(adapter["repo_id"], revision=adapter["revision"],
                                         allow_patterns=list(adapter["files"])))
    for name, expected in adapter["files"].items():
        checked_file(adapter_root / name, expected)
    return Path(root) / base["subfolder"], adapter_root
