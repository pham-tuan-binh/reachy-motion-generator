"""Default model resolution, state propagation and numerical provenance without downloads."""
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from common.provenance import digest
from inference import defaults
from inference import pipeline as pipeline_module
from planner.state import ORIGIN_FIELDS

POSE = {"roll": 1., "pitch": -4., "yaw": 10., "body": 5., "right": -10., "left": 10., "x": .001, "y": 0., "z": 0.}
RECIPE = "go 1 y=5 E=0 | go 1 y=0 E=0"
INTENT = {"gesture": "look_left", "endpoint": "absolute", "absolute_deg": 15.0}
COMPILED = "hold 0.08 E=0 | go 5.4 y=5 b=5 E=0 | hold 0.52 E=0"


def test_default_selection_is_pinned_and_independent_between_calls():
    config = defaults.default_config()
    assert config["generator"]["kind"] == "residual"
    assert config["generator"]["step"] == 500
    assert config["state_contract"] == "settled-relative-v1"
    for name in ("base", "adapter", "generator"):
        revision = config[name]["revision"]
        assert len(revision) == 40 and int(revision, 16) > 0
    config["generator_settings"]["cfg"] = 99
    assert defaults.default_config()["generator_settings"]["cfg"] == 1.


def test_generator_download_uses_pin_and_rejects_corruption(monkeypatch, tmp_path):
    checkpoint = tmp_path / "generator.pt"
    checkpoint.write_bytes(b"checked model bytes")
    config = defaults.default_config()
    config["generator"]["sha256"] = digest(checkpoint)
    monkeypatch.setattr(defaults, "default_config", lambda: copy.deepcopy(config))
    calls = []

    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(checkpoint)

    monkeypatch.setattr(defaults, "hf_hub_download", download)
    assert defaults.generator_checkpoint() == checkpoint
    spec = config["generator"]
    assert calls == [(spec["repo_id"], spec["filename"], {"revision": spec["revision"]})]
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash differs"):
        defaults.generator_checkpoint()


def test_planner_download_pins_both_models_and_checks_adapter_files(monkeypatch, tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    (base / "planner").mkdir(parents=True)
    adapter.mkdir()
    config = defaults.default_config()
    for name in config["adapter"]["files"]:
        (adapter / name).write_text(name)
        config["adapter"]["files"][name] = digest(adapter / name)
    monkeypatch.setattr(defaults, "default_config", lambda: copy.deepcopy(config))
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(base if repo == config["base"]["repo_id"] else adapter)

    monkeypatch.setattr(defaults, "snapshot_download", download)
    assert defaults.planner_checkpoints() == (base / "planner", adapter)
    assert calls == [
        (config["base"]["repo_id"], {"revision": config["base"]["revision"], "allow_patterns": ["planner/*"]}),
        (config["adapter"]["repo_id"], {"revision": config["adapter"]["revision"],
                                         "allow_patterns": list(config["adapter"]["files"])}),
    ]
    (adapter / "adapter_model.safetensors").write_bytes(b"wrong weights")
    with pytest.raises(ValueError, match="hash differs"):
        defaults.planner_checkpoints()


@pytest.fixture
def stub_pipeline(monkeypatch):
    from planner import retained

    calls = {"planner_load": [], "generator_load": [], "planner": [], "expansion": [], "sample": []}
    network, stats = SimpleNamespace(maxlen=720), {"normalization": "fixture"}
    metadata = {"version": "state-conditioned-generator-v4", "kind": "residual", "device": "cpu"}
    actual_expand = pipeline_module.recipe_to_plan

    def load_decoder(self, device, config):
        calls["planner_load"].append((device, config["adapter"]["repo_id"]))

    def complete(self, messages):
        calls["planner"].append(copy.deepcopy(messages))
        if isinstance(self, retained.IntentPlanner):
            return json.dumps(INTENT)
        return json.dumps({"idea": "Look left and return", "recipe": RECIPE})

    def load(checkpoint, device):
        calls["generator_load"].append((checkpoint, device))
        return network, stats, metadata

    def expand(recipe, seed, mode):
        calls["expansion"].append((recipe, seed, mode))
        return actual_expand(recipe, seed, mode)

    def sample(net, normalization, plans, origins, seeds, **kwargs):
        calls["sample"].append({"net": net, "stats": normalization, "plans": plans,
                                    "origins": copy.deepcopy(origins), "seeds": seeds, "settings": kwargs})
        return [np.arange(len(pipeline_module.frames(plans[0])) * 9, dtype=np.float32).reshape(-1, 9) / 1000]

    monkeypatch.setattr(retained.TextDecoder, "__init__", load_decoder)
    monkeypatch.setattr(retained.TextDecoder, "complete", complete)
    monkeypatch.setattr(pipeline_module, "generator_checkpoint", lambda config: Path("pinned-generator.pt"))
    monkeypatch.setattr(pipeline_module, "load_generator", load)
    monkeypatch.setattr(pipeline_module, "recipe_to_plan", expand)
    monkeypatch.setattr(pipeline_module, "sample_batch", sample)
    return pipeline_module.MotionPipeline(device="cpu"), calls, network, stats


@pytest.mark.parametrize("pose", [None, {}, {**POSE, "yaw": True}, {**POSE, "yaw": "10"},
                                  {**POSE, "x": float("inf")}, {**POSE, "pitch": float("nan")},
                                  {**POSE, "extra": 0}, {**POSE, "yaw": 10**1000}])
def test_bad_state_rejected_before_loading_any_model(stub_pipeline, pose):
    pipeline, calls, _, _ = stub_pipeline
    with pytest.raises(ValueError):
        pipeline.generate("Look left", current_pose=pose)
    assert not calls["planner_load"] and not calls["generator_load"]


@pytest.mark.parametrize("prompt,seed", [("", 1), (" ", 1), (None, 1), ("Look left", True),
                                       ("Look left", -1), ("Look left", 2**63), ("Look left", 1.5)])
def test_bad_request_rejected_before_loading_any_model(stub_pipeline, prompt, seed):
    pipeline, calls, _, _ = stub_pipeline
    with pytest.raises(ValueError):
        pipeline.generate(prompt, current_pose=POSE, seed=seed)
    assert not calls["planner_load"] and not calls["generator_load"]


def test_default_pipeline_forwards_state_and_exact_sampling_contract(stub_pipeline):
    pipeline, calls, network, stats = stub_pipeline
    pose = dict(POSE)
    result = pipeline.generate("Look left and return", current_pose=pose, seed=123)
    pipeline.generate("Look left again", current_pose=pose, seed=124)
    assert calls["planner_load"] == [("cpu", "diwank/reachy-mini-gesture-intent-lora-v8-20261117")]
    assert calls["generator_load"] == [(Path("pinned-generator.pt"), "cpu")]
    assert json.loads(calls["planner"][0][-1]["content"]) == {"instruction": "Look left and return", "current_pose": POSE}
    assert result.receipt["intent"] == INTENT
    assert result.receipt["recipe"] == COMPILED
    assert calls["expansion"] == [(COMPILED, 123, "nominal"), (COMPILED, 124, "nominal")]
    sample = calls["sample"][0]
    assert sample["net"] is network and sample["stats"] is stats
    assert sample["origins"] == [POSE] and sample["seeds"] == [123]
    assert sample["settings"] == {"steps": 8, "cfg": 1., "lowpass_hz": 4.}
    assert result.receipt["pipeline"] == defaults.default_config()
    assert result.receipt["origin"] == POSE and result.receipt["seed"] == 123
    assert result.receipt["shape"] == list(result.trajectory.shape)
    assert not result.receipt["anchored_to_origin"]
    assert not result.receipt["safety_checked"]
    assert not result.receipt["physical_robot_tested"]
    pose["yaw"] = -40.
    assert result.receipt["origin"] == POSE


def test_too_long_recipe_is_rejected_without_sampling(stub_pipeline):
    pipeline, calls, network, _ = stub_pipeline
    network.maxlen = 2
    with pytest.raises(ValueError, match="maximum duration"):
        pipeline.generate("Look left", current_pose=POSE)
    assert not calls["sample"]


def test_result_save_preserves_native_output_and_hash_bound_receipt(stub_pipeline, tmp_path):
    pipeline, _, _, _ = stub_pipeline
    result = pipeline.generate("Look left", current_pose=POSE)
    out = tmp_path / "nested" / "result.npz"
    receipt_path = result.save(out)
    assert receipt_path == out.with_suffix(".json")
    with np.load(out, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["trajectory"], result.trajectory)
        np.testing.assert_array_equal(saved["origin"], np.array([POSE[key] for key in ORIGIN_FIELDS], dtype=np.float32))
    receipt = json.loads(receipt_path.read_text())
    assert receipt == {**result.receipt, "trajectory_sha256": digest(out)}
    with pytest.raises(FileExistsError):
        result.save(out)
    out.unlink()
    with pytest.raises(FileExistsError):
        result.save(out)  # The receipt alone also protects the previous result.
    with pytest.raises(ValueError, match="npz"):
        result.save(tmp_path / "bad.json")


def test_cli_requires_explicit_pose_before_any_model_load(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "inference.pipeline", "--prompt", "Look left", "--out", str(tmp_path / "out.npz")],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30, check=False,
    )
    assert completed.returncode == 2
    assert "--origin" in completed.stderr and "required" in completed.stderr
    assert not (tmp_path / "out.npz").exists()


def test_mutating_returned_pipeline_config_cannot_change_sampling_or_provenance(stub_pipeline):
    pipeline, calls, _, _ = stub_pipeline
    config = pipeline.config
    config["generator_settings"]["mode"] = "randomized"
    config["generator_settings"]["cfg"] = 99.
    config["generator"]["sha256"] = "untrusted"
    result = pipeline.generate("Look left", current_pose=POSE)
    assert calls["expansion"] == [(COMPILED, 123, "nominal")]
    assert calls["sample"][0]["settings"] == {"steps": 8, "cfg": 1., "lowpass_hz": 4.}
    assert result.receipt["pipeline"] == defaults.default_config()


def test_recipe_only_default_resolves_and_verifies_selected_checkpoint(monkeypatch, tmp_path):
    import torch

    from generator.conditioned import checkpoint_payload, create_model
    from generator.conditioned_cli import sample_recipe

    checkpoint = tmp_path / "generator.pt"
    net = create_model("residual", config={"d": 16, "heads": 2, "layers": 1, "maxlen": 80})
    stats = {"MU": [0.] * 9, "SD": [1.] * 9, "PMU": [0.] * 8, "PSD": [1.] * 8}
    torch.save(checkpoint_payload(net, stats), checkpoint)
    config = defaults.default_config()
    config["generator"]["sha256"] = digest(checkpoint)
    monkeypatch.setattr(defaults, "default_config", lambda: copy.deepcopy(config))
    calls = []

    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(checkpoint)

    monkeypatch.setattr(defaults, "hf_hub_download", download)
    result = sample_recipe(None, RECIPE, POSE, tmp_path / "sample.npz")
    spec = config["generator"]
    assert calls == [(spec["repo_id"], spec["filename"], {"revision": spec["revision"]})]
    assert result["checkpoint_sha256"] == spec["sha256"]
    assert result["settings"] == {"mode": "nominal", "steps": 8, "cfg": 1., "lowpass_hz": 4.}
    assert result["origin"] == POSE and not result["anchored_to_origin"]


@pytest.fixture
def stub_retained_planner():
    import torch

    from planner.retained import RetainedPlanner

    calls = {"template": [], "generate": []}
    control = {"input_length": 10, "tokens": [7, 2], "completion": json.dumps({"idea": "Look left", "recipe": RECIPE})}

    class Encoding(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = pad_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            calls["template"].append((copy.deepcopy(messages), kwargs))
            return "rendered state-aware prompt"

        def __call__(self, text, **kwargs):
            assert text == "rendered state-aware prompt"
            assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
            return Encoding(input_ids=torch.ones((1, control["input_length"]), dtype=torch.long))

        def decode(self, tokens, **kwargs):
            assert tokens == control["tokens"] and kwargs == {"skip_special_tokens": True}
            return control["completion"]

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[3, 2])

        def generate(self, **kwargs):
            calls["generate"].append(kwargs)
            return torch.cat([kwargs["input_ids"], torch.tensor([control["tokens"]])], dim=1)

    planner = object.__new__(RetainedPlanner)
    planner.settings = defaults.default_config()["planner_settings"]
    planner.tokenizer, planner.model, planner.device = Tokenizer(), Model(), "cpu"
    return planner, calls, control


def test_retained_planner_uses_greedy_state_prompt_and_preserves_stop_tokens(stub_retained_planner):
    from planner.state import state_messages

    planner, calls, _ = stub_retained_planner
    idea, recipe, completion = planner.plan("Look left", POSE)
    assert idea == "Look left" and recipe == RECIPE
    assert json.loads(completion)["recipe"] == RECIPE
    assert calls["template"] == [(state_messages("Look left", POSE),
                                    {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})]
    assert len(calls["generate"]) == 1
    settings = {key: value for key, value in calls["generate"][0].items() if key != "input_ids"}
    assert settings == {"max_new_tokens": 512, "do_sample": False, "eos_token_id": [2, 3],
                            "pad_token_id": 2, "use_cache": True}


def test_retained_planner_rejects_long_prompt_without_truncating_state(stub_retained_planner):
    planner, calls, control = stub_retained_planner
    control["input_length"] = 1537
    with pytest.raises(ValueError, match="refusing state truncation"):
        planner.plan("Look left", POSE)
    assert not calls["generate"]


@pytest.mark.parametrize("tokens,completion,message", [
    ([7], json.dumps({"recipe": RECIPE}), "truncated completion"),
    ([7, 2], "invalid JSON", "valid recipe"),
    ([7, 2], json.dumps({"recipe": "not a DSL recipe"}), "valid recipe"),
])
def test_retained_planner_rejects_failed_completion_without_retry(stub_retained_planner, tokens, completion, message):
    planner, calls, control = stub_retained_planner
    control.update(tokens=tokens, completion=completion)
    with pytest.raises(ValueError, match=message):
        planner.plan("Look left", POSE)
    assert len(calls["generate"]) == 1

def test_explicit_recipe_mode_uses_previous_adapter_and_receipt(stub_pipeline):
    from planner.state import state_messages

    _, calls, _, _ = stub_pipeline
    pipeline = pipeline_module.MotionPipeline(device="cpu", planner_mode="recipe")
    result = pipeline.generate("Look left and return", current_pose=POSE)
    assert calls["planner_load"] == [("cpu", "diwank/reachy-mini-planner-repair-v6-20261116")]
    assert calls["planner"] == [state_messages("Look left and return", POSE)]
    assert result.receipt["pipeline"]["id"] == "retained-rl-residual-500-v1"
    assert result.receipt["recipe"] == RECIPE and "intent" not in result.receipt


def test_invalid_intent_fails_before_generator_load(stub_pipeline, monkeypatch):
    from planner.retained import TextDecoder

    monkeypatch.setattr(TextDecoder, "complete", lambda self, messages: '{"gesture":"unsupported"}')
    pipeline, calls, _, _ = stub_pipeline
    with pytest.raises(ValueError, match="Unsupported"):
        pipeline.generate("Fly away", current_pose=POSE)
    assert len(calls["planner_load"]) == 1
    assert not calls["generator_load"] and not calls["sample"]


def test_intent_asset_binding_and_pose_offsets(stub_pipeline):
    config = defaults.default_config()
    config["intent_assets"]["files"]["intent_examples.json"] = "0" * 64
    with pytest.raises(ValueError, match="hash differs"):
        defaults.intent_asset("intent_examples.json", config)

    pipeline, _, _, _ = stub_pipeline
    result = pipeline.generate("Look left", current_pose={**POSE, "yaw": -4.})
    assert "y=19 b=19" in result.receipt["recipe"]


def test_unknown_planner_mode_rejected():
    with pytest.raises(ValueError, match="planner_mode"):
        pipeline_module.MotionPipeline(planner_mode="auto")
