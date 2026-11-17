import hashlib
import json
from importlib.resources import files

import pytest

from planner.dsl import check
from planner.intent import compile_recipe, intent_messages, parse_intent

POSE = {
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 6.0,
    "body": 0.0,
    "right": 0.0,
    "left": 0.0,
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
}
DEFAULTS = json.loads(files("planner").joinpath("intent_defaults.json").read_text())
EXAMPLES = json.loads(files("planner").joinpath("intent_examples.json").read_text())


@pytest.mark.parametrize(("yaw", "offset"), [(6.0, 12), (-4.0, 22)])
def test_absolute_yaw_resolves_from_settled_pose(yaw, offset):
    intent = {"gesture": "look_left", "endpoint": "absolute", "absolute_deg": 18}
    recipe = compile_recipe(intent, {**POSE, "yaw": yaw}, defaults=DEFAULTS)
    assert recipe == f"hold 0.08 E=0 | go 5.4 y={offset} b={offset} E=0 | hold 0.52 E=0"
    assert check(recipe) is None


def test_counted_nod_and_duration_use_whole_frames():
    intent = {
        "gesture": "nod",
        "endpoint": "return",
        "cycles": 2,
        "duration_s": [3.49, 3.54],
    }
    recipe = compile_recipe(intent, POSE, defaults=DEFAULTS)
    assert recipe == (
        "hold 0.08 E=0 | go 0.76 p=7.5 E=0 | go 0.72 p=0 E=0 | "
        "go 0.72 p=7.5 E=0 | go 0.72 p=0 E=0 | hold 0.52 E=0"
    )
    assert check(recipe) is None


@pytest.mark.parametrize(
    "completion",
    [
        'extra {"gesture":"tilt_left","endpoint":"hold"}',
        '{"gesture":"nod","gesture":"shake","endpoint":"return"}',
        '{"gesture":"nod","endpoint":"return","cycles":2.0}',
        '{"gesture":"unsupported"}',
        '{"gesture":"look_left","endpoint":"absolute","absolute_deg":NaN}',
        '{"gesture":"look_up","endpoint":"hold","extra":1}',
        '{"gesture":"wiggle","endpoint":"hold"}',
    ],
)
def test_strict_intent_parser_rejects_invalid_completions(completion):
    with pytest.raises(ValueError):
        parse_intent(completion)


def test_sparse_schema_accepts_null_for_unstated_fields():
    assert parse_intent('{"gesture":"tilt_left","endpoint":"hold","cycles":null}') == {
        "gesture": "tilt_left",
        "endpoint": "hold",
    }


@pytest.mark.parametrize(
    "constraints",
    [
        {"duration_s": [3.49, 3.51]},
        {"absolute_deg": 18, "endpoint": "absolute"},
        {"excursion_deg": [51, 60]},
    ],
)
def test_compiler_reports_incompatible_constraints(constraints):
    with pytest.raises(ValueError):
        compile_recipe(
            {"gesture": "look_right", "endpoint": "hold", **constraints},
            POSE,
            defaults=DEFAULTS,
        )


def test_few_shot_prompt_matches_published_checkpoint_format():
    messages = intent_messages("Turn left and hold.", POSE, EXAMPLES)
    assert hashlib.sha256(
        json.dumps(messages, sort_keys=True).encode()
    ).hexdigest() == (
        "187816f3334d77dcd1513ddbb760630e67517a01c53e9141b8ad5106bd618527"
    )


def test_prompt_rejects_incomplete_pose_and_heldout_examples():
    with pytest.raises(ValueError):
        intent_messages("Turn left.", {"yaw": 6.0}, EXAMPLES)
    with pytest.raises(ValueError):
        intent_messages("Turn left.", POSE, [{**EXAMPLES[0], "split": "test"}])
