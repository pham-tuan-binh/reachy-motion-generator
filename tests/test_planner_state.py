"""Pose prompting reaches actual Planner calls, without loading an LLM."""
import copy
import json
import math

import pytest

from inference.engine import Planner
from planner.prompt import student_messages
from planner.state import STATE_NOTE, state_messages

POSE = {"roll": 1., "pitch": -4., "yaw": 3., "body": 2., "right": -10., "left": 10., "x": 0., "y": 0., "z": 0.}


def test_state_prompt_copies_inputs_and_keeps_original_user_request():
    before = copy.deepcopy(POSE)
    base = student_messages("Look left")
    result = state_messages("Look left", POSE)
    assert result[0]["content"] == base[0]["content"] + "\n\n" + STATE_NOTE
    assert result[-1]["content"] == base[-1]["content"] + "\ncurrent_pose=" + json.dumps(POSE, sort_keys=True)
    assert POSE == before and student_messages("Look left") == base


@pytest.mark.parametrize("pose", [{}, {**POSE, "pitch": math.nan}, {**POSE, "z": math.inf},
                                  {**POSE, "yaw": True}, {**POSE, "pitch": "-4"}, {**POSE, "extra": 1}])
def test_invalid_pose_rejected(pose):
    with pytest.raises(ValueError):
        state_messages("Look left", pose)


def test_planner_state_reaches_completion_and_retries():
    planner = object.__new__(Planner)
    planner.backend = "hf"
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)

    planner.tok = Tokenizer()
    assert json.loads(planner.text("Look left")) == student_messages("Look left")

    def complete(texts, temperature, seed):
        calls.append((texts, temperature, seed))
        return ["invalid" if len(calls) == 1 else json.dumps({"idea": "Look left", "recipe": "go 1 y=5"})]

    planner.complete = complete
    _, recipe = planner.plan("Look left", retries=1, current_pose=POSE)
    assert recipe == "go 1 y=5" and len(calls) == 2
    expected = state_messages("Look left", POSE)
    assert all(json.loads(call[0][0]) == expected for call in calls)
