"""Settled-pose prompt contract used by the tuned text planners.

This is prompting, not motion anchoring or a runtime safety guard.
"""
import json
import math

from planner.prompt import student_messages

CONTRACT = "settled-relative-v1"
ORIGIN_FIELDS = ("roll", "pitch", "yaw", "body", "right", "left", "x", "y", "z")
STATE_NOTE = 'State-aware motion contract (settled-relative-v1):\nThe robot is settled at the supplied current_pose. Angles are degrees and x/y/z are metres.\nPositive pitch looks down, positive yaw looks left, positive roll tilts right.\nYour recipe still starts at the DSL neutral pose. Playback converts the generated clip\ninto offsets from its first frame and adds them to current_pose; recipe angles are NOT\nabsolute robot targets. Preserve the requested gesture after this anchoring.\nChoose amplitude, timing and body coordination to fit the current pose. Head yaw is\nworld-frame; move body yaw with head yaw when needed for head-shell clearance.\nDo not assume lifting the head increases the modeled clearance. Avoid fast moves,\nlarge translations and excessive energy. Return only the usual JSON with idea and recipe.\n'


def validate_pose(current_pose):
    if not isinstance(current_pose, dict) or set(current_pose) != set(ORIGIN_FIELDS):
        raise ValueError("current_pose requires all nine named pose fields")
    try:
        valid = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in current_pose.values())
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("current_pose must contain finite numeric degrees/metres")


def state_messages(prompt, current_pose):
    validate_pose(current_pose)
    messages = student_messages(prompt)
    messages[0]["content"] += "\n\n" + STATE_NOTE
    messages[-1]["content"] += "\ncurrent_pose=" + json.dumps(current_pose, sort_keys=True)
    return messages
