"""Render gesture-intent prompts and compile settled-pose intents into motion recipes."""

from __future__ import annotations

import json
import math

from common.motion import FPS
from planner.dsl import LIMITS
from planner.state import validate_pose

INITIAL_HOLD_FRAMES = 2
FINAL_HOLD_FRAMES = 13
AXIS_CHANNELS = {"pitch": "p", "roll": "r", "yaw": "y"}
CYCLIC = {"nod": ("pitch", (1,)), "shake": ("yaw", (1, -1)), "wiggle": ("yaw", (1, -1))}
DIRECTIONS = {
    "tilt_left": ("roll", -1),
    "tilt_right": ("roll", 1),
    "look_up": ("pitch", -1),
    "look_down": ("pitch", 1),
    "look_left": ("yaw", 1),
    "look_right": ("yaw", -1),
    "sleepy": ("pitch", 1),
}
FAMILIES = tuple(CYCLIC) + tuple(DIRECTIONS)
FIELDS = {
    "gesture",
    "endpoint",
    "cycles",
    "absolute_deg",
    "excursion_deg",
    "duration_s",
    "tolerance_deg",
}
SYSTEM = """Translate one Reachy Mini instruction to one JSON gesture intent. Use only these keys:
gesture: nod, shake, wiggle, look_up, look_down, look_left, look_right, tilt_left, tilt_right, or sleepy.
endpoint: "hold" for a directional hold, "absolute" for a named final robot angle, or "return" for returning to the starting angle.
Optional fields: cycles (whole cycles), absolute_deg (signed final robot angle), excursion_deg ([minimum,maximum] in degrees), duration_s ([minimum,maximum] in seconds), tolerance_deg (stated endpoint tolerance).
Copy only constraints given by the instruction. Copy absolute targets exactly; a compiler handles current_pose. For a single specified excursion or duration use equal bounds. A nod is a down-and-back dip; a shake says no; a wiggle is playful. All angles are degrees: down/left/rightward tilt are positive pitch/yaw/roll respectively. Use {"gesture":"unsupported"} for incompatible instructions. Output JSON only."""


def _number(value, field):
    try:
        finite = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{field} requires a finite number")
    return value


def validate_intent(intent):
    if not isinstance(intent, dict) or intent.keys() - FIELDS:
        raise ValueError("Expected a gesture intent with schema fields only")
    intent = {key: value for key, value in intent.items() if value is not None}
    gesture, endpoint = intent.get("gesture"), intent.get("endpoint")
    if gesture not in FAMILIES or endpoint not in ("hold", "absolute", "return"):
        raise ValueError("Unsupported gesture or endpoint")
    if ("absolute_deg" in intent) != (endpoint == "absolute"):
        raise ValueError("Absolute endpoints require an absolute angle")
    if gesture in CYCLIC and endpoint != "return":
        raise ValueError("A cyclic gesture must return to its starting angle")
    if "cycles" in intent and (
        gesture not in CYCLIC
        or type(intent["cycles"]) is not int
        or intent["cycles"] < 1
    ):
        raise ValueError("Cycle count must be a positive integer for a cyclic gesture")
    if "absolute_deg" in intent:
        _number(intent["absolute_deg"], "absolute_deg")
    if "tolerance_deg" in intent and (
        _number(intent["tolerance_deg"], "tolerance_deg") < 0 or endpoint == "hold"
    ):
        raise ValueError(
            "Tolerance requires a defined endpoint and a nonnegative angle"
        )
    for key in ("excursion_deg", "duration_s"):
        if key in intent:
            bounds = intent[key]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"{key} must contain two bounds")
            lo, hi = (_number(value, key) for value in bounds)
            if not 0 < lo <= hi:
                raise ValueError(f"{key} requires increasing positive bounds")
    return intent


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Repeated intent field {key}")
        result[key] = value
    return result


def parse_intent(completion):
    if not isinstance(completion, str) or len(completion) > 6000:
        raise ValueError("Expected a bounded text completion")
    try:
        return validate_intent(json.loads(completion, object_pairs_hook=_unique_keys))
    except RecursionError as error:
        raise ValueError("Intent JSON exceeds the nesting limit") from error


def _within(preferred, lower, upper):
    if lower > upper:
        raise ValueError("Requested bounds are infeasible for this pose or recipe")
    return min(max(preferred, lower), upper)


def _waypoints(intent, pose, amplitude):
    gesture = intent["gesture"]
    axis = CYCLIC[gesture][0] if gesture in CYCLIC else DIRECTIONS[gesture][0]
    limit = LIMITS[AXIS_CHANNELS[axis]][1]
    lower, upper = intent.get("excursion_deg", (0.0, math.inf))
    if gesture in CYCLIC:
        pattern = CYCLIC[gesture][1]
        # A two-sided excursion is measured from the negative peak to the positive peak.
        peak = _within(amplitude, lower, min(upper, limit * len(pattern))) / len(
            pattern
        )
        count = intent.get("cycles", 1)
        if 2 * len(pattern) * count + 2 > 24:
            raise ValueError("Cycle count exceeds the 24-segment contract")
        return axis, [
            value
            for _ in range(count)
            for sign in pattern
            for value in (sign * peak, 0.0)
        ]
    sign = DIRECTIONS[gesture][1]
    if intent["endpoint"] == "absolute":
        desired = (intent["absolute_deg"] - pose[axis]) * sign
        tolerance = intent.get("tolerance_deg", 0.0)
        distance = _within(
            desired,
            max(lower, desired - tolerance),
            min(upper, limit, desired + tolerance),
        )
    else:
        distance = _within(amplitude, lower, min(upper, limit))
    if distance <= 0:
        raise ValueError(
            "Requested direction and endpoint are incompatible with the current pose"
        )
    waypoints = [sign * distance]
    if intent["endpoint"] == "return":
        waypoints.append(0.0)
    return axis, waypoints


def _format(value):
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def compile_recipe(intent, current_pose, *, defaults):
    """Emit nominal DSL from public intent, a nine-field pose, and frozen training defaults."""
    intent = validate_intent(intent)
    validate_pose(current_pose)
    spec = defaults[intent["gesture"]]
    amplitude = _number(spec["excursion_deg"], "default excursion")
    duration = _number(spec["duration_s"], "default duration")
    if min(duration, amplitude) <= 0:
        raise ValueError("Compiler defaults must be positive")
    axis, waypoints = _waypoints(intent, current_pose, amplitude)
    # Use the teacher's end holds and distribute remaining integer frames over its legs.
    lower, upper = intent.get("duration_s", (0.5, 16.0))
    holds = INITIAL_HOLD_FRAMES + FINAL_HOLD_FRAMES
    lower, upper = max(0.5, lower), min(16.0, upper)
    if lower > upper:
        raise ValueError("Duration falls outside the recipe contract")
    min_frames = max(math.ceil(lower * FPS - 1e-9), holds + 2 * len(waypoints))
    max_frames = min(math.floor(upper * FPS + 1e-9), holds + 10 * FPS * len(waypoints))
    frames = _within(
        math.floor(min(duration, 16.0) * FPS + 0.5), min_frames, max_frames
    )
    per_leg, extra = divmod(frames - holds, len(waypoints))
    segments = [f"hold {_format(INITIAL_HOLD_FRAMES / FPS)} E=0"]
    channel = AXIS_CHANNELS[axis]
    for index, target in enumerate(waypoints):
        seconds = (per_leg + (index < extra)) / FPS
        command = f"go {_format(seconds)} {channel}={_format(target)}"
        if axis == "yaw":
            command += f" b={_format(target)}"
        segments.append(command + " E=0")
    segments.append(f"hold {_format(FINAL_HOLD_FRAMES / FPS)} E=0")
    return " | ".join(segments)


def _user_message(instruction, current_pose):
    validate_pose(current_pose)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Expected a nonempty instruction")
    return {
        "role": "user",
        "content": json.dumps(
            {"instruction": instruction, "current_pose": current_pose},
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def intent_messages(instruction, current_pose, examples=()):
    """Format the same input for few-shot inference and completion-only SFT."""
    messages = [{"role": "system", "content": SYSTEM}]
    for example in examples:
        if example.get("split") != "train":
            raise ValueError(
                "Only training demonstrations can enter the extraction prompt"
            )
        messages.append(_user_message(example["instruction"], example["origin"]))
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    validate_intent(example["intent"]), separators=(",", ":")
                ),
            }
        )
    messages.append(_user_message(instruction, current_pose))
    return messages
