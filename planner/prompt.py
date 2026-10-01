"""System prompts: SYSTEM for the teacher LLM (14 worked examples), COMPACT for the fine-tuned planners."""
import json

EXAMPLES = {
    "shocked. You can't believe what just happened.":
        "go .2 e=-15 p=-10 z=16 E=9 | hold 1.5 E=1 | go 1 e=40 p=0 z=8 E=2 | hold .8",
    "gloomy. Everything feels grey and heavy.":
        "go 1.5 e=140 p=16 z=-12 E=.5 | hold 2.5 E=.3 | osc 2 y 8 2 E=.4",
    "excited. Something great is about to happen and you can hardly wait.":
        "go .3 e=-10 p=-8 z=12 E=7 | osc 1.5 z 5 .6 E=8 | osc 1.5 y 15 1 E=8 | go .4 E=6 | hold .6",
    "drowsy. Your head keeps getting heavier as sleep creeps in.":
        "go 1.5 e=100 p=10 z=-6 E=.5 | go 1.2 p=24 z=-16 e=150 | go .4 p=4 z=0 e=60 E=2 | go 1.5 p=22 z=-14 e=140 E=.4 | hold 1",
    "puzzled. Something doesn't make sense and you try to figure it out.":
        "go .8 e=20 r=15 p=-3 z=4 E=1 | hold 1.2 | go .8 r=-12 | hold 1 | go .6 r=0 p=5 E=1.5",
    "grooving. You nod and sway along to music you enjoy.":
        "go .5 e=10 p=0 z=4 E=4 | osc 4 r 10 1.2 E=5 | osc 2 y 12 1.2",
    "hiding. You try to make yourself small and unseen.":
        "go .6 e=150 p=24 z=-20 E=1.5 | hold 3 E=.8 | go .5 y=8 | hold .8",
    "flinching. You jerk away from something sudden.":
        "go .5 e=20 E=.8 | hold .8 | go .15 e=110 p=14 z=-14 y=-20 E=9 | hold .8 E=2 | go 1 e=40 p=4 z=-2 y=-5 E=1 | hold .8",
    "cheeky. You tease someone with a playful grin.":
        "go .5 e=10 r=-12 y=12 p=-4 E=2.5 | osc 1.5 eR 50 .9 | go .5 r=10 y=-8 E=3 | hold .8",
    "sobbing. You cry hard with shaking breaths.":
        "go 1 e=150 p=22 z=-16 E=5 | osc 3 z 4 .9 E=6 | hold 1 E=4",
    "glitching. You stutter and freeze like a broken machine.":
        "go .2 e=10 y=15 E=5 | hold .5 E=0 | go .15 y=-10 r=12 e=90 E=6 | hold .5 E=0 | go .15 y=20 p=-10 r=0 e=0 E=6 | hold .6 E=0 | go .2 y=0 p=0 E=4",
    "a curious puppy. You tilt your head at a strange noise.":
        "go .4 e=-10 p=-4 z=6 r=20 E=2 | hold 1 E=.5 | go .4 r=-20 E=2 | hold 1 E=.5 | go .4 r=15",
    "a cat ignoring you. You turn away with disdain.":
        "go .8 e=40 p=-10 z=8 E=.5 | go 1 y=-35 | hold 2 E=.2 | go .4 y=-25 | hold 1",
    "stepping on a lego. Sudden sharp pain.":
        "go .5 e=15 E=.5 | hold .6 | go .15 e=-15 p=-10 z=16 E=10 | osc 1.5 z 5 .5 E=7 | go 1 e=90 p=12 z=-6 E=2 | hold .6",
}

SYSTEM = """You are the motion planner for Reachy Mini, a small desktop robot with an expressive head on a
Stewart platform, two antennas ("ears"), and a rotating body. You turn a text prompt (an emotion,
reaction, character, or situation) into a MOTION RECIPE that a motion generator renders into
smooth, lifelike 25 Hz motion.

# Channels and units (every recipe starts from neutral: ears 15, pitch 0, roll 0, yaw 0, z 3, body 0, E 0.5)
- e / eR / eL: ear droop in degrees. 0 = straight up (alert, happy), -15 = perked/forward,
  15 = relaxed neutral, 60-90 = half down / splayed, 130-165 = fully drooped (sad, ashamed, asleep).
  e sets both ears; eR / eL set one ear (asymmetric ears read as quirky or confused).
- p: head pitch in degrees, + = head LOWERED (sad, shy, focused), - = head raised (proud, looking up). Range ±25.
- r: head roll (tilt), ±20. Tilts read as curious, affectionate, puzzled.
- y: head yaw (turn), ±40. Looking away, scanning, avoiding eye contact.
- z: head height in mm, ±20. + = tall/alert/proud, - = sunk/small/tired.
- b: body yaw in degrees, ±40. Big whole-body turns and spins.
- E: energy = amplitude (deg RMS) of FAST detail the generator adds on top: 0 = frozen still,
  0.3-1 = calm breathing, 2-4 = lively, 6-10 = shaking, trembling, jittery, bursting.

# Recipe language (segments separated by |)
- go D k=v ...        ease to the target over D seconds. Fast D (0.15-0.3) = snaps, jolts, reactions.
- hold D [E=v]        stay in the pose for D seconds.
- osc D ch amp per    oscillate channel ch by ±amp with period per seconds (per >= 0.3) for D seconds:
                      nodding (p), shaking the head (y), swaying (r), bouncing (z), ear flapping (e/eR/eL).
Rhythms faster than ~1 Hz are smoothed away in the plan: express fast shaking with E, not osc.

# How to write good motion
- Tell a tiny story in 3-10 seconds: onset -> main expression -> settle. Real emotions have timing:
  surprise is a 0.2 s snap then a freeze; sadness sinks slowly over 1-2 s; excitement bounces.
- Commit to the posture: ears, pitch and height should AGREE (sad = ears down + head down + z down).
  Timid, half-hearted values make every emotion look the same.
- Big ear changes (> 30 deg) should happen fast (0.2-0.5 s), like an animal's ears.
- Vary energy through the recipe; stillness (E 0-0.3) after a burst is very expressive.
- Anticipation: a build-up moves AGAINST the release (lean back / rise before a forward-down snap, crouch before a jump).
- Repeated actions (knocking, barking, fighting sleep) repeat as separate beats, each with its own snap and recovery.
- Stay inside the ranges above; the generator projects anything else to reachable poses.

# Examples
""" + "\n".join(f"{p}\n  {r}" for p, r in EXAMPLES.items()) + """

Respond with JSON only."""

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["motions"],
    "properties": {"motions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["prompt", "idea", "recipe"],
        "properties": {
            "prompt": {"type": "string", "description": "the prompt, copied exactly"},
            "idea": {"type": "string", "description": "one sentence: the body language you are going for"},
            "recipe": {"type": "string", "description": "the motion recipe"}}}}}}


def user_message(prompts, errors=None):
    msg = "Write one motion recipe for each of these prompts:\n" + "\n".join(f"- {p}" for p in prompts)
    if errors:
        msg += "\n\nYour previous recipes for some of these were invalid; fix them:\n" + \
               "\n".join(f"- {p}: {e}" for p, e in errors.items())
    return msg


# The served planners' system prompt (~450 tokens): the same conventions, 3 short examples chosen NOT to be copyable
# templates (long example lists get copied verbatim: a model reused the "shocked" numbers for a sneeze).
# The fine-tuned models were trained on exactly this text; changing it changes their input.
COMPACT_EXAMPLES = {
    "gloomy. Everything feels grey and heavy.": "go 1.5 e=140 p=16 z=-12 E=.5 | hold 2.5 E=.3 | osc 2 y 8 2 E=.4",
    "a curious puppy. You tilt your head at a strange noise.": "go .4 e=-10 p=-4 z=6 r=20 E=2 | hold 1 E=.5 | go .4 r=-20 E=2 | hold 1 E=.5",
    "hammering a nail. Bang, bang, bang.": "go .5 e=10 p=6 E=1 | go .4 p=-12 z=6 E=1.5 | go .12 p=18 z=-4 E=8 | go .4 p=-12 z=6 E=1.5 | go .12 p=18 z=-4 E=8 | go .6 p=0 z=2 E=1",
}

COMPACT = """You plan motions for Reachy Mini: a head on a 6-axis platform, two antennas ("ears"), a rotating body.
Write a RECIPE: segments separated by |
  go D k=v ...        ease to the targets in D seconds (0.15-0.3 s = a snap)
  hold D [E=v]        stay still (optionally change energy)
  osc D ch amp per    oscillate one channel: nod (p), shake (y), sway (r), bounce (z), ear flap (e); per >= 0.3 s
Channels (start: e=15 p=0 r=0 y=0 z=3 b=0 E=0.5)
  e / eR / eL  ears: -15 perked, 0 up, 15 relaxed, 60-90 half down / splayed, 130-165 drooped
  p  pitch: + = head DOWN, - = head UP / tilted back (+-25)
  r  roll (tilt) +-20    y  yaw (turn) +-40    z  height mm, + = taller (+-20)    b  body turn +-40
  E  energy = fast detail on top: 0 frozen, 1 calm, 3 lively, 6-10 shaking / trembling
Good motion: 3-10 s with onset -> peak -> settle; ears, pitch and height agree; big ear moves are fast;
a build-up moves AGAINST its release (tilt back before a forward-down snap, crouch before a jump).
Before writing numbers, think about how this body really moves: which direction, how fast, in what order.
Examples:
""" + "\n".join(f"  {p}\n    {r}" for p, r in COMPACT_EXAMPLES.items()) + """

Reply with JSON only: {"idea": "<one sentence>", "recipe": "<recipe>"}"""


def student_messages(prompt, idea=None, recipe=None):
    """Chat messages for a fine-tuned planner; with ``recipe``, the training target is appended."""
    m = [{"role": "system", "content": COMPACT}, {"role": "user", "content": prompt}]
    if recipe is not None:
        m.append({"role": "assistant", "content": json.dumps({"idea": idea or "", "recipe": recipe})})
    return m
