# Reachy Mini text-to-motion

The default pipeline converts an instruction and a settled robot pose into a
25 Hz trajectory. Binh's 4B planner with the **v8 step-20 intent LoRA** extracts
the gesture and its constraints. A deterministic compiler converts the intent
into a relative recipe. The **v4 residual generator at checkpoint 500** adds
bounded motion detail to the nominal plan. The package pins the model revisions
and verifies the adapter, generator and bundled intent assets by hash.
[Base model](https://huggingface.co/binhpham/reachy-mini-motion-planner-4b) ·
[Intent adapter](https://huggingface.co/diwank/reachy-mini-gesture-intent-lora-v8-20261117) ·
[Residual generator](https://huggingface.co/diwank/reachy-mini-trajectory-generator-v4-20261116)

## Install and generate

```bash
pip install 'git+https://github.com/creatorrr/reachy-motion-generator.git'
```

The first call downloads the pinned public models anonymously. Use a CUDA GPU
for the 4B planner. The package also supports CPU inference.

```python
from inference import MotionPipeline

# Replace these example values with the robot's measured, settled pose.
pose = dict(roll=0., pitch=0., yaw=6., body=0.,
            right=0., left=0., x=0., y=0., z=0.)

pipeline = MotionPipeline()
result = pipeline.generate("Look left and hold at an absolute yaw of +18 degrees.", current_pose=pose, seed=123)
result.save("runs/look.npz")
```

With this example pose the compiler requests a +12° yaw change. To use the CLI,
save the same nine fields as `pose.json` and run the following command.

```bash
reachy-motion --prompt 'Look left and hold at an absolute yaw of +18 degrees.' \
  --origin-file pose.json --out runs/look.npz --device cuda
```

`python -m inference.pipeline` and the checkout's `python pipeline.py` use the
same default. Supply all nine settled pose fields, with angles in degrees and
translations in metres. Velocity and interruption history are outside the
settled-pose contract. [Axis conventions](docs/STATE_CONDITIONED.md#both-state-interfaces)

The NPZ contains an unanchored native `(T,9)` trajectory and the starting pose.
The JSON receipt records the model identities, intent, recipe and settings.
Output channels are xyz metres, roll/pitch/yaw Euler radians, signed right/left
antenna radians and body-yaw radians. Before playback, the application must
derive offsets from the first frame, anchor them to its measured pose and apply
its clearance, IK and controller checks.

## Planner modes

The default handles one of ten familiar gestures per request. Use a nod, shake,
wiggle, sleepy head droop, left or right tilt, or a look left, right, up or down.
The model extracts any specified endpoint, count, angle and duration constraints.
An unsupported model intent or infeasible recipe raises an error. The caller
chooses whether to try a different planner.

Select `MotionPipeline(planner_mode="recipe")` or pass `--planner-mode recipe` to
`reachy-motion` for the v1.3 state-aware planner that writes free-form recipes.
This mode also requires a settled pose and uses residual checkpoint 500.

## Generate from a recipe

This skips the text planner and uses the same default residual checkpoint:

```bash
python -m generator.conditioned_cli \
  --recipe 'go 1 y=5 E=0 | go 1 y=0 E=0' \
  --origin-file pose.json --out runs/recipe.npz
```

Supply `--checkpoint path/to/model.pt` to choose another generator explicitly.
See [the v4 guide](docs/STATE_CONDITIONED.md) for lower-level APIs, checkpoints,
portable training and numerical validation.

## Evidence for this default

In numerical replay on **200 held-out tasks** across ten gesture families, the
v8 default achieved **90.38% joint task and safety success**. The previous
recipe planner scored **28.25%** on the same tasks with the same residual
generator. Direct playback of the v8 recipes scored 90.50%. V8 produced a recipe
for 198 of 200 tasks. Of the 792 generated draws, 69 failed raw mechanical checks.
Physical motion quality and safety still require attended tests.
[Held-out evaluation](https://huggingface.co/datasets/diwank/reachy-mini-gesture-intent-v8-20261117/blob/1fd034f9bab46daf65b7b938f0f4f9bbaaa0c18e/report.md)

## Legacy serving and upstream work

The original Docker/HTTP server, `MotionEngine` and explicit
`python -m inference.legacy_pipeline` remain available with their original
sampling path. They do not forward the settled pose required by this default.
Their usage, benchmarks, renderer and architecture are documented in
[Legacy upstream pipeline and serving](docs/LEGACY.md).

This fork builds on [Pham Tuan Binh's implementation](https://github.com/pham-tuan-binh/reachy-motion-generator)
at `18ff98574a8f0652b21db32f204a8029f8930d88`. The state-conditioned experiments
and port derive from [Loopy](https://github.com/creatorrr/loopy).
Apache-2.0; see [LICENSE](LICENSE) and the [attribution](docs/STATE_CONDITIONED.md#attribution).
