# Reachy Mini text-to-motion

The primary default is the **retained state-aware RL planner plus v4 bounded
residual generator, checkpoint 500**. An instruction and a settled current pose
produce a recipe, a deterministic nominal plan, then a 25 Hz numerical trajectory.
The package pins the base model, adapter and generator revisions and checks the
adapter and generator hashes when loading.

- **Planner:** [retained RL adapter](https://huggingface.co/diwank/reachy-mini-planner-repair-v6-20261116)
  on [binhpham's 4B base](https://huggingface.co/binhpham/reachy-mini-motion-planner-4b).
  The adapter is the unchanged initializer retained by the v6 experiment (step 0).
- **Generator:** [v4 residual-500](https://huggingface.co/diwank/reachy-mini-trajectory-generator-v4-20261116).
- **Settings:** greedy planning, no retries, at most 512 new tokens; nominal recipe
  expansion; eight generator steps, CFG 1.0 and 4 Hz filtering.

## Install and generate

```bash
pip install 'git+https://github.com/creatorrr/reachy-motion-generator.git'
```

The first call downloads the pinned public models. No API key is required.
Use a CUDA GPU for the 4B planner; CPU inference is supported but slower.

```python
from inference import MotionPipeline

# Replace these example values with the robot's measured, settled pose.
pose = dict(roll=0., pitch=-4., yaw=10., body=5.,
            right=-10., left=10., x=.001, y=0., z=0.)

pipeline = MotionPipeline(device="cuda")
result = pipeline.generate("Look gently left, then return.", current_pose=pose, seed=123)
result.save("runs/look.npz")
```

Or save the nine-field pose as `pose.json` and run:

```bash
reachy-motion --prompt 'Look gently left, then return.' \
  --origin-file pose.json --out runs/look.npz --device cuda
```

`python -m inference.pipeline` and, from a checkout, `python pipeline.py` use the
same default. A current pose is required: angles are degrees, translations are
metres, and the contract assumes the robot is settled. It does not model velocity
or interruption history. See [the state contract](docs/STATE_CONDITIONED.md#both-state-interfaces)
for axis conventions.

The result is an NPZ containing an unanchored native `(T,9)` trajectory and origin,
plus a JSON receipt with model identities and inference settings. Output channels
are xyz metres, roll/pitch/yaw Euler radians, signed right/left antenna radians,
and body-yaw radians. The application must derive offsets from the first frame,
anchor them to the measured pose using its rotation convention, and apply its
clearance, IK and controller checks before playback. The package does not connect
to the robot or provide those runtime guards.

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

On the 200-task, 50-group v7 retrospective evaluation, the retained RL planner
with residual-500 scored **42.875% joint task/safety success**, versus **40.5%**
with deterministic direct-plan output. This is a rescore of existing trajectories
after evaluation-boundary fixes; it is not independent confirmation or a new
training gain. No physical robot acceptance or blinded naturalness assessment
has been established by these checks. The [v7 evaluation artifacts](https://huggingface.co/datasets/diwank/reachy-mini-eval-boundaries-v7-20261117)
record the comparison and its limitations.

The earlier generator-only 100% result supplied known-correct recipes and also
had a 100% direct-plan control. It does not measure end-to-end language planning.

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
