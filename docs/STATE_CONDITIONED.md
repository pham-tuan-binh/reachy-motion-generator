# State-conditioned trajectory generator v4

This fork separates the reusable generator experiment from
[Loopy](https://github.com/creatorrr/loopy). It adds three flow-model variants:
`full` warm-starts the original 21.8M-parameter generator with nine zero-initialized
state input columns; `residual` learns bounded detail around a deterministic
plan; `large` trains a roughly 57M-parameter generator from scratch. The original
generator, sampler, DSL and explicit legacy planner paths remain available.
The primary package default pairs the v8 intent planner and deterministic recipe
compiler with `residual-500`.
Exact dimensions and parameter counts are recorded in each checkpoint.

The code accepts state as an input. This experiment did not include a
shuffled-origin ablation isolating the benefit of supplying the correct state.
Conditioning does not itself anchor the output to that pose or enforce runtime
safety.
An [actual-checkpoint numerical probe](STATE_CONDITIONED_STATE_RESPONSE.json)
held recipe and seed fixed while changing pose: all three trained arms changed
their output, while the original generator produced identical output. That
confirms input use, without showing better task success or safety.

## Install and download

Python 3.11+ is required by the default planner's dependencies. Numerical parity was checked with the
pilot's PyTorch 2.15.0, NumPy 2.2.6, SciPy 1.18.1 and Rust kinematics 1.0.3.

```bash
git clone https://github.com/creatorrr/reachy-motion-generator.git
cd reachy-motion-generator
pip install -e .
hf download diwank/reachy-mini-trajectory-generator-v4-20261116 \
  --revision c33fe3d2f54a6dc0d90c271a2adeda89021d7581 --local-dir bundles/generator-v4
hf download diwank/reachy-mini-generator-curriculum-v4-20261116 --repo-type dataset \
  --revision f2dc9dc620f160fbc6eefdbc2e0bb25d6024589d --local-dir runs/curriculum-v4
python -m generator.curriculum --extract runs/curriculum-v4
```

The dataset contains standalone JSONL indexes/tasks and `trajectories.tar.gz`.
Extraction verifies the exact archive member set and every member's SHA256
against the published manifests, and refuses an existing `arrays/` directory.
The loader checks training-index and NPZ hashes again. The NPZ schema is native
target `X[T,9]`, conditioning `Q[T,8]`, and deterministic backbone `B[T,9]`, all at
25 Hz. The index supplies split, source, clip/group identity and settled origin.
The published synthetic targets are constructed mathematically; real targets
come from Pollen clips projected by upstream IK. These are different sources of
evidence, and real clips are not all certified by the pilot clearance checks.

The selected checkpoint is `bundles/generator-v4/generator.pt`; best-per-arm
comparisons are in `comparisons/{full,residual,large}/generator.pt`. They are
custom PyTorch flow models, separate from the 4B text planner. Only load trusted
PyTorch checkpoint files. Download provenance and byte hashes are in
`generator-manifest.json` and the linked training receipts.

## Primary default

`MotionPipeline`, `reachy-motion`, `python -m inference.pipeline` and the root
`pipeline.py` use the 4B text decoder with the v8 step-20 LoRA. The planner
receives the instruction, the settled pose and three frozen training examples.
It emits an intent for one of ten gesture families. The compiler resolves the
endpoint against the starting pose, constructs cycles and holds, and uses
training-derived defaults when excursion or duration is omitted. An unsupported
model intent or infeasible recipe raises an error. The package pins immutable
model revisions and verifies the adapter, generator and intent assets by hash.
[V8 adapter](https://huggingface.co/diwank/reachy-mini-gesture-intent-lora-v8-20261117)

```python
from inference import MotionPipeline

pose = dict(roll=0., pitch=0., yaw=6., body=0.,
            right=0., left=0., x=0., y=0., z=0.)
pipeline = MotionPipeline()
result = pipeline.generate("Look left and hold at an absolute yaw of +18 degrees.",
                           current_pose=pose, seed=123)
result.save("runs/look.npz")
```

The compiler requests a +12° yaw change for this pose. Planning is greedy and
allows up to 512 new tokens in one attempt. Nominal recipe expansion feeds
residual checkpoint 500 with eight steps, CFG 1.0 and 4 Hz filtering. The
NPZ and JSON receipt preserve the pose, intent, recipe, model identities and
settings. The application must anchor and check the trajectory before playback.

The v1.3 state-aware planner writes a free-form recipe from the instruction and
pose. Select it explicitly with `MotionPipeline(planner_mode="recipe")` or
`reachy-motion --planner-mode recipe`. It uses the retained v6 adapter and the
same residual generator. [V6 adapter](https://huggingface.co/diwank/reachy-mini-planner-repair-v6-20261116)

## Both state interfaces

Every state object has **all nine fields**. Angles `roll`, `pitch`, `yaw`, `body`,
`right` and `left` are degrees; `x`, `y`, `z` are metres. Head yaw is world-frame;
positive pitch looks down, positive yaw looks left and positive roll tilts right.
The contract is for a settled starting pose, without velocities or live feedback.

```python
from inference.engine import Planner
from generator.conditioned import load_generator, sample_batch
from generator.plans import recipe_to_plan

pose = dict(roll=0., pitch=-4., yaw=10., body=5.,
            right=-10., left=10., x=.001, y=0., z=0.)

# For explicit merged-model experiments; use MotionPipeline for the pinned default.
planner = Planner("path/to/merged-state-aware-planner", backend="hf")
idea, recipe = planner.plan("Look left and return", seed=123, retries=0, current_pose=pose)

# Or supply a known recipe, without calling any text model.
recipe = "go 1 p=6 E=0 | go 1 p=0 E=0 | hold .5 E=0"
plan = recipe_to_plan(recipe, seed=123, mode="nominal")
net, stats, metadata = load_generator("bundles/generator-v4/generator.pt", device="cpu")
trajectory = sample_batch(net, stats, [plan], origins=[pose], seeds=[123],
                          steps=8, cfg=1., lowpass_hz=4.)[0]
```

`Planner.text(..., current_pose=pose)` and `Planner.plan(..., current_pose=pose)`
use the exact `settled-relative-v1` prompt contract from the planner experiment,
including retries. Omitting `current_pose` preserves upstream prompting.
The neural generator passes the normalized nine-field pose to every flow-model
frame; original generator checkpoints accept the same API but ignore pose.

`trajectory` is an unanchored native `(T,9)` array: xyz metres, roll/pitch/yaw
Euler radians, signed right/left antenna radians, body radians. The settled
relative playback contract first derives offsets from the clip's first frame
and applies them to the current pose with the application's rotation convention.
That anchoring, clearance admission, lag evaluation and motor writer remain the
caller's responsibility. This port includes no robot connection or playback.

The existing `MotionEngine`, `/generate-dense`, `/generate-sparse` and explicit
`python -m inference.legacy_pipeline` use the original sampling path and do not
forward a settled pose. The primary `MotionPipeline`/`pipeline.py` and the recipe
CLI below support state; the legacy HTTP routes do not.

## Numerical recipe CLI

```bash
python -m generator.conditioned_cli \
  --recipe 'go 1 p=6 E=0 | go 1 p=0 E=0 | hold .5 E=0' \
  --origin '{"roll":0,"pitch":-4,"yaw":10,"body":5,"right":-10,"left":10,"x":0.001,"y":0,"z":0}' \
  --mode nominal --seed 123 --steps 8 --cfg 1 --lowpass-hz 4 \
  --out runs/sample-v4.npz --device cpu
```

Without `--checkpoint`, the CLI downloads the package's pinned `residual-500`
checkpoint and verifies its hash. Use `--checkpoint bundles/generator-v4/generator.pt`
or another explicit path to override it. The CLI writes `trajectory` and `origin`
arrays plus a JSON receipt with the checkpoint/output hashes, recipe and settings.
It rejects overwrite and recipe truncation. It does not anchor or project the sample or send it to hardware.
`--origin-file pose.json` can replace inline JSON. `--mode randomized` reproduces
the upstream amplitude/tempo/antenna RNG stream; nominal mode disables those
jitters. Both use the pilot's 2 Hz filter and 0.25 s conditioning key interval.

## Training

Download the original bundle's generator for normalization statistics and the
`full` warm start. Keep this file separate from the selected v4 checkpoint:

```bash
hf download binhpham/reachy-mini-motion-planner-4b generator.pt \
  --revision 1b60d8f16b85473ceb20e3b62e464780c532b519 --local-dir bundles/original-4b
python -m generator.finetune --data runs/curriculum-v4 \
  --original-checkpoint bundles/original-4b/generator.pt \
  --out runs/full-v4 --kind full --device cuda \
  --steps 1000 --batch-size 16 --lr 0.0001 --real-fraction 0.2
```

For `residual` or `large`, set `--kind` accordingly and use the pilot's learning
rate `0.0003`. Each call requires a fresh output directory and writes input/source
hashes, loss settings, source mixture counts, training logs and checkpoints.
Defaults are one device, AdamW, gradient clipping at 1, FP32 weights/optimizer and
CUDA BF16 activations. `--model-config small.json` permits explicit architecture
overrides for software smokes; it changes the experiment. `--real-fraction 0`
or `1` explicitly permits a single-source curriculum.

The losses preserve the pilot's masked flow, fidelity, plan, endpoint and velocity
math. **Clearance supervision defaults to disabled with weight zero.** To enable
the approximate safety loss, supply an external calibrated file:

```bash
python -m generator.finetune ... --clearance-table path/to/calibrated-clearance.npz
```

No Loopy calibration asset is distributed or fetched by this fork. The external
NPZ needs uniform increasing axes `rel_yaw`, `pitch`, `roll`, `z_mm`, `x_mm` and a
matching five-dimensional `gap_decimm` table. The surrogate uses Euler-offset
anchoring and omits exact SO(3), IK, lag and controller checks. Without the pilot's
calibration file, this is a different training objective, explicitly recorded in
the manifest. The task construction, replay/rubric and held-out model-selection
controller remain in Loopy, rather than being represented as portable safety
evaluation here.

Residual detail is clipped after filtering to 0.5 mm translation, 1 degree head
and body rotation and 5 degree antenna rotation, with zero detail at endpoints
and 400 ms ramps. Residual training uses the attainable clipped detail target;
it is not unrestricted imitation of real clips. These architectural bounds are
not a safety proof.

## Results and evidence limits

On 200 fresh held-out tasks across ten gesture families, the v8 end-to-end
pipeline scored **90.38% joint task and safety success**. The previous v1.3
recipe planner with the same generator scored **28.25%**. Direct playback of
the v8 recipes scored 90.50%. The paired 95% interval for the v8 gain over
v1.3 was +53.00 to +70.88 percentage points across 50 pose/wording groups.
These figures measure numerical replay. Physical motion quality and safety
require attended evaluation.
[V8 held-out report](https://huggingface.co/datasets/diwank/reachy-mini-gesture-intent-v8-20261117/blob/1fd034f9bab46daf65b7b938f0f4f9bbaaa0c18e/report.md)

The earlier v7 retrospective rescored a different set of outputs. On that set,
the v1.3 pipeline scored 42.875% joint success with residual-500 and 40.5%
with direct-plan output. [V7 results](https://huggingface.co/datasets/diwank/reachy-mini-eval-boundaries-v7-20261117)

The following results are the separate, original generator-only experiment:

The [public result report](https://huggingface.co/datasets/diwank/reachy-mini-generator-curriculum-v4-20261116/blob/main/reports/analysis.json)
contains dev-selected checkpoint choices, fresh held-out results and provenance.
With known-correct recipes on the fixed mathematical pilot suite, joint success
was 14.625% for the original baseline, 24% after inference tuning, 93.75% for the
full fine-tune, 100% for the bounded residual and 93.75% for the larger model.
The **deterministic direct-plan control also scored 100%**. These are generator
recipe-fidelity checks, not an end-to-end text planner benchmark or hardware acceptance.
The residual comparison combines training with its deterministic backbone, hard
bounds and endpoint gates; an untrained bounded-residual control was not run.
Real-clip reconstruction changes were small and mixed. No human naturalness
ratings or physical robot tests are claimed.

[Validation receipts](STATE_CONDITIONED_VALIDATION.json) record bit-for-bit CPU
sampling parity for actual trained full/residual/large checkpoints with differing
plan lengths and origins. The original checkpoint is separately compared against
the pinned upstream sampler. Portable tests cover checkpoint round trips,
padding masks, nonzero training gradients, bounded residuals after filtering,
frozen input rejection, and two optimizer steps per arm using tiny models.
These short training tests establish software behavior, not pilot reproduction:

```bash
pip install pytest
MOTION_ORIGINAL_CHECKPOINT=bundles/original-4b/generator.pt python -m pytest tests -q
```

## Attribution

The original implementation and all unchanged upstream files are by
[Pham Tuan Binh](https://github.com/pham-tuan-binh/reachy-motion-generator), under
the retained Apache-2.0 [LICENSE](../LICENSE). This branch builds on upstream
commit `18ff98574a8f0652b21db32f204a8029f8930d88`, including the flow transformer,
sampler, DSL and motion conventions. The state-conditioned experiment and port
derive from [creatorrr/loopy](https://github.com/creatorrr/loopy); frozen source
hashes and public model/dataset links are in
[STATE_CONDITIONED_SOURCE.json](STATE_CONDITIONED_SOURCE.json). Reference motion
data comes from Pollen Robotics' Apache-2.0 emotion/dance libraries, as documented
in the published dataset. The original source and dataset licenses remain
applicable; no model weights or calibration table are committed in this fork.
