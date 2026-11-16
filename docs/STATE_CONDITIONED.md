# State-conditioned trajectory generator v4

This branch separates the reusable generator experiment from
[Loopy](https://github.com/creatorrr/loopy). It adds three flow-model variants:
`full` warm-starts the original 21.8M-parameter generator with nine zero-initialized
state input columns; `residual` learns bounded detail around a deterministic
plan; `large` trains a roughly 57M-parameter generator from scratch. The original
generator, sampler, DSL, planner defaults and checkpoints remain compatible.
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

Python 3.10+ is supported by the package. Numerical parity was checked with the
pilot's PyTorch 2.15.0, NumPy 2.2.6, SciPy 1.18.1 and Rust kinematics 1.0.3.

```bash
git clone --branch state-conditioned-generator-v4 https://github.com/creatorrr/reachy-motion-generator.git
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

# Use an appropriate state-aware tuned text-planner checkpoint here.
planner = Planner("path/to/tuned-planner", backend="hf")
idea, recipe = planner.plan("Look left and return", seed=123, current_pose=pose)

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

The existing `MotionEngine`, `/generate-dense`, `/generate-sparse` and
`pipeline.py` still use the original sampling path and do not forward a settled
pose. State support here is the explicit planner/generator Python interfaces
above and the recipe CLI below, not the legacy HTTP routes.

## Numerical recipe CLI

```bash
python -m generator.conditioned_cli \
  --checkpoint bundles/generator-v4/generator.pt \
  --recipe 'go 1 p=6 E=0 | go 1 p=0 E=0 | hold .5 E=0' \
  --origin '{"roll":0,"pitch":-4,"yaw":10,"body":5,"right":-10,"left":10,"x":0.001,"y":0,"z":0}' \
  --mode nominal --seed 123 --steps 8 --cfg 1 --lowpass-hz 4 \
  --out runs/sample-v4.npz --device cpu
```

The CLI writes `trajectory` and `origin` arrays plus a JSON receipt with the
checkpoint/output hashes, recipe and settings. It rejects overwrite and recipe
truncation. It does not anchor or project the sample or send it to hardware.
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
