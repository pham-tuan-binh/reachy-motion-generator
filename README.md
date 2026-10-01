# Reachy Mini text-to-motion

Type what [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) should express (*"sneezing"*, *"a cat stalking
prey"*, *"heartbroken"*) and get a motion the robot can play right away: head, antennas and body at 25 Hz, already
fitted to what the robot can physically reach.

- **Models:** [27B](https://huggingface.co/binhpham/reachy-mini-motion-planner-27b) (best quality) ·
  [4B](https://huggingface.co/binhpham/reachy-mini-motion-planner-4b) (balanced) ·
  [0.8B](https://huggingface.co/binhpham/reachy-mini-motion-planner-0.8b) (fastest)
- **Try it in the browser:** [reachy-mini-motion-generator](https://huggingface.co/spaces/binhpham/reachy-mini-motion-generator)
- **Dataset:** [reachy-mini-massive-motion-library](https://huggingface.co/datasets/binhpham/reachy-mini-massive-motion-library),
  10,872 prompts performed by the robot (15 h of motion, with video)
- **Python client:** [reachy-motion-generator-api](https://github.com/pham-tuan-binh/reachy-motion-generator-api), with an example that plays results on the robot

## Run it

You need Linux, an NVIDIA GPU and Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

```bash
git clone https://github.com/pham-tuan-binh/reachy-motion-generator && cd reachy-motion-generator
docker compose up -d        # builds the image, serves the 4B on port 8000; prints "ready: ..." when listening
curl -s localhost:8000/generate-dense -H 'content-type: application/json' \
     -d '{"prompt": "sneezing. You build up and then sneeze loudly."}'
```

The model downloads from Hugging Face on first start, and is cached with the compiled GPU kernels in the
`reachy-motion-cache` volume. Startup takes a few minutes. Times are on an RTX PRO 6000 (96 GB, fits all three); an
RTX 5090 (32 GB) runs the 4B but not the 27B.

| model | `effort` | GPU memory | time per request |
|---|---|---|---|
| 0.8B | `low` | ~6 GB | ~0.17 s |
| 4B | `medium` | ~12 GB | ~0.29 s |
| 27B | `high` | ~40 GB | ~0.8 s |
| all three | per request | ~85 GB | |

To serve other models, put a `compose.override.yaml` next to `compose.yaml`:

```yaml
services:
  reachy-motion:
    command: ["--bundle", "high=binhpham/reachy-mini-motion-planner-27b",
              "--bundle", "medium=binhpham/reachy-mini-motion-planner-4b",
              "--bundle", "low=binhpham/reachy-mini-motion-planner-0.8b"]
```

Without Docker: `pip install -e ".[serve]"`, then
`python -m inference.server --bundle medium=binhpham/reachy-mini-motion-planner-4b`.

**No NVIDIA GPU?** The planners also run with plain transformers on a Mac or CPU, more slowly, through the offline
pipeline. It writes the moves as JSON and can render videos:

```bash
pip install -e ".[render]"
python pipeline.py --planner binhpham/reachy-mini-motion-planner-0.8b --prompt "proud. You finally solved the puzzle." --out runs/proud
```

### API

Two endpoints, same input:

- **`POST /generate-dense`** returns `moves`: finished trajectories in the Reachy Mini SDK's recorded-move format
  (`{"time": [...], "set_target_data": [{"head": 4x4, "antennas": [r, l], "body_yaw"}, ...]}`), ready to play.
- **`POST /generate-sparse`** returns `plans`: the keyframes the motion is built from. It skips the motion model,
  so it is faster. With the same seed, `plans[i]` is the plan behind `moves[i]`.

| field | default | |
|---|---|---|
| `prompt` | required | `word. one sentence of context.` works best |
| `n` | 1 | how many variations to return (1–16) |
| `seed` | 0 | same seed, same result |
| `effort` | `high`, or the only one loaded | which planner answers: `high` = 27B, `medium` = 4B, `low` = 0.8B |
| `retries` | 2 | extra attempts when the planner's first answer is invalid (0–8) |
| `batched_retries` | `false` | decode the first answer and the retries together: a bad answer costs no extra round, every request is ~30–50% slower |

Both also return the `recipe` and `idea` the planner wrote, and `timing_ms`. To watch a result, drop a move JSON onto
the [visualizer](https://huggingface.co/spaces/binhpham/reachy-mini-motion-generator), or point the page at your
server (the page is HTTPS, so the server must be too).

## How it works

The only real expressive motion recorded for Reachy Mini is Pollen's emotions library: 85 clips, 8.6 minutes, plus
19 dances. No model trained on nine minutes of motion learns what *heartbroken* or *a cat stalking prey* means. So the
system splits the problem in two, and puts each kind of knowledge where it already lives:

![pipeline](docs/figures/1_pipeline.png)

- **What to express** is world knowledge. A **planner**, a fine-tuned LLM, writes a short motion script.
- **How Reachy Mini moves** (organic timing, overshoot, antenna flicks, micro-motion) is learned from the real clips by
  a small **generator** that never sees text, so it can train on every clip, including the dances.

### 1. The planner writes a recipe

The planner answers each prompt with one sentence of intent and a **recipe** in a tiny language:

```
go D k=v ...        ease to a pose over D seconds (0.15-0.3 s = a snap)
hold D [E=v]        stay put
osc D ch amp per    oscillate one channel: nod, shake, sway, bounce, ear flap
```

Channels are the ears (`e`, or `eR`/`eL`, degrees of droop), head pitch/roll/yaw (`p r y`), head height (`z`, mm),
body yaw (`b`) and **energy** (`E`): how much fast, organic detail the generator should add on top. The 27B's
*sneezing*: rise and tilt back for 1.6 s ("ah… ah…"), tense, snap down in 0.12 s with the ears flaring ("choo"),
recover:

```
go .6 e=40 p=-6 z=4 E=1 | go 1 p=-16 z=10 e=70 E=3 | hold .6 E=6 | go .12 p=18 z=-8 e=100 E=10 | go 1 p=2 z=2 e=30 E=1 | hold .6
```

A recipe is about 10× fewer tokens than raw keyframes, and it is easy to check: a validator rejects anything
malformed or out of range, and the server resamples (temperature 0.7) until the answer is valid.

The planners are Qwen models fine-tuned with LoRA, unchanged in architecture. Qwen3.5 and Qwen3.8 are hybrid decoders:
three in four layers use linear attention (Gated DeltaNet), every fourth full attention. Each has a multi-token
prediction (MTP) head, used at serving time to draft tokens.

| | base | layers | trained on |
|---|---|---|---|
| `high` | Qwen3.8-27B | 64 (48 linear + 16 full attention) | 1× RTX PRO 6000, 3.5 h |
| `medium` | Qwen3.5-4B | 32 (24 linear + 8 full attention) | 1× RTX 5090, 1.9 h |
| `low` | Qwen3.5-0.8B | 24 (18 linear + 6 full attention) | 1× RTX 5090, 24 min |

**Teacher data.** The training targets come from frontier LLMs acting as teachers: 585 hand-written prompt → recipe
rows (emotions at three intensities, reactions, social behaviours, animals, characters, robot states, build-up →
release events, long multi-phase stories), 287 seed recipes, and 5,000 individually authored scenarios across about
600 families. The 5,000 scenarios were first written as precise choreography with frozen holds; they were re-authored
in a livelier style (constant organic energy, rhythm, expressive ears) because that is what reads as alive on the
robot. All 10,872 rows are in `distill_data/dataset.jsonl`.

**Fine-tuning.** Each row becomes one chat example: a compact system prompt (~330 words: units, the recipe grammar,
four motion rules, three examples), the user prompt, and the JSON answer. Every prompt is also trained as its bare word
and as its sentence alone, so short prompts work. The loss is on the answer only. LoRA rank 32 on the attention and MLP
projections, AdamW 8-bit at lr 1e-4, batch 32, 1.5–2 epochs, keeping the checkpoint with the best held-out loss. After
merging, the MTP head is restored from the base model and then fine-tuned on the planner's own answers, so it drafts
recipes well: planner time drops 19–26% with identical outputs.

**Keeping the evaluation honest.** Training prompts close to any evaluation prompt (cosine > 0.72 with
Qwen3-Embedding-0.6B) are removed, and so are any that mention the core test concepts (sneeze, startle, drunk,
toddler, stalking, heartbroken, ecstatic). The models never see a sneeze.

### 2. The recipe becomes a plan

A recipe expands into a **plan**: 8 channels (both ears, pitch, roll, yaw, height, body yaw, energy) as keyframes every
0.25 s. One recipe yields as many variants as you ask for (`n`): amplitude ×0.75–1.25, tempo ×0.8–1.25, per-segment
timing jitter, slight ear asymmetry, and a left-right mirror on every other variant.

![recipe to plan](docs/figures/2_recipe_to_plan.png)

The same plan can be **extracted** from any recorded clip by low-passing its posture and measuring the fast detail
left over as energy. That is what makes every real clip, even a dance with no description, a training example for the
generator.

### 3. The generator turns the plan into motion

A 21.8M-parameter **flow-matching transformer** turns the plan into 25 Hz motion for the 9 degrees of freedom (head
position and orientation, two antennas, body yaw).

- **Input.** Each frame is one token: the noisy motion (9 numbers), the plan interpolated to that frame (8), and a
  has-plan flag, projected to 384 dimensions. The plan is concatenated to every frame rather than cross-attended, so
  the model cannot ignore it. 8 transformer blocks attend over all frames, so each frame sees the whole plan:
  anticipation, follow-through and transitions come from there. The flow time conditions every block through AdaLN.
- **Duration** is set by the plan: sampling starts from duration × 25 frames of noise and returns exactly that many.
  Clips are capped at 28.8 s.
- **Training.** The real clips (12 emotions held out for evaluation), each mirrored and time-stretched ×0.8/1/1.25,
  and each paired with its own extracted plan. Loss: flow matching plus a velocity term on the implied clean motion;
  without it, head and antenna speeds came out half as fast as real ones. 10% plan dropout enables classifier-free
  guidance. 5,000 steps on one GPU in ~3 min, keeping the best held-out checkpoint.
- **Sampling.** 8 Euler steps from noise with guidance 1.5 on the plan, then a 4 Hz low-pass. Given the true plan of a
  held-out real emotion, the generated motion is closer to that emotion's real clip than to the other 11 held-out
  clips 89% of the time.

### 4. The motion is made reachable

The head sits on a Stewart platform, whose reachable poses are coupled: a tilt that is reachable at one height is not
at another, so clipping each channel cannot guarantee a valid pose, and the robot freezes on an unreachable target.
Every frame is checked with the SDK's analytical inverse kinematics; an unreachable frame is pulled back along a
line search from the last reachable pose toward the target. Timing and antennas are untouched.

### Results

Scored on prompts no model trained on. **Probes**: 16 out-of-distribution prompts, each with an automatic physical
check (does the sneeze release move the head *down*? does the sleepy toddler droop *and* recover?), 12 samples each.
**Real clips**: the planner writes recipes for the 12 held-out real emotions, and we check whether the generated motion
is closest to its own real clip (chance: 8%).

| | OOD probes | skill probes | real clips top-1 | sneeze correct | time (n = 1) |
|---|---|---|---|---|---|
| 27B (`high`) | **0.96** | **0.97** | 27% | **24/24** | 0.79 s |
| 4B (`medium`) | 0.91 | 0.875 | 32% | 21/24 | 0.29 s |
| 0.8B (`low`) | 0.66 | 0.57 | 17% | 12/24 | 0.17 s |

The 0.8B handles single-feeling prompts but misses multi-phase ones (yawns, bows, build-ups). With 12 emotions,
real-clip scores move by ±7 points from noise.

**Why it is fast.** FP8 weights (decoding on one GPU is bound by memory traffic); speculative decoding with the
fine-tuned MTP head; all `n` motions and both guidance branches in one batched generator pass, replayed as CUDA graphs;
8 flow steps. Planner decoding is ~90% of the time.

The [technical report](docs/TECHNICAL_REPORT.md) has the full story: every design that did not work (end-to-end
text-to-motion models, reasoning before the recipe, retrieval), how the probes caught a sneeze that every early model
got backwards, the step-by-step speedups, and the limitations.

## Repository

```
common/       move format <-> arrays, the plan, reachability (SDK inverse kinematics)
planner/      the recipe language, the prompts, the teacher client; distill/ fine-tunes and scores the planners
generator/    the flow-matching model: data, training, sampling, evaluation
inference/    the server (engine, REST API), serving bundles, MTP head tools, dataset export
renderer/     MuJoCo playback of the official model: videos, contact sheets, grids
visualizer/   the browser viewer (the Hugging Face Space)
pipeline.py   prompts -> recipes -> motions -> videos, offline
distill_data/ the teacher dataset and the fixed validation set
docs/         technical report, training guide, figures
```

To retrain anything (generator, planners, MTP heads) or build your own bundle, see [docs/TRAINING.md](docs/TRAINING.md).

## License

Apache-2.0. `common/assets/kinematics_data.json` is from [pollen-robotics/reachy_mini](https://github.com/pollen-robotics/reachy_mini),
and the visualizer's 3D model and meshes from [8bitkick/reachy_mini_3d_web_viz](https://huggingface.co/spaces/8bitkick/reachy_mini_3d_web_viz),
both Apache-2.0. The generator is trained on Pollen's
[emotions](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library) and
[dances](https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library) libraries.
