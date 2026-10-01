# Training and reproducing

Everything behind the published models, from the generator to a serving bundle. To just run the server, see the
[README](../README.md); for why things are the way they are, the [technical report](TECHNICAL_REPORT.md).

```bash
pip install -e ".[render,teacher]"     # + ".[train]" (Unsloth, TRL, PEFT) on a CUDA machine to fine-tune planners
cp .env.example .env && set -a && source .env && set +a     # only for the teacher LLM: AI_GATEWAY_API_KEY
```

## Offline pipeline

`pipeline.py` runs prompts → recipes → plans → motions → MuJoCo videos into one folder (`recipes.json`, `plans.jsonl`,
`motions/*.json`, `videos/*.mp4`, `grid.mp4`). The recipes come from one of three places:

```bash
python pipeline.py --recipes planner/examples/recipes.json --out runs/examples                        # 287 shipped recipes
python pipeline.py --prompts prompts.txt --planner binhpham/reachy-mini-motion-planner-4b --out runs/a  # fine-tuned planner
python pipeline.py --prompts prompts.txt --out runs/b [--model anthropic/claude-opus-5.5]              # teacher LLM
```

The fine-tuned planner runs on transformers here (CUDA, Apple silicon or CPU). The teacher is any model on the
[Vercel AI Gateway](https://vercel.com/ai-gateway), called with the full teacher prompt (`planner/prompt.py: SYSTEM`).
Rendering needs the Reachy Mini model files (`pip install reachy-mini`, or `REACHY_MINI_ROOT`); on a headless Linux
machine set `MUJOCO_GL=egl`.

Each stage on its own:

```bash
python -m planner write     --prompts prompts.txt --out recipes.json       # teacher LLM, resumable, validated
python -m planner expand    --recipes recipes.json --variants 4 --out plans.jsonl
python -m generator sample  --plans plans.jsonl --out motions/ [--seeds 2] [--cfg 1.5]
python -m generator extract --moves some_moves/ --out plans.jsonl         # the plans of existing motion
python -m renderer video motions/ --out videos/
python -m renderer sheet motions/sad__0.json motions/sad__1.json --out sad.png
python -m renderer grid  motions/ --out all.mp4 --cols 6
```

## Generator

```bash
python -m generator train --out checkpoints/generator.pt     # ~3 min on an RTX 5090, ~15 min on Apple silicon
python -m generator.evaluate                                 # identification + motion speeds on the 12 held-out emotions
```

Pollen's emotions and dances libraries download from the Hugging Face Hub. The 12 emotions in `common/data.py:
HELD_OUT` are kept out for evaluation (`--no-held-out` trains on everything). The checkpoint with the best held-out
loss is kept; the shipped one is from step 2,250. Without a local `checkpoints/generator.pt`, every command uses the
published one.

## Planners

The teacher data is `distill_data/dataset.jsonl` (10,872 rows: `id, prompt, idea, recipe, source, family, weight`).
`distill_data/val.jsonl` holds the 39 validation prompts, each with two independent teacher recipes; they are never
trained on.

```bash
# 1. the SFT set: served sources, leak filter + OOD blocklist, input variants  (17,367 rows)
python -m planner.distill.sft --out distill/sft
# 2. LoRA fine-tune + verified merge
python -m planner.distill.train --data distill/sft --out distill/4b  --model Qwen/Qwen3.5-4B      # 1.9 h, RTX 5090
python -m planner.distill.train --data distill/sft --out distill/08b --model Qwen/Qwen3.5-0.8B    # 24 min, RTX 5090
python -m planner.distill.train --data distill/sft --out distill/27b --model Qwen/Qwen3.8-27B \
       --epochs 1.5 --bs 8 --grad-accum 4                                                        # 3.5 h, RTX PRO 6000
# 3. score: probes, agreement with the teacher, real-clip identification
python -m planner.distill.evaluate distill/4b/merged
```

The SFT set's size can differ by a few rows between machines: the leak filter's similarity threshold catches a
borderline prompt or not depending on fp16 vs fp32 embeddings.

The `source` column: `claude` (520 hand-written rows), `claude_events` (65 build-up/release events and long
stories, weight 3), `seed` (the 287 recipes of `planner/examples/recipes.json`), `astra` (5,000 scenarios, written as
precise choreography) and `astra_lively` (the same prompts, re-authored in the lively style). The served models use
all but `astra`, which is kept for comparison.

## MTP head, bundle, serve

```bash
# restore the base model's MTP head (the merge drops it), then fine-tune it on the planner's own answers
python -m inference.mtp add   distill/4b/merged --base Qwen/Qwen3.5-4B
python -m inference.mtp data  distill/4b/merged distill/sft/train.jsonl distill/4b/mtp.jsonl
python -m inference.mtp train distill/4b/merged distill/4b/mtp.jsonl distill/4b/served
# planner + generator + serving settings in one directory, then serve it
python -m inference.bundle --planner distill/4b/served --base Qwen/Qwen3.5-4B --out bundles/my-4b      # 27B: --spec-tokens 4
python -m inference.server --bundle medium=bundles/my-4b
python -m inference.bench --effort medium                                                             # latency
```

A bundle is `planner/` (with its MTP head), `generator.pt` and `serve.json` (FP8, drafts per step, flow steps, plan
expansion). Upload the directory to a Hugging Face model repo to serve it by id.

## Motion library

The [motion library](https://huggingface.co/datasets/binhpham/reachy-mini-massive-motion-library) is every dataset row
performed and rendered:

```bash
python -m inference.export_motions --out build/motions.pkl              # GPU: one reachable trajectory per row
MUJOCO_GL=egl python -m inference.lerobot_export --motions build/motions.pkl \
       --repo <user>/<dataset> --root build/lerobot --workers 20 [--push]     # LeRobot episodes with video
```

## Figures

`python docs/make_figures.py` redraws `docs/figures/` from the code and the shipped generator.
