"""Text prompts -> reference motions for Reachy Mini, end to end, with videos. For serving, see inference/server.py.

  python pipeline.py --recipes planner/examples/recipes.json --out runs/examples          # 287 shipped recipes, no LLM
  python pipeline.py --prompt "startled. A door slams behind you." --out runs/one         # teacher LLM (AI Gateway)
  python pipeline.py --prompts my_prompts.txt --planner binhpham/reachy-mini-motion-planner-4b --out runs/local

Stages (each writes into --out):
  1. planner   recipes.json   a recipe per prompt: the teacher LLM, or a fine-tuned planner with --planner
  2. planner   plans.jsonl    each recipe -> N randomised plans (amplitude, tempo, mirror)
  3. generator motions/       plan -> 25 Hz motion (flow model) -> projected onto the reachable set
  4. renderer  videos/        MuJoCo playback -> mp4 per motion + grid.mp4
"""
import argparse
import json
import os

from planner import read_lines, slug


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompts", help="text file, one prompt per line"); src.add_argument("--prompt", action="append")
    src.add_argument("--recipes", help="skip the planner: a {prompt: recipe} JSON file")
    ap.add_argument("--out", required=True); ap.add_argument("--model", help="teacher: AI Gateway model id")
    ap.add_argument("--planner", help="fine-tuned planner instead of the teacher: a bundle (HF repo id or dir) or a merged model dir")
    ap.add_argument("--variants", type=int, default=2); ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--cfg", type=float, default=1.5); ap.add_argument("--ckpt", default="checkpoints/generator.pt")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    rec_path = os.path.join(a.out, "recipes.json")
    prompts = read_lines(a.prompts) if a.prompts else a.prompt
    if a.recipes:
        recipes = json.load(open(a.recipes)); json.dump(recipes, open(rec_path, "w"), indent=1)
    elif a.planner:
        from inference.engine import Planner
        print(f"[1/4] planner: {len(prompts)} prompts -> recipes ({a.planner}, transformers)")
        P = Planner(a.planner, backend="hf"); recipes = {}
        for p in prompts:
            try: recipes[p] = P.plan(p)[1]
            except ValueError as e: print(f"  FAILED {p[:60]!r}: {e}")
        json.dump(recipes, open(rec_path, "w"), indent=1)
    else:
        from planner.write import write_recipes
        print(f"[1/4] planner: {len(prompts)} prompts -> recipes ({a.model or 'default teacher model'})")
        recipes = write_recipes(prompts, rec_path, model=a.model)

    from planner.dsl import variants
    plans_path = os.path.join(a.out, "plans.jsonl")
    print(f"[2/4] planner: {len(recipes)} recipes x {a.variants} variants -> plans")
    with open(plans_path, "w") as fh:
        for i, (prompt, rec) in enumerate(recipes.items()):
            for v, pl in enumerate(variants(rec, a.variants, seed=i, fc=2.0, kdt=0.25)):    # as served: fast events stay fast
                fh.write(json.dumps(dict(name=f"{slug(prompt)}__{v}", prompt=prompt, **pl)) + "\n")

    from generator.sample import read_plans, sample_plans
    mdir = os.path.join(a.out, "motions")
    print("[3/4] generator: plans -> motions")
    sample_plans(read_plans(plans_path), a.ckpt, mdir, seeds=a.seeds, cfg=a.cfg)

    if not a.no_render:
        from renderer.outputs import grid, videos
        paths = sorted(os.path.join(mdir, f) for f in os.listdir(mdir) if f.endswith(".json"))
        print(f"[4/4] renderer: {len(paths)} motions -> videos")
        videos(paths, os.path.join(a.out, "videos"))
        grid(paths[:36], os.path.join(a.out, "grid.mp4"))
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
