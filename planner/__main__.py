"""Planner CLI.

  python -m planner write  --prompts prompts.txt --out recipes.json     # teacher LLM writes recipes (AI Gateway)
  python -m planner expand --recipes recipes.json --variants 4 --out plans.jsonl
"""
import argparse
import json

from planner import read_lines, slug


def main():
    ap = argparse.ArgumentParser(prog="planner", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="teacher LLM writes one recipe per prompt")
    w.add_argument("--prompts", required=True, help="text file, one prompt per line ('word. one sentence.')")
    w.add_argument("--out", required=True); w.add_argument("--model"); w.add_argument("--batch", type=int, default=8)
    w.add_argument("--workers", type=int, default=4)
    e = sub.add_parser("expand", help="recipes -> randomised plans")
    e.add_argument("--recipes", required=True); e.add_argument("--out", required=True)
    e.add_argument("--variants", type=int, default=4); e.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.cmd == "write":
        from planner.write import write_recipes
        r = write_recipes(read_lines(a.prompts), a.out, model=a.model, batch=a.batch, workers=a.workers)
        print(f"{len(r)} recipes -> {a.out}")
    else:
        from planner.dsl import variants
        recipes = json.load(open(a.recipes)); n = 0
        with open(a.out, "w") as fh:
            for i, (prompt, rec) in enumerate(recipes.items()):
                for v, pl in enumerate(variants(rec, a.variants, seed=a.seed * 100003 + i)):
                    fh.write(json.dumps(dict(name=f"{slug(prompt)}__{v}", prompt=prompt, **pl)) + "\n"); n += 1
        print(f"{n} plans from {len(recipes)} recipes -> {a.out}")


if __name__ == "__main__":
    main()
