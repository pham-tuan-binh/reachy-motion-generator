"""Score a fine-tuned planner on prompts it never trained on.

  python -m planner.distill.evaluate distill/4b/merged            # or a bundle: binhpham/reachy-mini-motion-planner-4b
  python -m planner.distill.evaluate <planner> --backend hf       # without vLLM

  probes   16 out-of-distribution prompts with a physical check each ("does the sneeze release move the head down?"),
           --samples per prompt at temperature 0.7: OOD-core (concepts in no training data) and skill pass rates
  agree    mean per-descriptor Pearson r between the planner's greedy plans and the teacher's on the 39 val prompts,
           next to the teacher's agreement with itself (its second recipe): the ceiling
  real     its recipes for the 12 held-out real emotions -> generator -> identification among Pollen's real clips
           (top-1 and mean rank; chance 8.3% / 6.5), independent of any teacher's taste
  valid    share of answers that parse and pass the recipe checker

Differences under ~0.1 in probe pass rate, and ~7 points in real-clip top-1, are noise.
"""
import argparse
import json

import numpy as np

from planner.distill.common import DESC_NAMES, plan_descriptors
from planner.distill.probes import PROBES, split
from planner.distill.probes import score as probe_score
from planner.dsl import variants


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return [float(np.corrcoef(a[:, j], b[:, j])[0, 1]) if a[:, j].std() > 1e-6 and b[:, j].std() > 1e-6 else float("nan")
            for j in range(a.shape[1])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("planner"); ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--samples", type=int, default=12); ap.add_argument("--val-file", default="distill_data/val.jsonl")
    ap.add_argument("--ckpt", default="checkpoints/generator.pt"); ap.add_argument("--out", help="write the full report as JSON")
    a = ap.parse_args()
    from inference.engine import Planner
    P = Planner(a.planner, backend=a.backend, gpu_memory_utilization=0.6)

    def recipes(prompts, temperature=0.0, seed=0):
        outs = P.complete([P.text(p) for p in prompts], temperature, [seed + i for i in range(len(prompts))])
        return [P.parse(o)[1] for o in outs]

    val = [json.loads(l) for l in open(a.val_file)]
    got = recipes([v["prompt"] for v in val]); ok = [(v, r) for v, r in zip(val, got) if r]
    st = corr([plan_descriptors(r) for _, r in ok], [plan_descriptors(v["labels"][0]["recipe"]) for v, _ in ok])
    tt = corr(*zip(*[(plan_descriptors(v["labels"][0]["recipe"]), plan_descriptors(v["labels"][1]["recipe"])) for v in val]))

    per = {p: recipes([p] * a.samples, 0.7, seed=1) for p, _, _ in PROBES}
    _, per_rate = probe_score(per); ood, skill = split(per_rate)

    from generator.evaluate import HeldOut
    from generator.sample import generate, load
    H = HeldOut(); held = dict(zip(H.prompts, recipes(list(H.prompts.values()))))
    net, stats, dev = load(a.ckpt); ranks = []
    for h, rec in held.items():
        if rec:   # the settings of the published numbers: 5 training-style (1 Hz) plans, 100 flow steps
            ranks += [H.rank(generate(net, stats, pl, dev, seed=i, steps=100), h) for i, pl in enumerate(variants(rec, 5, seed=1))]
    r = np.array(ranks) if ranks else np.array([np.nan])

    n = len(val) + a.samples * len(PROBES) + len(held)
    valid = (len(ok) + sum(x is not None for rs in per.values() for x in rs) + sum(x is not None for x in held.values())) / n
    print(f"probes OOD-core {ood:.2f}  skill {skill:.2f}")
    for p, v in per_rate.items(): print(f"  {v:4.2f}  {p}")
    print(f"agree {np.nanmean(st):.2f} (teacher vs itself {np.nanmean(tt):.2f})")
    print(f"real clips top-1 {100 * (r == 1).mean():.0f}%  mean rank {r.mean():.2f}")
    print(f"valid {100 * valid:.1f}%")
    if a.out:
        json.dump(dict(ood=ood, skill=skill, per_probe=per_rate, agree=float(np.nanmean(st)), agree_per_descriptor=dict(zip(DESC_NAMES, st)),
                       real_top1=float((r == 1).mean()), real_rank=float(r.mean()), valid=valid, probe_recipes=per), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
