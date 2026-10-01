"""distill_data/dataset.jsonl -> chat SFT set for the planner (train.jsonl, val.jsonl).

  python -m planner.distill.sft --out distill/sft

- rows are repeated by their `weight` (build-up/release events x3); only the --sources the served models used
- rows whose prompt or family mentions a core out-of-distribution probe concept are dropped (--block), and so are
  prompts within cosine --leak of any evaluation or probe prompt (Qwen3-Embedding-0.6B), so the probes and the
  held-out emotions stay a real generalisation test
- the 39 prompts of distill_data/val.jsonl are never trained on; val.jsonl holds their first teacher recipe
- every "word. sentence." training prompt is also trained as "word." and as "sentence." alone
"""
import argparse
import json
import os
import random
import re

from planner import read_lines
from planner.distill.probes import PROBES
from planner.dsl import check
from planner.prompt import student_messages

SOURCES = "claude,claude_events,seed,astra_lively"
BLOCK = r"sneez|startl|drunk|dizz|toddler|stalk|pounc|heartbr|ecstat"


def write_jsonl(rows, path):
    with open(path, "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="distill_data/dataset.jsonl"); ap.add_argument("--val-file", default="distill_data/val.jsonl")
    ap.add_argument("--sources", default=SOURCES, help="comma list of dataset sources to keep ('' = all)")
    ap.add_argument("--eval", default="planner/examples/eval_prompts.txt", help="evaluation prompts, besides the probes")
    ap.add_argument("--leak", type=float, default=0.72); ap.add_argument("--block", default=BLOCK)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    keep = set(filter(None, a.sources.split(","))); rows = []
    for r in map(json.loads, open(a.dataset)):
        if not keep or r["source"] in keep: rows += [r] * int(r.get("weight", 1))
    n0 = len(rows); rows = [r for r in rows if not re.search(a.block, r["prompt"] + " " + r.get("family", ""), re.I)]
    print(f"blocklist: dropped {n0 - len(rows)} rows")
    rows = [r for r in rows if not check(r["recipe"])]

    from planner.distill.common import LocalEmbedder
    emb = LocalEmbedder(); ps = sorted({r["prompt"] for r in rows}); ev = read_lines(a.eval) + [p for p, _, _ in PROBES]
    sim = dict(zip(ps, (emb(ps) @ emb(ev).T).max(1))); leaked = {p for p in ps if sim[p] > a.leak}
    print(f"leak filter: dropped {len(leaked)} prompts > {a.leak} to an eval prompt, e.g. {sorted(leaked)[:6]}")

    val = [json.loads(l) for l in open(a.val_file)]; val_p = {v["prompt"] for v in val}
    tr = [r for r in rows if r["prompt"] not in leaked and r["prompt"] not in val_p]
    extra = []
    for r in tr:
        w, _, rest = r["prompt"].partition(". ")
        if rest: extra += [dict(r, prompt=w + "."), dict(r, prompt=rest)]
    tr += list({(e["prompt"], e["recipe"]): e for e in extra}.values())     # repeated rows need not repeat variants
    random.Random(a.seed).shuffle(tr)

    os.makedirs(a.out, exist_ok=True)
    write_jsonl([dict(messages=student_messages(r["prompt"], r["idea"], r["recipe"]), source=r["source"]) for r in tr], f"{a.out}/train.jsonl")
    write_jsonl([dict(messages=student_messages(v["prompt"], v["labels"][0]["idea"], v["labels"][0]["recipe"]), source="val") for v in val],
                f"{a.out}/val.jsonl")
    print(f"train {len(tr)} rows | val {len(val)} prompts -> {a.out}")


if __name__ == "__main__":
    main()
