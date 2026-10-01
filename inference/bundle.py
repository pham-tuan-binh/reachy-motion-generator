"""Assemble one self-contained serving bundle: planner weights (+ MTP head) + generator + serving settings.

  python -m inference.bundle --planner distill/4b/merged --base Qwen/Qwen3.5-4B --out bundles/my-4b
  python -m inference.server --bundle bundles/my-4b

bundle/
  planner/        merged fine-tuned LLM (hard links when on the same disk, so no extra space)
  generator.pt    plan -> motion flow model
  serve.json      the settings the service was tuned with (FP8, MTP drafts, flow steps, plan expansion)
"""
import argparse
import glob
import json
import os
import shutil

DEFAULTS = dict(fp8=True, spec_tokens=3, steps=8, cfg=1.5, expand_fc=2.0, expand_kdt=0.25, max_tokens=600)


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--planner", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True, help="the model the planner was fine-tuned from (source of the MTP head)")
    ap.add_argument("--generator", default="checkpoints/generator.pt")
    ap.add_argument("--spec-tokens", type=int, default=3, help="MTP drafts per step (the 27B was tuned at 4)")
    ap.add_argument("--card", help="optional model card (README.md) to include")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/planner", exist_ok=True)
    for f in glob.glob(f"{a.planner}/*"):
        if os.path.isfile(f): link_or_copy(os.path.realpath(f), f"{a.out}/planner/{os.path.basename(f)}")
    if not os.path.exists(f"{a.out}/planner/model-mtp.safetensors"):
        from inference.mtp import add_head
        add_head(f"{a.out}/planner", a.base)
    link_or_copy(os.path.realpath(a.generator), f"{a.out}/generator.pt")
    json.dump(dict(DEFAULTS, spec_tokens=a.spec_tokens, base_model=a.base), open(f"{a.out}/serve.json", "w"), indent=1)
    if a.card: shutil.copy(a.card, f"{a.out}/README.md")
    print(f"bundle -> {a.out}  ({sum(os.path.getsize(f) for f in glob.glob(f'{a.out}/**', recursive=True) if os.path.isfile(f)) / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
