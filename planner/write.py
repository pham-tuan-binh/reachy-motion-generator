"""Ask the LLM (through the AI Gateway) for a recipe per prompt; validate, retry invalid ones."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

from planner import gateway
from planner.dsl import check
from planner.prompt import SCHEMA, SYSTEM, user_message


def _batch(prompts, model, max_fix=2):
    got, errors = {}, None
    todo = list(prompts)
    for _ in range(max_fix + 1):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(todo, errors)}]
        out = gateway.chat_json(msgs, SCHEMA, "motion_recipes", model=model)
        by_prompt = {m["prompt"].strip(): m for m in out.get("motions", [])}
        errors = {}
        for p in todo:
            m = by_prompt.get(p.strip())
            if m is None:
                errors[p] = "missing from your answer"; continue
            err = check(m["recipe"])
            if err: errors[p] = err
            else: got[p] = dict(recipe=m["recipe"], idea=m.get("idea", ""))
        todo = list(errors)
        if not todo:
            break
    return got, errors


def write_recipes(prompts, out_path, model=None, batch=8, workers=4):
    """Resumable: prompts already in ``out_path`` are skipped. Returns the full {prompt: recipe} dict."""
    have = json.load(open(out_path)) if os.path.exists(out_path) else {}
    todo = [p for p in prompts if p not in have]
    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    failed = {}
    with ThreadPoolExecutor(workers) as ex:
        for got, errs in ex.map(lambda c: _batch(c, model), chunks):
            have.update({p: v["recipe"] for p, v in got.items()}); failed.update(errs)
            json.dump(have, open(out_path, "w"), indent=1)
            print(f"  {len(have)} recipes written", flush=True)
    for p, e in failed.items():
        print(f"  FAILED {p[:60]!r}: {e}")
    return have
