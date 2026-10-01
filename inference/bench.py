"""Latency benchmark against a running server: python -m inference.bench [--url http://localhost:8000] [--n 1] [--effort low|medium|high]"""
import argparse
import json
import statistics
import time
import urllib.request

PROMPTS = ["sneezing. You build up and then sneeze loudly.", "greeting a friend at the door. You are delighted to see them.",
           "sleepy toddler. You fight to stay awake.", "a cat stalking prey. You crouch low and freeze.",
           "startled. A door slams behind you.", "proud. You finally solved the puzzle.",
           "drunk. You are unsteady and wobbly.", "listening closely to a question."]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://localhost:8000"); ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--effort", choices=["low", "medium", "high"]); ap.add_argument("--sparse", action="store_true")
    a = ap.parse_args(); wall, parts = [], []
    for r in range(a.rounds):
        for p in PROMPTS:
            body = json.dumps({"prompt": p, "n": a.n, "seed": r, **({"effort": a.effort} if a.effort else {})}).encode()
            t0 = time.perf_counter()
            res = json.load(urllib.request.urlopen(urllib.request.Request(f"{a.url}/generate-{'sparse' if a.sparse else 'dense'}", body, {"content-type": "application/json"})))
            wall.append((time.perf_counter() - t0) * 1e3); parts.append(res["timing_ms"])
    med = lambda k: statistics.median(x.get(k, 0) for x in parts)
    print(f"{len(wall)} requests, n={a.n}, effort={a.effort or 'default'}: wall median {statistics.median(wall):.0f} ms, p90 {sorted(wall)[int(.9 * len(wall)) - 1]:.0f} ms | "
          f"planner {med('planner'):.0f} ms, generator {med('generator'):.0f} ms, reachability {med('reachability'):.0f} ms")


if __name__ == "__main__":
    main()
