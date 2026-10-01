"""REST service: prompt in, Reachy Mini motion out. Everything, including the diffusion, runs here; clients only play the result.

Two endpoints, same input, different output:

  POST /generate-sparse   planner only -> the recipe and its keyframe plans (8 channels every 0.25 s)
  POST /generate-dense    full pipeline -> 25 Hz trajectories, reachable, ready to play on the robot

  request (both): {"prompt": str, "n": int = 1 (variants), "seed": int = 0, "effort": "high" | "medium" | "low" (default: high),
                   "retries": int = 2 (0-8), "batched_retries": bool = false}
                  retries: extra attempts if the planner's first answer is invalid. batched_retries: decode the first
                  answer and every retry together in one call, so an invalid first answer costs no extra round
                  (the first answer still wins when valid; every request decodes a little slower).

  sparse response: {"prompt", "effort", "idea", "recipe", "plans": [{"duration", "keys": [{"t", "earR", "earL", "pitch",
                    "roll", "yaw", "z", "body", "energy"}, ...]}, ...], "durations_s", "timing_ms"}
                    (angles in deg, z in mm, energy = RMS deg of the fast detail the generator adds)
  dense response : {"prompt", "effort", "idea", "recipe", "moves": [Reachy move, ...], "durations_s", "timing_ms"}
                    each move is the SDK's recorded-move format {"description", "time", "set_target_data": [{"head": 4x4,
                    "antennas", "body_yaw"}]}, already projected onto the reachable set

The same prompt, seed and effort give the same recipe and plans in both, so plans[i] is the plan behind moves[i].

  # one planner
  python -m inference.server --bundle binhpham/reachy-mini-motion-planner-27b
  # all three planners on one GPU, picked per request with "effort"
  python -m inference.server --bundle high=binhpham/reachy-mini-motion-planner-27b \\
      --bundle medium=binhpham/reachy-mini-motion-planner-4b --bundle low=binhpham/reachy-mini-motion-planner-0.8b

  curl -s localhost:8000/generate-dense -H 'content-type: application/json' \\
       -d '{"prompt": "sneezing. You build up and then sneeze loudly.", "n": 2, "effort": "medium"}'

A bundle is a local directory or a Hugging Face repo id (planner/ + generator.pt + serve.json; see inference/bundle.py).
--backend hf runs the planner with plain transformers instead of vLLM (slow; for CUDA machines without vLLM).
"""
import argparse
import json
import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference.engine import MotionEngine, Planner, bundle_root

app = FastAPI(title="reachy-motion", docs_url=None, redoc_url=None)
# browsers (e.g. the visualizer Space) call the service directly; the page itself is static
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST"], allow_headers=["content-type"])
engine: MotionEngine = None

# GPU share per planner when several are loaded (vLLM pre-allocates weights + KV cache up to this fraction)
GPU_SHARE = {"high": 0.55, "medium": 0.22, "low": 0.08}


class Request(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    n: int = Field(default=1, ge=1, le=16)
    seed: int = 0
    effort: Optional[Literal["high", "medium", "low"]] = None
    retries: int = Field(default=2, ge=0, le=8)       # extra attempts when the planner's first answer is invalid
    batched_retries: bool = False                     # decode the first answer and the retries together, in one call


def run(fn, req):
    try:
        return fn(req.prompt, req.n, req.seed, req.effort, req.retries, req.batched_retries)
    except ValueError as e:        # effort not loaded, or the planner failed to write a valid recipe 3 times
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/generate-sparse")
def generate_sparse(req: Request):
    return run(engine.sparse, req)


@app.post("/generate-dense")
def generate_dense(req: Request):
    return run(engine.dense, req)


def main():
    global engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", action="append", required=True,
                    help="[effort=]bundle (dir or HF repo id); repeat to load several: high=... medium=... low=...")
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--bf16", action="store_true", help="serve bf16 weights (default from serve.json: FP8)")
    a = ap.parse_args()
    specs = [b.split("=", 1) if "=" in b.split("/")[0] else ("high", b) for b in a.bundle]
    planners, gen_cfg, generator = {}, None, None
    for effort, b in specs:
        root = bundle_root(b); cfg = json.load(open(os.path.join(root, "serve.json")))
        if a.bf16: cfg["fp8"] = False
        share = GPU_SHARE.get(effort, 0.3) if len(specs) > 1 else 0.80
        planners[effort] = Planner(os.path.join(root, "planner"), backend=a.backend, gpu_memory_utilization=share, **cfg)
        gen_cfg = gen_cfg or cfg; generator = generator or os.path.join(root, "generator.pt")
    engine = MotionEngine(planners, generator, **gen_cfg)
    import uvicorn
    print(f"ready: POST http://{a.host}:{a.port}/generate-dense and /generate-sparse (effort: {', '.join(sorted(planners))})", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
