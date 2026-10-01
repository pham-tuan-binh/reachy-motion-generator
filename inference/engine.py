"""MotionEngine: prompt -> recipe (fine-tuned LLM planner) -> plans -> motion (flow generator) -> reachable moves.

Speed choices (settings come from each bundle's serve.json):
- planner on vLLM: FP8 weights (decode on one GPU is bound by memory traffic), speculative decoding with the model's
  own fine-tuned multi-token-prediction head (lossless), prefix caching for the shared system prompt
- generator: all n samples and both guidance branches in ONE batched forward per step, 8 Euler steps, replayed as
  CUDA graphs: the 21.8M-parameter net is launch-bound, so capturing each (batch, length) bucket once is ~4x faster
- recipes are expanded with a 2 Hz / 0.25 s plan (instead of the 1 Hz / 0.5 s used to extract training plans), so
  0.12 s snaps survive as snaps; this doubled peak release speed on "sneezing" (124 -> 258 deg/s)
"""
import json
import os
import re
import threading
import time

import numpy as np
import torch

from common.motion import FPS, to_move
from common.reach import Reach
from generator.sample import generate_batch, load
from planner.dsl import check, variants
from planner.prompt import student_messages


def bundle_root(bundle):
    """A bundle (planner/ + generator.pt + serve.json): a local directory, or a Hugging Face repo id (downloaded once)."""
    if os.path.isdir(bundle):
        return bundle
    from huggingface_hub import snapshot_download
    return snapshot_download(bundle)


T_BUCKETS = [100, 150, 200, 300, 450, 720]
B_BUCKETS = [2, 8, 32]   # CFG doubles the batch: 2 = one motion, 8 covers n <= 4, 32 covers n <= 16


def graphed(net):
    """Wrap the generator so every call runs on a fixed (batch, length) bucket through a CUDA graph."""
    import torch.nn.functional as Fn
    # one compiled graph per (batch, length) bucket: 18 > dynamo's default limit of 8, past which it silently runs eager
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 4 * len(T_BUCKETS) * len(B_BUCKETS))
    fwd = torch.compile(net.forward, mode="reduce-overhead", dynamic=False)

    def call(x, t, P, has, pad):
        B, T = x.shape[:2]; Tb = next(b for b in T_BUCKETS if b >= T); Bb = next(b for b in B_BUCKETS if b >= B)
        x = Fn.pad(x, (0, 0, 0, Tb - T, 0, Bb - B)); P = Fn.pad(P, (0, 0, 0, Tb - T, 0, Bb - B))
        has = Fn.pad(has, (0, 0, 0, 0, 0, Bb - B)); t = Fn.pad(t, (0, Bb - B), value=0.5)
        pad = Fn.pad(pad, (0, Tb - T, 0, Bb - B), value=True); pad[B:, 0] = False   # padded rows: keep one key unmasked
        return fwd(x, t, P, has, pad)[:B, :T].clone()
    return call


def _round(x, k=5):
    """Round every float in a move to 1e-5 (0.01 mm / 0.00057 deg): ~40% smaller JSON, far below what the robot resolves."""
    if isinstance(x, float): return round(x, k)
    if isinstance(x, list): return [_round(v, k) for v in x]
    if isinstance(x, dict): return {a: _round(b, k) for a, b in x.items()}
    return x


class Planner:
    """One fine-tuned LLM planner. ``path``: a merged model directory, or a bundle (directory or HF repo id).
    backend "vllm" (serving: FP8, MTP drafts, prefix cache) or "hf" (plain transformers, any device)."""
    def __init__(self, path, backend="vllm", fp8=False, spec_tokens=0, max_tokens=600, gpu_memory_utilization=0.80, **_):
        path = bundle_root(path)
        if os.path.isdir(os.path.join(path, "planner")): path = os.path.join(path, "planner")
        self.backend, self.max_tokens, self.name = backend, max_tokens, path
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        if backend == "vllm":
            from vllm import LLM
            self.llm = LLM(model=path, enable_prefix_caching=True, max_model_len=2048, dtype="bfloat16",
                           quantization="fp8" if fp8 else None, gpu_memory_utilization=gpu_memory_utilization,
                           limit_mm_per_prompt={"image": 0, "video": 0}, max_num_seqs=16,
                           # Qwen3.5/3.8 ship a multi-token-prediction head: draft k tokens, verify in one pass (lossless)
                           speculative_config={"method": "mtp", "num_speculative_tokens": spec_tokens} if spec_tokens else None)
        else:
            from generator.model import device
            from planner.distill.common import lm_class
            self.llm = lm_class(path).from_pretrained(path, dtype=torch.bfloat16).to(device()).eval()

    def text(self, prompt):
        return self.tok.apply_chat_template(student_messages(prompt), tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)

    def complete(self, texts, temperature, seed):
        """One completion per text. ``temperature`` and ``seed`` are one value for all, or one per text."""
        if self.backend == "vllm":
            from vllm import SamplingParams
            ts = temperature if isinstance(temperature, list) else [temperature] * len(texts)
            ss = seed if isinstance(seed, list) else [seed] * len(texts)
            sp = [SamplingParams(temperature=t, top_p=0.95 if t else 1.0, max_tokens=self.max_tokens, seed=s) for t, s in zip(ts, ss)]
            return [o.outputs[0].text for o in self.llm.generate(texts, sp, use_tqdm=False)]
        t = temperature[0] if isinstance(temperature, list) else temperature   # one value for the whole batch
        torch.manual_seed(seed[0] if isinstance(seed, list) else seed)
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.llm.device)
        kw = dict(do_sample=True, temperature=t, top_p=0.95) if t else dict(do_sample=False)
        with torch.no_grad():
            g = self.llm.generate(**enc, max_new_tokens=self.max_tokens, pad_token_id=self.tok.pad_token_id, **kw)
        return self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    @staticmethod
    def parse(text):
        m = re.search(r"\{.*\}", text.split("</think>")[-1], re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
        r = d.get("recipe")
        return (d.get("idea", ""), r) if r and not check(r) else (d.get("idea", ""), None)

    def plan(self, prompt, seed=0, retries=2, batched=False):
        """Greedy recipe; if it is invalid, up to ``retries`` samples at temperature 0.7.

        ``batched=False`` tries them one after another, only when needed. ``batched=True`` decodes the greedy answer
        and all retries together in one call: the greedy answer still wins when it is valid (same output), and an
        invalid one costs no extra round, at the price of a slightly slower decode for every request.
        """
        text = self.text(prompt)
        if batched and retries and self.backend == "vllm":
            outs = self.complete([text] * (1 + retries), [0.0] + [0.7] * retries, [seed + k for k in range(1 + retries)])
            for idea, rec in map(self.parse, outs):
                if rec:
                    return idea, rec
            raise ValueError("planner did not produce a valid recipe")
        idea, rec = self.parse(self.complete([text], 0.0, seed)[0])
        for k in range(retries):
            if rec: break
            idea, rec = self.parse(self.complete([text], 0.7, seed + k + 1)[0])
        if rec is None:
            raise ValueError("planner did not produce a valid recipe")
        return idea, rec


class MotionEngine:
    """planners: {effort: Planner}, e.g. {"high": 27B, "low": 4B}; one shared generator (plan -> motion)."""
    def __init__(self, planners, generator="checkpoints/generator.pt", steps=8, cfg=1.5, expand_fc=2.0, expand_kdt=0.25, **_):
        self.planners = planners; self.default = "high" if "high" in planners else next(iter(planners))
        self.steps, self.cfg, self.fc, self.kdt = steps, cfg, expand_fc, expand_kdt
        self.lock = threading.Lock()
        self.net, self.stats, self.dev = load(generator, "cuda"); self.reach = Reach()
        self.net.forward = graphed(self.net)
        self._warm()

    def _warm(self):
        """Prefill (and cache) each planner's system prompt, and capture every generator CUDA graph up front
        (3 batch buckets x 6 length buckets), so no request ever pays a capture."""
        for effort, p in self.planners.items():
            self.dense("warm-up. You stretch after waking.", n=1, effort=effort)
            p.complete([p.text("warm-up. You stretch after waking.")], 0.7, 1)       # the resample path's kernels
            p.plan("warm-up. You stretch after waking.", 0, retries=2, batched=True)  # and the batched-retry path
        from planner.dsl import variants as V
        for T in T_BUCKETS:
            dur = T / 25 - 0.3; k = int(np.ceil(dur / 5))          # segments are capped at 10 s by the checker
            pl = V(" | ".join(f"go {dur / k:.2f} e={20 + 5 * (i % 2)}" for i in range(k)), 1, fc=self.fc, kdt=self.kdt)
            for n in (1, 4, 16):
                for _ in range(2): generate_batch(self.net, self.stats, pl * n, self.dev, seeds=list(range(n)), steps=self.steps, cfg=self.cfg)

    def _plan(self, prompt, n, seed, effort, retries=2, batched=False):
        """prompt -> planner LLM -> recipe -> n plan variants (sparse keyframes)."""
        effort = effort or self.default
        if effort not in self.planners:
            raise ValueError(f"effort {effort!r} not loaded; available: {sorted(self.planners)}")
        t0 = time.perf_counter()
        idea, rec = self.planners[effort].plan(prompt, seed, retries, batched)
        plans = variants(rec, n, seed=seed, fc=self.fc, kdt=self.kdt)
        return effort, idea, rec, plans, round((time.perf_counter() - t0) * 1e3)

    def sparse(self, prompt, n=1, seed=0, effort=None, retries=2, batched_retries=False):
        """Planner only: the recipe and its n keyframe plans. Same input as dense(); the same seed gives the plans dense() uses."""
        with self.lock:
            effort, idea, rec, plans, planner_ms = self._plan(prompt, n, seed, effort, retries, batched_retries)
        return dict(prompt=prompt, effort=effort, idea=idea, recipe=rec, plans=plans,
                    durations_s=[p["duration"] for p in plans], timing_ms=dict(planner=planner_ms, total=planner_ms))

    def dense(self, prompt, n=1, seed=0, effort=None, retries=2, batched_retries=False):
        """Full pipeline: plans -> 25 Hz motion (flow-matching generator, one batched pass) -> reachable Reachy moves."""
        with self.lock:
            effort, idea, rec, plans, planner_ms = self._plan(prompt, n, seed, effort, retries, batched_retries)
            t1 = time.perf_counter()
            trajs = generate_batch(self.net, self.stats, plans, self.dev, seeds=[seed * 1000 + i for i in range(n)],
                                   steps=self.steps, cfg=self.cfg)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            moves = [_round(self.reach.project(to_move(A, prompt))[0]) for A in trajs]
            t3 = time.perf_counter()
        gen_ms, reach_ms = round((t2 - t1) * 1e3), round((t3 - t2) * 1e3)
        return dict(prompt=prompt, effort=effort, idea=idea, recipe=rec, moves=moves,
                    durations_s=[round(len(A) / FPS, 2) for A in trajs],
                    timing_ms=dict(planner=planner_ms, generator=gen_ms, reachability=reach_ms, total=planner_ms + gen_ms + reach_ms))
