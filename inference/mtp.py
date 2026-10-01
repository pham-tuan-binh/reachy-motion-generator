"""The planner's multi-token-prediction (MTP) head, which vLLM uses to draft tokens (speculative decoding, lossless).

  python -m inference.mtp add   PLANNER_DIR [--base Qwen/Qwen3.5-4B]       # restore the base model's head (the merge drops it)
  python -m inference.mtp data  PLANNER_DIR SFT_TRAIN.jsonl OUT.jsonl      # the planner answers its own training prompts
  python -m inference.mtp train PLANNER_DIR DATA.jsonl OUT_DIR [--epochs 2] [--steps 3]

The stock head was trained on general text, so it drafts our recipes poorly. Fine-tuning only the head on the
planner's own answers (the planner frozen) raises accepted tokens per step from ~2.3 to ~3.2-3.7 and cuts planner
time by 19-26%, with identical outputs. OUT_DIR is a hard-linked copy of PLANNER_DIR with a new model-mtp.safetensors.

Training matches vLLM's qwen3_5_mtp: x = fc([norm_e(embed(tok[t+1])), norm_h(h[t])]) -> one full-attention decoder layer -> norm
-> shared lm_head predicts tok[t+2], where h is the target's post-final-norm hidden state. Draft steps 2..k reuse the
same layer on its own output hidden, so those are trained by unrolling k steps (teacher-forced tokens).
"""
import argparse
import glob
import json
import math
import os
import random
import time


def add_head(planner, base):
    """Copy the base checkpoint's mtp.* tensors into a merged fine-tune (into model-mtp.safetensors + the index)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    root = snapshot_download(base, allow_patterns=["*.json", "*.safetensors"])
    mtp = {k: v for k, v in json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"].items() if k.startswith("mtp.")}
    tensors = {}
    for shard in set(mtp.values()):
        with safe_open(f"{root}/{shard}", "pt") as f:
            tensors.update({k: f.get_tensor(k) for k, s in mtp.items() if s == shard})
    save_file(tensors, f"{planner}/model-mtp.safetensors", metadata={"format": "pt"})
    ip = glob.glob(f"{planner}/*.index.json")
    if ip:
        idx = json.load(open(ip[0]))
    else:   # single-file checkpoint: build an index
        with safe_open(f"{planner}/model.safetensors", "pt") as f: idx = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in f.keys()}}
        ip = [f"{planner}/model.safetensors.index.json"]
    idx["weight_map"].update({k: "model-mtp.safetensors" for k in tensors})
    json.dump(idx, open(ip[0], "w"), indent=1)
    print(f"added {len(tensors)} MTP tensors from {base} -> {planner}/model-mtp.safetensors")


def make_data(planner, sft, out):
    """On-policy data: the planner's greedy answers to its training prompts, plus 1/3 of them again at T = 0.7.
    Probe and val prompts are excluded, so speed benchmarks on them stay honest."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from planner.distill.probes import PROBES
    from planner.prompt import student_messages
    tok = AutoTokenizer.from_pretrained(planner)
    held = {p for p, _, _ in PROBES} | {json.loads(l)["prompt"] for l in open("distill_data/val.jsonl")}
    prompts = sorted({json.loads(l)["messages"][1]["content"] for l in open(sft)} - held)
    texts = [tok.apply_chat_template(student_messages(p), tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
    llm = LLM(model=planner, quantization="fp8", max_model_len=2048, gpu_memory_utilization=0.85, max_num_seqs=64,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0}, mamba_ssm_cache_dtype="bfloat16")
    greedy = llm.generate(texts, SamplingParams(temperature=0, max_tokens=600))
    idx = random.Random(0).sample(range(len(texts)), len(texts) // 3)
    sampled = llm.generate([texts[i] for i in idx], SamplingParams(temperature=0.7, top_p=0.95, max_tokens=600, seed=1))
    with open(out, "w") as f:
        for t, o in zip(texts, greedy): f.write(json.dumps({"prompt": t, "answer": o.outputs[0].text}) + "\n")
        for i, o in zip(idx, sampled): f.write(json.dumps({"prompt": texts[i], "answer": o.outputs[0].text}) + "\n")
    print(f"{len(texts) + len(idx)} answers to {len(prompts)} prompts -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("add"); s.add_argument("planner"); s.add_argument("--base", default="Qwen/Qwen3.5-4B")
    s = sub.add_parser("data"); s.add_argument("planner"); s.add_argument("sft"); s.add_argument("out")
    s = sub.add_parser("train"); s.add_argument("planner"); s.add_argument("data"); s.add_argument("out")
    s.add_argument("--epochs", type=float, default=2); s.add_argument("--lr", type=float, default=5e-5)
    s.add_argument("--steps", type=int, default=3); s.add_argument("--bs", type=int, default=16); s.add_argument("--max-len", type=int, default=1024)
    a = ap.parse_args()
    if a.cmd == "add": add_head(a.planner, a.base)
    elif a.cmd == "data": make_data(a.planner, a.sft, a.out)
    else: train(a)


def train(a):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
    dev = "cuda"; torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(a.planner)
    target = AutoModelForImageTextToText.from_pretrained(a.planner, dtype=torch.bfloat16, device_map={"": dev}).eval().requires_grad_(False)
    lm = target.model.language_model; cfg = target.config.text_config; cfg._attn_implementation = "sdpa"
    embed, lm_head, rotary = lm.embed_tokens, target.lm_head, lm.rotary_emb
    H = cfg.hidden_size; full_idx = cfg.layer_types.index("full_attention")

    class MTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2 * H, H, bias=False)
            self.pre_fc_norm_embedding = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
            self.pre_fc_norm_hidden = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
            self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, full_idx)])
            self.norm = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)

        def forward(self, e, h, pos_emb):
            x = self.fc(torch.cat([self.pre_fc_norm_embedding(e), self.pre_fc_norm_hidden(h)], -1))
            return self.norm(self.layers[0](x, position_embeddings=pos_emb))

    sd = load_file(os.path.join(a.planner, "model-mtp.safetensors"))
    mtp = MTP().to(dev)
    missing, unexpected = mtp.load_state_dict({k[len("mtp."):]: v.float() for k, v in sd.items()}, strict=False)
    assert not missing and not unexpected, (missing, unexpected)   # fp32 master weights, bf16 autocast compute

    def encode(r):
        p = tok(r["prompt"], add_special_tokens=False).input_ids
        ans = tok(r["answer"] + "<|im_end|>", add_special_tokens=False).input_ids
        return (p + ans)[: a.max_len], len(p)
    data = [encode(json.loads(l)) for l in open(a.data)]; random.Random(0).shuffle(data)
    val, train = data[:256], data[256:]
    print(f"{len(train)} train / {len(val)} val sequences", flush=True)

    def batch(items):
        T = max(len(i) for i, _ in items)
        ids = torch.full((len(items), T), tok.pad_token_id or 0); m = torch.zeros(len(items), T, dtype=torch.bool)
        for b, (i, p) in enumerate(items):
            ids[b, : len(i)] = torch.tensor(i); m[b, p : len(i)] = True       # loss on answer tokens only
        return ids.to(dev), m.to(dev)

    def losses(ids, m):
        with torch.no_grad():
            h = lm(input_ids=ids).last_hidden_state                              # post-final-norm, as vLLM passes it
            pos = torch.arange(ids.shape[1], device=dev)[None].expand(ids.shape[0], -1)
            cos, sin = rotary(h, pos)
        ce, acc, hid = [], [], h
        for k in range(1, a.steps + 1):
            T = ids.shape[1] - 1 - k
            if T <= 0: break
            tgt, mk = ids[:, k + 1 : k + 1 + T], m[:, k + 1 : k + 1 + T]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = mtp(embed(ids[:, k : k + T]), hid[:, :T], (cos[:, :T], sin[:, :T]))
                logits = lm_head(y[mk]).float()                                  # answer positions only (248k vocab)
            ce.append(F.cross_entropy(logits, tgt[mk])); acc.append((logits.argmax(-1) == tgt[mk]).float().mean().item())
            hid = y
        return ce, acc

    def evaluate():
        mtp.eval(); A = []
        with torch.no_grad():
            for i in range(0, len(val), a.bs): A.append(losses(*batch(val[i : i + a.bs]))[1])
        mtp.train(); return [round(sum(x[k] for x in A) / len(A), 3) for k in range(len(A[0]))]

    opt = torch.optim.AdamW(mtp.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    n_steps = math.ceil(len(train) / a.bs * a.epochs); W = [1.0, 0.8, 0.6, 0.5, 0.4][: a.steps]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, n_steps) / n_steps)))
    order = [i for e in range(math.ceil(a.epochs)) for i in random.Random(e).sample(range(len(train)), len(train))]
    print("val per-step top-1 acc before:", evaluate(), flush=True); t0 = time.time()
    for s in range(n_steps):
        ce, _ = losses(*batch([train[j] for j in order[s * a.bs : (s + 1) * a.bs]]))
        loss = sum(w * c for w, c in zip(W, ce))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0); opt.step(); sched.step()
        if (s + 1) % 50 == 0: print(f"step {s + 1}/{n_steps} loss {loss.item():.3f} {[round(c.item(), 3) for c in ce]} {time.time() - t0:.0f}s", flush=True)
        if (s + 1) % 300 == 0 or s + 1 == n_steps: print("val per-step top-1 acc:", evaluate(), flush=True)

    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.planner):
        d = os.path.join(a.out, f)
        if f != "model-mtp.safetensors" and not os.path.lexists(d): os.link(os.path.realpath(os.path.join(a.planner, f)), d)   # HF cache files are symlinks
    new = {"mtp." + k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in mtp.state_dict().items()}
    assert set(new) == set(sd), set(new) ^ set(sd)
    save_file(new, os.path.join(a.out, "model-mtp.safetensors"), metadata={"format": "pt"})
    print("MTP_DONE", a.out)


if __name__ == "__main__":
    main()
