"""Unsloth LoRA SFT of a planner (loss on the assistant answer only), then a verified 16-bit merge.

  python -m planner.distill.train --data distill/sft --out distill/4b --model Qwen/Qwen3.5-4B
  python -m planner.distill.train --data distill/sft --out distill/27b --model Qwen/Qwen3.8-27B --epochs 1.5 --bs 8 --grad-accum 4

Keeps the adapter with the best val loss (evaluated every --eval-steps) and writes the merged model to <out>/merged.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B"); ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1536); ap.add_argument("--eval-steps", type=int, default=40)
    ap.add_argument("--merge-only", action="store_true", help="just merge <out>/adapter into <out>/merged")
    a = ap.parse_args()
    if not a.merge_only:
        train(a)
    merge(a.model, f"{a.out}/adapter", f"{a.out}/merged")


def train(a):
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=a.max_len, load_in_4bit=False, dtype=None)
    # q/k/v/o and the MLPs: every MLP and the full-attention layers. Qwen3.5's Gated DeltaNet layers name their
    # projections in_proj_qkv / in_proj_z / out_proj, so those stay frozen (adding them is the obvious next experiment).
    model = FastLanguageModel.get_peft_model(
        model, r=a.r, lora_alpha=a.r, lora_dropout=0.0, bias="none", use_gradient_checkpointing="unsloth", random_state=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ds = load_dataset("json", data_files={"train": f"{a.data}/train.jsonl", "val": f"{a.data}/val.jsonl"})
    fmt = lambda b: {"text": [tok.apply_chat_template(m, tokenize=False) for m in b["messages"]]}
    ds = ds.map(fmt, batched=True, remove_columns=ds["train"].column_names)

    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds["train"], eval_dataset=ds["val"],
        args=SFTConfig(output_dir=a.out, dataset_text_field="text", max_seq_length=a.max_len, per_device_train_batch_size=a.bs,
                       per_device_eval_batch_size=2, gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
                       warmup_ratio=0.05, weight_decay=0.01, logging_steps=10, eval_strategy="steps", eval_steps=a.eval_steps,
                       save_strategy="steps", save_steps=a.eval_steps, save_total_limit=2, load_best_model_at_end=True,
                       metric_for_best_model="eval_loss", greater_is_better=False, bf16=True, optim="adamw_8bit", seed=0, report_to="none"))
    # loss only on the assistant JSON: the prompt and system text are not what the planner must learn
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
    trainer.train()
    print(f"best checkpoint {trainer.state.best_model_checkpoint} (eval_loss {trainer.state.best_metric:.4f})")
    model.save_pretrained(f"{a.out}/adapter"); tok.save_pretrained(f"{a.out}/adapter")


def merge(base, adapter, out):
    """Merge the LoRA into a 16-bit model with plain PEFT (Unsloth's merged save copies read-only base shards from the
    HF cache and then fails to patch them). Loads the base with the class it was trained as (see ``lm_class``) and
    refuses to save unless LoRA layers were injected and the weights actually changed."""
    import shutil
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from planner.distill.common import lm_class
    shutil.rmtree(out, ignore_errors=True)
    m = lm_class(base).from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    pm = PeftModel.from_pretrained(m, adapter)
    injected = sum(1 for n, _ in pm.named_modules() if n.endswith("lora_A"))
    if injected == 0:
        raise SystemExit(f"merge: no LoRA layers matched {base}; adapter keys do not fit the model")
    name, w = next((n, p) for n, p in pm.named_parameters() if "base_layer.weight" in n and "lora" not in n.split(".")[-2])
    before = w.detach().float().clone()
    m = pm.merge_and_unload()
    after = dict(m.named_parameters())[name.replace("base_model.model.", "").replace(".base_layer", "")].detach().float()
    delta = (after - before).abs().max().item()
    if delta == 0:
        raise SystemExit("merge: weights unchanged after merging; refusing to save the base model as the planner")
    m.save_pretrained(out, safe_serialization=True); AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    print(f"merged {injected} LoRA layers (max weight change {delta:.2e}) -> {out}")


if __name__ == "__main__":
    main()
