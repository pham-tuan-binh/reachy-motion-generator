"""Fine-tune open LLMs to write recipes like the teacher (Unsloth LoRA on one GPU).

  1. sft.py       distill_data/dataset.jsonl -> chat SFT set (leak filter, OOD blocklist, input variants)
  2. train.py     LoRA SFT (loss on the answer only), best checkpoint by val loss, verified 16-bit merge
  3. evaluate.py  probes (physical checks on OOD prompts), plan agreement with the teacher, real-clip identification
"""
