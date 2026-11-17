"""Load the retained RL adapter using the text-only extraction used in evaluation."""
import json

import torch

from inference.defaults import default_config, planner_checkpoints
from inference.engine import Planner
from planner.state import state_messages


def load_text_model(path, dtype):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    raw = json.loads((path / "config.json").read_text())
    config = Qwen3_5TextConfig.from_dict(raw.get("text_config", raw))
    kwargs = {"config": config, "dtype": dtype, "attn_implementation": "sdpa", "output_loading_info": True}
    if "text_config" in raw:
        kwargs["key_mapping"] = {r"^model\.language_model\.": "model."}
    model, info = Qwen3_5ForCausalLM.from_pretrained(str(path), **kwargs)
    unexpected = [key for key in info["unexpected_keys"]
                  if not key.startswith(("model.visual.", "mtp.", "model.mtp."))]
    if info["missing_keys"] or info.get("mismatched_keys") or unexpected:
        raise RuntimeError(f"Unmatched text weights: {info}")
    return model


class RetainedPlanner:
    """Pinned base plus retained RL LoRA, merged for greedy inference."""

    def __init__(self, device):
        from peft import PeftModel
        from transformers import AutoTokenizer

        base_path, adapter_path = planner_checkpoints()
        self.settings = default_config()["planner_settings"]
        self.tokenizer = AutoTokenizer.from_pretrained(base_path, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = str(device)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        base = load_text_model(base_path, dtype)
        self.model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False).merge_and_unload()
        self.model = self.model.to(device).eval()

    def plan(self, prompt, current_pose):
        tokenizer, model = self.tokenizer, self.model
        text = tokenizer.apply_chat_template(state_messages(prompt, current_pose), tokenize=False,
            add_generation_prompt=True, enable_thinking=self.settings["enable_thinking"])
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        if encoded.input_ids.shape[1] > self.settings["max_prompt_tokens"]:
            raise ValueError("Prompt exceeds context allowance; refusing state truncation")
        original = model.generation_config.eos_token_id
        original = [] if original is None else [original] if isinstance(original, int) else original
        stops = list(dict.fromkeys([tokenizer.eos_token_id, *original]))
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=self.settings["max_new_tokens"],
                do_sample=self.settings["do_sample"], eos_token_id=stops,
                pad_token_id=tokenizer.pad_token_id, use_cache=True)
        tokens = generated[0, encoded.input_ids.shape[1]:].tolist()
        if not tokens or tokens[-1] not in stops:
            raise ValueError("Planner exhausted its completion allowance; refusing a truncated recipe")
        completion = tokenizer.decode(tokens, skip_special_tokens=True)
        idea, recipe = Planner.parse(completion)
        if recipe is None:
            raise ValueError("Planner did not produce a valid recipe")
        return idea, recipe, completion
