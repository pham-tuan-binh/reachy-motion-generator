"""Frozen text decoder used by the two state-aware planners."""
import json

import torch

from inference.defaults import default_config, intent_asset, planner_checkpoints
from inference.engine import Planner
from planner.intent import compile_recipe, intent_messages, parse_intent
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


class TextDecoder:
    """Load the pinned text decoder and merge the selected adapter for inference."""

    def __init__(self, device, config):
        from peft import PeftModel
        from transformers import AutoTokenizer

        base_path, adapter_path = planner_checkpoints(config)
        self.settings = config["planner_settings"]
        self.tokenizer = AutoTokenizer.from_pretrained(base_path, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = str(device)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        base = load_text_model(base_path, dtype)
        self.model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False).merge_and_unload()
        self.model = self.model.to(device).eval()

    def complete(self, messages):
        tokenizer, model = self.tokenizer, self.model
        text = tokenizer.apply_chat_template(messages, tokenize=False,
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
            raise ValueError("Planner exhausted its completion allowance; refusing a truncated completion")
        return tokenizer.decode(tokens, skip_special_tokens=True)


class RetainedPlanner(TextDecoder):
    """The v1.3 free-form recipe planner with the retained v6 RL adapter."""

    def __init__(self, device):
        super().__init__(device, default_config("recipe"))

    def plan(self, prompt, current_pose):
        completion = self.complete(state_messages(prompt, current_pose))
        idea, recipe = Planner.parse(completion)
        if recipe is None:
            raise ValueError("Planner did not produce a valid recipe")
        return idea, recipe, completion

    def plan_details(self, prompt, current_pose):
        idea, recipe, completion = self.plan(prompt, current_pose)
        return {"idea": idea, "recipe": recipe, "completion": completion}


class IntentPlanner(TextDecoder):
    """Extract one v8 gesture intent and compile it against the settled pose."""

    def __init__(self, device):
        config = default_config()
        self.examples = intent_asset("intent_examples.json", config)
        self.defaults = intent_asset("intent_defaults.json", config)
        super().__init__(device, config)

    def plan_details(self, prompt, current_pose):
        completion = self.complete(intent_messages(prompt, current_pose, self.examples))
        intent = parse_intent(completion)
        recipe = compile_recipe(intent, current_pose, defaults=self.defaults)
        return {"idea": intent["gesture"], "intent": intent, "recipe": recipe, "completion": completion}
