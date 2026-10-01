"""Shared by the distillation scripts and the server: model loading, the local embedder, plan descriptors."""
import numpy as np


def lm_class(path):
    """The transformers class a checkpoint was saved as. Multimodal checkpoints (...ForConditionalGeneration, e.g.
    Qwen3.5 / 3.8) must NOT be opened as a text-only CausalLM: their weights live under model.language_model.*, so
    every key would miss and be silently re-initialised (or every LoRA key silently fail to apply)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    arch = (AutoConfig.from_pretrained(path).architectures or [""])[0]
    return AutoModelForImageTextToText if "ConditionalGeneration" in arch else AutoModelForCausalLM


class LocalEmbedder:
    """Qwen3-Embedding-0.6B, last-token pooling: the encoder the 0.72 leak threshold was calibrated on."""
    def __init__(self, name="Qwen/Qwen3-Embedding-0.6B", device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch; self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.m = AutoModel.from_pretrained(name, torch_dtype=torch.float16 if self.dev == "cuda" else torch.float32).to(self.dev).eval()

    def __call__(self, texts, batch=128):
        out = []
        for i in range(0, len(texts), batch):
            b = self.tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=96).to(self.dev)
            with self.torch.no_grad():
                h = self.m(**b).last_hidden_state[:, -1].float()      # left padding -> last token
            out.append(self.torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(out)


DESC_NAMES = ["duration", "ear_mean", "ear_min", "ear_max", "pitch_mean", "pitch_min", "pitch_max",
              "z_mean", "z_max", "z_min", "yaw_range", "roll_range", "energy_mean", "energy_max"]


def plan_descriptors(recipe):
    """Deterministic expansion (no jitter, amp = tempo = 1) -> the descriptor vector named by DESC_NAMES."""
    from planner.dsl import expand
    F = expand(recipe, np.random.default_rng(0)); T = len(F) / 25
    ears = F[:, :2].mean(1)
    return np.array([T, ears.mean(), ears.min(), ears.max(), F[:, 2].mean(), F[:, 2].min(), F[:, 2].max(),
                     F[:, 5].mean(), F[:, 5].max(), F[:, 5].min(), np.ptp(F[:, 4]), np.ptp(F[:, 3]), F[:, 7].mean(), F[:, 7].max()])
