"""Text -> motion recipe -> motion plans. The recipe language, the prompts, and the teacher LLM client."""
import re


def read_lines(path):
    """Prompts file: one prompt per line; blank lines and # comments skipped."""
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


def slug(prompt):
    return re.sub(r"[^a-z0-9]+", "_", prompt.split(".")[0].lower()).strip("_")[:48]
