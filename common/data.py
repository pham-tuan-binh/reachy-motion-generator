"""Real reference motion: Pollen's emotions and dances libraries on the Hugging Face Hub."""
import copy
import glob
import json
import os
import re

from huggingface_hub import snapshot_download

from .reach import Reach

EMOTIONS = "pollen-robotics/reachy-mini-emotions-library"
DANCES = "pollen-robotics/reachy-mini-dances-library"

# Emotions kept out of generator training and used for evaluation. Pick emotions that span the space
# (sad/low, angry/high-energy, positive, calm) so the held-out score means something.
HELD_OUT = ["disgusted1", "downcast1", "electric1", "exhausted1", "frustrated1", "lonely1",
            "rage1", "relief1", "surprised1", "thoughtful1", "welcoming1", "impatient1"]


def library(repo, project=True, cache_dir=None):
    """Download a moves library and return ``[(name, move)]``, optionally projected onto the
    reachable set (~5% of emotion-library frames are not IK-reachable as recorded)."""
    root = snapshot_download(repo, repo_type="dataset", allow_patterns=["*.json"], cache_dir=cache_dir)
    R = Reach() if project else None; out = []
    for p in sorted(glob.glob(os.path.join(root, "*.json"))):
        m = json.load(open(p))
        if "set_target_data" not in m:
            continue
        if R is not None:
            m, _ = R.project(copy.deepcopy(m))
        out.append((os.path.basename(p)[:-5], m))
    return out


def caption(name, move):
    """Library clip -> prompt, e.g. ("downcast1", move) -> "downcast. <the clip's description>"."""
    return f"{re.sub(r'\d+$', '', name).replace('_', ' ')}. {move.get('description', '')}".strip()
