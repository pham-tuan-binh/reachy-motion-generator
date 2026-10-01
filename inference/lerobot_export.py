"""Exported trajectories -> a LeRobot dataset (one episode per teacher row) with a rendered MuJoCo video.

  MUJOCO_GL=egl REACHY_MINI_ROOT=... python -m inference.lerobot_export --motions build/motions.pkl \
      --repo binhpham/reachy-mini-massive-motion-library --root build/lerobot --workers 20 [--push] [--limit 20]

Each episode:
  observation.images.front  MuJoCo render of the official Reachy Mini model (320x256, 25 fps)
  observation.state/action  9-DoF trajectory [x y z roll pitch yaw antenna_right antenna_left body_yaw]
  task                      the prompt
meta/teacher_rows.jsonl maps episode_index -> id, source, family, idea, recipe (so the viewer can be filtered by teacher).
"""
import argparse
import json
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


from common.motion import DOF, FPS

W, H = 320, 256
FEATURES = {
    "observation.images.front": {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (9,), "names": DOF},
    "action": {"dtype": "float32", "shape": (9,), "names": DOF},
}


def build_shard(args):
    k, rows, repo, root = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from common.motion import to_move
    from renderer.sim import Sim
    sim = Sim(width=W, height=H)
    shard_root = Path(root) / f"shard_{k:02d}"; shutil.rmtree(shard_root, ignore_errors=True)
    ds = LeRobotDataset.create(repo_id=f"{repo}-shard{k:02d}", fps=FPS, features=FEATURES, root=shard_root,
                               robot_type="reachy_mini", use_videos=True, streaming_encoding=True)
    for r in rows:
        A = r["traj"]; frames = sim.play(to_move(A, r["prompt"]), FPS)
        for f, s in zip(frames, A):
            ds.add_frame({"observation.images.front": f, "observation.state": s, "action": s, "task": r["prompt"]})
        ds.save_episode()
    ds.finalize() if hasattr(ds, "finalize") else None
    return str(shard_root), [r["id"] for r in rows]


def card(repo, n, frames, counts):
    return f"""---
license: apache-2.0
task_categories: [robotics]
tags: [LeRobot, reachy_mini, text-to-motion, synthetic]
---

# Reachy Mini Massive Motion Library

{n:,} episodes ({frames / FPS / 3600:.1f} h at {FPS} fps). Each one is a prompt performed by Reachy Mini: the teacher's motion
*recipe*, turned into 25 Hz motion by the plan-to-motion flow generator, projected onto the robot's reachable set,
and rendered in MuJoCo with the official model.

| source | episodes | what |
|---|---|---|
""" + "\n".join(f"| `{s}` | {c:,} | {DESC.get(s, '')} |" for s, c in counts.items()) + """

- `observation.state` / `action`: the 9-DoF target trajectory `[x y z roll pitch yaw antenna_right antenna_left body_yaw]`
  (metres / radians). Play it with the SDK by converting to a recorded move.
- `task`: the prompt. `meta/teacher_rows.jsonl` gives each episode's `source`, `family`, `idea` and `recipe`.
- `astra` and `astra_lively` contain the same 5,000 prompts performed in two styles: the original choreography, and
  a re-authored "lively" version with more energy, rhythm and ear expression.

Pipeline: LLM planner -> motion recipe -> plan -> flow-matching generator -> reachability projection -> MuJoCo render.
"""


DESC = {"claude": "hand-written by Claude (Anthropic)", "claude_events": "build-up/release events and long multi-phase stories (Claude)",
        "seed": "the original 287 hand-written recipes", "astra": "5,000 individually authored scenarios (Astra)",
        "astra_lively": "the same 5,000 prompts re-authored in the lively house style (Codex, gpt-6-astra)"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--motions", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--root", required=True); ap.add_argument("--workers", type=int, default=20); ap.add_argument("--limit", type=int)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    rows = pickle.load(open(a.motions, "rb"))[: a.limit]
    shards = [rows[i::a.workers] for i in range(a.workers)]
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(build_shard, [(k, s, a.repo, a.root) for k, s in enumerate(shards) if s]))
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    final = Path(a.root) / "final"; shutil.rmtree(final, ignore_errors=True)
    aggregate_datasets(repo_ids=[f"{a.repo}-shard{k:02d}" for k in range(len(results))], aggr_repo_id=a.repo,
                       roots=[Path(p) for p, _ in results], aggr_root=final)
    order = [i for _, ids in results for i in ids]; by_id = {r["id"]: r for r in rows}
    with open(final / "meta" / "teacher_rows.jsonl", "w") as fh:
        for ep, i in enumerate(order):
            r = by_id[i]; fh.write(json.dumps(dict(episode_index=ep, id=i, prompt=r["prompt"], source=r["source"], family=r.get("family", ""),
                                                   idea=r.get("idea", ""), recipe=r["recipe"])) + "\n")
    from collections import Counter
    counts = Counter(by_id[i]["source"] for i in order)
    (final / "README.md").write_text(card(a.repo, len(order), sum(len(r["traj"]) for r in rows), dict(counts)))
    print(f"{len(order)} episodes -> {final}")
    if a.push:
        ds = LeRobotDataset(a.repo, root=final)
        ds.push_to_hub(private=False)
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(path_or_fileobj=str(final / "meta" / "teacher_rows.jsonl"), path_in_repo="meta/teacher_rows.jsonl", repo_id=a.repo, repo_type="dataset")
        api.upload_file(path_or_fileobj=str(final / "README.md"), path_in_repo="README.md", repo_id=a.repo, repo_type="dataset")
        print(f"pushed -> https://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()
