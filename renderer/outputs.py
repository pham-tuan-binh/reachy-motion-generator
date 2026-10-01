"""Videos, single-motion contact sheets, and labelled grid videos."""
import os

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

from common.motion import FPS
from renderer.sim import Sim


def _label(img, text, h=16):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, h], fill=(0, 0, 0)); d.text((4, 2), text, fill=(255, 255, 0))
    return np.asarray(im)


def videos(paths, outdir, width=520, height=420):
    os.makedirs(outdir, exist_ok=True); s = Sim(width, height); out = []
    for p in paths:
        fr = s.play_file(p); nm = os.path.basename(p)[:-5]
        o = os.path.join(outdir, nm + ".mp4"); iio.mimwrite(o, fr, fps=FPS, quality=8, macro_block_size=1)
        print(f"  {nm:48s} {len(fr)/FPS:5.1f}s -> {o}" + (f"  ({s.ik_fail} unreachable frames held)" if s.ik_fail else ""))
        out.append(o)
    return out


def sheet(paths, out, ncols=12, tile=170):
    """One row per motion, ``ncols`` evenly spaced frames: high temporal resolution for ONE prompt."""
    s = Sim(tile, tile); rows = []
    for p in paths:
        fr = s.play_file(p); idx = np.linspace(0, len(fr) - 1, ncols).astype(int)
        rows.append(_label(np.concatenate([fr[i] for i in idx], 1), f"{os.path.basename(p)[:-5][:60]}  [{len(fr)/FPS:.1f}s]", 12))
    Image.fromarray(np.concatenate(rows, 0)).save(out); print(f"sheet -> {out}")
    return out


def grid(paths, out, ncols=6, tile=260):
    """Tile several motions into one labelled video; shorter clips loop."""
    s = Sim(tile, int(tile * 420 / 520)); clips = []
    for p in paths:
        lab = os.path.basename(p)[:-5].replace("_", " ")
        clips.append(np.stack([_label(f, lab) for f in s.play_file(p)]))
    T = max(len(c) for c in clips); h, w = clips[0].shape[1:3]; nrow = -(-len(clips) // ncols)
    frames = np.zeros((T, nrow * h, ncols * w, 3), np.uint8)
    for i, c in enumerate(clips):
        r, k = divmod(i, ncols)
        for t in range(T): frames[t, r * h:(r + 1) * h, k * w:(k + 1) * w] = c[t % len(c)]
    iio.mimwrite(out, list(frames), fps=FPS, quality=8, macro_block_size=1); print(f"grid ({len(clips)} motions) -> {out}")
    return out
