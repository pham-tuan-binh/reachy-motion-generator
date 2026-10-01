"""Renderer CLI.

  python -m renderer video motions/*.json --out videos/
  python -m renderer sheet motions/sad__0.json motions/sad__1.json --out sad.png
  python -m renderer grid  motions/*.json --out all.mp4 [--cols 6]
"""
import argparse
import glob
import os


def expand(paths):
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    return out


def main():
    ap = argparse.ArgumentParser(prog="renderer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["video", "sheet", "grid"]); ap.add_argument("moves", nargs="+", help="move .json files or folders")
    ap.add_argument("--out", required=True); ap.add_argument("--cols", type=int, default=None)
    a = ap.parse_args(); paths = expand(a.moves)
    from renderer import outputs
    if a.mode == "video": outputs.videos(paths, a.out)
    elif a.mode == "sheet": outputs.sheet(paths, a.out, ncols=a.cols or 12)
    else: outputs.grid(paths, a.out, ncols=a.cols or 6)


if __name__ == "__main__":
    main()
