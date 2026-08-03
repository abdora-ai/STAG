#!/usr/bin/env python3
"""
Peek at GraSP annotation JSONs (COCO-style) to figure out which files give us
per-keyframe *tool presence* labels for the tool-detection task.

Usage:
    python tools/peek_grasp.py --root .
"""
import argparse
import json
import os
import sys
from collections import Counter


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def summarize_coco_like(path, data):
    """Dump the top-level structure of a COCO-style file."""
    print(f"\n[file] {path}")
    print(f"[size] {os.path.getsize(path)/1e6:.2f} MB")

    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"[keys] {keys}")

        for k in keys:
            v = data[k]
            if isinstance(v, list):
                print(f"  '{k}' : list of {len(v)}")
                if v:
                    first = v[0]
                    print(f"     first element type = {type(first).__name__}")
                    if isinstance(first, dict):
                        print(f"     first element keys = {list(first.keys())}")
                        # Print the first element compactly
                        s = json.dumps(first, ensure_ascii=False, indent=None)
                        print(f"     first element : {s[:300]}{'...' if len(s)>300 else ''}")
            elif isinstance(v, dict):
                print(f"  '{k}' : dict of {len(v)} entries")
                dk = list(v.keys())
                print(f"     first keys = {dk[:6]}")
            else:
                print(f"  '{k}' = {repr(v)[:200]}")

        # Category listings if present (surgical instruments etc.)
        for cat_key in ("categories", "instrument_categories", "phase_categories",
                        "step_categories", "action_categories"):
            if cat_key in data and isinstance(data[cat_key], list):
                cats = data[cat_key]
                # Try to render as (id, name)
                rendered = []
                for c in cats[:60]:
                    if isinstance(c, dict):
                        rendered.append((c.get("id"), c.get("name")))
                    else:
                        rendered.append(c)
                print(f"\n[{cat_key}]  {len(cats)} entries")
                for r in rendered:
                    print(f"    {r}")

        # For an "images" list (COCO keyframes), show a few file_names
        if "images" in data and isinstance(data["images"], list):
            imgs = data["images"]
            fname_counter = Counter()
            for im in imgs:
                if isinstance(im, dict):
                    fn = im.get("file_name") or im.get("name") or ""
                    # Take the top-level parent (case dir)
                    top = fn.split("/")[0] if "/" in fn else "(no subdir)"
                    fname_counter[top] += 1
            print(f"\n[images] {len(imgs)} entries; parent-dir histogram:")
            for k, v in fname_counter.most_common():
                print(f"    {k}: {v}")
            if imgs:
                s = json.dumps(imgs[0], ensure_ascii=False)
                print(f"    first image entry: {s[:400]}{'...' if len(s)>400 else ''}")

        # For an "annotations" list, show a few examples
        if "annotations" in data and isinstance(data["annotations"], list):
            anns = data["annotations"]
            print(f"\n[annotations] {len(anns)} entries")
            for a in anns[:2]:
                s = json.dumps(a, ensure_ascii=False)
                print(f"    {s[:500]}{'...' if len(s)>500 else ''}")


def main(args):
    root = os.path.abspath(args.root)
    ann_dir = os.path.join(root, "GraSP", "GraSP_1fps", "annotations")
    if not os.path.isdir(ann_dir):
        print(f"[error] annotations dir not found: {ann_dir}")
        sys.exit(1)

    entries = sorted(os.listdir(ann_dir))
    print(f"[annotations dir] {ann_dir}")
    print(f"[entries] {entries}")

    # README first
    readme = os.path.join(ann_dir, "README.txt")
    if os.path.isfile(readme):
        print("\n===== README.txt =====")
        with open(readme, "r") as f:
            for line in f:
                print("  " + line.rstrip("\n"))

    # JSON files
    for fn in entries:
        if not fn.endswith(".json"):
            continue
        path = os.path.join(ann_dir, fn)
        try:
            data = load_json(path)
        except Exception as e:
            print(f"\n[error] cannot parse {path}: {e}")
            continue
        print("\n" + "=" * 78)
        summarize_coco_like(path, data)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    main(p.parse_args())
