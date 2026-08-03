#!/usr/bin/env python3
"""
Walk the Cholec80, CholecT50, and GraSP dataset folders and report their layout.
Also probes common label-file locations and prints their schema, so we can plan
the feature-caching + head-training scripts precisely.

Usage:
    python tools/inspect_datasets.py --root /path/to/LEMON
"""
import argparse
import json
import os
import sys
from collections import Counter


DATASETS = ("Cholec80", "CholecT50", "GraSP")


def _safe_listdir(path):
    try:
        return sorted(os.listdir(path))
    except OSError as e:
        print(f"  [error] cannot list {path}: {e}")
        return []


def walk_summary(root, max_depth=3, dir_indent=0, max_dirs=8, max_files=5):
    if not os.path.isdir(root):
        print(("  " * dir_indent) + f"(missing) {root}")
        return
    entries = _safe_listdir(root)
    dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(root, e))]
    ext_counter = Counter(os.path.splitext(f)[1].lower() for f in files)
    prefix = "  " * dir_indent
    label = os.path.basename(root) or root
    print(f"{prefix}{label}/  ({len(dirs)} dirs, {len(files)} files, "
          f"ext={dict(ext_counter)})")
    if max_depth <= 0:
        return
    for d in dirs[:max_dirs]:
        walk_summary(os.path.join(root, d), max_depth - 1, dir_indent + 1)
    if len(dirs) > max_dirs:
        print(f"{prefix}  ... and {len(dirs)-max_dirs} more dirs")
    for f in files[:max_files]:
        print(f"{prefix}  {f}")
    if len(files) > max_files:
        print(f"{prefix}  ... and {len(files)-max_files} more files")


def inspect_json(path, max_keys=15):
    print(f"  [json] {path}")
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"    parse error: {e}")
        return
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"    dict of {len(keys)} entries")
        print(f"    first keys: {keys[:max_keys]}")
        if keys:
            first_val = data[keys[0]]
            print(f"    first value type = {type(first_val).__name__}, "
                  f"sample = {str(first_val)[:200]}")
    elif isinstance(data, list):
        print(f"    list of {len(data)} entries")
        if data:
            print(f"    first entry type = {type(data[0]).__name__}")
            print(f"    first entry: {str(data[0])[:250]}")


def find_label_files(root, patterns=("label", "annot", "split", "phase", "tool", "verb")):
    """Return small candidate list of .json/.csv/.txt files whose name matches any pattern."""
    hits = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if not (low.endswith(".json") or low.endswith(".csv")
                    or low.endswith(".txt") or low.endswith(".pkl")):
                continue
            if any(p in low for p in patterns):
                hits.append(os.path.join(dirpath, fn))
        # keep it shallow to avoid dumping every frame-level annotation
        if dirpath.count(os.sep) - root.count(os.sep) >= 3:
            _dirnames.clear()
    return hits[:20]


def count_files_by_ext(root, exts=(".jpg", ".jpeg", ".png", ".mp4", ".json", ".csv", ".txt")):
    counts = Counter()
    for _dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                counts[ext] += 1
    return counts


def main(args):
    root = os.path.abspath(args.root)
    print(f"[root] {root}")

    if not os.path.isdir(root):
        print(f"[error] root does not exist: {root}")
        sys.exit(1)

    for name in DATASETS:
        print(f"\n===================== {name} =====================")
        dpath = os.path.join(root, name)
        if not os.path.isdir(dpath):
            print(f"  (not found under root)")
            continue
        print(f"Path: {dpath}")

        print("\n[layout] shallow walk (max_depth=3):")
        walk_summary(dpath, max_depth=args.max_depth)

        print("\n[counts] file counts (full recursive) by extension:")
        counts = count_files_by_ext(dpath)
        for k, v in counts.most_common():
            print(f"  {k}: {v}")

        print("\n[labels] candidate label / annotation / split files:")
        candidates = find_label_files(dpath)
        if not candidates:
            print("  (none matched)")
        for cand in candidates:
            print(f"  - {cand}")

        # Show the schema of the first few JSONs (limited to short files to be safe)
        json_hits = [c for c in candidates if c.endswith(".json")]
        for j in json_hits[:3]:
            size = os.path.getsize(j)
            if size > 5_000_000:
                print(f"\n[json-skip] {j} ({size/1e6:.1f} MB) - too large to preview safely")
                continue
            inspect_json(j)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="Path to LEMON root (contains Cholec80, CholecT50, GraSP)")
    p.add_argument("--max-depth", type=int, default=3)
    main(p.parse_args())
