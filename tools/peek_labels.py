#!/usr/bin/env python3
"""
Peek at the label formats of Cholec80, CholecT50 and (if extracted) GraSP so
we can pin the parsers used by the feature-caching and head-training scripts.

Usage:
    python tools/peek_labels.py --root .
"""
import argparse
import json
import os
import sys


def head_file(path, n=20):
    print(f"\n--- head({n}) of {path} ---")
    try:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= n:
                    print(f"    ... (truncated at {n} lines)")
                    break
                print("    " + line.rstrip("\n"))
    except Exception as e:
        print(f"    [error] {e}")


def peek_cholec80(root):
    base = os.path.join(root, "Cholec80")
    print(f"\n========== Cholec80 : {base} ==========")
    phase_dir = os.path.join(base, "phase_annotations")
    tool_dir = os.path.join(base, "tool_annotations")
    if os.path.isdir(phase_dir):
        files = sorted(os.listdir(phase_dir))
        print(f"phase_annotations/  {len(files)} files, e.g. {files[:3]}")
        if files:
            head_file(os.path.join(phase_dir, files[0]))
    if os.path.isdir(tool_dir):
        files = sorted(os.listdir(tool_dir))
        print(f"tool_annotations/  {len(files)} files, e.g. {files[:3]}")
        if files:
            head_file(os.path.join(tool_dir, files[0]))

    # Count frames of the same video to compare with annotation-row count
    frames_dir = os.path.join(base, "frames", "video01")
    if os.path.isdir(frames_dir):
        n = len(os.listdir(frames_dir))
        print(f"\nvideo01 frame PNGs: {n}")


def peek_cholect50(root):
    base = os.path.join(root, "CholecT50")
    print(f"\n========== CholecT50 : {base} ==========")

    # label_mapping.txt: verb / triplet class names
    lm = os.path.join(base, "label_mapping.txt")
    if os.path.isfile(lm):
        print(f"\n[label_mapping.txt]")
        head_file(lm, n=80)

    # First few JSON label files
    labels_dir = os.path.join(base, "labels")
    if os.path.isdir(labels_dir):
        js = sorted([f for f in os.listdir(labels_dir) if f.endswith(".json")])
        print(f"\n[labels/] {len(js)} files, e.g. {js[:5]}")
        if js:
            path = os.path.join(labels_dir, js[0])
            print(f"\n--- {path} ---")
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                print(f"    top-level keys: {list(data.keys())[:20]}")
                # CholecT50: expect "annotations" dict frame -> list of labels
                if "annotations" in data:
                    ann = data["annotations"]
                    keys = list(ann.keys())
                    print(f"    annotations: {len(keys)} entries, "
                          f"first frame ids = {keys[:5]}")
                    if keys:
                        sample = ann[keys[0]]
                        print(f"    first entry: {sample}")
                        print(f"    entry type/shape: type={type(sample).__name__}, "
                              f"len={len(sample) if hasattr(sample,'__len__') else 'n/a'}")
                        if isinstance(sample, list) and sample:
                            print(f"    inner type: {type(sample[0]).__name__}, "
                                  f"inner len: {len(sample[0])}")
                # Any auxiliary keys?
                for k, v in data.items():
                    if k == "annotations":
                        continue
                    if isinstance(v, (dict, list)):
                        print(f"    aux '{k}' -> {type(v).__name__} of len {len(v)}")
                    else:
                        print(f"    aux '{k}' = {v!r}")
            except Exception as e:
                print(f"    [error] {e}")

    # First VID's frame count
    vid_dir = os.path.join(base, "videos", "VID01")
    if os.path.isdir(vid_dir):
        n = len([f for f in os.listdir(vid_dir) if f.endswith(".png")])
        print(f"\nVID01 frame PNGs: {n}")


def peek_grasp(root):
    base = os.path.join(root, "GraSP")
    print(f"\n========== GraSP : {base} ==========")

    for cand in ("GraSP_1fps", "GraSP_30fps"):
        p = os.path.join(base, cand)
        if not os.path.isdir(p):
            continue
        print(f"\n{p}/ contents: {sorted(os.listdir(p))}")

    readme = os.path.join(base, "GraSP_1fps", "README.txt")
    if os.path.isfile(readme):
        head_file(readme, n=80)

    # After extraction, look for annotations directory
    for cand_root in ("GraSP_1fps",):
        for ann_candidate in ("annotations", "labels", "annots"):
            p = os.path.join(base, cand_root, ann_candidate)
            if os.path.isdir(p):
                print(f"\nFound {p}/  ({len(os.listdir(p))} entries)")
                # Peek at first few
                for f in sorted(os.listdir(p))[:5]:
                    fp = os.path.join(p, f)
                    if os.path.isfile(fp) and os.path.getsize(fp) < 500_000:
                        head_file(fp, n=15)


def main(args):
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"[error] root does not exist: {root}")
        sys.exit(1)
    peek_cholec80(root)
    peek_cholect50(root)
    peek_grasp(root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    main(p.parse_args())
