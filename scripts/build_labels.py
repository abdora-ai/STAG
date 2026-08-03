#!/usr/bin/env python3
"""
Build unified per-video label files and the split file for the three downstream
tasks we care about:

  - Cholec80 phase recognition (7 classes, multi-class, per-frame)
  - Cholec80 tool presence detection (7 classes, multi-label, per-frame)
  - CholecT50 verb recognition (10 classes, multi-label, per-frame, verb slot only)
  - GraSP tool presence detection (7 classes, multi-label, short-term keyframes)

Reads from:
  <root>/Cholec80/frames/videoNN/videoNN_KKKKKK.png
  <root>/Cholec80/phase_annotations/videoNN-phase.txt   (row per 25-fps frame)
  <root>/Cholec80/tool_annotations/videoNN-tool.txt     (row per 25 frames = per second)
  <root>/CholecT50/labels/VIDNN.json                    ({annotations, categories, ...})
  <root>/CholecT50/videos/VIDNN/KKKKKK.png
  <root>/GraSP/GraSP_1fps/annotations/grasp_short-term_{train,test}.json
  <root>/GraSP/GraSP_1fps/frames/CASEXXX/KKKKK.jpg

Writes to:
  <root>/preprocessed/labels/cholec80_phase.json
  <root>/preprocessed/labels/cholec80_tool.json
  <root>/preprocessed/labels/cholect50_verb.json
  <root>/preprocessed/labels/grasp_tool.json
  <root>/preprocessed/splits.json

Per-task label file schema:
  {
    "dataset": <str>,
    "task": <str>,
    "class_names": [<str>, ...],
    "num_classes": <int>,
    "label_type": "multi-class" | "multi-label",
    "videos": {
       <video_id>: {
          "frames": [<filename>, ...],       # sorted, filenames only (no path)
          "labels": [<int>, ...] |           # multi-class: int per frame
                    [[0/1, ...], ...]         # multi-label: fixed-length vector per frame
       }
    }
  }

Usage:
    python scripts/build_labels.py --root .
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict


# --------------------------------------------------------- class name tables ---

CHOLEC80_PHASE_CLASSES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]
CHOLEC80_TOOL_CLASSES = [
    "Grasper", "Bipolar", "Hook", "Scissors",
    "Clipper", "Irrigator", "SpecimenBag",
]

# Order matches paper's dataloader_cholecT50.py class_mapping.
CHOLECT50_VERB_CLASSES = [
    "grasp", "retract", "dissect", "coagulate", "clip",
    "cut", "aspirate", "irrigate", "pack", "null_verb",
]

# Order matches GraSP short-term "categories" (ids 1..7 → indices 0..6).
GRASP_TOOL_CLASSES = [
    "Bipolar Forceps",
    "Prograsp Forceps",
    "Large Needle Driver",
    "Monopolar Curved Scissors",
    "Suction Instrument",
    "Clip Applier",
    "Laparoscopic Grasper",
]


# ----------------------------------------------------------------- Cholec80 ---

_CHOLEC80_PNG_RE = re.compile(r"^(video\d+)_(\d+)\.png$")


def _parse_cholec80_phase_txt(path):
    """Return dict {frame_idx: phase_class_idx}. Frame idx is at original 25 fps."""
    labels = {}
    with open(path, "r") as f:
        f.readline()  # header: "Frame\tPhase"
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            frame_idx = int(parts[0])
            phase = parts[1]
            if phase not in CHOLEC80_PHASE_CLASSES:
                raise ValueError(f"[cholec80] unknown phase {phase!r} in {path}")
            labels[frame_idx] = CHOLEC80_PHASE_CLASSES.index(phase)
    return labels


def _parse_cholec80_tool_txt(path):
    """Return dict {frame_idx: 7-hot list}. Frame idx is at original 25 fps, stride 25."""
    labels = {}
    with open(path, "r") as f:
        f.readline()  # header: "Frame\tGrasper\t..."
        for line in f:
            parts = line.split()
            if len(parts) < 1 + 7:
                continue
            frame_idx = int(parts[0])
            tool_vec = [int(x) for x in parts[1:8]]
            labels[frame_idx] = tool_vec
    return labels


def _list_cholec80_pngs(frames_root, video_id):
    d = os.path.join(frames_root, video_id)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".png"))


def build_cholec80(root):
    frames_root = os.path.join(root, "Cholec80", "frames")
    phase_dir = os.path.join(root, "Cholec80", "phase_annotations")
    tool_dir = os.path.join(root, "Cholec80", "tool_annotations")

    video_ids = sorted(v for v in os.listdir(frames_root) if v.startswith("video"))

    phase_videos = {}
    tool_videos = {}
    misaligned = []

    for vid in video_ids:
        pngs = _list_cholec80_pngs(frames_root, vid)
        if not pngs:
            print(f"  [warn] no PNGs for {vid}", file=sys.stderr)
            continue

        phase_labels = _parse_cholec80_phase_txt(
            os.path.join(phase_dir, f"{vid}-phase.txt"))
        tool_labels = _parse_cholec80_tool_txt(
            os.path.join(tool_dir, f"{vid}-tool.txt"))

        # Extract K from filename "video01_KKKKKK.png"; align to original frame = (K-1)*25
        phase_per = []
        tool_per = []
        for f in pngs:
            m = _CHOLEC80_PNG_RE.match(f)
            if not m:
                raise ValueError(f"unexpected png filename: {f}")
            k = int(m.group(2))
            orig = (k - 1) * 25
            if orig in phase_labels:
                phase_per.append(phase_labels[orig])
            else:
                misaligned.append(f"{vid} k={k} orig={orig} not-in-phase")
                phase_per.append(-1)
            if orig in tool_labels:
                tool_per.append(tool_labels[orig])
            else:
                misaligned.append(f"{vid} k={k} orig={orig} not-in-tool")
                tool_per.append([-1] * 7)

        phase_videos[vid] = {"frames": pngs, "labels": phase_per}
        tool_videos[vid] = {"frames": pngs, "labels": tool_per}

    return phase_videos, tool_videos, misaligned


# ---------------------------------------------------------------- CholecT50 ---

def build_cholect50_verb(root):
    labels_dir = os.path.join(root, "CholecT50", "labels")
    videos_dir = os.path.join(root, "CholecT50", "videos")

    out = {}
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.endswith(".json"):
            continue
        vid = fn[:-5]  # "VID01"
        with open(os.path.join(labels_dir, fn), "r") as f:
            data = json.load(f)
        ann = data.get("annotations", {})

        v_dir = os.path.join(videos_dir, vid)
        if not os.path.isdir(v_dir):
            print(f"  [warn] no frames dir for {vid}", file=sys.stderr)
            continue
        pngs = sorted(f for f in os.listdir(v_dir) if f.endswith(".png"))

        labels = []
        for png in pngs:
            frame_id = str(int(png[:-4]))  # "000000.png" -> "0"
            instances = ann.get(frame_id, [])
            vec = [0] * 10
            for inst in instances:
                if isinstance(inst, list) and len(inst) >= 8:
                    v_idx = inst[7]
                    if isinstance(v_idx, int) and 0 <= v_idx < 10:
                        vec[v_idx] = 1
            labels.append(vec)

        out[vid] = {"frames": pngs, "labels": labels}

    return out


# --------------------------------------------------------------------- GraSP ---

def build_grasp_tool(root):
    """Per-keyframe 7-hot instrument presence, from GraSP short-term COCO files."""
    ann_dir = os.path.join(root, "GraSP", "GraSP_1fps", "annotations")
    frames_dir = os.path.join(root, "GraSP", "GraSP_1fps", "frames")

    keyframes = defaultdict(dict)  # case -> {file_name: 7-hot list}

    for fn in ("grasp_short-term_train.json", "grasp_short-term_test.json"):
        p = os.path.join(ann_dir, fn)
        if not os.path.isfile(p):
            continue
        with open(p, "r") as f:
            data = json.load(f)
        images_by_id = {im["id"]: im for im in data["images"]}
        # ensure all keyframes present (even those with no annotations)
        for im in data["images"]:
            case = im["video_name"]
            keyframes[case].setdefault(im["file_name"], [0] * 7)
        for a in data.get("annotations", []):
            cat = a.get("category_id")
            im = images_by_id.get(a["image_id"])
            if im is None or not (1 <= cat <= 7):
                continue
            case = im["video_name"]
            keyframes[case][im["file_name"]][cat - 1] = 1

    # per-case: keyframes sorted by name; store just the filename (no case prefix)
    out = {}
    for case, fmap in keyframes.items():
        # sanity-check the frame directory exists
        d = os.path.join(frames_dir, case)
        if not os.path.isdir(d):
            print(f"  [warn] frames dir missing for {case}: {d}", file=sys.stderr)
        names_sorted = sorted(fmap.keys())
        frames = [os.path.basename(n) for n in names_sorted]  # "00000.jpg"
        labels = [fmap[n] for n in names_sorted]
        out[case] = {"frames": frames, "labels": labels}

    return out


# ------------------------------------------------------------------- splits ---

def build_splits():
    return {
        # Standard Cholec80 protocol (Twinanda 2017 / TeCNO / LoViT).
        "cholec80": {
            "train": [f"video{n:02d}" for n in range(1, 41)],
            "val":   [f"video{n:02d}" for n in range(41, 49)],
            "test":  [f"video{n:02d}" for n in range(49, 81)],
        },
        # Rendezvous cross-validation split, per paper dataloader_cholecT50.py.
        # Keys are fold numbers (1..5); values are the *test* video ids for that fold.
        # Training videos = union of the other four folds.
        "cholect50_crossval_test_vids": {
            "1": [f"VID{str(n).zfill(2)}" for n in [79,  2, 51,  6, 25, 14, 66, 23, 50, 111]],
            "2": [f"VID{str(n).zfill(2)}" for n in [80, 32,  5, 15, 40, 47, 26, 48, 70,  96]],
            "3": [f"VID{str(n).zfill(2)}" for n in [31, 57, 36, 18, 52, 68, 10,  8, 73, 103]],
            "4": [f"VID{str(n).zfill(2)}" for n in [42, 29, 60, 27, 65, 75, 22, 49, 12, 110]],
            "5": [f"VID{str(n).zfill(2)}" for n in [78, 43, 62, 35, 74,  1, 56,  4, 13,  92]],
        },
        # GraSP official 4 / 4 / 5 split.
        "grasp_tool": {
            "train_fold1": ["CASE001", "CASE004", "CASE014", "CASE015"],
            "train_fold2": ["CASE002", "CASE003", "CASE007", "CASE021"],
            "test":        ["CASE041", "CASE047", "CASE050", "CASE051", "CASE053"],
        },
    }


# ---------------------------------------------------------------------- I/O ---

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def _summ_videos(vids):
    n_vids = len(vids)
    n_frames = sum(len(v["frames"]) for v in vids.values())
    return n_vids, n_frames


def main(args):
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, "preprocessed", "labels")
    print(f"[root] {root}")
    print(f"[out ] {out_dir}")

    # Cholec80 phase + tool
    print("\n[cholec80] parsing phase + tool annotations")
    phase, tool, mis = build_cholec80(root)
    if mis:
        print(f"  [warn] {len(mis)} alignment issues (first 10):")
        for m in mis[:10]:
            print(f"    - {m}")
    write_json(os.path.join(out_dir, "cholec80_phase.json"), {
        "dataset": "cholec80", "task": "phase",
        "class_names": CHOLEC80_PHASE_CLASSES,
        "num_classes": len(CHOLEC80_PHASE_CLASSES),
        "label_type": "multi-class",
        "videos": phase,
    })
    write_json(os.path.join(out_dir, "cholec80_tool.json"), {
        "dataset": "cholec80", "task": "tool",
        "class_names": CHOLEC80_TOOL_CLASSES,
        "num_classes": len(CHOLEC80_TOOL_CLASSES),
        "label_type": "multi-label",
        "videos": tool,
    })
    nv, nf = _summ_videos(phase)
    print(f"  cholec80_phase.json  →  {nv} videos, {nf} frames")
    nv, nf = _summ_videos(tool)
    print(f"  cholec80_tool.json   →  {nv} videos, {nf} frames")

    # CholecT50 verb
    print("\n[cholect50] parsing verb annotations")
    verb = build_cholect50_verb(root)
    write_json(os.path.join(out_dir, "cholect50_verb.json"), {
        "dataset": "cholect50", "task": "verb",
        "class_names": CHOLECT50_VERB_CLASSES,
        "num_classes": len(CHOLECT50_VERB_CLASSES),
        "label_type": "multi-label",
        "videos": verb,
    })
    nv, nf = _summ_videos(verb)
    print(f"  cholect50_verb.json  →  {nv} videos, {nf} frames")

    # GraSP tool
    print("\n[grasp] parsing short-term tool annotations")
    grasp = build_grasp_tool(root)
    write_json(os.path.join(out_dir, "grasp_tool.json"), {
        "dataset": "grasp", "task": "tool",
        "class_names": GRASP_TOOL_CLASSES,
        "num_classes": len(GRASP_TOOL_CLASSES),
        "label_type": "multi-label",
        "videos": grasp,
    })
    nv, nf = _summ_videos(grasp)
    print(f"  grasp_tool.json      →  {nv} cases, {nf} keyframes")

    # Splits
    splits_path = os.path.join(root, "preprocessed", "splits.json")
    write_json(splits_path, build_splits())
    print(f"\n[splits] wrote {splits_path}")

    print("\n[done]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Repo/data root")
    main(p.parse_args())
