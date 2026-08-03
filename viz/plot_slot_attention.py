#!/usr/bin/env python3
"""
Slot-attention figures for the paper.

Produces:

 1. viz_out/slot_attention/slot_attention_unified.{pdf,png}
       4 rows × 4 cols  overview - one row per (dataset, task), one column per
       representative class.  This is the compact main-paper figure.

 2. viz_out/slot_attention/slot_attention_{task}_detailed.{pdf,png}
       4 rows × 4 cols  per-task detail - 16 real frames from the target
       dataset with class-specific slot attention overlaid.  Good for the
       supplement or a large main-paper figure.

Both use the same real-image sampling logic. For phase (K_s = 1) the single
context slot is used; for tool/verb (K_s = N_c) slot_c → class_c.

Usage:
    python viz/plot_slot_attention.py --root . --checkpoint lemonfm.pth
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from _style import apply_paper_style, TASK_PRETTY  # noqa: E402
from lemonfm_backbone import FrozenLemonFM         # noqa: E402
from stag_head import STAG                          # noqa: E402


# ---------------------------- Task configuration ---------------------------

TASKS = [
    {
        "key":            "cholec80_phase",
        "row_label":      "Cholec80\nPhase",
        "label_file":     "preprocessed/labels/cholec80_phase.json",
        "frame_root":     "Cholec80/frames",
        "num_slots":      1,
        "num_classes":    7,
        "label_type":     "multi-class",
        "window":         256, "depth": 8,
        "unified_classes": ["Preparation", "CalotTriangleDissection",
                            "ClippingCutting", "GallbladderRetraction"],
        "detail_classes":  ["Preparation", "CalotTriangleDissection",
                            "ClippingCutting", "GallbladderDissection"],
        "detail_per_class": 4,
    },
    {
        "key":            "cholec80_tool",
        "row_label":      "Cholec80\nTool",
        "label_file":     "preprocessed/labels/cholec80_tool.json",
        "frame_root":     "Cholec80/frames",
        "num_slots":      7,
        "num_classes":    7,
        "label_type":     "multi-label",
        "window":         64, "depth": 4,
        "unified_classes": ["Grasper", "Hook", "Clipper", "SpecimenBag"],
        "detail_classes":  ["Grasper", "Bipolar", "Hook", "Scissors",
                            "Clipper", "Irrigator", "SpecimenBag"],
        "detail_per_class": 3,
    },
    {
        "key":            "grasp_tool",
        "row_label":      "GraSP\nTool",
        "label_file":     "preprocessed/labels/grasp_tool.json",
        "frame_root":     "GraSP/GraSP_1fps/frames",
        "num_slots":      7,
        "num_classes":    7,
        "label_type":     "multi-label",
        "window":         8, "depth": 2,
        "unified_classes": ["Bipolar Forceps", "Prograsp Forceps",
                            "Monopolar Curved Scissors", "Suction Instrument"],
        "detail_classes":  ["Bipolar Forceps", "Prograsp Forceps",
                            "Large Needle Driver", "Monopolar Curved Scissors",
                            "Suction Instrument", "Clip Applier",
                            "Laparoscopic Grasper"],
        "detail_per_class": 3,
    },
    {
        "key":            "cholect50_verb",
        "row_label":      "CholecT50\nVerb",
        "label_file":     "preprocessed/labels/cholect50_verb.json",
        "frame_root":     "CholecT50/videos",
        "num_slots":      10,
        "num_classes":    10,
        "label_type":     "multi-label",
        "window":         8, "depth": 1,
        "unified_classes": ["grasp", "dissect", "clip", "cut"],
        "detail_classes":  ["grasp", "retract", "dissect", "coagulate",
                            "clip", "cut", "aspirate", "irrigate"],
        "detail_per_class": 2,
    },
]


PREPROC = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


# ------------------------ Checkpoint / head loader --------------------------

def find_latest_full_checkpoint(runs_root, task_key):
    """Return the newest full-mode checkpoint for a task (handles legacy layout)."""
    cands = []
    cands += glob.glob(os.path.join(runs_root, task_key,           "full", "*", "checkpoint_best.pt"))
    cands += glob.glob(os.path.join(runs_root, f"{task_key}_fold*", "full", "*", "checkpoint_best.pt"))
    cands += glob.glob(os.path.join(runs_root, task_key,            "20*", "checkpoint_best.pt"))
    cands += glob.glob(os.path.join(runs_root, f"{task_key}_fold*", "20*", "checkpoint_best.pt"))
    cands = sorted(cands)
    if not cands:
        raise FileNotFoundError(f"No 'full' checkpoint for {task_key} under {runs_root}")
    return cands[-1]


def load_head(runs_root, task_key, cfg, device):
    ck_path = find_latest_full_checkpoint(runs_root, task_key)
    ckpt = torch.load(ck_path, map_location=device, weights_only=False)
    tc = ckpt.get("cfg") or {}
    head = STAG(
        in_dim=1536,
        num_classes=cfg["num_classes"],
        task_type=cfg["label_type"],
        num_slots=cfg["num_slots"],
        slot_dim=256,
        mode="full",
        window=tc.get("window", cfg["window"]),
        depth=tc.get("depth", cfg["depth"]),
    ).to(device).eval()
    head.load_state_dict(ckpt["model_state"], strict=True)
    return head, ck_path


# ------------------------ Frame picking -------------------------------------

def _load_lab(root, cfg):
    with open(os.path.join(root, cfg["label_file"]), "r") as f:
        return json.load(f)


def pick_frames(cfg, root, class_names_wanted, n_per_class, seed=0, max_scan=250):
    """
    Return list of dicts: {class_idx, class_name, video, frame_name, frame_path}.
    Picks up to `n_per_class` frames for each requested class from the label JSON.
    """
    rng = random.Random(seed)
    lab = _load_lab(root, cfg)
    class_names = lab["class_names"]
    videos = lab["videos"]

    out = []
    for cname in class_names_wanted:
        if cname not in class_names:
            print(f"  [warn] {cfg['key']}: class '{cname}' not in class_names")
            continue
        c_idx = class_names.index(cname)

        candidates = []
        for vid, vdata in videos.items():
            frames = vdata["frames"]
            labels = vdata["labels"]
            for j, y in enumerate(labels):
                is_pos = (
                    (cfg["label_type"] == "multi-class" and y == c_idx)
                    or (cfg["label_type"] == "multi-label"
                        and isinstance(y, (list, tuple))
                        and len(y) > c_idx and y[c_idx] == 1)
                )
                if is_pos:
                    candidates.append((vid, frames[j]))
                    if len(candidates) >= max_scan:
                        break
            if len(candidates) >= max_scan:
                break

        if not candidates:
            print(f"  [warn] {cfg['key']}: no frame found for class '{cname}'")
            continue

        rng.shuffle(candidates)
        n = min(n_per_class, len(candidates))
        for vid, frame_name in candidates[:n]:
            path = os.path.join(root, cfg["frame_root"], vid, frame_name)
            if not os.path.isfile(path):
                continue
            out.append({
                "class_idx":  c_idx,
                "class_name": cname,
                "video":      vid,
                "frame_name": frame_name,
                "frame_path": path,
            })
    return out


# ------------------------ Attention extraction ------------------------------

@torch.no_grad()
def compute_slot_attention(head, spatial):
    """Return attention weights [T, K_s, S=49] from head's slot_pool."""
    sp = head.slot_pool
    T_ = spatial.size(0)
    k = sp.k_proj(spatial)
    v = sp.v_proj(spatial)
    q = sp.queries.unsqueeze(0).expand(T_, -1, -1)
    _, attn = sp.attn(q, k, v, need_weights=True, average_attn_weights=True)
    return attn.detach()


@torch.no_grad()
def frame_attention_map(backbone, head, image_path, target_slot_idx, device):
    """Return (frame_224_np, heatmap_7x7_np) for one image + target slot."""
    img = Image.open(image_path).convert("RGB")
    frame224 = np.asarray(img.resize((224, 224)))
    x = PREPROC(img).unsqueeze(0).to(device)
    _, spatial = backbone(x)                         # [1, 49, 1536]
    attn = compute_slot_attention(head, spatial)      # [1, K_s, 49]
    heat = attn[0, target_slot_idx].cpu().numpy().reshape(7, 7)
    return frame224, heat


# ------------------------ Rendering ---------------------------------------

def render_cell(ax, frame_224, attn_map_7x7, title=None,
                is_leftmost=False, row_label=None):
    heat = np.kron(attn_map_7x7, np.ones((32, 32)))   # 7*32=224
    hmin, hmax = heat.min(), heat.max()
    heat_disp = (heat - hmin) / (hmax - hmin + 1e-8)

    ax.imshow(frame_224)
    ax.imshow(heat_disp, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is not None:
        ax.set_title(title, fontsize=9, pad=2.5)

    if is_leftmost and row_label is not None:
        ax.set_ylabel(row_label, fontsize=11, rotation=0,
                      labelpad=44, ha="right", va="center")


# ---------- Unified 4x4 figure --------------------------------------------

def make_unified(backbone, args, device, out_dir):
    apply_paper_style()
    n_rows = len(TASKS)
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols + 1.5, 2.5 * n_rows),
        squeeze=False,
    )

    for r, cfg in enumerate(TASKS):
        try:
            head, ck_path = load_head(args.runs, cfg["key"], cfg, device)
        except FileNotFoundError as e:
            print(f"[skip] {cfg['key']}: {e}")
            for c in range(n_cols):
                axes[r, c].set_axis_off()
            continue
        print(f"[unified] {cfg['key']}  <- {ck_path}")

        picks = pick_frames(cfg, args.root, cfg["unified_classes"],
                            n_per_class=1, seed=args.seed + r)

        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(picks):
                ax.set_axis_off()
                continue
            pick = picks[c]
            slot_idx = 0 if cfg["num_slots"] == 1 else min(
                pick["class_idx"], cfg["num_slots"] - 1)
            frame224, heat = frame_attention_map(
                backbone, head, pick["frame_path"], slot_idx, device
            )
            render_cell(ax, frame224, heat,
                        title=pick["class_name"],
                        is_leftmost=(c == 0),
                        row_label=cfg["row_label"])

    fig.suptitle(
        "Slot-attention on frozen LemonFM - STAG class queries localize "
        "task-relevant regions", fontsize=12,
    )

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"slot_attention_unified.{ext}")
        fig.savefig(p)
        print(f"[ok] wrote {p}")
    plt.close(fig)


# ---------- Per-task detailed figure -----------------------------------------

def make_detailed(backbone, cfg, args, device, out_dir):
    """A 4x4 grid of frame + attention cells for one task."""
    apply_paper_style()
    try:
        head, ck_path = load_head(args.runs, cfg["key"], cfg, device)
    except FileNotFoundError as e:
        print(f"[skip] detailed for {cfg['key']}: {e}")
        return
    print(f"[detailed] {cfg['key']}  <- {ck_path}")

    picks = pick_frames(cfg, args.root, cfg["detail_classes"],
                        n_per_class=cfg["detail_per_class"], seed=args.seed)

    # Ensure exactly 16 cells (4x4). If fewer, top up by resampling with different seeds.
    trial = 1
    while len(picks) < 16 and trial < 5:
        extra = pick_frames(cfg, args.root, cfg["detail_classes"],
                            n_per_class=cfg["detail_per_class"],
                            seed=args.seed + 100 * trial)
        # dedupe by (video, frame_name)
        seen = {(p["video"], p["frame_name"]) for p in picks}
        for e in extra:
            if (e["video"], e["frame_name"]) in seen:
                continue
            picks.append(e); seen.add((e["video"], e["frame_name"]))
            if len(picks) >= 16:
                break
        trial += 1
    picks = picks[:16]

    n = 4
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n), squeeze=False)

    for i, ax in enumerate(axes.flat):
        if i >= len(picks):
            ax.set_axis_off()
            continue
        pick = picks[i]
        slot_idx = 0 if cfg["num_slots"] == 1 else min(
            pick["class_idx"], cfg["num_slots"] - 1)
        frame224, heat = frame_attention_map(
            backbone, head, pick["frame_path"], slot_idx, device
        )
        render_cell(ax, frame224, heat, title=pick["class_name"])

    fig.suptitle(
        f"Slot attention - {TASK_PRETTY.get(cfg['key'], cfg['key'])} "
        f"(STAG class-conditional queries)", fontsize=12,
    )

    os.makedirs(out_dir, exist_ok=True)
    stub = os.path.join(out_dir, f"slot_attention_{cfg['key']}_detailed")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stub}.{ext}")
        print(f"[ok] wrote {stub}.{ext}")
    plt.close(fig)


# ---------------------------- Main -----------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dev ] {device}")

    print("[load] backbone")
    backbone = FrozenLemonFM(args.checkpoint).to(device).eval()

    make_unified(backbone, args, device, args.out_dir)

    for cfg in TASKS:
        make_detailed(backbone, cfg, args, device, args.out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--checkpoint", required=True, help="Path to lemonfm.pth")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out-dir", default="viz_out/slot_attention")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
