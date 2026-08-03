#!/usr/bin/env python3
"""
Cache frozen LemonFM features for every frame listed in preprocessed/labels/*.json.

Per dataset group, writes:
    preprocessed/features/<group>/pooled/<video>.npy      # (T, 1536)    fp16
    preprocessed/features/<group>/spatial/<video>.npy     # (T, 49, 1536) fp16
in the same frame order as the corresponding labels JSON, so features[t]
aligns with labels[t] one-to-one at head-training time - no filename lookup
needed.

Groups:
    cholec80   -> shared frames for cholec80_phase and cholec80_tool
    cholect50  -> cholect50_verb
    grasp      -> grasp_tool (short-term keyframes only)

Cholec80 phase and tool reference the same 184,498 PNGs, so we cache once and
reuse for both tasks.

Usage:
    python scripts/cache_features.py --root . --checkpoint lemonfm.pth

Options:
    --only-pooled       skip 49-token spatial features (≈50x smaller cache)
    --only-group X      cache only 'cholec80' | 'cholect50' | 'grasp'
    --batch-size N      total batch across all GPUs (default 512)
    --num-workers N     DataLoader workers (default 8)
    --single-gpu        disable DataParallel
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Make lemonfm_backbone importable regardless of how the script is invoked
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lemonfm_backbone import FrozenLemonFM  # noqa: E402


# Group definition. Each group corresponds to one physical frame set.
# The label file just provides the frame ordering + per-video list.
GROUPS = {
    "cholec80":  {
        "label_rel": "preprocessed/labels/cholec80_phase.json",
        "frame_rel": "Cholec80/frames",
    },
    "cholect50": {
        "label_rel": "preprocessed/labels/cholect50_verb.json",
        "frame_rel": "CholecT50/videos",
    },
    "grasp": {
        "label_rel": "preprocessed/labels/grasp_tool.json",
        "frame_rel": "GraSP/GraSP_1fps/frames",
    },
}


class FrameDataset(Dataset):
    """Read PNGs/JPEGs from disk and produce ImageNet-normalized 224² tensors."""

    def __init__(self, paths, size=224):
        self.paths = paths
        self.tf = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img)


def cache_one_video(model, paths, batch_size, num_workers, only_pooled, device):
    ds = FrameDataset(paths)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), drop_last=False,
    )
    pooled_chunks, spatial_chunks = [], []
    for batch in dl:
        batch = batch.to(device, non_blocking=True).half()
        with torch.no_grad():
            p, s = model(batch)
        pooled_chunks.append(p.detach().float().cpu().numpy().astype(np.float16))
        if not only_pooled:
            spatial_chunks.append(s.detach().float().cpu().numpy().astype(np.float16))
    pooled = np.concatenate(pooled_chunks, axis=0)
    spatial = np.concatenate(spatial_chunks, axis=0) if not only_pooled else None
    return pooled, spatial


def main(args):
    root = os.path.abspath(args.root)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"[load] LemonFM teacher from {args.checkpoint}")
    model = FrozenLemonFM(args.checkpoint).to(device).half()
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1 and not args.single_gpu:
        model = nn.DataParallel(model)
        print(f"[gpu ] DataParallel across {n_gpu} GPUs")
    else:
        print(f"[gpu ] single GPU (visible={n_gpu})")

    total_t0 = time.time()

    for group_key, cfg in GROUPS.items():
        if args.only_group and args.only_group != group_key:
            continue

        label_file = os.path.join(root, cfg["label_rel"])
        frame_root = os.path.join(root, cfg["frame_rel"])
        if not os.path.isfile(label_file):
            print(f"[skip] {group_key}: label file missing ({label_file})")
            continue
        if not os.path.isdir(frame_root):
            print(f"[skip] {group_key}: frame root missing ({frame_root})")
            continue

        with open(label_file, "r") as f:
            data = json.load(f)
        videos = data["videos"]

        pooled_dir = os.path.join(root, "preprocessed", "features", group_key, "pooled")
        spatial_dir = os.path.join(root, "preprocessed", "features", group_key, "spatial")
        os.makedirs(pooled_dir, exist_ok=True)
        if not args.only_pooled:
            os.makedirs(spatial_dir, exist_ok=True)

        video_ids = sorted(videos.keys())
        print(f"\n[group] {group_key}  {len(video_ids)} videos, "
              f"{sum(len(v['frames']) for v in videos.values())} frames")

        for vid in video_ids:
            p_path = os.path.join(pooled_dir, f"{vid}.npy")
            s_path = os.path.join(spatial_dir, f"{vid}.npy")
            pooled_ok = os.path.isfile(p_path)
            spatial_ok = args.only_pooled or os.path.isfile(s_path)
            if pooled_ok and spatial_ok:
                if args.verbose:
                    print(f"  [skip] {vid} (already cached)")
                continue

            frames = videos[vid]["frames"]
            paths = [os.path.join(frame_root, vid, f) for f in frames]
            if paths and not os.path.isfile(paths[0]):
                print(f"  [warn] {vid}: first frame missing at {paths[0]}, skipping")
                continue

            t0 = time.time()
            pooled_np, spatial_np = cache_one_video(
                model, paths, args.batch_size, args.num_workers,
                args.only_pooled, device,
            )
            elapsed = time.time() - t0

            np.save(p_path, pooled_np)
            line = (f"  {vid:>10s}  T={len(paths):5d}  "
                    f"pooled {tuple(pooled_np.shape)} "
                    f"({pooled_np.nbytes/1e6:.1f} MB)")
            if spatial_np is not None:
                np.save(s_path, spatial_np)
                line += (f"  spatial {tuple(spatial_np.shape)} "
                         f"({spatial_np.nbytes/1e6:.1f} MB)")
            line += f"  [{elapsed:.1f}s, {len(paths)/max(elapsed,1e-3):.0f} fps]"
            print(line)

    print(f"\n[done] total elapsed {time.time()-total_t0:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=512,
                   help="Total batch across all GPUs (split by DataParallel)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--only-pooled", action="store_true",
                   help="Skip 49-token spatial features (saves ~50x disk)")
    p.add_argument("--only-group", type=str, default=None,
                   choices=list(GROUPS.keys()) + [None],
                   help="Cache only one group")
    p.add_argument("--single-gpu", action="store_true",
                   help="Disable DataParallel even if multiple GPUs are visible")
    p.add_argument("--verbose", action="store_true")
    main(p.parse_args())
