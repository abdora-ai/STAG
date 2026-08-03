#!/usr/bin/env python3
"""
Inspect the LemonFM checkpoint:
  1. Show top-level keys and their types.
  2. Summarize teacher/student state dicts.
  3. Instantiate torchvision.convnext_large, load teacher weights (after stripping
     `module.` and `backbone.` prefixes), and run a dummy forward pass.
  4. Report the pre-avgpool feature map shape (used by our slot pooling module).

Usage:
    python tools/inspect_checkpoint.py --checkpoint /path/to/lemonfm.pth
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision


def summarize_state_dict(sd, name, max_show=25):
    total_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    print(f"\n[{name}] entries = {len(sd)}, total params = {total_params/1e6:.2f} M")
    print(f"[{name}] first {min(max_show, len(sd))} keys and shapes:")
    for i, (k, v) in enumerate(sd.items()):
        if i >= max_show:
            print(f"  ... and {len(sd)-max_show} more")
            break
        shape = tuple(v.shape) if hasattr(v, "shape") else "scalar"
        dtype = v.dtype if hasattr(v, "dtype") else type(v).__name__
        print(f"  {k:80s}  {shape}  {dtype}")


def key_prefixes(sd, depth=1):
    prefixes = {}
    for k in sd:
        parts = k.split(".")
        p = ".".join(parts[:depth])
        prefixes[p] = prefixes.get(p, 0) + 1
    return prefixes


def strip_prefixes(sd, prefixes=("module.", "backbone.")):
    out = {}
    for k, v in sd.items():
        nk = k
        for prefix in prefixes:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


def main(args):
    print(f"[load] checkpoint = {args.checkpoint}")
    if not os.path.isfile(args.checkpoint):
        print(f"[error] file not found: {args.checkpoint}")
        sys.exit(1)

    # weights_only=False is required because the paper saves the full args namespace.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"[load] top-level type = {type(ckpt).__name__}")

    if not isinstance(ckpt, dict):
        print("[error] checkpoint is not a dict; unexpected format.")
        sys.exit(1)

    print(f"[load] top-level keys = {list(ckpt.keys())}")
    for k, v in ckpt.items():
        if isinstance(v, dict):
            print(f"  '{k}' -> dict, len={len(v)}")
        elif hasattr(v, "shape"):
            print(f"  '{k}' -> tensor, shape={tuple(v.shape)}")
        else:
            s = repr(v)
            print(f"  '{k}' -> {type(v).__name__}: {s[:200]}")

    for key in ("teacher", "student"):
        if key not in ckpt:
            continue
        sd = ckpt[key]
        summarize_state_dict(sd, key)
        print(f"[{key}] level-1 prefixes: {key_prefixes(sd, depth=1)}")

    if "teacher" not in ckpt:
        print("\n[error] no 'teacher' key found in checkpoint; cannot verify backbone.")
        sys.exit(1)

    teacher = ckpt["teacher"]
    stripped = strip_prefixes(teacher, ("module.", "backbone."))
    print(f"\n[teacher] after stripping 'module.'/'backbone.': {len(stripped)} keys")
    print(f"[teacher] level-1 prefixes after strip: {key_prefixes(stripped, depth=1)}")

    print("\n[model] instantiating torchvision.models.convnext_large(weights=None)")
    model = torchvision.models.convnext_large(weights=None)
    input_dim = model.classifier[2].in_features
    model.classifier[2] = nn.Identity()

    msg = model.load_state_dict(stripped, strict=False)
    n_missing = len(msg.missing_keys)
    n_unexpected = len(msg.unexpected_keys)
    print("[load_state_dict] result:")
    print(f"  missing_keys    ({n_missing})")
    for k in msg.missing_keys[:10]:
        print(f"    - {k}")
    if n_missing > 10:
        print(f"    ... and {n_missing-10} more")
    print(f"  unexpected_keys ({n_unexpected})")
    for k in msg.unexpected_keys[:10]:
        print(f"    - {k}")
    if n_unexpected > 10:
        print(f"    ... and {n_unexpected-10} more")

    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        fmap = model.features(x)
    print(f"\n[forward] output shape = {tuple(y.shape)} (expected (1, {input_dim}) = (1, 1536))")
    print(f"[forward] pre-avgpool feature map shape = {tuple(fmap.shape)} "
          f"(expected (1, 1536, 7, 7))")

    # Sanity summary
    ok = (
        y.shape == torch.Size([1, 1536])
        and tuple(fmap.shape) == (1, 1536, 7, 7)
        and n_missing == 0
    )
    print(f"\n[verdict] ConvNeXt-L teacher backbone load: {'OK' if ok else 'CHECK ABOVE'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to lemonfm.pth")
    main(p.parse_args())
