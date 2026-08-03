"""
Data utilities for STAG training on cached LemonFM features.
"""
import json
import os
from typing import List, Tuple

import numpy as np
import torch


def load_labels(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_splits(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_video_features(features_root: str, group: str, vid: str,
                        load_spatial: bool = True) -> dict:
    """
    Load pooled (and optionally spatial) features for a single video.
    Returns dict with keys 'pooled' (np.float16 [T, 1536]) and, if requested,
    'spatial' (np.float16 [T, 49, 1536]).
    """
    p = np.load(os.path.join(features_root, group, "pooled", f"{vid}.npy"),
                mmap_mode="r").astype(np.float16, copy=False)
    out = {"pooled": p}
    if load_spatial:
        s = np.load(os.path.join(features_root, group, "spatial", f"{vid}.npy"),
                    mmap_mode="r").astype(np.float16, copy=False)
        out["spatial"] = s
    return out


def to_torch_labels(labels_list, label_type: str) -> torch.Tensor:
    """
    multi-class : list[int] -> LongTensor [T]
    multi-label : list[list[int]] -> FloatTensor [T, N_c]
    """
    if label_type == "multi-class":
        return torch.tensor(labels_list, dtype=torch.long)
    if label_type == "multi-label":
        return torch.tensor(labels_list, dtype=torch.float32)
    raise ValueError(f"unknown label_type: {label_type}")


def get_split_partitions(splits: dict, dataset_task: str, fold: int = None
                         ) -> Tuple[List[str], List[str], List[str]]:
    """
    Return (train_vids, val_vids, test_vids) as lists of video ids.

    - cholec80_phase / cholec80_tool : official 40 / 8 / 32
    - cholect50_verb                 : 5-fold cross-val (fold=1..5 required);
                                       test = fold's test list, train = union of
                                       the other four folds' lists, val = last 5
                                       of train (matches paper dataloader).
    - grasp_tool                     : train_fold1 / train_fold2 / test.
    """
    if dataset_task in ("cholec80_phase", "cholec80_tool"):
        s = splits["cholec80"]
        return s["train"], s["val"], s["test"]

    if dataset_task == "cholect50_verb":
        assert fold is not None, "cholect50_verb requires --fold 1..5"
        all_folds = splits["cholect50_crossval_test_vids"]
        assert str(fold) in all_folds, f"fold {fold} not in {list(all_folds)}"
        test = list(all_folds[str(fold)])
        train_all = []
        for k, vs in all_folds.items():
            if int(k) != int(fold):
                train_all.extend(vs)
        val = train_all[-5:]
        train = train_all[:-5]
        return train, val, test

    if dataset_task == "grasp_tool":
        s = splits["grasp_tool"]
        return s["train_fold1"], s["train_fold2"], s["test"]

    raise ValueError(f"unknown dataset_task: {dataset_task}")


def preload_features(features_root: str, group: str, vids: List[str],
                     load_spatial: bool = True, verbose: bool = False) -> dict:
    """Preload features for a set of videos into RAM (as fp16 np arrays)."""
    out = {}
    for vid in vids:
        try:
            out[vid] = load_video_features(features_root, group, vid,
                                           load_spatial=load_spatial)
            if verbose:
                p = out[vid]["pooled"]
                s = out[vid].get("spatial")
                msg = f"  {vid}  pooled {p.shape}"
                if s is not None:
                    msg += f"  spatial {s.shape}"
                print(msg)
        except FileNotFoundError as e:
            print(f"  [warn] {vid}: {e}")
    return out
