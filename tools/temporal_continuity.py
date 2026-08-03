#!/usr/bin/env python3
"""
Compute the REAL temporal-continuity statistics and learned temporal parameters
for the STAG ablation figure, directly from the run outputs on the server.

For each task it reports, using the ground-truth test sequences saved in each
run's predictions_test.npz (respecting per-video boundaries via video_offsets):

  - autocorr    : lag-1 label autocorrelation
                    * multi-class (phase): mean over one-hot class channels
                    * multi-label (tool/verb/grasp): mean over classes with
                      non-zero variance
  - stay_prob   : P(label_{t+1} == label_t)  (exact vector match)
  - run_length  : expected run length = 1 / (1 - stay_prob)
  - gain_pp     : temporal-module gain = 100 * (full - slot_proto) primary metric
  - alpha, sigma: learned temporal parameters at the best-val epoch of the
                  'full' run (from metrics.jsonl / best.json)

CholecT50 verb is averaged over its folds.

Usage (on the server, from the repo root that contains 'runs/'):
    python tools/temporal_continuity.py --runs runs --out viz_out/temporal_continuity.json

Then copy viz_out/temporal_continuity.json back to the local repo and re-run
viz/plot_ablation.py; the figure will use these real numbers automatically.

Only needs numpy.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


TASKS = ["cholec80_phase", "cholec80_tool", "cholect50_verb", "grasp_tool"]
PRIMARY = {
    "cholec80_phase": "accuracy",
    "cholec80_tool":  "mAP",
    "cholect50_verb": "mAP",
    "grasp_tool":     "mAP",
}
MULTICLASS = {"cholec80_phase"}


# ------------------------------------------------------------- run lookup ---

def _latest(paths):
    """Pick the most recent run dir (they are timestamped; max name works)."""
    return sorted(paths)[-1] if paths else None


MODE_DIRS = {"linear", "prototype", "slot_proto", "full"}


def find_run(runs_root, task, mode, fold=None):
    """Return newest matching run dir, handling two layouts:

        runs/<slug>/<mode>/<timestamp>/          (ablation modes, and newer 'full')
        runs/<slug>/<timestamp>/                 ('full' runs saved without a mode dir)
    """
    slug = task if fold is None else f"{task}_fold{fold}"
    base = os.path.join(runs_root, slug)

    # Standard: runs/<slug>/<mode>/<ts>/
    cands = [p for p in glob.glob(os.path.join(base, mode, "*")) if os.path.isdir(p)]

    # Legacy 'full': timestamp dirs sitting directly under the slug dir.
    if mode == "full":
        for p in glob.glob(os.path.join(base, "*")):
            if os.path.isdir(p) and os.path.basename(p) not in MODE_DIRS:
                cands.append(p)

    return _latest(cands)


def task_folds(runs_root, task):
    """Return [None] for single-run tasks, or [1..k] if fold dirs exist."""
    folds = set()
    for p in glob.glob(os.path.join(runs_root, f"{task}_fold*")):
        base = os.path.basename(p)
        if os.path.isdir(p) and "_fold" in base:
            try:
                folds.add(int(base.split("_fold")[-1]))
            except ValueError:
                pass
    return sorted(folds) if folds else [None]


# ------------------------------------------------------------- statistics ---

def _segments(y, offsets):
    off = np.asarray(offsets).astype(int)
    for i in range(len(off) - 1):
        yield y[off[i]:off[i + 1]]


def autocorr_multiclass(y, offsets, num_classes):
    """Mean lag-1 autocorrelation over one-hot channels; also stay prob."""
    y = np.asarray(y).astype(int).ravel()
    C = int(num_classes if num_classes else (y.max() + 1))
    a_prev, a_next = [], []
    stay_hits, stay_tot = 0, 0
    for seg in _segments(y, offsets):
        if len(seg) < 2:
            continue
        oh = np.eye(C)[seg]                 # [T, C]
        a_prev.append(oh[:-1]); a_next.append(oh[1:])
        stay_hits += int((seg[1:] == seg[:-1]).sum())
        stay_tot += len(seg) - 1
    if not a_prev:
        return 0.0, 0.0
    P = np.concatenate(a_prev, 0); N = np.concatenate(a_next, 0)
    ac = _channelwise_corr(P, N)
    stay = stay_hits / max(stay_tot, 1)
    return float(np.nanmean(ac)), float(stay)


def autocorr_multilabel(y, offsets):
    """Mean lag-1 autocorrelation over classes w/ variance; exact-vector stay prob."""
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    a_prev, a_next = [], []
    stay_hits, stay_tot = 0, 0
    for seg in _segments(y, offsets):
        if len(seg) < 2:
            continue
        b = (seg > 0.5).astype(int)
        a_prev.append(b[:-1]); a_next.append(b[1:])
        stay_hits += int((b[1:] == b[:-1]).all(axis=1).sum())
        stay_tot += len(b) - 1
    if not a_prev:
        return 0.0, 0.0
    P = np.concatenate(a_prev, 0); N = np.concatenate(a_next, 0)
    ac = _channelwise_corr(P, N)
    stay = stay_hits / max(stay_tot, 1)
    return float(np.nanmean(ac)), float(stay)


def _channelwise_corr(P, N):
    """Pearson correlation per column between P[:,c] and N[:,c]; NaN if no variance."""
    out = []
    for c in range(P.shape[1]):
        x, z = P[:, c].astype(float), N[:, c].astype(float)
        if x.std() < 1e-8 or z.std() < 1e-8:
            out.append(np.nan)
        else:
            out.append(np.corrcoef(x, z)[0, 1])
    return np.array(out)


def autocorr_curve(y, offsets, is_multiclass, num_classes, max_lag):
    """Mean per-channel label autocorrelation as a function of lag (1..max_lag).

    Returns (lags, values) where values[k] is the mean autocorrelation at lag
    lags[k], averaged over class channels (one-hot channels for multi-class),
    and over all within-video pairs separated by that lag. NaN where undefined.
    """
    y = np.asarray(y)
    if is_multiclass:
        yy = y.astype(int).ravel()
        C = int(num_classes if num_classes else (yy.max() + 1))
        onehot = np.eye(C)
        seg_mats = [onehot[s] for s in _segments(yy, offsets)]   # each [T, C]
    else:
        if y.ndim == 1:
            y = y[:, None]
        seg_mats = [(s > 0.5).astype(float) for s in _segments(y, offsets)]

    lags = list(range(1, max_lag + 1))
    values = []
    for k in lags:
        prev, nxt = [], []
        for m in seg_mats:
            if len(m) > k:
                prev.append(m[:-k]); nxt.append(m[k:])
        if not prev:
            values.append(float("nan")); continue
        P = np.concatenate(prev, 0); N = np.concatenate(nxt, 0)
        ac = _channelwise_corr(P, N)
        values.append(float(np.nanmean(ac)) if np.isfinite(ac).any() else float("nan"))
    return lags, values


# ------------------------------------------------------------- run reading ---

def read_test_metric(run_dir, primary):
    bj = os.path.join(run_dir, "best.json")
    if os.path.isfile(bj):
        with open(bj) as f:
            d = json.load(f)
        tm = d.get("test_metrics") or {}
        if primary in tm:
            return float(tm[primary])
    return None


def read_alpha_sigma(run_dir):
    """Learned alpha/sigma at the best-val epoch of a 'full' run."""
    bj = os.path.join(run_dir, "best.json")
    best_ep = None
    if os.path.isfile(bj):
        with open(bj) as f:
            best_ep = json.load(f).get("best_epoch")
    mj = os.path.join(run_dir, "metrics.jsonl")
    rows = []
    if os.path.isfile(mj):
        with open(mj) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        return None, None
    row = next((r for r in rows if r.get("epoch") == best_ep), rows[-1])
    return row.get("alpha"), row.get("sigma")


def read_ytrue(run_dir):
    npz = os.path.join(run_dir, "predictions_test.npz")
    if not os.path.isfile(npz):
        return None, None
    d = np.load(npz, allow_pickle=True)
    return d["y_true"], d["video_offsets"]


# ------------------------------------------------------------------- main ---

def main(args):
    runs = os.path.abspath(args.runs)
    result = {}

    for task in TASKS:
        folds = task_folds(runs, task)
        primary = PRIMARY[task]
        acs, stays, gains, alphas, sigmas = [], [], [], [], []
        curve_lags, curve_vals = None, []   # accumulate per-fold curves

        for fold in folds:
            full_dir = find_run(runs, task, "full", fold)
            slot_dir = find_run(runs, task, "slot_proto", fold)
            if not full_dir:
                print(f"[warn] no 'full' run for {task} fold={fold}")
                continue

            # continuity from ground-truth test sequences
            y_true, offsets = read_ytrue(full_dir)
            if y_true is not None:
                is_mc = task in MULTICLASS
                if is_mc:
                    ac, stay = autocorr_multiclass(y_true, offsets, num_classes=None)
                else:
                    ac, stay = autocorr_multilabel(y_true, offsets)
                acs.append(ac); stays.append(stay)
                lags, vals = autocorr_curve(y_true, offsets, is_mc,
                                            num_classes=None, max_lag=args.max_lag)
                curve_lags = lags
                curve_vals.append(vals)

            # temporal gain
            f_metric = read_test_metric(full_dir, primary)
            s_metric = read_test_metric(slot_dir, primary) if slot_dir else None
            if f_metric is not None and s_metric is not None:
                gains.append(100.0 * (f_metric - s_metric))

            # learned params
            a, s = read_alpha_sigma(full_dir)
            if a is not None:
                alphas.append(a)
            if s is not None:
                sigmas.append(s)

        def _m(xs):
            return float(np.mean(xs)) if xs else None

        def _s(xs):
            return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0

        stay = _m(stays)
        curve_mean = None
        if curve_lags is not None and curve_vals:
            arr = np.array(curve_vals, dtype=float)          # [n_folds, L]
            curve_mean = {"lags": curve_lags,
                          "values": np.nanmean(arr, axis=0).tolist()}
        result[task] = {
            "autocorr":       _m(acs),
            "stay_prob":      stay,
            "run_length":     (1.0 / (1.0 - stay)) if (stay is not None and stay < 1) else None,
            "gain_pp":        _m(gains),
            "gain_pp_std":    _s(gains),
            "alpha":          _m(alphas),
            "sigma":          _m(sigmas),
            "n_folds":        len([f for f in folds]),
            "autocorr_by_lag": curve_mean,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    # human-readable dump
    print(f"\n[written] {args.out}\n")
    hdr = f"{'task':16s} {'autocorr':>9s} {'stay':>7s} {'runlen':>8s} {'gain_pp':>8s} {'alpha':>7s} {'sigma':>7s}"
    print(hdr); print("-" * len(hdr))
    for task in TASKS:
        r = result[task]
        def fmt(x, p=3):
            return f"{x:.{p}f}" if isinstance(x, (int, float)) else "  n/a"
        print(f"{task:16s} {fmt(r['autocorr']):>9s} {fmt(r['stay_prob']):>7s} "
              f"{fmt(r['run_length'],1):>8s} {fmt(r['gain_pp'],2):>8s} "
              f"{fmt(r['alpha']):>7s} {fmt(r['sigma']):>7s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="Path to the runs/ directory")
    ap.add_argument("--out", default="viz_out/temporal_continuity.json")
    ap.add_argument("--max-lag", type=int, default=64,
                    help="Maximum lag (in sampling steps) for the autocorrelation curve.")
    main(ap.parse_args())
