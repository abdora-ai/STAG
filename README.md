# STAG: A Spatio-Temporal Affinity Framework for Surgical Video Understanding

Official implementation of **STAG**, a compact head that reuses a frozen surgical
foundation model's own embedding geometry instead of relearning structure from
scarce downstream labels.

> A frozen backbone's standard probing protocol discards two things: the spatial
> layout of the pre-pool token grid, and the temporal continuity already encoded
> in the embedding geometry by self-supervised distillation. STAG recovers both
> with three lightweight modules and **1-2 M trainable parameters**, never
> unfreezing the backbone.

## Method

STAG stacks three modules on the frozen backbone's per-frame outputs: the pooled
feature `p_t` (1536-D) and the `7x7` spatial token grid `Z_t` (49 x 1536).

1. **Class-conditional slot pooling.** `K_s` learnable queries cross-attend once
   to the 49 spatial tokens, restoring localized tool and interaction evidence
   that global average pooling discards.
2. **Cosine prototype layer.** Logits are temperature-scaled cosine similarities
   to learned prototypes, placing task classes inside the same normalized metric
   space the backbone was distilled in.
3. **Temporal affinity graph.** Non-parametric Zhou-style label propagation over
   a banded video graph whose edges are the backbone's own cosine similarities.
   Only two scalars (`alpha`, `sigma`) are learned; the temporal receptive field
   is inherited from pretraining.

## Results

All numbers use a **frozen** LemonFM ConvNeXt-L backbone.

| Method | Phase Acc | Phase F1 | Cholec80 Tool mAP | GraSP Tool mAP | CholecT50 Verb mAP |
|---|---|---|---|---|---|
| LemonFM baseline | 75.8 | 68.6 | 76.8 | 76.4 | 50.4 |
| **STAG (ours)** | **85.3** | **78.2** | **91.2** | **81.6** | **56.4** |
| Δ | +9.5 | +9.6 | +14.4 | +5.2 | +6.0 |

Comparison against a learned parametric temporal head on the same frozen features:

| Head | Temporal params | Phase Acc | Cholec80 Tool mAP | GraSP Tool mAP |
|---|---|---|---|---|
| MS-TCN (TeCNO config for phase) | 0.12-0.76 M | 88.7 | 65.1 | 54.6 |
| **STAG (ours)** | 2 scalars | 85.3 | **91.2** | **81.6** |

## Repository layout

```
scripts/
  stag_head.py            # STAG head: slot pooling + cosine prototypes + temporal graph
  train_stag.py           # STAG training (modes: linear | prototype | slot_proto | full)
  mstcn_head.py           # MS-TCN / TeCNO baseline architecture
  train_mstcn.py          # MS-TCN baseline training (--causal for the online variant)
  reproduce_baseline.py   # shuffled-minibatch linear probe, for protocol comparison
  lemonfm_backbone.py     # frozen LemonFM loader -> (pooled, spatial) outputs
  cache_features.py       # one-pass frozen feature extraction to .npy
  build_labels.py         # per-task label JSONs + official splits
  data_utils.py           # feature/label loading
  metrics.py              # accuracy / macro-F1 / macro-mAP
  run_temporal_baselines.sh   # end-to-end baseline runner (tmux, causal + acausal)
  launch_tmux.sh              # STAG training across GPUs
  launch_ablation_tmux.sh     # 4-mode ablation across GPUs
tools/
  temporal_continuity.py  # label autocorrelation vs. lag (Fig. 2 data)
  extract_grasp.sh        # GraSP tarball extraction
  inspect_checkpoint.py   # verify the LemonFM checkpoint loads
  inspect_datasets.py     # verify dataset layout
viz/
  plot_ablation.py        # Fig. 2: autocorrelation decay curves
  plot_slot_attention.py  # Fig. 3: slot attention overlays
  aggregate.py            # collect runs/ into a summary CSV
env/
  setup_env.sh            # uv-based venv + pinned CUDA 12.4 PyTorch
```

## Setup

### 1. Environment

```bash
bash env/setup_env.sh
source .venv-lemon/bin/activate
```

### 2. Backbone checkpoint

STAG uses the **LemonFM** ConvNeXt-L teacher as a frozen feature extractor. This
repository does **not** redistribute it. Obtain `lemonfm.pth` from the LemonFM
authors ([HuggingFace](https://huggingface.co/visurg/LemonFM)) and place it at the
repository root. Verify it loads:

```bash
python tools/inspect_checkpoint.py --checkpoint lemonfm.pth
```

> **Trained STAG weights.** The head is small (1-2 M parameters) and trains in
> minutes per task from cached features, so it is fastest to reproduce directly
> with the commands below. Trained checkpoints are not included in this release.

### 3. Datasets

Each dataset requires its own access agreement and must be obtained from its
original providers. Expected layout:

```
<root>/
├── lemonfm.pth
├── Cholec80/
│   ├── frames/            videoNN/videoNN_KKKKKK.png
│   ├── phase_annotations/ videoNN-phase.txt
│   └── tool_annotations/  videoNN-tool.txt
├── CholecT50/
│   ├── videos/            VIDNN/*.png
│   └── labels/            VIDNN.json
└── GraSP/
    └── GraSP_1fps/
        ├── annotations/   grasp_short-term_{train,test}.json
        └── frames/        CASEXXX/*.jpg
```

| Dataset | Access |
|---|---|
| Cholec80 | [CAMMA request form](http://camma.u-strasbg.fr/datasets) |
| CholecT50 | [CholecT50 request form](https://forms.gle/GbMj8TwNoNpMUJuv9) |
| GraSP | [BCV-Uniandes/GraSP](https://github.com/BCV-Uniandes/GraSP) |

Verify the layout with `python tools/inspect_datasets.py --root .`

## Usage

### Build labels and cache frozen features

Feature extraction runs the frozen backbone over every frame **once**; all
subsequent training reads only the cached `.npy` files, so no backbone forward
pass occurs at training time.

```bash
python scripts/build_labels.py --root .

for group in cholec80 cholect50 grasp; do
    python scripts/cache_features.py --root . \
        --checkpoint lemonfm.pth --only-group $group
done
```

### Train STAG

```bash
python scripts/train_stag.py --root . --task cholec80_phase --mode full --epochs 50
python scripts/train_stag.py --root . --task cholec80_tool  --mode full --epochs 50
python scripts/train_stag.py --root . --task grasp_tool     --mode full --epochs 50

for fold in 1 2 3 4 5; do
    python scripts/train_stag.py --root . --task cholect50_verb \
        --fold $fold --mode full --epochs 50
done
```

Each run writes a self-contained directory containing `config.json`, `log.txt`,
`metrics.jsonl`, `best.json`, `checkpoint_best.pt`, and `predictions_test.npz`.

### Ablation

`--mode` selects the cumulative configuration, matching the rows of the paper's
ablation table:

| `--mode` | Configuration |
|---|---|
| `linear` | plain classifier on the pooled feature (internal anchor) |
| `prototype` | + cosine prototype layer |
| `slot_proto` | + class-conditional slot pooling |
| `full` | + temporal affinity graph (complete STAG) |

All modes share one training regime (one full video per optimization step, no
frame shuffling) because the temporal graph requires the whole video sequence.
This makes the `linear` anchor internally comparable across rows but **not**
comparable to published shuffled-minibatch probing numbers; use
`reproduce_baseline.py` for that protocol.

### MS-TCN baseline

```bash
bash scripts/run_temporal_baselines.sh . lemonfm.pth
```

Trains MS-TCN (TeCNO configuration for phase) on the identical cached features,
in both causal and acausal variants, each in its own tmux window.

### Figures

```bash
python viz/aggregate.py --runs runs --out viz_out/runs_summary.csv
python tools/temporal_continuity.py --runs runs --out viz_out/temporal_continuity.json
bash viz/run_all.sh
```

## Citation

```bibtex
@inproceedings{stag2026,
  title     = {STAG: A Spatio-Temporal Affinity Framework for Surgical Video Understanding},
  author    = {Anonymous},
  booktitle = {MICCAI Workshop},
  year      = {2026}
}
```

STAG builds on the LemonFM frozen backbone. If you use this code, please also
cite LEMON:

```bibtex
@inproceedings{che2026lemon,
  title     = {LEMON: A Large Endoscopic MONocular Dataset and Foundation Model
               for Perception in Surgical Settings},
  author    = {Che, Chengan and Wang, Chao and Vercauteren, Tom and Tsoka, Sophia
               and Garcia-Peraza-Herrera, Luis C.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition (CVPR)},
  year      = {2026},
  pages     = {42659--42669}
}
```

Please also cite the datasets (Cholec80, CholecT50, GraSP) and the MS-TCN/TeCNO
baselines where appropriate.

## License

Our code is released under the [MIT License](LICENSE). The LemonFM checkpoint and
the Cholec80, CholecT50, and GraSP datasets are governed by their own licenses and
access agreements, which this repository neither alters nor supersedes.
