#!/usr/bin/env python3
"""
MS-TCN head for frozen LemonFM features - a fair parametric temporal baseline.

This is the temporal baseline requested by the supervisor review: an MS-TCN
(Multi-Stage Temporal Convolutional Network, Farha & Gall 2019 / TeCNO 2020)
trained on the same frozen LemonFM features as STAG, under the same per-video
training regime. This isolates whether STAG's temporal affinity graph is better
than a standard learned temporal head when both operate on the same frozen features.

Architecture
------------
We implement the standard MS-TCN as described in:
  - Farha & Gall (2019) "MS-TCN: Multi-Stage Temporal Convolutional Network
    for Action Segmentation." CVPR.
  - Czempiel et al. (2020) "TeCNO: Surgical Phase Recognition with Multi-Stage
    Temporal Convolutional Networks." MICCAI.

One input projection (pooled 1536-D feature → num_filters) followed by
num_stages stages of dilated causal convolutions.

Parameters (~0.5-2M depending on config):
  - input: pooled feature p_t ∈ R^1536 (same as STAG, NOT the spatial grid)
  - stages: 4, filters: 64, layers_per_stage: 10 - matches TeCNO for phase
  - For tool/verb we use a lighter config: stages=2, filters=32, layers_per_stage=8

Integration
-----------
This script uses the same data_utils, metrics, and cached features as train_stag.py.
The interface is intentionally parallel so results are directly comparable.

Usage:
    # Cholec80 phase
    python scripts/train_mstcn.py --root . --task cholec80_phase --epochs 50

    # Cholec80 tool
    python scripts/train_mstcn.py --root . --task cholec80_tool  --epochs 50

    # GraSP tool
    python scripts/train_mstcn.py --root . --task grasp_tool     --epochs 50

    # CholecT50 verb (fold 1)
    python scripts/train_mstcn.py --root . --task cholect50_verb --fold 1 --epochs 50
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- building blocks ---

class _SingleStageModel(nn.Module):
    """One TCN stage: projection + R dilated residual layers."""

    def __init__(self, num_layers, num_f_maps, input_dim, num_classes, causal=True):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            _DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, causal=causal)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        # x: [B, C, T]
        f = self.conv_1x1(x)
        for layer in self.layers:
            f = layer(f)
        return self.conv_out(f), f   # (logits [B, N_c, T], features [B, F, T])


class _DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, causal=True):
        super().__init__()
        self.causal = causal
        self.dilation = dilation
        # causal convolution: pad only on the left
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3,
            padding=0 if causal else dilation,
            dilation=dilation,
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        if self.causal:
            # left-only padding = dilation * (kernel_size - 1) = dilation * 2
            x_pad = F.pad(x, (self.dilation * 2, 0))
        else:
            x_pad = x
        out = F.relu(self.conv_dilated(x_pad))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class MSTCN(nn.Module):
    """
    Multi-Stage TCN that operates on per-video feature sequences.

    Input:  pooled [T, 1536]  (the same p_t used by STAG; spatial grid ignored)
    Output: [T, num_classes]

    The first stage takes the raw features; subsequent stages refine the
    logit predictions from the previous stage (standard MS-TCN formulation).

    causal=True (default) for a fair online comparison with STAG's acausal graph.
    Set causal=False to match STAG's bidirectional setting.
    """

    def __init__(
        self,
        in_dim: int = 1536,
        num_classes: int = 7,
        num_stages: int = 4,
        num_layers: int = 10,
        num_filters: int = 64,
        causal: bool = True,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.num_classes = num_classes

        # Stage 1: raw features → logits + intermediate features
        self.stage1 = _SingleStageModel(
            num_layers, num_filters, in_dim, num_classes, causal=causal)

        # Stages 2+: refine using previous stage's (soft) predictions
        self.stages = nn.ModuleList([
            _SingleStageModel(
                num_layers, num_filters,
                num_classes,   # input = previous stage's class logits
                num_classes, causal=causal)
            for _ in range(num_stages - 1)
        ])

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        pooled: [T, 1536]   (one video, unbatched, same as STAG.forward)
        Returns: [T, num_classes]
        """
        # Add batch dim, transpose to [1, C, T] for Conv1d
        x = pooled.unsqueeze(0).transpose(1, 2)   # [1, 1536, T]

        outputs = []
        out, f = self.stage1(x)
        outputs.append(out)

        for stage in self.stages:
            out, f = stage(F.softmax(out, dim=1))
            outputs.append(out)

        # Return the final stage's output, squeezed back to [T, num_classes]
        return outputs[-1].squeeze(0).transpose(0, 1), outputs   # final + all stages

    def all_stage_outputs(self, pooled: torch.Tensor):
        """Returns list of [T, N_c] tensors for all stages (needed for multi-stage loss)."""
        _, all_outs = self.forward(pooled)
        # each entry: [1, N_c, T] → [T, N_c]
        return [o.squeeze(0).transpose(0, 1) for o in all_outs]


# ---------------------------------------------------------------- task configs ---

MSTCN_TASK_CONFIGS = {
    # ---- TeCNO config (Czempiel 2020): used for phase recognition.
    # 4 stages × 10 layers × 64 filters - matches the published TeCNO architecture.
    "cholec80_phase": dict(num_stages=4, num_layers=10, num_filters=64, causal=True),

    # ---- Standard MS-TCN config (Farha & Gall 2019): used for tool / verb.
    # Lighter than TeCNO because tool-presence and verb sequences have different
    # structure (persistent / transient rather than long monotonic phases).
    "cholec80_tool":  dict(num_stages=2, num_layers=8,  num_filters=32, causal=True),
    "grasp_tool":     dict(num_stages=2, num_layers=8,  num_filters=32, causal=True),
    "cholect50_verb": dict(num_stages=2, num_layers=8,  num_filters=32, causal=True),
}

# Human-readable name used in run directory and summary CSV
# (both are variants of the same MS-TCN family; TeCNO = MS-TCN tuned for phase)
MSTCN_RUN_NAME = {
    "cholec80_phase": "tecno",    # TeCNO config
    "cholec80_tool":  "mstcn",
    "grasp_tool":     "mstcn",
    "cholect50_verb": "mstcn",
}


def build_mstcn(task: str, num_classes: int, causal: bool = False) -> MSTCN:
    """Build an MS-TCN/TeCNO head for the given task.

    causal=False (default) matches STAG's acausal/bidirectional setting -
    the fair comparison for the paper.
    causal=True gives the standard online TeCNO/MS-TCN setting.
    Run both and report both so reviewers see the full picture.
    """
    cfg = dict(MSTCN_TASK_CONFIGS[task])   # copy so we can override causal
    cfg["causal"] = causal
    return MSTCN(in_dim=1536, num_classes=num_classes, **cfg)
