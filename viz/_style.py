"""Shared matplotlib style + palette for paper-quality figures.

Applies constrained_layout globally so labels don't collide.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_paper_style():
    mpl.rcParams.update({
        # Layout - this is what fixes label collisions
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.06,
        "figure.constrained_layout.w_pad": 0.06,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,

        # DPI / saving
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # Fonts
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
    })


# Colorblind-safe palette (Wong 2011)
PALETTE = {
    "paper_linear": "#D55E00",
    "linear":       "#999999",
    "prototype":    "#0072B2",
    "slot_proto":   "#E69F00",
    "full":         "#009E73",
    "paper":        "#D55E00",
    "ours":         "#009E73",
}

MODE_PRETTY = {
    "paper_linear": "Linear (paper regime)",
    "linear":       "Linear (our regime)",
    "prototype":    "Prototype",
    "slot_proto":   "Slot + Prototype",
    "full":         "STAG (full)",
}

TASK_PRETTY = {
    "cholec80_phase": "Cholec80 Phase",
    "cholec80_tool":  "Cholec80 Tool",
    "grasp_tool":     "GraSP Tool",
    "cholect50_verb": "CholecT50 Verb",
}

TASK_METRIC_LABEL = {
    "cholec80_phase": "Accuracy (%)",
    "cholec80_tool":  "mAP (%)",
    "grasp_tool":     "mAP (%)",
    "cholect50_verb": "mAP (%)",
}
