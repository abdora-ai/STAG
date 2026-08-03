"""
Downstream metrics for STAG training.

- phase_metrics       : frame-wise accuracy + macro-F1 (Cholec80 phase)
- multilabel_metrics  : macro mAP (Cholec80/GraSP tool, CholecT50 verb)
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, average_precision_score, precision_score, recall_score
)


def phase_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Multi-class frame-wise:
        y_pred : [N] int
        y_true : [N] int
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def multilabel_metrics(y_prob: np.ndarray, y_true: np.ndarray, thr: float = 0.5) -> dict:
    """
    Multi-label frame-wise:
        y_prob : [N, C] float in [0, 1]
        y_true : [N, C] binary
    Primary metric: macro mAP (matches paper Tables 2/3 for tool / verb).
    """
    y_pred = (y_prob >= thr).astype(np.int32)
    return {
        "mAP": float(average_precision_score(y_true, y_prob, average="macro")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
