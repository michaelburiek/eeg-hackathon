"""
src/evaluation/metrics.py
────────────────────────────────────────────────────────────────────────────────
Evaluation metrics for EEG classification.

Primary metric: Balanced Accuracy (BAcc) — standard in EEG literature because
class sizes are almost always unequal. All public EEG benchmarks (TUAB, SEED,
BCIC IV-2a) report BAcc as the main metric.

Secondary metrics: macro-F1, Cohen's κ, one-vs-rest AUC.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


CLASS_NAMES = ["AD", "FTD", "CN"]


def _resolve_class_names(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    class_names: Optional[List[str]],
) -> List[str]:
    if class_names is not None:
        return class_names
    if y_prob is not None and y_prob.ndim == 2:
        n_classes = int(y_prob.shape[1])
    else:
        max_label = int(max(np.max(y_true), np.max(y_pred))) if len(y_true) else 0
        n_classes = max_label + 1
    if n_classes <= len(CLASS_NAMES):
        return CLASS_NAMES[:n_classes]
    return [f"class_{i}" for i in range(n_classes)]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute standard EEG classification metrics.

    Parameters
    ----------
    y_true      : integer ground-truth labels (N,)
    y_pred      : integer predicted labels (N,)
    y_prob      : softmax probability matrix (N, C) — needed for AUC
    class_names : display names for classes

    Returns
    -------
    dict of metric_name → float
    """
    names = _resolve_class_names(y_true, y_pred, y_prob, class_names)

    metrics: Dict[str, float] = {}

    # Balanced accuracy — most important
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    # Macro F1
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Cohen's kappa
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
    for i, name in enumerate(names):
        metrics[f"acc_{name}"] = float(per_class_acc[i]) if i < len(per_class_acc) else 0.0

    # ROC-AUC (requires probabilities)
    if y_prob is not None:
        try:
            n_classes = y_prob.shape[1]
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro",
                labels=list(range(n_classes)),
            )
        except Exception:  # noqa: BLE001
            metrics["roc_auc_ovr"] = float("nan")

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """Return a single-line string summary of key metrics."""
    return (
        f"BAcc={metrics.get('balanced_accuracy', float('nan')):.3f}  "
        f"F1={metrics.get('f1_macro', float('nan')):.3f}  "
        f"κ={metrics.get('cohen_kappa', float('nan')):.3f}"
    )


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """Return sklearn classification report string."""
    names = _resolve_class_names(y_true, y_pred, None, class_names)
    return classification_report(
        y_true, y_pred,
        target_names=names,
        digits=3,
    )


class MetricsAccumulator:
    """
    Accumulate per-fold metrics across a CV run and compute mean ± std.

    Usage
    -----
    acc = MetricsAccumulator()
    for fold_metrics in fold_results:
        acc.update(fold_metrics)
    summary = acc.summary()
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, float]] = []

    def update(self, metrics: Dict[str, float]) -> None:
        self._records.append(metrics)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return {metric_name: {mean, std, values}} over all folds."""
        if not self._records:
            return {}
        keys = self._records[0].keys()
        result = {}
        for k in keys:
            vals = [r[k] for r in self._records if not np.isnan(r.get(k, float("nan")))]
            result[k] = {
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "std":  float(np.std(vals))  if vals else float("nan"),
                "values": vals,
            }
        return result

    def formatted_summary(self) -> str:
        """Human-readable CV summary."""
        s = self.summary()
        lines = ["Cross-Validation Summary"]
        lines.append("=" * 40)
        for k in ["balanced_accuracy", "f1_macro", "cohen_kappa", "roc_auc_ovr"]:
            if k in s:
                lines.append(f"  {k:<30} {s[k]['mean']:.4f} ± {s[k]['std']:.4f}")
        return "\n".join(lines)
