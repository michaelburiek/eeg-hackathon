"""
src/training/losses.py
────────────────────────────────────────────────────────────────────────────────
Loss functions for imbalanced EEG classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.

    Parameters
    ----------
    smoothing : float in [0, 1) — fraction of probability mass redistributed
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_targets * log_probs).sum(dim=-1).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced classification.

    Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017).

    Parameters
    ----------
    alpha   : per-class weight tensor or scalar
    gamma   : focusing parameter (0 = standard CE, 2 is typical)
    """

    def __init__(
        self,
        alpha: float | list | torch.Tensor = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, (list, torch.Tensor)):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))
        else:
            self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce

        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def build_loss(cfg: dict) -> nn.Module:
    """Build loss function from config."""
    loss_type  = cfg.get("training", {}).get("loss", "label_smoothing")
    smoothing  = cfg.get("training", {}).get("label_smoothing", 0.1)

    if loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    if loss_type == "focal":
        return FocalLoss(gamma=2.0)
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss type: {loss_type!r}")
