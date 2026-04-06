"""
src/training/trainer.py
────────────────────────────────────────────────────────────────────────────────
Training loop with W&B tracking, early stopping, and LR scheduling.

Public API
----------
Trainer                — stateful trainer object
train_fold(model, ...)  — run one CV fold and return metrics
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LinearLR, SequentialLR
from torch.utils.data import DataLoader, SubsetRandomSampler

from .losses import build_loss
from ..evaluation.metrics import compute_metrics
from ..tracking import finish_wandb_run, init_wandb_run, log_wandb_metrics, set_wandb_summary

log = logging.getLogger(__name__)


# ─── Optimizer / scheduler builders ──────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    t_cfg = cfg.get("training", {})
    name  = t_cfg.get("optimizer", "adamw").lower()
    lr    = t_cfg.get("lr", 1e-4)
    wd    = t_cfg.get("weight_decay", 0.05)

    # Support differential LR for backbone / head (LaBraM)
    if hasattr(model, "trainable_parameter_groups"):
        params = model.trainable_parameter_groups(
            lr_backbone=lr * 0.1,
            lr_head=lr,
        )
    else:
        params = model.parameters()

    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd,
                     betas=t_cfg.get("betas", (0.9, 0.999)),
                     eps=t_cfg.get("eps", 1e-8))
    if name == "sgd":
        return SGD(params, lr=lr, momentum=t_cfg.get("momentum", 0.9),
                   weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    steps_per_epoch: int,
) -> Optional[object]:
    t_cfg      = cfg.get("training", {})
    name       = t_cfg.get("lr_scheduler", "cosine").lower()
    epochs     = t_cfg.get("epochs", 50)
    warmup_ep  = t_cfg.get("warmup_epochs", 5)
    min_lr     = t_cfg.get("min_lr", 1e-6)

    if name == "cosine":
        main_sched = CosineAnnealingLR(
            optimizer,
            T_max=max(epochs - warmup_ep, 1),
            eta_min=min_lr,
        )
        if warmup_ep > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=t_cfg.get("warmup_lr", 1e-6) / t_cfg.get("lr", 1e-4),
                end_factor=1.0,
                total_iters=warmup_ep,
            )
            return SequentialLR(optimizer, schedulers=[warmup, main_sched],
                                milestones=[warmup_ep])
        return main_sched
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", patience=5,
                                 factor=0.5, min_lr=min_lr)
    if name == "none":
        return None
    raise ValueError(f"Unknown LR scheduler: {name!r}")


# ─── Core training loop ───────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience   = patience
        self.mode       = mode
        self.min_delta  = min_delta
        self.best       = -float("inf") if mode == "max" else float("inf")
        self.counter    = 0
        self.best_state: Optional[dict] = None

    def __call__(self, metric: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        improved = (
            (self.mode == "max" and metric > self.best + self.min_delta) or
            (self.mode == "min" and metric < self.best - self.min_delta)
        )
        if improved:
            self.best   = metric
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            log.info("Restored best model weights (score=%.4f).", self.best)


class Trainer:
    """
    Stateful trainer with MLflow logging, early stopping, and gradient clipping.

    Parameters
    ----------
    model     : nn.Module
    cfg       : merged config dict
    device    : torch device string
    run_name  : MLflow run name
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        device: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> None:
        self.model    = model
        self.cfg      = cfg
        self.run_name = run_name or cfg.get("experiment", {}).get("name", "run")
        self.device   = torch.device(
            device or cfg.get("project", {}).get("device", "cpu")
        )
        self.model.to(self.device)

        self.optimizer  = build_optimizer(model, cfg)
        self.criterion  = build_loss(cfg)
        self.t_cfg      = cfg.get("training", {})
        self.clip_grad  = self.t_cfg.get("clip_grad", 1.0)
        self.accum_steps = self.t_cfg.get("gradient_accumulation_steps", 1)

        save_dir = cfg.get("checkpointing", {}).get("save_dir", "experiments/checkpoints")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.early_stopping = EarlyStopping(
            patience=self.t_cfg.get("patience", 10),
            mode=self.t_cfg.get("mode", "max"),
        )

    # ── Single epoch ──────────────────────────────────────────────────────────

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool = True,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.train(train)
        total_loss = 0.0
        all_preds, all_labels = [], []
        self.optimizer.zero_grad()

        for step, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)

            with torch.set_grad_enabled(train):
                logits = self.model(x)
                loss   = self.criterion(logits, y) / self.accum_steps

            if train:
                loss.backward()
                if (step + 1) % self.accum_steps == 0:
                    if self.clip_grad > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.cpu().numpy())

        avg_loss = total_loss / len(loader)
        return avg_loss, np.concatenate(all_preds), np.concatenate(all_labels)

    # ── Full training loop ────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        scheduler=None,
        fold: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Run training loop.

        Returns
        -------
        history : dict of {metric_name → list of per-epoch values}
        """
        epochs = epochs or self.t_cfg.get("epochs", 50)
        scheduler = scheduler or build_scheduler(
            self.optimizer, self.cfg, steps_per_epoch=len(train_loader)
        )

        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_balanced_accuracy": [], "val_balanced_accuracy": [],
        }

        run_name = f"{self.run_name}_fold{fold}" if fold is not None else self.run_name
        wandb_run = init_wandb_run(
            self.cfg,
            run_name,
            "finetune",
            extra_config={"fold": fold if fold is not None else "all", "epochs": epochs},
            tags=[
                self.cfg.get("experiment", {}).get("model", "unknown"),
                self.cfg.get("experiment", {}).get("mode", "unknown"),
                "finetune",
            ],
        )

        try:
            best_val_bacc = 0.0
            for epoch in range(1, epochs + 1):
                t0 = time.time()

                tr_loss, tr_pred, tr_true = self._run_epoch(train_loader, train=True)
                vl_loss, vl_pred, vl_true = self._run_epoch(val_loader, train=False)

                tr_metrics = compute_metrics(tr_true, tr_pred)
                vl_metrics = compute_metrics(vl_true, vl_pred)

                # LR scheduler step
                if scheduler is not None:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(vl_metrics["balanced_accuracy"])
                    else:
                        scheduler.step()

                log_dict = {
                    "train/loss": tr_loss, "val/loss": vl_loss,
                    **{f"train/{k}": v for k, v in tr_metrics.items()},
                    **{f"val/{k}": v for k, v in vl_metrics.items()},
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "epoch_time_s": time.time() - t0,
                }
                log_wandb_metrics(wandb_run, log_dict, step=epoch)

                # Update history
                history["train_loss"].append(tr_loss)
                history["val_loss"].append(vl_loss)
                history["train_balanced_accuracy"].append(tr_metrics["balanced_accuracy"])
                history["val_balanced_accuracy"].append(vl_metrics["balanced_accuracy"])

                val_bacc = vl_metrics["balanced_accuracy"]
                log.info(
                    "Epoch %3d/%d | tr_loss=%.4f | vl_loss=%.4f | "
                    "tr_bacc=%.3f | vl_bacc=%.3f | lr=%.2e | %.1fs",
                    epoch, epochs, tr_loss, vl_loss,
                    tr_metrics["balanced_accuracy"], val_bacc,
                    self.optimizer.param_groups[0]["lr"],
                    time.time() - t0,
                )

                # Checkpoint best model
                if val_bacc > best_val_bacc:
                    best_val_bacc = val_bacc
                    ckpt_name = f"best_fold{fold}.pt" if fold is not None else "best.pt"
                    torch.save(self.model.state_dict(), self.save_dir / ckpt_name)

                # Early stopping
                if self.early_stopping(val_bacc, self.model):
                    log.info("Early stopping triggered at epoch %d.", epoch)
                    break

            # Restore best weights
            self.early_stopping.restore_best(self.model)
            set_wandb_summary(
                wandb_run,
                {
                    "best_val_balanced_accuracy": best_val_bacc,
                    "fold": fold if fold is not None else "all",
                },
            )
        finally:
            finish_wandb_run(wandb_run)

        return history


# ─── CV fold helper ───────────────────────────────────────────────────────────

def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          dict,
    fold:         int,
    device:       Optional[str] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Train one cross-validation fold and return the trained model + metrics.

    This is a convenience wrapper around Trainer.fit() intended to be called
    from the CV loop in scripts/04_finetune.py.
    """
    trainer = Trainer(
        model=model,
        cfg=cfg,
        device=device,
        run_name=cfg.get("experiment", {}).get("name", "run"),
    )
    history = trainer.fit(train_loader, val_loader, fold=fold)
    return trainer.model, history
