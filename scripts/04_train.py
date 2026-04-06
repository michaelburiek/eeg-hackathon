#!/usr/bin/env python3
"""
scripts/04_train.py
────────────────────────────────────────────────────────────────────────────────
Full pipeline: baseline evaluation → train from scratch → final evaluation.

Steps
-----
1. Load LEAD window store and subject-level splits.
2. Build EEGConformer with random (untrained) weights.
3. Evaluate the untrained model on the test set → baseline metrics.
4. Train on the training set, early-stopping on val balanced-accuracy.
5. Evaluate the best checkpoint on the test set → final metrics.
6. Print a side-by-side comparison.

Usage
-----
  # Local (CPU / MPS):
  python scripts/04_train.py --config configs/lead_train.yaml

  # KOA GPU cluster:
  koa run scripts/train.slurm --watch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.corpus.splits import load_splits
from src.data.dataset import EEGDataset, load_window_store
from src.evaluation.metrics import compute_metrics, format_metrics, get_classification_report
from src.models.eeg_conformer import build_eegconformer
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Evaluation helper ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Run inference on a DataLoader and return metrics."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
        all_probs.append(probs)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    return compute_metrics(y_true, y_pred, y_prob)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train EEGConformer from scratch on LEAD.")
    parser.add_argument("--config",      default="configs/lead_train.yaml")
    parser.add_argument("--index-csv",   help="Override paths.index_csv from config.")
    parser.add_argument("--splits-json", help="Override paths.splits_json from config.")
    parser.add_argument("--output-dir",  help="Override paths.results_dir from config.")
    parser.add_argument("--device",      help="Override project.device (cuda/mps/cpu).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths     = cfg.get("paths", {})
    ds_cfg    = cfg.get("dataset", {})
    dl_cfg    = cfg.get("dataloader", {})

    index_csv   = Path(args.index_csv   or paths.get("index_csv",    "data/lead/window_store/index.csv"))
    splits_json = Path(args.splits_json or paths.get("splits_json",  "data/lead/splits.json"))
    output_dir  = Path(args.output_dir  or paths.get("results_dir",  "experiments/results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device_str = args.device or cfg.get("project", {}).get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available — falling back to CPU.")
        device_str = "cpu"
    if device_str == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS requested but not available — falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    log.info("Using device: %s", device)

    # ── Seed ──────────────────────────────────────────────────────────────────
    seed = cfg.get("project", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading window store from %s", index_csv)
    label_map = ds_cfg.get("label_map", {"CN": 0, "AD": 1, "FTD": 2})

    splits = load_splits(splits_json)
    log.info(
        "Splits: train=%d  val=%d  test=%d subjects",
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )

    log.info("Loading train windows...")
    train_win, train_lbl, train_sids = load_window_store(
        index_csv, subject_keys=splits["train"], label_map=label_map
    )
    log.info("Loading val windows...")
    val_win, val_lbl, val_sids = load_window_store(
        index_csv, subject_keys=splits["val"], label_map=label_map
    )
    log.info("Loading test windows...")
    test_win, test_lbl, test_sids = load_window_store(
        index_csv, subject_keys=splits["test"], label_map=label_map
    )

    for split_name, lbl in [("train", train_lbl), ("val", val_lbl), ("test", test_lbl)]:
        unique, counts = np.unique(lbl, return_counts=True)
        inv_map = {v: k for k, v in label_map.items()}
        dist = {inv_map.get(int(u), str(u)): int(c) for u, c in zip(unique, counts)}
        log.info("%s class distribution: %s", split_name, dist)

    train_ds = EEGDataset(train_win, train_lbl, train_sids)
    val_ds   = EEGDataset(val_win,   val_lbl,   val_sids)
    test_ds  = EEGDataset(test_win,  test_lbl,  test_sids)

    batch_size  = dl_cfg.get("batch_size",  32)
    num_workers = dl_cfg.get("num_workers", 4)
    pin_memory  = dl_cfg.get("pin_memory",  True) and (device_str == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    n_channels = train_ds.n_channels
    n_times    = train_ds.n_times
    log.info("Building EEGConformer (n_channels=%d, n_times=%d, n_classes=%d)",
             n_channels, n_times, len(label_map))

    cfg["model"]["n_channels"] = n_channels
    cfg["model"]["n_classes"]  = len(label_map)
    cfg["preprocessing"] = {"sfreq": 200}
    cfg["windowing"]     = {"window_size_sec": n_times / 200.0}
    cfg["dataset"]["label_map"] = label_map
    model = build_eegconformer(cfg)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("Model has %.2fM parameters (random initialization).", n_params)

    # ── Step 1: Baseline (untrained) evaluation ───────────────────────────────
    log.info("=" * 60)
    log.info("STEP 1 — Baseline evaluation (untrained model)")
    log.info("=" * 60)
    baseline_metrics = evaluate(model, test_loader, device)
    log.info("Baseline test metrics: %s", format_metrics(baseline_metrics))
    print("\n── Baseline (untrained) ─────────────────────────────────────")
    print(get_classification_report(
        np.concatenate([b for b in [test_lbl]]),
        np.concatenate([
            model(x.to(device)).argmax(dim=-1).cpu().numpy()
            for x, _ in test_loader
        ]),
        class_names=list(label_map.keys()),
    ))

    # ── Step 2: Training ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 2 — Training from scratch")
    log.info("=" * 60)
    t_start = time.time()

    trainer = Trainer(
        model=model,
        cfg=cfg,
        device=device_str,
        run_name=cfg.get("experiment", {}).get("name", "lead-eegconformer"),
    )
    history = trainer.fit(train_loader, val_loader)

    elapsed = time.time() - t_start
    log.info("Training complete in %.1f minutes.", elapsed / 60)

    # ── Step 3: Final evaluation ──────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 3 — Final evaluation (best checkpoint)")
    log.info("=" * 60)
    final_metrics = evaluate(trainer.model, test_loader, device)
    log.info("Final test metrics:    %s", format_metrics(final_metrics))

    # ── Side-by-side comparison ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':>10} {'Trained':>10}")
    print("-" * 52)
    for key in ["balanced_accuracy", "f1_macro", "cohen_kappa"]:
        b = baseline_metrics.get(key, float("nan"))
        f = final_metrics.get(key, float("nan"))
        print(f"{key:<30} {b:>10.4f} {f:>10.4f}")
    print("=" * 60)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "baseline": {k: float(v) for k, v in baseline_metrics.items()},
        "trained":  {k: float(v) for k, v in final_metrics.items()},
        "config": args.config,
        "splits": str(splits_json),
        "n_train_windows": int(len(train_ds)),
        "n_val_windows":   int(len(val_ds)),
        "n_test_windows":  int(len(test_ds)),
        "n_params_M": float(f"{n_params:.2f}"),
        "training_seconds": float(f"{elapsed:.1f}"),
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Results saved to: %s", results_path)


if __name__ == "__main__":
    main()
