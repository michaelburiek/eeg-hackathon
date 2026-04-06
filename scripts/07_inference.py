#!/usr/bin/env python3
"""
scripts/07_inference.py
────────────────────────────────────────────────────────────────────────────────
Run every subject through the trained EEGConformer and generate a
clinician-style LLM report for each one.

For each subject this script:
  1. Loads all EEG windows from the window store
  2. Runs them through the trained model (batch inference)
  3. Aggregates window-level predictions → per-subject classification summary
  4. Calls the Claude API to generate a natural-language clinical report
  5. Saves the report as a markdown file under --output-dir

By default runs on every subject in ALL splits (train + val + test).
Pass --split test to restrict to the held-out test subjects only.

Usage
-----
  # All subjects (generates a report for every EEG in the dataset):
  python scripts/07_inference.py \
      --config      configs/lead_train.yaml \
      --checkpoint  experiments/checkpoints/best.pt

  # Test subjects only:
  python scripts/07_inference.py \
      --config      configs/lead_train.yaml \
      --checkpoint  experiments/checkpoints/best.pt \
      --split       test

  # Single subject:
  python scripts/07_inference.py \
      --config      configs/lead_train.yaml \
      --checkpoint  experiments/checkpoints/best.pt \
      --subject     ADFTD-RS:sub-0001

Output
------
  experiments/reports/
  ├── sub-0001_report.md
  ├── sub-0002_report.md
  ├── ...
  └── inference_summary.csv   ← one row per subject with key metrics
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.config import load_config
from src.corpus.splits import load_splits
from src.data.dataset import EEGDataset, load_window_store
from src.models.eeg_conformer import build_eegconformer
from src.reasoning.llm_reasoner import SubjectClassification, generate_subject_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Per-subject inference ────────────────────────────────────────────────────

@torch.no_grad()
def run_subject_inference(
    model: torch.nn.Module,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run all windows for one subject through the model.

    Returns
    -------
    preds : int array (N,)    — argmax prediction per window
    probs : float array (N, C) — softmax probabilities per window
    """
    model.eval()
    dataset = EEGDataset(windows, np.zeros(len(windows), dtype=np.int64))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_probs = [], []
    for x, _ in loader:
        x      = x.to(device)
        logits = model(x)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_probs.append(probs)

    return np.concatenate(all_preds), np.concatenate(all_probs)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-subject inference + LLM report generation."
    )
    parser.add_argument("--config",     default="configs/lead_train.yaml")
    parser.add_argument("--checkpoint", default="experiments/checkpoints/best.pt",
                        help="Trained model checkpoint (.pt).")
    parser.add_argument("--index-csv",  help="Override paths.index_csv from config.")
    parser.add_argument("--splits-json", help="Override paths.splits_json from config.")
    parser.add_argument("--output-dir", default="experiments/reports",
                        help="Directory for per-subject markdown reports.")
    parser.add_argument("--split",
                        choices=["train", "val", "test", "all"],
                        default="all",
                        help="Which split of subjects to run (default: all).")
    parser.add_argument("--subject",    default=None,
                        help="Run a single subject_key instead of a full split.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device",     help="Override project.device from config.")
    parser.add_argument("--no-llm",     action="store_true",
                        help="Skip Claude API calls — save raw classification only.")
    parser.add_argument("--env",        default=".env")
    args = parser.parse_args()

    load_dotenv(args.env)

    cfg   = load_config(args.config)
    paths = cfg.get("paths", {})
    ds_cfg = cfg.get("dataset", {})

    index_csv   = Path(args.index_csv   or paths.get("index_csv",   "data/lead/window_store/index.csv"))
    splits_json = Path(args.splits_json or paths.get("splits_json", "data/lead/splits.json"))
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run scripts/04_train.py first to train the model."
        )

    # ── Device ────────────────────────────────────────────────────────────────
    device_str = args.device or cfg.get("project", {}).get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    log.info("Using device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("Loading checkpoint: %s", checkpoint_path)
    label_map = ds_cfg.get("label_map", {"CN": 0, "AD": 1, "FTD": 2})
    class_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]

    # Infer n_times from config
    pp_cfg  = cfg.get("preprocessing", {"sfreq": 200})
    win_cfg = cfg.get("windowing", {"window_size_sec": 2.0})
    sfreq   = pp_cfg.get("sfreq", 200)
    n_times = int(win_cfg.get("window_size_sec", 2.0) * sfreq)

    cfg["model"]["n_classes"] = len(label_map)
    cfg["model"]["n_channels"] = cfg.get("model", {}).get("n_channels", 19)
    cfg["preprocessing"] = {"sfreq": sfreq}
    cfg["windowing"]     = {"window_size_sec": n_times / sfreq}
    cfg["dataset"]["label_map"] = label_map

    model = build_eegconformer(cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    log.info("Model loaded.")

    # ── Determine subject list ─────────────────────────────────────────────
    index_df = pd.read_csv(index_csv)

    if args.subject:
        subject_keys = [args.subject]
        log.info("Single-subject mode: %s", args.subject)
    elif args.split == "all":
        subject_keys = index_df["subject_key"].tolist()
        log.info("Running all %d subjects.", len(subject_keys))
    else:
        splits = load_splits(splits_json)
        subject_keys = splits[args.split]
        log.info("Running %d subjects from the '%s' split.", len(subject_keys), args.split)

    # Build a subject → true label map
    true_label_map = dict(zip(index_df["subject_key"], index_df["label"]))

    # ── Per-subject loop ──────────────────────────────────────────────────────
    summary_rows = []
    n_total = len(subject_keys)

    for i, subject_key in enumerate(subject_keys, 1):
        log.info("[%d/%d] Processing %s", i, n_total, subject_key)

        # Load this subject's windows only
        try:
            windows, _, _ = load_window_store(
                index_csv, subject_keys=[subject_key], label_map=label_map
            )
        except Exception as exc:
            log.warning("  Skipping %s — failed to load windows: %s", subject_key, exc)
            continue

        # Run inference
        preds, probs = run_subject_inference(model, windows, device, args.batch_size)

        # Build classification summary
        sc = SubjectClassification(
            subject_id=subject_key,
            n_windows=len(preds),
            window_preds=preds.tolist(),
            window_probs=probs.tolist(),
            class_names=class_names,
            true_label=true_label_map.get(subject_key),
        )

        primary     = sc.primary_class()
        confidence  = sc.primary_confidence()
        mean_probs  = sc.mean_probs()
        true_label  = sc.true_label or "unknown"
        correct     = (primary == true_label) if sc.true_label else None

        log.info(
            "  → Predicted: %s (%.1f%% windows)  |  True: %s  |  Correct: %s",
            primary, confidence, true_label,
            str(correct) if correct is not None else "n/a",
        )

        # Generate LLM report
        report_text = None
        if not args.no_llm:
            try:
                report_text = generate_subject_report(sc)
                log.info("  → LLM report generated.")
            except Exception as exc:
                log.warning("  LLM report failed for %s: %s", subject_key, exc)
                report_text = (
                    f"# EEG Classification Report — {subject_key}\n\n"
                    f"**Predicted:** {primary} ({confidence:.1f}% window agreement)\n"
                    f"**True label:** {true_label}\n\n"
                    f"*LLM report generation failed: {exc}*\n"
                )

        if report_text is None:
            # --no-llm mode: write a minimal classification-only report
            report_text = (
                f"# EEG Classification Report — {subject_key}\n\n"
                f"**Predicted:** {primary} ({confidence:.1f}% window agreement)\n"
                f"**True label:** {true_label}\n"
                f"**Windows analysed:** {len(preds)}\n\n"
                "## Window Prediction Distribution\n\n"
                + "\n".join(
                    f"- {name}: {int((np.array(preds) == idx).sum())} windows "
                    f"({100 * (np.array(preds) == idx).mean():.1f}%)"
                    for idx, name in enumerate(class_names)
                )
                + "\n\n## Mean Class Probabilities\n\n"
                + "\n".join(
                    f"- {name}: {prob * 100:.1f}%"
                    for name, prob in sorted(mean_probs.items(), key=lambda x: -x[1])
                )
            )

        # Save report
        safe_id    = subject_key.replace(":", "_").replace("/", "_")
        report_path = output_dir / f"{safe_id}_report.md"
        report_path.write_text(report_text, encoding="utf-8")

        summary_rows.append({
            "subject_key":       subject_key,
            "true_label":        true_label,
            "predicted_label":   primary,
            "correct":           correct,
            "primary_confidence": round(confidence, 2),
            "n_windows":         len(preds),
            **{f"mean_prob_{name}": round(mean_probs.get(name, 0.0) * 100, 2)
               for name in class_names},
            "report_path":       str(report_path),
        })

    # ── Save summary CSV ──────────────────────────────────────────────────────
    summary_path = output_dir / "inference_summary.csv"
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        log.info("Summary saved to: %s", summary_path)

        correct_rows = [r for r in summary_rows if r["correct"] is True]
        if correct_rows:
            acc = len(correct_rows) / len(summary_rows) * 100
            log.info("Overall accuracy across processed subjects: %.1f%%", acc)

    log.info(
        "Done. %d reports written to: %s",
        len(summary_rows), output_dir,
    )
    print(f"\nReports: {output_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
