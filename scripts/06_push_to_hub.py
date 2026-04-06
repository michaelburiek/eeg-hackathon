#!/usr/bin/env python3
"""
scripts/06_push_to_hub.py
────���─────────────────────────────────��─────────────────────────────────────────
Upload the trained EEGConformer checkpoint to HuggingFace Hub.

Uploads:
  - model weights      (pytorch_model.pt)
  - architecture config (config.json)
  - model card          (README.md)
  - results summary     (results.json)

Requires HF_TOKEN in .env and write access to the target repo.

Usage
-----
  python scripts/06_push_to_hub.py \
      --checkpoint experiments/checkpoints/best.pt \
      --results    experiments/results/results.json \
      --config     configs/lead_train.yaml

Target repo: https://huggingface.co/michaelburiek/eeg-hackathon-eegconformer
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.config import load_config
from src.models.eeg_conformer import build_eegconformer

HF_REPO_ID = "michaelburiek/eeg-hackathon-eegconformer"


# ─── Model card ───────────────────────────────────────────────────────────────

def build_model_card(cfg: dict, results: dict | None) -> str:
    label_map = cfg.get("dataset", {}).get("label_map", {"CN": 0, "AD": 1, "FTD": 2})
    classes = ", ".join(f"{k} ({v})" for k, v in label_map.items())

    baseline = (results or {}).get("baseline", {})
    trained  = (results or {}).get("trained",  {})

    def fmt(d: dict, key: str) -> str:
        v = d.get(key)
        return f"{float(v):.4f}" if v is not None else "—"

    metrics_table = ""
    if results:
        metrics_table = f"""
## Results

Evaluated on a held-out test set (15% of subjects, subject-level split).

| Metric | Baseline (untrained) | Trained |
|---|---|---|
| Balanced Accuracy | {fmt(baseline, 'balanced_accuracy')} | {fmt(trained, 'balanced_accuracy')} |
| Macro F1 | {fmt(baseline, 'f1_macro')} | {fmt(trained, 'f1_macro')} |
| Cohen's κ | {fmt(baseline, 'cohen_kappa')} | {fmt(trained, 'cohen_kappa')} |
| OvR AUC | {fmt(baseline, 'roc_auc_ovr')} | {fmt(trained, 'roc_auc_ovr')} |
"""

    t_cfg = cfg.get("training", {})
    dl_cfg = cfg.get("dataloader", {})

    return f"""---
license: apache-2.0
tags:
  - eeg
  - brain-computer-interface
  - medical
  - neuroscience
  - alzheimer
  - dementia
  - pytorch
  - braindecode
library_name: braindecode
---

# EEGConformer — Trained from Scratch on LEAD (ADFTD-RS)

EEGConformer trained from random initialization on the [LEAD open-source EEG dataset](https://github.com/DL4mHealth/LEAD) for three-class dementia classification.

**Task:** Classify resting-state EEG into {classes}

**Architecture:** [EEGConformer](https://braindecode.org/stable/generated/braindecode.models.EEGConformer.html) (convolutional transformer) from [braindecode](https://braindecode.org)

**Paper:** Song et al., "EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization", IEEE TNNLS 2023. [https://ieeexplore.ieee.org/document/10178045](https://ieeexplore.ieee.org/document/10178045)
{metrics_table}
## Dataset

**LEAD — ADFTD-RS subset**
- Source: [DL4mHealth/LEAD](https://github.com/DL4mHealth/LEAD)
- 88 subjects (36 AD · 23 FTD · 29 CN)
- 19 EEG channels (10-20 system)
- 121,825 windows · 2 s each · 200 Hz
- Subject-level train / val / test split (70 / 15 / 15)

## Training

| Hyperparameter | Value |
|---|---|
| Epochs | {t_cfg.get('epochs', 50)} |
| Optimizer | {t_cfg.get('optimizer', 'adamw').upper()} |
| Learning rate | {t_cfg.get('lr', 1e-4)} |
| Weight decay | {t_cfg.get('weight_decay', 0.05)} |
| LR schedule | {t_cfg.get('lr_scheduler', 'cosine')} |
| Batch size | {dl_cfg.get('batch_size', 32)} |
| Early stopping patience | {t_cfg.get('patience', 10)} epochs |
| Loss | Label smoothing (ε={t_cfg.get('label_smoothing', 0.1)}) |

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from braindecode.models import EEGConformer

# Download weights and config
ckpt_path   = hf_hub_download("{HF_REPO_ID}", "pytorch_model.pt")
config_path = hf_hub_download("{HF_REPO_ID}", "config.json")

import json
cfg = json.load(open(config_path))
m   = cfg["model"]

# Rebuild architecture
model = EEGConformer(
    n_outputs           = m["n_classes"],
    n_chans             = m["n_channels"],
    n_times             = m["n_times"],
    sfreq               = m["sfreq"],
    n_filters_time      = m["n_filters_time"],
    filter_time_length  = m["filter_time_length"],
    pool_time_length    = m["pool_time_length"],
    pool_time_stride    = m["pool_time_stride"],
    drop_prob           = m["drop_prob"],
    att_depth           = m["att_depth"],
    att_heads           = m["att_heads"],
    att_drop_prob       = m["att_drop_prob"],
    final_fc_length     = m["final_fc_length"],
)

# Load weights
state = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state)
model.eval()

# Inference — input shape: (batch, n_channels, n_times)
x      = torch.randn(1, {cfg.get('model', {}).get('n_channels', 19)}, {int(cfg.get('model', {}).get('n_times', 400))})
logits = model(x)          # (1, 3)
pred   = logits.argmax(1)  # 0=CN  1=AD  2=FTD
```

## Label Map

| Integer | Class |
|---|---|
{chr(10).join(f'| {v} | {k} |' for k, v in label_map.items())}

## Repository

Training code: [michaelburiek/eeg-hackathon](https://github.com/michaelburiek/eeg-hackathon) *(or update with your actual repo URL)*
"""


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Push trained EEGConformer to HuggingFace Hub.")
    parser.add_argument("--checkpoint", default="experiments/checkpoints/best.pt",
                        help="Path to the saved model state dict (.pt).")
    parser.add_argument("--results",    default="experiments/results/results.json",
                        help="Path to results.json (optional, added to model card).")
    parser.add_argument("--config",     default="configs/lead_train.yaml",
                        help="Training config used to reconstruct model architecture.")
    parser.add_argument("--repo-id",    default=HF_REPO_ID,
                        help=f"HuggingFace repo ID (default: {HF_REPO_ID}).")
    parser.add_argument("--private",    action="store_true",
                        help="Create the repo as private.")
    parser.add_argument("--env",        default=".env")
    args = parser.parse_args()

    load_dotenv(args.env)
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise SystemExit("HF_TOKEN not found in .env — add it and re-run.")

    from huggingface_hub import HfApi

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}\nRun scripts/04_train.py first.")

    cfg = load_config(args.config)

    results = None
    results_path = Path(args.results)
    if results_path.exists():
        results = json.loads(results_path.read_text())
        print(f"Loaded results from {results_path}")
    else:
        print(f"No results.json found at {results_path} — uploading without metrics.")

    # ── Rebuild model to get n_times ─────────────────────────────────────────
    # Load checkpoint to infer input shape
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Infer n_times from first conv weight shape if possible, else use config
    m_cfg  = cfg.get("model", {})
    pp_cfg = cfg.get("preprocessing", {"sfreq": 200})
    win_cfg = cfg.get("windowing", {"window_size_sec": 2.0})
    sfreq   = pp_cfg.get("sfreq", 200)
    n_times = int(win_cfg.get("window_size_sec", 2.0) * sfreq)

    model_config = {
        "architecture": "EEGConformer",
        "library": "braindecode",
        "model": {
            "n_channels":        m_cfg.get("n_channels", 19),
            "n_times":           n_times,
            "n_classes":         m_cfg.get("n_classes", 3),
            "sfreq":             float(sfreq),
            "n_filters_time":    m_cfg.get("n_filters_time", 40),
            "filter_time_length": m_cfg.get("filter_time_length", 25),
            "pool_time_length":  m_cfg.get("pool_time_length", 75),
            "pool_time_stride":  m_cfg.get("pool_time_stride", 15),
            "drop_prob":         m_cfg.get("drop_prob", 0.5),
            "att_depth":         m_cfg.get("att_depth", 6),
            "att_heads":         m_cfg.get("att_heads", 10),
            "att_drop_prob":     m_cfg.get("att_drop_prob", 0.5),
            "final_fc_length":   m_cfg.get("final_fc_length", "auto"),
        },
        "label_map": cfg.get("dataset", {}).get("label_map", {"CN": 0, "AD": 1, "FTD": 2}),
        "dataset": "LEAD ADFTD-RS L400",
        "training_config": cfg.get("training", {}),
    }

    model_card = build_model_card(cfg, results)

    # ── Upload ────────────────────────────────────────────────────────────────
    api = HfApi(token=hf_token)

    print(f"Creating / verifying repo: {args.repo_id}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        exist_ok=True,
        private=args.private,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # config.json
        config_out = tmp / "config.json"
        config_out.write_text(json.dumps(model_config, indent=2))

        # README.md (model card)
        card_out = tmp / "README.md"
        card_out.write_text(model_card, encoding="utf-8")

        # results.json
        if results:
            results_out = tmp / "results.json"
            results_out.write_text(json.dumps(results, indent=2))

        files_to_upload = [
            (checkpoint_path, "pytorch_model.pt"),
            (config_out,      "config.json"),
            (card_out,        "README.md"),
        ]
        if results:
            files_to_upload.append((results_out, "results.json"))

        for local_path, repo_filename in files_to_upload:
            print(f"Uploading {repo_filename} ...")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_filename,
                repo_id=args.repo_id,
                repo_type="model",
                token=hf_token,
            )

    print(f"\nDone. Model live at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
