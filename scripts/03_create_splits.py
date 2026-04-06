#!/usr/bin/env python3
"""
scripts/03_create_splits.py
────────────────────────────────────────────────────────────────────────────────
Create deterministic subject-level train / val / test splits and save them
to a JSON file.

Splitting at the subject level (not the window level) ensures that no subject's
EEG windows appear in more than one partition — essential for fair evaluation.

Usage
-----
  python scripts/03_create_splits.py --config configs/lead_train.yaml

Output
------
  data/lead/splits.json   (path from config)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.corpus.splits import create_train_val_test_splits, save_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test splits.")
    parser.add_argument("--config", default="configs/lead_train.yaml")
    parser.add_argument("--index-csv", help="Override paths.index_csv from config.")
    parser.add_argument("--output",    help="Override paths.splits_json from config.")
    parser.add_argument("--val-frac",  type=float, help="Override splits.val_frac.")
    parser.add_argument("--test-frac", type=float, help="Override splits.test_frac.")
    parser.add_argument("--seed",      type=int,   help="Override splits.seed.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths      = cfg.get("paths", {})
    splits_cfg = cfg.get("splits", {})

    index_csv   = Path(args.index_csv or paths.get("index_csv",   "data/lead/window_store/index.csv"))
    output_path = Path(args.output    or paths.get("splits_json", "data/lead/splits.json"))
    val_frac    = args.val_frac  if args.val_frac  is not None else splits_cfg.get("val_frac",  0.15)
    test_frac   = args.test_frac if args.test_frac is not None else splits_cfg.get("test_frac", 0.15)
    seed        = args.seed      if args.seed      is not None else splits_cfg.get("seed",       42)

    splits = create_train_val_test_splits(
        index_csv=index_csv,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    save_splits(splits, output_path)

    n_train = len(splits["train"])
    n_val   = len(splits["val"])
    n_test  = len(splits["test"])
    print(f"Subjects — train: {n_train}  val: {n_val}  test: {n_test}")
    print(f"Splits written to: {output_path}")
    print(f"\nNext step: python scripts/04_train.py --config {args.config}")


if __name__ == "__main__":
    main()
