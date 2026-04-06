#!/usr/bin/env python3
"""
scripts/02_import_lead.py
────────────────────────────────────────────────────────────────────────────────
Convert LEAD-format .dat files into this repo's .npz window store.

LEAD stores EEG data as memory-mapped binary files (X.dat / y.dat) with a
meta.json descriptor. This script reads those files and writes one .npz per
subject under --output-dir, plus an index.csv that all downstream scripts use.

Usage
-----
  python scripts/02_import_lead.py \
      --config configs/lead_train.yaml \
      --source-root data/lead/source \
      --datasets ADFTD-RS \
      --output-dir data/lead/window_store \
      --index-out data/lead/window_store/index.csv

LEAD label encoding (L400 / ADFTD datasets)
--------------------------------------------
  y.dat columns: [label, subject_id, sampling_rate]
  label values : 0 = CN, 1 = AD, 2 = FTD  (ADFTD subsets)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.corpus.lead import import_lead_processed_datasets, parse_label_map
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Import LEAD dataset into .npz window store.")
    parser.add_argument("--config", default="configs/lead_train.yaml")
    parser.add_argument(
        "--source-root",
        help="Root directory containing LEAD dataset folders (e.g. data/lead/source). "
             "Overrides paths.lead_source in config.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Subset of LEAD directories to import (e.g. ADFTD-RS). "
             "Defaults to the dataset.name value in config.",
    )
    parser.add_argument("--output-dir", help="Override paths.window_store from config.")
    parser.add_argument("--index-out",  help="Override paths.index_csv from config.")
    parser.add_argument(
        "--label-map", nargs="*", default=None,
        help="Optional label overrides: e.g. 0:CN 1:AD 2:FTD",
    )
    parser.add_argument("--x-dtype", default="float32")
    parser.add_argument("--y-dtype", default="int32")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    dataset_cfg = cfg.get("dataset", {})

    source_root = Path(args.source_root or paths.get("lead_source", "data/lead/source"))
    output_dir  = Path(args.output_dir  or paths.get("window_store", "data/lead/window_store"))
    index_out   = Path(args.index_out   or paths.get("index_csv",    "data/lead/window_store/index.csv"))
    datasets    = args.datasets or ([dataset_cfg["name"]] if dataset_cfg.get("name") else None)

    label_map = parse_label_map(args.label_map) if args.label_map else None

    manifest_df, index_df = import_lead_processed_datasets(
        source_root=source_root,
        output_dir=output_dir,
        datasets=datasets,
        label_map=label_map,
        x_dtype=args.x_dtype,
        y_dtype=args.y_dtype,
    )

    index_out.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(index_out, index=False)

    print(f"Imported {len(manifest_df)} subjects → {len(index_df)} window-store entries")
    print(f"Index written to: {index_out}")
    label_counts = index_df.groupby("label")["n_windows"].sum().to_dict()
    print(f"Windows per class: {label_counts}")
    print(f"\nNext step: python scripts/03_create_splits.py --config {args.config}")


if __name__ == "__main__":
    main()
