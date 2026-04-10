#!/usr/bin/env python3
"""
Gather per-subject biomarker CSVs from SLURM array jobs into a single file.

Usage
-----
    python slurm/gather.py --input-dir data/ADFTD-RS_biomarkers --out data/ADFTD-RS_biomarkers.csv

Then run scripts/biomarkers/generate_summaries.py on the merged CSV to compute norms and z-scores.
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Gather per-subject CSVs into one file.')
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Directory containing per-subject CSV files.')
    parser.add_argument('--out', type=Path, required=True,
                        help='Output merged CSV path.')
    args = parser.parse_args()

    csvs = sorted(args.input_dir.glob('*.csv'))
    if not csvs:
        raise FileNotFoundError(f'No CSV files found in {args.input_dir}')

    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    df = df.drop_duplicates(subset='subject', keep='last')
    df = df.sort_values('subject').reset_index(drop=True)

    print(f'Gathered {len(df)} subjects from {len(csvs)} files.')
    if 'group' in df.columns:
        print(df['group'].value_counts().to_string(header=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()