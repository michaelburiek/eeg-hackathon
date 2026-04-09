#!/usr/bin/env python3
"""
Generate CN norms, z-scores, and text summaries from biomarker CSVs.

Run this AFTER extracting biomarkers from all datasets.  Takes one or more
biomarker CSV files, computes CN (healthy control) distribution from the
pooled data, then produces:

  1. CN norms JSON           — mean/std per biomarker from CN group
  2. CSV with z-score columns — original biomarkers + z-score per biomarker
  3. Per-subject text summaries (optional, --summaries flag)

Usage
-----
    # Single dataset
    python generate_summaries.py --input data/ADFTD-RS_biomarkers.csv

    # Multiple datasets pooled (CN norms computed across all)
    python generate_summaries.py --input data/ADFTD-RS_biomarkers.csv data/APAVA_biomarkers.csv

    # Also write per-subject text summaries
    python generate_summaries.py --input data/ADFTD-RS_biomarkers.csv --summaries

    # Use pre-computed norms instead of deriving from CN group
    python generate_summaries.py --input data/ADFTD-RS_biomarkers.csv --norms data/cn_norms.json
"""

import argparse
import json
import pandas as pd
from pathlib import Path

from extract_biomarkers import (
    compute_cn_norms,
    append_zscores,
    generate_biomarker_summary,
    BIOMARKER_KEYS,
)


def load_and_merge(input_paths: list[Path]) -> pd.DataFrame:
    """Load one or more biomarker CSVs and merge into a single DataFrame."""
    frames = []
    for p in input_paths:
        df = pd.read_csv(p)
        df['source_file'] = p.name
        frames.append(df)
        print(f'  Loaded {len(df)} subjects from {p}')
    merged = pd.concat(frames, ignore_index=True)
    print(f'Total: {len(merged)} subjects')
    if 'group' in merged.columns:
        print(merged['group'].value_counts().to_string(header=False))
    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Compute CN norms, z-scores, and text summaries from biomarker CSVs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input', type=Path, nargs='+', required=True,
        help='One or more biomarker CSV files to process.',
    )
    parser.add_argument(
        '--norms', type=Path, default=None,
        help='Pre-computed CN norms JSON. If omitted, computed from CN subjects in the input.',
    )
    parser.add_argument(
        '--out-dir', type=Path, default=None,
        help='Output directory. Defaults to same directory as first input file.',
    )
    parser.add_argument(
        '--summaries', action='store_true',
        help='Write per-subject text summaries to a subdirectory.',
    )
    args = parser.parse_args()

    # Output directory
    out_dir = args.out_dir or args.input[0].parent
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print('Loading biomarker CSVs...')
    df = load_and_merge([p.resolve() for p in args.input])

    # Norms: load or compute
    if args.norms:
        with open(args.norms) as f:
            norms = json.load(f)
        print(f'\nLoaded pre-computed norms from {args.norms}')
    else:
        cn_count = (df['group'] == 'CN').sum()
        if cn_count < 2:
            print(f'\nOnly {cn_count} CN subject(s) — cannot compute norms.')
            print('Provide --norms with a pre-computed norms JSON, or include more CN subjects.')
            return
        norms = compute_cn_norms(df)
        norms_path = out_dir / 'cn_norms.json'
        with open(norms_path, 'w') as f:
            json.dump(norms, f, indent=2)
        print(f'\nComputed CN norms from {cn_count} CN subjects.')
        print(f'Saved: {norms_path}')

    # Z-scores
    df_z = append_zscores(df, norms)
    zscores_path = out_dir / 'biomarkers_with_zscores.csv'
    df_z.to_csv(zscores_path, index=False)
    print(f'Saved: {zscores_path}')

    # Text summaries
    if args.summaries:
        summaries_dir = out_dir / 'summaries'
        summaries_dir.mkdir(exist_ok=True)
        for _, row in df.iterrows():
            zscores, quant, text = generate_biomarker_summary(row, norms)
            subj = row['subject']
            summary_path = summaries_dir / f'{subj}_summary.txt'
            with open(summary_path, 'w') as f:
                f.write(text)
        print(f'Saved {len(df)} text summaries to {summaries_dir}/')

    print('\nDone.')


if __name__ == '__main__':
    main()