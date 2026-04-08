#!/usr/bin/env python3
"""
Gather per-subject biomarker CSVs into one file + compute CN norms & z-scores.

Usage
-----
    python slurm/gather.py --input-dir data/ds004504_biomarkers --out data/biomarkers_all_subjects.csv
    python slurm/gather.py --input-dir data/ADFTD-RS_biomarkers --out data/ADFTD-RS_biomarkers.csv
"""

import argparse
import json
import pandas as pd
from pathlib import Path

from extract_biomarkers import compute_cn_norms

ZSCORE_KEYS = [
    'pdr_hz', 'iaf_hz',
    'delta_power', 'theta_power', 'alpha_power', 'beta_power',
    'slowing_ratio', 'theta_alpha_ratio',
    'posterior_coherence', 'global_coherence', 'pli_alpha',
    'frontal_posterior_asym', 'perm_entropy', 'lz_complexity',
]


def main():
    parser = argparse.ArgumentParser(description='Gather per-subject CSVs and compute CN norms.')
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
        print(df['group'].value_counts().to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'Saved: {args.out}')

    # CN norms + z-scores
    cn_mask = df['group'] == 'CN'
    if cn_mask.sum() >= 2:
        norms = compute_cn_norms(df)
        norms_path = args.out.with_name(args.out.stem + '_cn_norms.json')
        with open(norms_path, 'w') as f:
            json.dump(norms, f, indent=2)
        print(f'Saved CN norms: {norms_path}')

        for key in ZSCORE_KEYS:
            df[f'{key}_z'] = (df[key] - norms[f'{key}_mean']) / (norms[f'{key}_std'] + 1e-9)

        zscores_path = args.out.with_name(args.out.stem + '_with_zscores.csv')
        df.to_csv(zscores_path, index=False)
        print(f'Saved with z-scores: {zscores_path}')
    else:
        print(f'Only {cn_mask.sum()} CN subjects — skipping norms. Need >=2.')


if __name__ == '__main__':
    main()
