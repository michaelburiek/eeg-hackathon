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

from extract_biomarkers import BIOMARKER_KEYS


# ── CN norms & z-scores ───────────────────────────────────────────────────────
def compute_cn_norms(biomarkers_df: pd.DataFrame) -> dict:
    """Compute mean/std for each biomarker using the CN (healthy control) group."""
    cn = biomarkers_df[biomarkers_df['group'] == 'CN']
    norms = {}
    for k in BIOMARKER_KEYS:
        norms[f'{k}_mean'] = float(cn[k].mean())
        norms[f'{k}_std']  = float(cn[k].std())
    return norms


def append_zscores(df: pd.DataFrame, norms: dict) -> pd.DataFrame:
    """Add z-score columns to a biomarkers DataFrame using precomputed CN norms."""
    df = df.copy()
    for key in BIOMARKER_KEYS:
        df[f'{key}_z'] = (df[key] - norms[f'{key}_mean']) / (norms[f'{key}_std'] + 1e-9)
    return df


# ── Text summary generation ──────────────────────────────────────────────────
def generate_biomarker_summary(row, norms):
    """
    Compute z-scores and generate a human-readable text summary for one subject.

    Parameters
    ----------
    row   : dict or pd.Series with all biomarker fields
    norms : dict with {biomarker}_mean / {biomarker}_std entries (from CN group)

    Returns
    -------
    zscores : dict
    quant   : dict  (actual values with units)
    text    : str
    """
    def _z(val, key):
        return (val - norms[f'{key}_mean']) / (norms[f'{key}_std'] + 1e-9)

    def _arrow(z):
        if z >  1.0: return '\u2191 elevated'
        if z < -1.0: return '\u2193 reduced'
        return '~ normal'

    pdr_z   = _z(row['pdr_hz'],                'pdr_hz')
    iaf_z   = _z(row['iaf_hz'],                'iaf_hz')
    delta_z = _z(row['delta_power'],            'delta_power')
    theta_z = _z(row['theta_power'],            'theta_power')
    alpha_z = _z(row['alpha_power'],            'alpha_power')
    beta_z  = _z(row['beta_power'],             'beta_power')
    sr_z    = _z(row['slowing_ratio'],          'slowing_ratio')
    ta_z    = _z(row['theta_alpha_ratio'],      'theta_alpha_ratio')
    coh_z   = _z(row['posterior_coherence'],    'posterior_coherence')
    gcoh_z  = _z(row['global_coherence'],       'global_coherence')
    pli_z   = _z(row['pli_alpha'],              'pli_alpha')
    fpa_z   = _z(row['frontal_posterior_asym'],  'frontal_posterior_asym')
    pe_z    = _z(row['perm_entropy'],           'perm_entropy')
    lzc_z   = _z(row['lz_complexity'],          'lz_complexity')

    zscores = {
        'pdr_z':                   round(float(pdr_z),   2),
        'iaf_z':                   round(float(iaf_z),   2),
        'delta_power_z':           round(float(delta_z), 2),
        'theta_power_z':           round(float(theta_z), 2),
        'alpha_power_z':           round(float(alpha_z), 2),
        'beta_power_z':            round(float(beta_z),  2),
        'slowing_ratio_z':         round(float(sr_z),    2),
        'theta_alpha_ratio_z':     round(float(ta_z),    2),
        'posterior_coherence_z':    round(float(coh_z),   2),
        'global_coherence_z':      round(float(gcoh_z),  2),
        'pli_alpha_z':             round(float(pli_z),   2),
        'frontal_posterior_asym_z': round(float(fpa_z),   2),
        'perm_entropy_z':          round(float(pe_z),    2),
        'lz_complexity_z':         round(float(lzc_z),   2),
        'PDR_clinically_slowed':   bool(row['pdr_hz'] < 8.0),
    }

    quant = {
        'pdr_hz':                round(float(row['pdr_hz']),                2),
        'iaf_hz':                round(float(row['iaf_hz']),                2),
        'delta_power_pct':       round(float(row['delta_power']) * 100,     1),
        'theta_power_pct':       round(float(row['theta_power']) * 100,     1),
        'alpha_power_pct':       round(float(row['alpha_power']) * 100,     1),
        'beta_power_pct':        round(float(row['beta_power'])  * 100,     1),
        'slowing_ratio':         round(float(row['slowing_ratio']),         3),
        'theta_alpha_ratio':     round(float(row['theta_alpha_ratio']),     3),
        'posterior_coherence':    round(float(row['posterior_coherence']),   3),
        'global_coherence':      round(float(row['global_coherence']),      3),
        'pli_alpha':             round(float(row['pli_alpha']),             3),
        'frontal_posterior_asym': round(float(row['frontal_posterior_asym']), 3),
        'perm_entropy':          round(float(row['perm_entropy']),          4),
        'lz_complexity':         round(float(row['lz_complexity']),         4),
    }

    text = (
        f"Subject: {row['subject']}  |  Age: {row.get('age', 'N/A')}  |  MMSE: {row.get('mmse', 'N/A')}\n"
        f"Group: {row['group']}\n"
        f"\n--- QUANTITATIVE BIOMARKERS ---\n"
        f"Posterior Dominant Rhythm:    {quant['pdr_hz']:.2f} Hz"
        f"  (CN norm: {norms['pdr_hz_mean']:.1f} \u00b1 {norms['pdr_hz_std']:.1f} Hz)"
        f"  [clinically slowed: {'YES' if row['pdr_hz'] < 8.0 else 'no'}]\n"
        f"Individual Alpha Frequency:  {quant['iaf_hz']:.2f} Hz"
        f"  (CN norm: {norms['iaf_hz_mean']:.1f} \u00b1 {norms['iaf_hz_std']:.1f} Hz)\n"
        f"Delta power:                 {quant['delta_power_pct']:.1f}%\n"
        f"Theta power:                 {quant['theta_power_pct']:.1f}%\n"
        f"Alpha power:                 {quant['alpha_power_pct']:.1f}%\n"
        f"Beta power:                  {quant['beta_power_pct']:.1f}%\n"
        f"Slowing ratio (d+t)/(a+b):   {quant['slowing_ratio']:.3f}"
        f"  (CN norm: {norms['slowing_ratio_mean']:.3f})\n"
        f"Theta/Alpha ratio:           {quant['theta_alpha_ratio']:.3f}"
        f"  (CN norm: {norms['theta_alpha_ratio_mean']:.3f})\n"
        f"Posterior alpha coherence:    {quant['posterior_coherence']:.3f}"
        f"  (CN norm: {norms['posterior_coherence_mean']:.3f})\n"
        f"Global alpha coherence:      {quant['global_coherence']:.3f}"
        f"  (CN norm: {norms['global_coherence_mean']:.3f})\n"
        f"Phase Lag Index (alpha):      {quant['pli_alpha']:.3f}"
        f"  (CN norm: {norms['pli_alpha_mean']:.3f})\n"
        f"Frontal/Posterior asymmetry:  {quant['frontal_posterior_asym']:.3f}"
        f"  (CN norm: {norms['frontal_posterior_asym_mean']:.3f})\n"
        f"Permutation entropy:         {quant['perm_entropy']:.4f}"
        f"  (CN norm: {norms['perm_entropy_mean']:.4f})\n"
        f"Lempel-Ziv complexity:       {quant['lz_complexity']:.4f}"
        f"  (CN norm: {norms['lz_complexity_mean']:.4f})\n"
        f"\n--- Z-SCORES vs CN NORMS ---\n"
        f"{'PDR':38s}  z = {pdr_z:+.2f}  {_arrow(pdr_z)}\n"
        f"{'Individual Alpha Frequency':38s}  z = {iaf_z:+.2f}  {_arrow(iaf_z)}\n"
        f"{'Delta power':38s}  z = {delta_z:+.2f}  {_arrow(delta_z)}\n"
        f"{'Theta power':38s}  z = {theta_z:+.2f}  {_arrow(theta_z)}\n"
        f"{'Alpha power':38s}  z = {alpha_z:+.2f}  {_arrow(alpha_z)}\n"
        f"{'Beta power':38s}  z = {beta_z:+.2f}  {_arrow(beta_z)}\n"
        f"{'Slowing ratio':38s}  z = {sr_z:+.2f}  {_arrow(sr_z)}\n"
        f"{'Theta/Alpha ratio':38s}  z = {ta_z:+.2f}  {_arrow(ta_z)}\n"
        f"{'Posterior coherence':38s}  z = {coh_z:+.2f}  {_arrow(coh_z)}\n"
        f"{'Global coherence':38s}  z = {gcoh_z:+.2f}  {_arrow(gcoh_z)}\n"
        f"{'Phase Lag Index (alpha)':38s}  z = {pli_z:+.2f}  {_arrow(pli_z)}\n"
        f"{'Frontal/Posterior asymmetry':38s}  z = {fpa_z:+.2f}  {_arrow(fpa_z)}\n"
        f"{'Permutation entropy':38s}  z = {pe_z:+.2f}  {_arrow(pe_z)}\n"
        f"{'Lempel-Ziv complexity':38s}  z = {lzc_z:+.2f}  {_arrow(lzc_z)}\n"
    )

    return zscores, quant, text


# ── Data loading ──────────────────────────────────────────────────────────────
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


# ── CLI ───────────────────────────────────────────────────────────────────────
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