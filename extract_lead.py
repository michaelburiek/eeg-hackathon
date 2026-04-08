#!/usr/bin/env python3
"""
EEG Biomarker Extraction — LEAD dataset adapter

Loads LEAD pre-segmented memmap data (X.dat / y.dat / meta.json),
reconstructs per-subject continuous signals, wraps them in mne.io.RawArray,
and runs the same biomarker functions from extract_biomarkers.py.

Usage
-----
    # All subjects in a LEAD dataset
    python extract_lead.py --dataset data/lead/L200/APAVA

    # Single subject (for SLURM array)
    python extract_lead.py --dataset data/lead/L400/ADFTD-RS --subject-idx 5

    # Pick a specific sampling rate (default: highest available)
    python extract_lead.py --dataset data/lead/L400/ADFTD-RS --sfreq 100

Caveats
-------
- LEAD segments are independently z-score normalised.  Concatenating them
  introduces discontinuities at boundaries.  Spectral power *ratios*
  (relative band power, slowing ratio, etc.) are still meaningful, but
  absolute PSD values are not comparable to raw-microvolt recordings.
- Phase-based connectivity (PLI, coherence) may have edge artifacts at
  segment junctions.  Interpret with caution for short recordings.
- Datasets with very few segments per subject (ADSZ: ~16 s, ADFSU: ~30 s)
  yield fewer Welch windows and noisier estimates.
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import mne
mne.set_log_level('WARNING')
warnings.filterwarnings('ignore')

from extract_biomarkers import (
    extract_subject_biomarkers,
    compute_cn_norms,
)

# Standard 19-channel 10-20 names (LEAD channel order)
CH_NAMES_19 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3',  'C3',  'Cz', 'C4', 'T4',
    'T5',  'P3',  'Pz', 'P4', 'T6',
    'O1',  'O2',
]

# LEAD label mappings
LABEL_MAPS = {
    2: {0: 'CN', 1: 'AD'},
    3: {0: 'AD', 1: 'FTD', 2: 'CN'},
}


def load_lead_dataset(dataset_path: Path, sfreq: float = None):
    """
    Load a LEAD dataset's metadata, X.dat, and y.dat.

    Parameters
    ----------
    dataset_path : Path to directory containing meta.json, X.dat, y.dat
    sfreq        : Desired sampling rate (Hz). If None, uses the highest available.

    Returns
    -------
    X     : np.memmap  shape (N_filtered, T, C)
    y     : np.ndarray shape (N_filtered, 3) — [label, subject_id, sfreq]
    meta  : dict
    sfreq : float — the sampling rate of returned segments
    label_map : dict {int: str}
    """
    meta_path = dataset_path / 'meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'meta.json not found at {dataset_path}')

    with open(meta_path) as f:
        meta = json.load(f)

    N, T, C = meta['N'], meta['T'], meta['C']

    y = np.fromfile(str(dataset_path / 'y.dat'), dtype=np.float32).reshape(N, 3)
    X = np.memmap(str(dataset_path / 'X.dat'), dtype=np.float32, mode='r').reshape(N, T, C)

    n_classes = len(np.unique(y[:, 0]))
    label_map = LABEL_MAPS.get(n_classes, {int(k): str(int(k)) for k in np.unique(y[:, 0])})

    # Filter to desired sampling rate
    available_srs = sorted(np.unique(y[:, 2]).tolist())
    if sfreq is None:
        sfreq = max(available_srs)
    elif sfreq not in available_srs:
        raise ValueError(f'Requested sfreq={sfreq} not in {available_srs}')

    mask = y[:, 2] == sfreq
    X_filt = X[mask]
    y_filt = y[mask]

    print(f'Dataset: {dataset_path.name}')
    print(f'  Shape: N={N}, T={T}, C={C}')
    print(f'  Sampling rates: {available_srs} Hz — using {sfreq} Hz')
    print(f'  Segments at {sfreq} Hz: {mask.sum()}')
    print(f'  Subjects: {len(np.unique(y_filt[:, 1]))}')
    print(f'  Classes: {label_map}')

    return X_filt, y_filt, meta, sfreq, label_map


def reconstruct_raw(segments: np.ndarray, sfreq: float, n_channels: int) -> mne.io.Raw:
    """
    Concatenate segments into a continuous signal and wrap as mne.io.RawArray.

    Parameters
    ----------
    segments   : (n_segments, T, C) array
    sfreq      : Sampling rate in Hz
    n_channels : Number of channels

    Returns
    -------
    mne.io.Raw
    """
    # Concatenate: (n_segs, T, C) → (n_segs*T, C) → (C, n_segs*T)
    continuous = segments.reshape(-1, n_channels).T.astype(np.float64)

    ch_names = CH_NAMES_19[:n_channels]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(continuous, info, verbose=False)

    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore', verbose=False)
    except Exception:
        pass

    return raw


def extract_subject(segments: np.ndarray, sfreq: float, n_channels: int,
                    subject_id, label: int, label_map: dict) -> dict:
    """Extract all biomarkers for one subject's concatenated segments."""
    raw = reconstruct_raw(segments, sfreq, n_channels)
    rec = extract_subject_biomarkers(raw)
    rec['subject']    = int(subject_id)
    rec['group']      = label_map.get(int(label), str(int(label)))
    rec['n_segments'] = len(segments)
    rec['duration_s'] = round(raw.times[-1], 1)
    rec['sfreq']      = sfreq
    return rec


def extract_all(dataset_path: Path, output_path: Path,
                sfreq: float = None, subject_idx: int = None,
                min_duration: float = 10.0):
    """
    Run biomarker extraction on a LEAD dataset.

    Parameters
    ----------
    dataset_path  : Path to LEAD dataset directory
    output_path   : Where to write the CSV
    sfreq         : Sampling rate to use (None = highest)
    subject_idx   : If set, only process this subject (0-indexed, for SLURM)
    min_duration  : Skip subjects with less than this many seconds of data
    """
    X, y, meta, sfreq, label_map = load_lead_dataset(dataset_path, sfreq)
    T, C = meta['T'], meta['C']

    subject_ids = np.unique(y[:, 1])
    if subject_idx is not None:
        if subject_idx >= len(subject_ids):
            raise ValueError(f'subject_idx={subject_idx} but only {len(subject_ids)} subjects')
        subject_ids = [subject_ids[subject_idx]]
        print(f'Processing single subject index {subject_idx} (id={int(subject_ids[0])})')

    records = []
    skipped = 0
    for sid in tqdm(subject_ids, desc='Extracting', disable=len(subject_ids) == 1):
        mask = y[:, 1] == sid
        segments = X[mask]   # (n_segs, T, C)
        label = int(y[mask][0, 0])

        duration = len(segments) * T / sfreq
        if duration < min_duration:
            skipped += 1
            continue

        try:
            rec = extract_subject(segments, sfreq, C, sid, label, label_map)
            records.append(rec)
        except Exception as e:
            print(f'  Error on subject {int(sid)}: {e}')

    if skipped:
        print(f'Skipped {skipped} subjects with <{min_duration}s of data.')

    df = pd.DataFrame(records)
    print(f'\nExtracted biomarkers for {len(df)} subjects.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Saved: {output_path}')

    # Compute CN norms if enough CN subjects
    cn_mask = df['group'] == 'CN'
    if cn_mask.sum() >= 2:
        norms = compute_cn_norms(df)
        norms_path = output_path.with_name(output_path.stem + '_cn_norms.json')
        with open(norms_path, 'w') as f:
            json.dump(norms, f, indent=2)
        print(f'Saved CN norms: {norms_path}')

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Extract EEG biomarkers from a LEAD dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dataset', type=Path, required=True,
        help='Path to LEAD dataset dir (containing meta.json, X.dat, y.dat).',
    )
    parser.add_argument(
        '--out', type=Path, default=None,
        help='Output CSV. Default: data/<dataset_name>_biomarkers.csv.',
    )
    parser.add_argument(
        '--sfreq', type=float, default=None,
        help='Sampling rate to use. Default: highest available.',
    )
    parser.add_argument(
        '--subject-idx', type=int, default=None,
        help='Process only this subject (0-indexed). For SLURM array jobs.',
    )
    parser.add_argument(
        '--min-duration', type=float, default=10.0,
        help='Skip subjects with fewer seconds of concatenated data.',
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    if args.out:
        output_path = args.out.resolve()
    elif args.subject_idx is not None:
        output_path = (Path('data') / f'{dataset_path.name}_biomarkers'
                       / f'subject_{args.subject_idx:04d}.csv').resolve()
    else:
        output_path = (Path('data') / f'{dataset_path.name}_biomarkers.csv').resolve()

    extract_all(
        dataset_path=dataset_path,
        output_path=output_path,
        sfreq=args.sfreq,
        subject_idx=args.subject_idx,
        min_duration=args.min_duration,
    )


if __name__ == '__main__':
    main()
