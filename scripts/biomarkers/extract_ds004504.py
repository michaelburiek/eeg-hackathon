#!/usr/bin/env python3
"""
EEG Biomarker Extraction — ds004504 (BIDS / EEGLAB .set)

Loads subjects from the ds004504 dataset (BIDS format, git-annex .set files),
preprocesses, and extracts biomarkers using src/extract_biomarkers.py.

Usage
-----
    python scripts/biomarkers/extract_ds004504.py                                    # all subjects
    python scripts/biomarkers/extract_ds004504.py --dataset /path/to/ds004504
    python scripts/biomarkers/extract_ds004504.py --subjects sub-001 sub-002         # specific subjects
    python scripts/biomarkers/extract_ds004504.py --out my_biomarkers.csv --duration 180
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import mne
mne.set_log_level('WARNING')
warnings.filterwarnings('ignore')

from src.extract_biomarkers import extract_subject_biomarkers


# ── Data loading & preprocessing ──────────────────────────────────────────────
def load_single_subject(subject_id: str, dataset_root: Path) -> mne.io.Raw:
    """
    Load one subject's raw EEG from EEGLAB .set format.

    Parameters
    ----------
    subject_id   : e.g. 'sub-001'
    dataset_root : Path to ds004504/

    Returns
    -------
    mne.io.Raw (preloaded)
    """
    set_file = dataset_root / subject_id / 'eeg' / f'{subject_id}_task-eyesclosed_eeg.set'
    if not set_file.exists():
        raise FileNotFoundError(
            f'EEG file not found: {set_file}\n'
            'Download the annexed payload first (git annex get).'
        )
    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
    raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore', verbose=False)
    except Exception:
        pass
    return raw


def preprocess_raw(raw: mne.io.Raw, l_freq=0.5, h_freq=45.0,
                   analysis_duration=120.0) -> mne.io.Raw:
    """
    Preprocess: bandpass filter -> average reference -> crop.

    Crops to [60 s, 60+analysis_duration s] to skip settling artefacts.
    Works on a copy; original is unchanged.
    """
    raw = raw.copy()
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    t_start = max(0.0, min(60.0, raw.times[-1] - 30.0))
    t_end   = min(t_start + analysis_duration, raw.times[-1])
    raw.crop(tmin=t_start, tmax=t_end)
    return raw


# ── Main pipeline ─────────────────────────────────────────────────────────────
def extract_all(dataset_root: Path, output_path: Path,
                analysis_duration: float = 120.0,
                subject_filter: list = None) -> pd.DataFrame:
    """
    Full pipeline: load -> preprocess -> extract -> save CSV.

    Parameters
    ----------
    dataset_root      : Path to ds004504/
    output_path       : Where to write biomarkers CSV
    analysis_duration : Seconds of EEG to analyse per subject (after 60 s skip)
    subject_filter    : Optional list of subject IDs to restrict to

    Returns
    -------
    pd.DataFrame with one row per subject
    """
    # Load participant metadata
    participants_file = dataset_root / 'participants.tsv'
    if not participants_file.exists():
        raise FileNotFoundError(f'participants.tsv not found at {dataset_root}')
    participants = pd.read_csv(participants_file, sep='\t')
    participants['Group'] = participants['Group'].map({'A': 'AD', 'C': 'CN', 'F': 'FTD'})
    subject_to_group = dict(zip(participants['participant_id'], participants['Group']))
    subject_to_mmse  = dict(zip(participants['participant_id'], participants['MMSE']))
    subject_to_age   = dict(zip(participants['participant_id'], participants['Age']))

    # Identify available subjects
    def _eeg_path(sid):
        return dataset_root / sid / 'eeg' / f'{sid}_task-eyesclosed_eeg.set'

    all_subjects = sorted(participants['participant_id'].tolist())
    if subject_filter:
        all_subjects = [s for s in all_subjects if s in subject_filter]
    available = [s for s in all_subjects if _eeg_path(s).exists()]

    if not available:
        raise RuntimeError(
            f'No EEG .set files found under {dataset_root}.\n'
            'Download the git-annex payloads first.'
        )
    print(f'Found {len(available)}/{len(all_subjects)} subjects with EEG files.')

    # Load, preprocess, extract
    records = []
    for subj in tqdm(available, desc='Processing'):
        try:
            raw = load_single_subject(subj, dataset_root)
            raw = preprocess_raw(raw, analysis_duration=analysis_duration)
            rec = extract_subject_biomarkers(raw)
            rec['subject'] = subj
            rec['group']   = subject_to_group.get(subj, 'Unknown')
            rec['mmse']    = subject_to_mmse.get(subj, np.nan)
            rec['age']     = subject_to_age.get(subj, np.nan)
            records.append(rec)
        except Exception as e:
            print(f'  Error on {subj}: {e}')

    df = pd.DataFrame(records)
    print(f'\nExtracted biomarkers for {len(df)} subjects.')

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Saved: {output_path}')

    return df


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Extract EEG biomarkers from ds004504 BIDS dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dataset', type=Path, default=None,
        help='Path to ds004504/ directory. Auto-detected if omitted.',
    )
    parser.add_argument(
        '--out', type=Path, default=None,
        help='Output CSV path. Defaults to data/biomarkers_all_subjects.csv.',
    )
    parser.add_argument(
        '--duration', type=float, default=120.0,
        help='Seconds of EEG to analyse per subject (after 60 s artefact skip).',
    )
    parser.add_argument(
        '--subjects', nargs='+', default=None,
        help='Restrict to specific subject IDs, e.g. --subjects sub-001 sub-002.',
    )
    args = parser.parse_args()

    if args.dataset:
        dataset_root = args.dataset.resolve()
    else:
        candidates = [Path('data/ds004504'), Path('../data/ds004504')]
        dataset_root = next(
            (p.resolve() for p in candidates if p.exists()),
            candidates[0].resolve()
        )

    if args.out:
        output_path = args.out.resolve()
    else:
        output_path = (dataset_root.parent / 'biomarkers_all_subjects.csv').resolve()

    print(f'Dataset: {dataset_root}')
    print(f'Output:  {output_path}')
    print(f'Duration per subject: {args.duration} s')

    extract_all(
        dataset_root=dataset_root,
        output_path=output_path,
        analysis_duration=args.duration,
        subject_filter=args.subjects,
    )


if __name__ == '__main__':
    main()
