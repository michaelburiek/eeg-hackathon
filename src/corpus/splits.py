from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_val_test_splits(
    index_csv: str | Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Create deterministic subject-level train / val / test splits.

    Splitting is done at the *subject* level so that no subject's windows
    appear in more than one partition (prevents data leakage).

    Parameters
    ----------
    index_csv  : path to the window store index CSV
    val_frac   : fraction of subjects to hold out for validation
    test_frac  : fraction of subjects to hold out for test
    seed       : random seed for reproducibility

    Returns
    -------
    dict with keys "train", "val", "test" each mapping to a list of subject_keys
    """
    index_df = pd.read_csv(index_csv)
    subject_df = (
        index_df[["subject_key", "label"]]
        .drop_duplicates()
        .sort_values("subject_key")
        .reset_index(drop=True)
    )

    subject_keys = subject_df["subject_key"].tolist()
    labels = subject_df["label"].tolist()

    # First split off test, then split remainder into train/val
    train_val_keys, test_keys, train_val_labels, _ = train_test_split(
        subject_keys, labels,
        test_size=test_frac,
        stratify=labels,
        random_state=seed,
    )

    relative_val_frac = val_frac / (1.0 - test_frac)
    train_keys, val_keys = train_test_split(
        train_val_keys,
        test_size=relative_val_frac,
        stratify=train_val_labels,
        random_state=seed,
    )

    return {
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "train": sorted(train_keys),
        "val": sorted(val_keys),
        "test": sorted(test_keys),
    }


def save_splits(splits: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return path


def load_splits(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
