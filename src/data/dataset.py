"""
src/data/dataset.py
────────────────────────────────────────────────────────────────────────────────
PyTorch Dataset wrapping windowed EEG arrays loaded from the LEAD window store.

Public API
----------
EEGDataset              — torch Dataset wrapping windowed EEG arrays
load_window_store       — load all .npz files from the window store index into arrays
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    """
    Torch Dataset wrapping a numpy array of EEG windows.

    Parameters
    ----------
    windows     : np.ndarray, shape (N, C, T)  — EEG windows
    labels      : np.ndarray, shape (N,)       — integer class labels
    subject_ids : list[str] of length N        — subject ID for each window
    transform   : optional callable applied to each window tensor
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[List[str]] = None,
        transform=None,
    ) -> None:
        assert len(windows) == len(labels), "windows and labels must match"
        self.windows = torch.from_numpy(windows.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.subject_ids = subject_ids or ["unknown"] * len(windows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.windows[idx]  # (C, T)
        y = self.labels[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    @property
    def n_channels(self) -> int:
        return self.windows.shape[1]

    @property
    def n_times(self) -> int:
        return self.windows.shape[2]

    @property
    def n_classes(self) -> int:
        return int(self.labels.max().item()) + 1


# ─── Window store loader ──────────────────────────────────────────────────────

def load_window_store(
    index_csv: str | Path,
    subject_keys: Optional[List[str]] = None,
    label_map: Optional[Dict[str, int]] = None,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load EEG windows from the .npz window store.

    Parameters
    ----------
    index_csv    : path to the window store index CSV produced by scripts/02_import_lead.py
    subject_keys : optional list of subject_key strings to filter; loads all if None
    label_map    : {label_str → int}, e.g. {"CN": 0, "AD": 1, "FTD": 2}
                   If None, integer labels from the .npz files are used directly.

    Returns
    -------
    windows      : float32 array of shape (N_windows_total, C, T)
    labels       : int64 array of shape (N_windows_total,)
    subject_ids  : list of length N_windows_total — subject ID for each window
    """
    index_df = pd.read_csv(index_csv)

    if subject_keys is not None:
        index_df = index_df[index_df["subject_key"].isin(subject_keys)].reset_index(drop=True)

    all_windows: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_subject_ids: List[str] = []

    for _, row in index_df.iterrows():
        npz = np.load(row["window_store_path"])
        windows = npz["windows"].astype(np.float32)  # (N, C, T)

        if label_map is not None:
            label_str = str(npz["label_str"])
            label_int = label_map[label_str]
        else:
            label_int = int(npz["label_int"])

        n = len(windows)
        all_windows.append(windows)
        all_labels.append(np.full(n, label_int, dtype=np.int64))
        all_subject_ids.extend([str(npz["subject_key"])] * n)

        log.debug("Loaded %s: %d windows, label=%s", row["subject_key"], n, label_int)

    windows_arr = np.concatenate(all_windows, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    log.info(
        "Loaded %d windows from %d subjects (%s).",
        len(windows_arr), len(index_df), index_csv,
    )
    return windows_arr, labels_arr, all_subject_ids
