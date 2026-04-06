from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MANIFEST_COLUMNS = [
    "dataset_id",
    "subject_id",
    "subject_key",
    "eeg_path",
    "file_format",
    "label",
    "age",
    "sex",
    "mmse",
]


def normalize_subject_key(dataset_id: str, subject_id: str) -> str:
    return f"{dataset_id}:{subject_id}"


DEFAULT_LABEL_MAP = {
    0: "CN",
    1: "AD",
    2: "FTD",
}


def parse_label_map(entries: Iterable[str] | None) -> dict[int, str]:
    if not entries:
        return DEFAULT_LABEL_MAP.copy()
    mapping: dict[int, str] = {}
    for entry in entries:
        key, value = entry.split(":", 1)
        mapping[int(key)] = value
    return mapping


def _dataset_dirs(source_root: Path, datasets: list[str] | None) -> list[Path]:
    if datasets:
        dirs = [source_root / dataset for dataset in datasets]
    else:
        dirs = [path for path in sorted(source_root.iterdir()) if path.is_dir()]
    existing = [path for path in dirs if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No LEAD dataset directories found under {source_root}")
    return existing


def _find_legacy_label_file(label_dir: Path, subject_token: str) -> Path:
    candidates = [
        label_dir / f"label_{subject_token}.npy",
        label_dir / f"labels_{subject_token}.npy",
        label_dir / f"y_{subject_token}.npy",
        label_dir / f"{subject_token}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find label file for subject token {subject_token} in {label_dir}")


def _normalize_windows_shape(windows: np.ndarray, n_channels: int | None = None) -> np.ndarray:
    windows = np.asarray(windows)
    if windows.ndim == 2:
        windows = windows[None, ...]

    if windows.ndim != 3:
        raise ValueError(f"Expected 2D or 3D EEG windows, got shape {windows.shape}")

    if n_channels is None:
        plausible = [axis for axis, size in enumerate(windows.shape) if 1 < size <= 64]
        n_channels = windows.shape[plausible[-1]] if plausible else windows.shape[-1]

    if windows.shape[-1] == n_channels:
        windows = np.transpose(windows, (0, 2, 1))
    elif windows.shape[1] == n_channels:
        pass
    else:
        channel_axis = min(
            range(1, windows.ndim),
            key=lambda axis: abs(windows.shape[axis] - n_channels),
        )
        if channel_axis == 2:
            windows = np.transpose(windows, (0, 2, 1))
    return windows.astype(np.float32)


def _memmap_dtype(dtype_name: str):
    return np.dtype(dtype_name)


def _meta_value(meta: dict, *keys, default=None):
    for key in keys:
        if key in meta:
            return meta[key]
    return default


def _subject_rows_from_memmap(
    dataset_dir: Path,
    dataset_id: str,
    label_map: dict[int, str],
    x_dtype: str,
    y_dtype: str,
) -> tuple[list[dict], list[dict]]:
    meta = json.loads((dataset_dir / "meta.json").read_text(encoding="utf-8"))
    x_path = dataset_dir / "X.dat"
    y_path = dataset_dir / "y.dat"

    n_samples = int(_meta_value(meta, "n_samples", "num_samples", "N_sample", "N"))
    n_times = int(_meta_value(meta, "n_timestamps", "timestamps", "seq_len", "N_timestamp", "T"))
    channel_names = list(_meta_value(meta, "channel_names", "channels", default=[]))
    if not channel_names:
        n_channels = int(_meta_value(meta, "n_channels", "num_channels", "N_channel", "C"))
        channel_names = [f"EEG{i+1}" for i in range(n_channels)]
    n_channels = len(channel_names)

    x = np.memmap(
        x_path,
        dtype=_memmap_dtype(x_dtype),
        mode="r",
        shape=(n_samples, n_times, n_channels),
    )
    y = np.memmap(
        y_path,
        dtype=_memmap_dtype(y_dtype),
        mode="r",
        shape=(n_samples, 3),
    )

    manifest_rows = []
    subject_records = []
    subject_ids = np.unique(y[:, 1].astype(np.int64))
    sfreq_values = _meta_value(meta, "sampling_rates", "sampling_rate_list", "SAMPLE_RATE_LIST", default=[200])

    for subject_id_num in subject_ids.tolist():
        sample_mask = y[:, 1].astype(np.int64) == int(subject_id_num)
        subject_labels = y[sample_mask, 0].astype(np.int64)
        label_int = int(subject_labels[0])
        if np.any(subject_labels != label_int):
            raise ValueError(
                f"Inconsistent labels for subject {subject_id_num} in {dataset_dir.name}"
            )

        subject_id = f"sub-{int(subject_id_num):04d}"
        subject_key = normalize_subject_key(dataset_id, subject_id)
        label = label_map.get(label_int, f"class_{label_int}")
        subject_windows = np.transpose(np.asarray(x[sample_mask]), (0, 2, 1)).astype(np.float32)

        manifest_rows.append(
            {
                "dataset_id": dataset_id,
                "subject_id": subject_id,
                "subject_key": subject_key,
                "eeg_path": str(dataset_dir),
                "file_format": "lead_processed_memmap",
                "label": label,
                "age": None,
                "sex": None,
                "mmse": None,
            }
        )
        subject_records.append(
            {
                "dataset_id": dataset_id,
                "subject_id": subject_id,
                "subject_key": subject_key,
                "label": label,
                "label_int": label_int,
                "windows": subject_windows,
                "sfreq": float(sfreq_values[0]) if sfreq_values else 200.0,
                "channel_names": channel_names,
            }
        )

    return manifest_rows, subject_records


def _subject_rows_from_legacy(
    dataset_dir: Path,
    dataset_id: str,
    label_map: dict[int, str],
) -> tuple[list[dict], list[dict]]:
    feature_dir = dataset_dir / "Feature"
    label_dir = dataset_dir / "Label"
    feature_files = sorted(feature_dir.glob("feature_*.npy"))
    if not feature_files:
        raise FileNotFoundError(f"No feature_*.npy files found in {feature_dir}")

    manifest_rows = []
    subject_records = []
    for feature_path in feature_files:
        subject_token = feature_path.stem.removeprefix("feature_")
        label_path = _find_legacy_label_file(label_dir, subject_token)

        windows = np.load(feature_path, allow_pickle=False)
        windows = _normalize_windows_shape(windows, n_channels=19)
        label_arr = np.load(label_path, allow_pickle=False)
        label_int = int(np.asarray(label_arr).reshape(-1)[0])

        subject_id = f"sub-{subject_token}"
        subject_key = normalize_subject_key(dataset_id, subject_id)
        label = label_map.get(label_int, f"class_{label_int}")
        channel_names = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T7", "C3", "Cz", "C4", "T8",
            "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
        ][: windows.shape[1]]

        manifest_rows.append(
            {
                "dataset_id": dataset_id,
                "subject_id": subject_id,
                "subject_key": subject_key,
                "eeg_path": str(dataset_dir),
                "file_format": "lead_processed_legacy",
                "label": label,
                "age": None,
                "sex": None,
                "mmse": None,
            }
        )
        subject_records.append(
            {
                "dataset_id": dataset_id,
                "subject_id": subject_id,
                "subject_key": subject_key,
                "label": label,
                "label_int": label_int,
                "windows": windows.astype(np.float32),
                "sfreq": 200.0,
                "channel_names": channel_names,
            }
        )

    return manifest_rows, subject_records


def import_lead_processed_datasets(
    source_root: str | Path,
    output_dir: str | Path,
    datasets: list[str] | None = None,
    label_map: dict[int, str] | None = None,
    x_dtype: str = "float32",
    y_dtype: str = "int32",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert LEAD-style processed datasets into this repo's manifest/window-store format.

    Supported formats:
    - `meta.json` + `X.dat` + `y.dat`
    - legacy `Feature/feature_*.npy` + `Label/*.npy`
    """
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_map = label_map or DEFAULT_LABEL_MAP.copy()

    manifest_rows: list[dict] = []
    index_rows: list[dict] = []

    for dataset_dir in _dataset_dirs(source_root, datasets):
        dataset_id = dataset_dir.name
        if (dataset_dir / "meta.json").exists():
            dataset_manifest, subject_records = _subject_rows_from_memmap(
                dataset_dir,
                dataset_id,
                label_map=label_map,
                x_dtype=x_dtype,
                y_dtype=y_dtype,
            )
        elif (dataset_dir / "Feature").exists() and (dataset_dir / "Label").exists():
            dataset_manifest, subject_records = _subject_rows_from_legacy(
                dataset_dir,
                dataset_id,
                label_map=label_map,
            )
        else:
            raise ValueError(
                f"Unsupported LEAD dataset layout at {dataset_dir}. "
                "Expected meta.json/X.dat/y.dat or Feature/ and Label/ directories."
            )

        manifest_rows.extend(dataset_manifest)

        dataset_out_dir = output_dir / dataset_id
        dataset_out_dir.mkdir(parents=True, exist_ok=True)
        for record in subject_records:
            out_path = dataset_out_dir / f"{record['subject_id']}.npz"
            np.savez_compressed(
                out_path,
                windows=record["windows"],
                label_int=np.array(record["label_int"], dtype=np.int64),
                label_str=np.array(str(record["label"])),
                dataset_id=np.array(str(record["dataset_id"])),
                subject_id=np.array(str(record["subject_id"])),
                subject_key=np.array(str(record["subject_key"])),
                ch_names=np.array(record["channel_names"], dtype=object),
                sfreq=np.array(float(record["sfreq"]), dtype=np.float32),
            )
            index_rows.append(
                {
                    "dataset_id": record["dataset_id"],
                    "subject_id": record["subject_id"],
                    "subject_key": record["subject_key"],
                    "label": record["label"],
                    "label_int": int(record["label_int"]),
                    "window_store_path": str(out_path),
                    "n_windows": int(record["windows"].shape[0]),
                    "n_channels": int(record["windows"].shape[1]),
                    "n_times": int(record["windows"].shape[2]),
                    "sfreq": float(record["sfreq"]),
                    "channel_names_json": json.dumps(list(record["channel_names"])),
                }
            )

    manifest_df = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    index_df = pd.DataFrame(index_rows)
    return manifest_df, index_df
