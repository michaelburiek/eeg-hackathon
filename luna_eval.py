"""Evaluate a pretrained LUNA model on the CAUEEG abnormal/dementia task.

Predictions are computed per-recording by averaging softmax outputs across
non-overlapping windows. Reports AUROC plus standard binary metrics and
saves the ROC curve.

Example invocations:
    conda run -n eeg312 python luna_eval.py --model aapo23/my-luna-model-paper --outdir output/ftd_ad_grouped/
    conda run -n eeg312 python luna_eval.py --model aapo23/my-luna-model-alz --outdir output/cn_ftd_grouped/
"""

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from braindecode.datasets import BaseConcatDataset, RawDataset
from braindecode.models import LUNA
from braindecode.preprocessing import (
    create_fixed_length_windows,
    exponential_moving_standardize,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader


def pick_device(requested):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_recording(
    model,
    raw,
    device,
    target_sfreq=200,
    window_len_sec=5.0,
    batch_size=32,
):
    """Return mean softmax probability vector over all windows in a recording."""
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(sfreq=target_sfreq, npad="auto")
    raw._data = exponential_moving_standardize(raw.get_data(), factor_new=0.001)

    window_samples = int(raw.info["sfreq"] * window_len_sec)
    windows = create_fixed_length_windows(
        BaseConcatDataset([RawDataset(raw)]),
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=window_samples,
        window_stride_samples=window_samples,
        drop_last_window=True,
        preload=True,
    )
    loader = DataLoader(windows, batch_size=batch_size, shuffle=False)

    probs = []
    with torch.no_grad():
        for batch_X, _, _ in loader:
            batch_X = batch_X.to(device).float()
            out = model(batch_X)
            probs.append(torch.softmax(out, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0).mean(axis=0)


def evaluate(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_roc(y_true, y_score, out_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/caueeg-dataset")
    parser.add_argument("--task", default="abnormal", choices=["abnormal", "dementia"])
    parser.add_argument("--split", default="test_split")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only first N subjects")
    parser.add_argument("--device", default=None,
                        help="cuda | mps | cpu (auto-detected by default)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default="aapo23/my-luna-model-paper",
                        help="HuggingFace LUNA checkpoint id")
    parser.add_argument("--outdir", default="output",
                        help="directory to write outputs into (created if missing)")
    parser.add_argument("--out-prefix", default="luna_eval")
    args = parser.parse_args()

    model_slug = args.model.split("/")[-1]

    dataset_dir = Path(args.dataset_dir)
    with open(dataset_dir / f"{args.task}.json") as f:
        infos = json.load(f)[args.split]
    if args.limit:
        infos = infos[: args.limit]
    edf_dir = dataset_dir / "signal" / "edf"

    device = pick_device(args.device)
    print(f"Device: {device}  |  model: {args.model}  |  task: {args.task}  "
          f"|  split: {args.split}  |  n={len(infos)}")

    model = LUNA.from_pretrained(args.model)
    model.eval()
    model.to(device)

    y_true, y_score = [], []
    for i, info in enumerate(infos, 1):
        serial, label = info["serial"], info["class_label"]
        edf_path = edf_dir / f"{serial}.edf"
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
            mean_probs = predict_recording(
                model, raw, device, batch_size=args.batch_size
            )
            prob_abnormal = float(mean_probs[1])
        except Exception as e:
            print(f"[{i}/{len(infos)}] {serial}: FAILED ({type(e).__name__}: {e})")
            continue
        y_true.append(label)
        y_score.append(prob_abnormal)
        print(f"[{i}/{len(infos)}] {serial}: label={label}  p(abnormal)={prob_abnormal:.3f}")

    y_true_arr = np.array(y_true)
    y_score_arr = np.array(y_score)

    metrics = evaluate(y_true_arr, y_score_arr)
    print("\n=== Subject-level evaluation ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.out_prefix}_{model_slug}"
    roc_path = outdir / f"{stem}_roc.png"
    preds_path = outdir / f"{stem}_preds.npz"
    metrics_path = outdir / f"{stem}_metrics.json"
    save_roc(
        y_true_arr,
        y_score_arr,
        roc_path,
        title=f"{model_slug} on CAUEEG-{args.task} ({metrics['n']} subjects)",
    )
    np.savez(preds_path, y_true=y_true_arr, y_score=y_score_arr)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {roc_path}, {preds_path}, {metrics_path}")


if __name__ == "__main__":
    main()
