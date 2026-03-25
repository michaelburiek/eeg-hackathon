"""SHAP-based explainability for LaBraM EEG classifier.

Computes SHAP values for a trained LaBraM model, identifies the most
salient EEG segments, and produces structured outputs ready for
``text_reasoning.build_prompt``.

Typical usage
-------------
>>> from shap_explainer import LaBraMExplainer
>>> explainer = LaBraMExplainer(model, background_data, ch_names=ch_names)
>>> result = explainer.explain(sample)
>>> from text_reasoning import build_prompt
>>> prompt = build_prompt(**result)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import shap
except ImportError:
    shap = None

import src  # noqa: F401 — sets up LaBraM path + monkey-patches
import utils


# ---------------------------------------------------------------------------
# Model wrapper: flatten 4-D EEG input for SHAP, restore inside forward
# ---------------------------------------------------------------------------

class _LaBraMWrapper(nn.Module):
    """Wraps a LaBraM ``NeuralTransformer`` so that SHAP explainers can
    operate on a flattened 2-D tensor ``(batch, N*A*T)`` while the real
    model still receives the original 4-D shape ``(batch, N, A, T)``.
    """

    def __init__(
        self,
        model: nn.Module,
        n_channels: int,
        n_patches: int,
        patch_size: int = 200,
        input_chans: Optional[List[int]] = None,
        is_binary: bool = True,
    ):
        super().__init__()
        self.model = model
        self.n_channels = n_channels
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.input_chans = input_chans
        self.is_binary = is_binary

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        B = x_flat.shape[0]
        x = x_flat.view(B, self.n_channels, self.n_patches, self.patch_size)
        logits = self.model(x, input_chans=self.input_chans)
        if self.is_binary:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Main explainer class
# ---------------------------------------------------------------------------

class LaBraMExplainer:
    """Compute SHAP values for a trained LaBraM model.

    Parameters
    ----------
    model : nn.Module
        A trained ``NeuralTransformer`` (from ``modeling_finetune``).
    background : torch.Tensor
        A small set of representative EEG samples used as the SHAP
        background / reference distribution.  Shape should match what
        the model expects **before** the ``/100`` scaling that the
        engine applies, i.e. ``(K, N, A*T)`` from the dataloader.
        Internally this is scaled and reshaped.
    ch_names : list[str] | None
        Channel (electrode) names for the background samples.  Used
        both for mapping to ``standard_1020`` indices and for
        human-readable feature names.
    device : str
        ``"cuda"`` or ``"cpu"``.
    n_background : int
        Max number of background samples to use (SHAP can be slow with
        many).
    """

    def __init__(
        self,
        model: nn.Module,
        background: torch.Tensor,
        ch_names: Optional[List[str]] = None,
        device: str = "cpu",
        n_background: int = 50,
        is_binary: bool = True,
    ):
        if shap is None:
            raise ImportError("shap is required: pip install shap")

        self.device = torch.device(device)
        self.ch_names = ch_names
        self.input_chans = None
        if ch_names is not None:
            self.input_chans = utils.get_input_chans(ch_names)

        # Prepare background: scale + reshape to 4-D then flatten for wrapper
        bg = background[:n_background].float().to(self.device) / 100.0
        if bg.ndim == 3:
            # (K, N, A*T) -> (K, N, A, T)
            n_channels = bg.shape[1]
            n_patches = bg.shape[2] // 200
            bg = bg.view(bg.shape[0], n_channels, n_patches, 200)
        else:
            n_channels = bg.shape[1]
            n_patches = bg.shape[2]

        self.n_channels = n_channels
        self.n_patches = n_patches
        self.patch_size = 200

        # Build wrapper model
        model.eval()
        self.wrapper = _LaBraMWrapper(
            model,
            n_channels=n_channels,
            n_patches=n_patches,
            patch_size=self.patch_size,
            input_chans=self.input_chans,
            is_binary=is_binary,
        ).to(self.device)

        # Flatten background for the wrapper
        bg_flat = bg.reshape(bg.shape[0], -1)
        self.explainer = shap.GradientExplainer(self.wrapper, bg_flat)

        # Feature name helpers
        self._build_feature_names()

    # ---- internal helpers -------------------------------------------------

    def _build_feature_names(self):
        """Build per-channel and per-channel-patch feature names."""
        if self.ch_names is not None:
            ch_labels = [ch.upper() for ch in self.ch_names]
        else:
            ch_labels = [f"ch_{i}" for i in range(self.n_channels)]

        # Per-channel feature names (aggregated across patches & samples)
        self.channel_names = ch_labels

        # Per channel-patch names  (e.g. "FP1_patch0")
        self.channel_patch_names: List[str] = []
        for ch in ch_labels:
            for p in range(self.n_patches):
                self.channel_patch_names.append(f"{ch}_patch{p}")

    # ---- public API -------------------------------------------------------

    def explain(
        self,
        sample: torch.Tensor,
        top_k_channels: int = 10,
        top_k_segments: int = 3,
    ) -> Dict[str, Any]:
        """Compute SHAP values for a single EEG sample.

        Parameters
        ----------
        sample : torch.Tensor
            Raw EEG from the dataloader, shape ``(N, A*T)`` (single sample,
            no batch dim) **before** the ``/100`` scaling.
        top_k_channels : int
            Number of top channels to include in the SHAP summary.
        top_k_segments : int
            Number of most-salient segments to return as raw data.

        Returns
        -------
        dict
            Keys match the arguments of ``text_reasoning.build_prompt``:
            ``prediction_prob``, ``shap_values``, ``feature_names``,
            ``top_segments``, ``segment_ids``.
        """
        # Scale and reshape
        x = sample.float().to(self.device) / 100.0
        if x.ndim == 2:
            x = x.view(1, self.n_channels, self.n_patches, self.patch_size)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        x_flat = x.reshape(1, -1)

        # --- prediction ---
        with torch.no_grad():
            prob = self.wrapper(x_flat).item()

        # --- SHAP values ---
        # GradientExplainer returns list (one per output); we have 1 output
        shap_vals = self.explainer.shap_values(x_flat)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_arr = np.array(shap_vals).reshape(
            self.n_channels, self.n_patches, self.patch_size
        )  # (N, A, T)

        # Aggregate: per-channel SHAP = mean |SHAP| across patches & time
        channel_shap = np.mean(np.abs(shap_arr), axis=(1, 2))  # (N,)

        # Per channel-patch SHAP (for segment ranking)
        patch_shap = np.mean(np.abs(shap_arr), axis=2)  # (N, A)

        # --- top segments by SHAP importance ---
        flat_patch_shap = patch_shap.flatten()  # (N*A,)
        top_indices = np.argsort(flat_patch_shap)[::-1][:top_k_segments]

        raw_eeg = x.cpu().numpy().reshape(
            self.n_channels, self.n_patches, self.patch_size
        )

        top_segments: List[List[float]] = []
        segment_ids: List[str] = []
        for idx in top_indices:
            ch_idx = idx // self.n_patches
            p_idx = idx % self.n_patches
            seg = raw_eeg[ch_idx, p_idx, :].tolist()
            top_segments.append(seg)
            segment_ids.append(self.channel_patch_names[idx])

        return {
            "prediction_prob": float(prob),
            "shap_values": channel_shap.tolist(),
            "feature_names": self.channel_names,
            "top_segments": top_segments,
            "segment_ids": segment_ids,
        }

    def explain_batch(
        self,
        samples: torch.Tensor,
        top_k_channels: int = 10,
        top_k_segments: int = 3,
    ) -> List[Dict[str, Any]]:
        """Run ``explain`` on each sample in a batch.

        Parameters
        ----------
        samples : torch.Tensor
            Shape ``(B, N, A*T)`` from the dataloader, before ``/100``.
        """
        results = []
        for i in range(samples.shape[0]):
            results.append(
                self.explain(
                    samples[i],
                    top_k_channels=top_k_channels,
                    top_k_segments=top_k_segments,
                )
            )
        return results

    def get_channel_importance(
        self, sample: torch.Tensor
    ) -> List[Tuple[str, float]]:
        """Return all channels sorted by SHAP importance (descending).

        Useful for quick inspection or visualization.
        """
        result = self.explain(sample)
        pairs = list(zip(result["feature_names"], result["shap_values"]))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return pairs


# ---------------------------------------------------------------------------
# Convenience: end-to-end explain + prompt generation
# ---------------------------------------------------------------------------

def explain_and_build_prompt(
    model: nn.Module,
    sample: torch.Tensor,
    background: torch.Tensor,
    ch_names: Optional[List[str]] = None,
    patient_meta: Optional[Dict[str, Any]] = None,
    few_shot_path: Optional[str] = None,
    device: str = "cpu",
    top_k_channels: int = 10,
    top_k_segments: int = 3,
    n_background: int = 50,
    is_binary: bool = True,
) -> str:
    """One-call helper: compute SHAP + build the LLM prompt.

    Parameters
    ----------
    model : nn.Module
        Trained LaBraM model.
    sample : torch.Tensor
        Single EEG sample ``(N, A*T)`` before ``/100`` scaling.
    background : torch.Tensor
        Background dataset ``(K, N, A*T)``.
    ch_names : list[str] | None
        Electrode names.
    patient_meta : dict | None
        Optional clinician notes / patient info.
    few_shot_path : str | None
        Path to ``few_shot_examples.json``.
    device : str
        ``"cuda"`` or ``"cpu"``.
    top_k_channels : int
        Number of top SHAP channels in the summary.
    top_k_segments : int
        Number of salient EEG segments to include.
    n_background : int
        Number of background samples for SHAP.
    is_binary : bool
        Whether the model is a binary classifier (AD vs healthy).

    Returns
    -------
    str
        The complete LLM prompt string.
    """
    from src.text_reasoning import build_prompt, load_few_shot_examples

    explainer = LaBraMExplainer(
        model, background, ch_names=ch_names,
        device=device, n_background=n_background,
        is_binary=is_binary,
    )
    result = explainer.explain(
        sample,
        top_k_channels=top_k_channels,
        top_k_segments=top_k_segments,
    )

    few_shot = None
    if few_shot_path is not None:
        few_shot = load_few_shot_examples(few_shot_path)

    return build_prompt(
        prediction_prob=result["prediction_prob"],
        shap_values=result["shap_values"],
        top_segments=result["top_segments"],
        segment_ids=result["segment_ids"],
        feature_names=result["feature_names"],
        patient_meta=patient_meta,
        few_shot_examples=few_shot,
        top_k_shap=top_k_channels,
    )


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors run_class_finetuning.py logic)
# ---------------------------------------------------------------------------

def load_checkpoint(model: nn.Module, path: str, device: str = "cpu",
                    model_key: str = "model|module",
                    model_filter_name: str = "") -> nn.Module:
    """Load a LaBraM checkpoint, handling the various key layouts.

    Reproduces the same loading logic used in ``run_class_finetuning.py``
    so that pretrained *and* fine-tuned checkpoints both work.
    """
    from collections import OrderedDict

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Try standard keys: "model", "module"
    checkpoint_model = None
    for key in model_key.split("|"):
        if key in ckpt:
            checkpoint_model = ckpt[key]
            print(f"Loaded state_dict via key '{key}'")
            break
    if checkpoint_model is None:
        checkpoint_model = ckpt

    # Strip "student." prefix if present (distillation checkpoints)
    if model_filter_name:
        new_dict = OrderedDict()
        for k in list(checkpoint_model.keys()):
            if k.startswith("student."):
                new_dict[k[8:]] = checkpoint_model[k]
        if new_dict:
            checkpoint_model = new_dict

    # Drop head weights if shapes mismatch (pretrained ckpt vs fine-tuned)
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from checkpoint (shape mismatch)")
            del checkpoint_model[k]

    # Drop relative_position_index buffers
    for key in list(checkpoint_model.keys()):
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

    utils.load_state_dict(model, checkpoint_model)
    return model


# ---------------------------------------------------------------------------
# Data loading helpers for real EEG data
# ---------------------------------------------------------------------------

def _load_tuab_sample(data_dir: str, index: int = 0):
    """Load a single sample + background batch from a TUAB pickle dir."""
    import os
    import pickle
    from scipy.signal import resample as scipy_resample

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {data_dir}")

    samples = []
    for f in files[:51]:  # 1 sample + up to 50 background
        with open(os.path.join(data_dir, f), "rb") as fh:
            rec = pickle.load(fh)
        samples.append(torch.FloatTensor(rec["X"]))

    sample = samples[index]
    background = torch.stack(samples[1:] if index == 0 else
                             samples[:index] + samples[index + 1:])
    return sample, background


def _load_hdf5_sample(hdf5_path: str, index: int = 0,
                       window_size: int = 800):
    """Load a single sample + background from an HDF5 dataset file."""
    import h5py

    f = h5py.File(hdf5_path, "r")
    subjects = list(f.keys())

    # Read ch_names from first subject
    ch_names_raw = f[subjects[0]]["eeg"].attrs.get("chOrder", None)
    ch_names = None
    if ch_names_raw is not None:
        ch_names = [n.split(" ")[-1].split("-")[0] for n in ch_names_raw]

    # Gather samples from subjects
    samples = []
    for subj in subjects[:51]:
        eeg = f[subj]["eeg"]
        length = eeg.shape[1]
        if length >= window_size:
            samples.append(torch.FloatTensor(eeg[:, :window_size]))
    f.close()

    if not samples:
        raise ValueError(f"No subjects with >= {window_size} samples in {hdf5_path}")

    sample = samples[index]
    bg_list = samples[1:] if index == 0 else samples[:index] + samples[index + 1:]
    background = torch.stack(bg_list)
    return sample, background, ch_names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os
    from timm.models import create_model

    parser = argparse.ArgumentParser(
        description="SHAP explanation for a LaBraM EEG checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a .pth checkpoint. If omitted, uses random weights (demo mode).")
    parser.add_argument("--model", type=str, default="labram_base_patch200_200",
                        choices=["labram_base_patch200_200",
                                 "labram_large_patch200_200",
                                 "labram_huge_patch200_200"])
    parser.add_argument("--nb-classes", type=int, default=1,
                        help="1 for binary AD detection, 6 for TUEV, etc.")
    parser.add_argument("--init-values", type=float, default=None,
                        help="Layer-scale init value (use value from training config).")
    # Data
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data: a directory of .pkl files (TUAB), "
                             "an .h5/.hdf5 file, or omit for synthetic demo.")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Which sample to explain.")
    parser.add_argument("--window-size", type=int, default=800,
                        help="EEG window length in samples (n_patches * 200).")
    parser.add_argument("--ch-names", type=str, nargs="+", default=None,
                        help="Channel names (auto-detected from HDF5 if omitted).")
    # Explainer
    parser.add_argument("--n-background", type=int, default=20)
    parser.add_argument("--top-k-channels", type=int, default=10)
    parser.add_argument("--top-k-segments", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    # Output
    parser.add_argument("--patient-age", type=int, default=None)
    parser.add_argument("--patient-notes", type=str, default=None)
    parser.add_argument("--few-shot-path", type=str, default=None)
    args = parser.parse_args()

    # ---- build model ----
    is_binary = (args.nb_classes == 1)
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        init_values=args.init_values,
    )
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device=args.device)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided — running with random weights (demo mode)")
    model.eval()

    # ---- load data ----
    ch_names = args.ch_names
    if args.data is not None:
        if args.data.endswith((".h5", ".hdf5")):
            sample, background, detected_ch = _load_hdf5_sample(
                args.data, index=args.sample_index, window_size=args.window_size)
            if ch_names is None:
                ch_names = detected_ch
        else:
            sample, background = _load_tuab_sample(
                args.data, index=args.sample_index)
    else:
        # Synthetic demo
        n_ch = len(ch_names) if ch_names else 19
        n_p = args.window_size // 200
        sample = torch.randn(n_ch, args.window_size) * 100
        background = torch.randn(args.n_background, n_ch, args.window_size) * 100

    if ch_names is None:
        n_ch = sample.shape[0]
        ch_names = utils.standard_1020[:n_ch]
        print(f"No channel names provided — using first {n_ch} standard 10-20 names")

    # ---- patient metadata ----
    patient_meta = {}
    if args.patient_age is not None:
        patient_meta["age"] = args.patient_age
    if args.patient_notes is not None:
        patient_meta["notes"] = args.patient_notes

    # ---- explain ----
    few_shot_path = args.few_shot_path
    if few_shot_path is None:
        default_path = os.path.join(os.path.dirname(__file__),
                                    "llm_prompts", "few_shot_examples.json")
        if os.path.exists(default_path):
            few_shot_path = default_path

    prompt = explain_and_build_prompt(
        model, sample, background,
        ch_names=ch_names,
        patient_meta=patient_meta or None,
        few_shot_path=few_shot_path,
        device=args.device,
        top_k_channels=args.top_k_channels,
        top_k_segments=args.top_k_segments,
        n_background=args.n_background,
        is_binary=is_binary,
    )
    print(prompt)
