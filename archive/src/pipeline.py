"""End-to-end pipeline: EEG sample -> LaBraM -> SHAP -> LLM -> clinical report.

Ties together all components of the system:
1. Load a trained LaBraM model
2. Run inference on an EEG sample
3. Compute SHAP explanations
4. Build a structured prompt with few-shot examples
5. Send to Qwen LLM for clinical interpretation

Usage
-----
    python pipeline.py \
        --checkpoint path/to/labram.pth \
        --eeg-file path/to/sample.h5 \
        --llm-model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from einops import rearrange

import src  # noqa: F401 — sets up LaBraM path + monkey-patches
import utils
from modeling_finetune import labram_base_patch200_200
from src.shap_explainer import LaBraMExplainer
from src.text_reasoning import build_prompt, load_few_shot_examples
from src.llm_inference import QwenInterpreter


def load_labram(
    checkpoint: Optional[str],
    num_classes: int = 1,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load a LaBraM model, optionally from a checkpoint."""
    model = labram_base_patch200_200(
        num_classes=num_classes, in_chans=1, out_chans=8, init_values=0.1,
    )
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
    model.to(device).eval()
    return model


def load_eeg_sample(path: str) -> tuple:
    """Load an EEG sample from an HDF5 file.

    Expects the file to contain:
      - ``eeg``: shape ``(N_channels, total_samples)``
      - ``ch_names`` (optional): list of channel name strings
      - ``label`` (optional): ground-truth label

    Returns (eeg_tensor, ch_names, label).
    """
    import h5py

    with h5py.File(path, "r") as f:
        eeg = np.array(f["eeg"])
        ch_names = None
        if "ch_names" in f:
            ch_names = [s.decode() if isinstance(s, bytes) else s for s in f["ch_names"]]
        label = None
        if "label" in f:
            label = int(np.array(f["label"]))
    return torch.from_numpy(eeg), ch_names, label


def run_pipeline(
    model: torch.nn.Module,
    eeg: torch.Tensor,
    background: torch.Tensor,
    llm: QwenInterpreter,
    ch_names: Optional[List[str]] = None,
    patient_meta: Optional[Dict[str, Any]] = None,
    few_shot_path: Optional[str] = None,
    device: str = "cpu",
    top_k_channels: int = 10,
    top_k_segments: int = 3,
    n_background: int = 50,
) -> Dict[str, Any]:
    """Execute the full pipeline and return all intermediate + final results.

    Returns
    -------
    dict with keys:
        ``prediction_prob``, ``shap_values``, ``feature_names``,
        ``top_segments``, ``segment_ids``, ``prompt``, ``interpretation``
    """
    # 1. SHAP explanation
    explainer = LaBraMExplainer(
        model, background, ch_names=ch_names,
        device=device, n_background=n_background,
    )
    shap_result = explainer.explain(
        eeg, top_k_channels=top_k_channels, top_k_segments=top_k_segments,
    )

    # 2. Build prompt
    few_shot = None
    if few_shot_path and os.path.exists(few_shot_path):
        few_shot = load_few_shot_examples(few_shot_path)

    prompt = build_prompt(
        prediction_prob=shap_result["prediction_prob"],
        shap_values=shap_result["shap_values"],
        top_segments=shap_result["top_segments"],
        segment_ids=shap_result["segment_ids"],
        feature_names=shap_result["feature_names"],
        patient_meta=patient_meta,
        few_shot_examples=few_shot,
        top_k_shap=top_k_channels,
    )

    # 3. LLM interpretation
    interpretation = llm.interpret(prompt)

    return {
        **shap_result,
        "prompt": prompt,
        "interpretation": interpretation,
    }


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end EEG -> LaBraM -> SHAP -> LLM pipeline"
    )
    # Model args
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LaBraM .pth checkpoint")
    parser.add_argument("--device", type=str, default="cpu")

    # EEG input
    parser.add_argument("--eeg-file", type=str, default=None,
                        help="Path to HDF5 file with EEG sample")
    parser.add_argument("--n-channels", type=int, default=19,
                        help="Number of channels (used for synthetic demo)")
    parser.add_argument("--n-patches", type=int, default=4,
                        help="Number of time patches (used for synthetic demo)")

    # LLM args
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--llm-device", type=str, default="auto")
    parser.add_argument("--llm-dtype", type=str, default="float32",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-tokens", type=int, default=1024)

    # SHAP args
    parser.add_argument("--top-k-channels", type=int, default=10)
    parser.add_argument("--top-k-segments", type=int, default=3)
    parser.add_argument("--n-background", type=int, default=50)

    # Patient metadata
    parser.add_argument("--patient-meta", type=str, default=None,
                        help='JSON string, e.g. \'{"age": 72, "notes": "memory decline"}\'')

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Save full results as JSON to this path")

    args = parser.parse_args()

    # --- Load LaBraM ---
    print("[1/4] Loading LaBraM model...")
    model = load_labram(args.checkpoint, device=args.device)

    # --- Load or synthesize EEG ---
    if args.eeg_file:
        print(f"[2/4] Loading EEG from {args.eeg_file}...")
        eeg, ch_names, label = load_eeg_sample(args.eeg_file)
        # Synthesize background from same file shape for demo
        background = torch.randn(args.n_background, eeg.shape[0], eeg.shape[1]) * 100
    else:
        print("[2/4] Using synthetic EEG data (no --eeg-file provided)...")
        n_ch, n_p, ps = args.n_channels, args.n_patches, 200
        eeg = torch.randn(n_ch, n_p * ps) * 100
        background = torch.randn(args.n_background, n_ch, n_p * ps) * 100
        ch_names = utils.standard_1020[:n_ch]
        label = None

    # --- Load LLM ---
    print(f"[3/4] Loading LLM ({args.llm_model})...")
    llm = QwenInterpreter(
        model_name=args.llm_model,
        device=args.llm_device,
        torch_dtype=args.llm_dtype,
        max_new_tokens=args.max_tokens,
    )

    # --- Run pipeline ---
    patient_meta = None
    if args.patient_meta:
        patient_meta = json.loads(args.patient_meta)

    few_shot_path = os.path.join(
        os.path.dirname(__file__), "llm_prompts", "few_shot_examples.json"
    )

    print("[4/4] Running SHAP + LLM pipeline...")
    result = run_pipeline(
        model=model,
        eeg=eeg,
        background=background,
        llm=llm,
        ch_names=ch_names,
        patient_meta=patient_meta,
        few_shot_path=few_shot_path,
        device=args.device,
        top_k_channels=args.top_k_channels,
        top_k_segments=args.top_k_segments,
        n_background=args.n_background,
    )

    # --- Output ---
    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)
    print(f"AD probability: {result['prediction_prob']:.4f}")
    if label is not None:
        print(f"Ground truth:   {'AD' if label == 1 else 'Healthy'}")

    print("\n" + "=" * 60)
    print("TOP SHAP CHANNELS")
    print("=" * 60)
    pairs = sorted(
        zip(result["feature_names"], result["shap_values"]),
        key=lambda p: abs(p[1]), reverse=True,
    )
    for name, val in pairs[:args.top_k_channels]:
        print(f"  {name:>12s}: {val:+.6f}")

    print("\n" + "=" * 60)
    print("LLM INTERPRETATION")
    print("=" * 60)
    print(result["interpretation"])

    if args.output:
        # Make segments JSON-serializable
        out = {
            "prediction_prob": result["prediction_prob"],
            "shap_values": dict(zip(result["feature_names"], result["shap_values"])),
            "top_segments": {sid: seg for sid, seg in zip(result["segment_ids"], result["top_segments"])},
            "interpretation": result["interpretation"],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
