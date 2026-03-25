"""Text-based reasoning utilities for LaBraM.

This module builds structured, human-readable prompts for LLMs that
encapsulate LaBraM model outputs (probabilistic predictions), SHAP
explanations, and the most salient raw EEG segments.

The primary entrypoint is `build_prompt(...)` which returns a single
string prompt suitable for few-shot LLM prompting.
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json


def _format_shap_summary(shap_vals: Sequence[float], feature_names: Optional[Sequence[str]] = None, top_k: int = 5) -> str:
    indices = sorted(range(len(shap_vals)), key=lambda i: abs(shap_vals[i]), reverse=True)[:top_k]
    lines = ["SHAP summary (top {} features):".format(top_k)]
    for i in indices:
        name = feature_names[i] if feature_names is not None and i < len(feature_names) else f"feature_{i}"
        lines.append(f"- {name}: contribution={shap_vals[i]:+.4f}")
    return "\n".join(lines)


def _format_segments(segments: Sequence[Sequence[float]], ids: Optional[Sequence[Any]] = None, max_len: int = 200) -> str:
    out = ["Top EEG segments (numeric sequences truncated):"]
    for idx, seg in enumerate(segments):
        label = f"seg_{ids[idx]}" if ids is not None and idx < len(ids) else f"seg_{idx}"
        seq = ",".join(str(x) for x in (seg[:max_len]))
        out.append(f"- {label}: [{seq}{'...' if len(seg) > max_len else ''}]")
    return "\n".join(out)


def build_prompt(
    prediction_prob: float,
    shap_values: Sequence[float],
    top_segments: Sequence[Sequence[float]],
    segment_ids: Optional[Sequence[Any]] = None,
    feature_names: Optional[Sequence[str]] = None,
    patient_meta: Optional[Dict[str, Any]] = None,
    few_shot_examples: Optional[Iterable[Dict[str, Any]]] = None,
    top_k_shap: int = 5,
    max_segment_len: int = 200,
) -> str:
    """Build a structured text prompt combining model outputs and SHAP.

    Args:
        prediction_prob: probability of AD (0..1) from LaBraM classifier.
        shap_values: sequence of SHAP values aligned with input features.
        top_segments: list/sequence of the most salient EEG segments (numeric arrays).
        segment_ids: optional identifiers for the segments.
        feature_names: optional human-readable feature names for SHAP values.
        patient_meta: optional patient/physician notes to include.
        few_shot_examples: optional iterable of labeled examples; each example should
            be a dict with keys like `label`, `prediction_prob`, `shap_values`, `notes`.
        top_k_shap: how many top SHAP items to include in the summary.
        max_segment_len: truncate each numeric segment to this many values.

    Returns:
        A single string prompt including the few-shot examples if provided.
    """
    parts: List[str] = []

    if few_shot_examples is not None:
        parts.append("Few-shot examples:")
        for ex in few_shot_examples:
            ex_label = ex.get("label", "UNKNOWN")
            ex_prob = ex.get("prediction_prob")
            ex_shap = ex.get("shap_values")
            ex_notes = ex.get("notes")
            parts.append(f"Example label: {ex_label}")
            if ex_prob is not None:
                parts.append(f"- model_prob: {ex_prob:.4f}")
            if ex_shap is not None:
                parts.append(_format_shap_summary(ex_shap, ex.get("feature_names"), top_k=top_k_shap))
            if ex_notes:
                parts.append(f"- notes: {ex_notes}")
            parts.append("---")

    parts.append("Patient input:")
    if patient_meta:
        parts.append("Patient metadata:")
        for k, v in (patient_meta.items() if isinstance(patient_meta, dict) else []):
            parts.append(f"- {k}: {v}")

    parts.append(f"LaBraM prediction (probability of AD): {prediction_prob:.4f}")
    parts.append(_format_shap_summary(shap_values, feature_names, top_k=top_k_shap))
    parts.append(_format_segments(top_segments, ids=segment_ids, max_len=max_segment_len))

    # Guidance for the LLM: explicit instructions about what to do with inputs
    parts.append("Instructions: Provide an interpretation of the model output, explain which features/segments drove the prediction using the SHAP summary, and suggest next clinical actions or additional tests. State uncertainty and possible confounders.")

    return "\n\n".join(parts)


def load_few_shot_examples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Minimal self-test helper when run directly
if __name__ == "__main__":
    demo_shap = [0.12, -0.05, 0.001, 0.9, -0.3, 0.04]
    demo_segments = [list(range(300)), list(range(50, 350)), list(range(-100, 200))]
    prompt = build_prompt(0.78, demo_shap, demo_segments, segment_ids=["A","B","C"], feature_names=[f"f{i}" for i in range(len(demo_shap))])
    print(prompt)
