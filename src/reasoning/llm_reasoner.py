"""
src/reasoning/llm_reasoner.py
──────────────────────────────────────────────────────────────────────��─────────
Clinician-style LLM reasoning layer.

Takes the trained EEGConformer's per-subject classification output (aggregated
over all EEG windows for that subject) and calls the Claude API to produce a
structured natural-language clinical report.

The LLM is grounded: the system prompt explicitly prohibits it from citing
findings not present in the structured input. It synthesises; it does not
diagnose from raw data.

Public API
----------
SubjectClassification   — dataclass holding aggregated model output for one subject
generate_subject_report — call Claude and return a markdown report string
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

CLASS_NAMES = ["CN", "AD", "FTD"]

SYSTEM_PROMPT = """You are a clinical EEG analyst assistant helping interpret the output \
of a deep learning model trained to classify resting-state EEG recordings into three \
diagnostic categories: Cognitively Normal (CN), Alzheimer's Disease (AD), and \
Frontotemporal Dementia (FTD).

You will be given structured classification data from an EEGConformer model — a \
convolutional transformer trained from scratch on the LEAD open-source EEG dataset \
(88 subjects, 200 Hz, 19 channels, 2-second windows).

Your role is to synthesise the model output into a clear, concise clinical-style report.

STRICT RULES you must follow:
1. Only reference findings that are present in the structured input provided to you.
2. Never invent biomarker findings, channel-level patterns, or clinical history not given.
3. Always express uncertainty proportional to the model's confidence scores.
4. Always include a differential diagnosis — never commit to a single label if confidence \
   is below 90%.
5. Always include a disclaimer that this is a research model output, not a clinical diagnosis.
6. Use precise, professional clinical language. Be concise — the report should be \
   readable in under two minutes.
7. Structure your report exactly using the section headers provided in the user message."""


USER_TEMPLATE = """Generate a clinical EEG interpretation report for the following subject.

## Subject Information
- Subject ID: {subject_id}
- Total EEG windows analysed: {n_windows}
- Window duration: 2 seconds (200 Hz, 19 channels, 10-20 system)

## Model Classification Summary
- **Primary prediction:** {primary_class} ({primary_confidence:.1f}% of windows)
- **Mean class probabilities (softmax):**
{class_prob_lines}

## Window-Level Prediction Distribution
{window_dist_lines}

## Prediction Consistency
- Consistency score: {consistency:.1f}% (% of windows matching the primary prediction)
- Interpretation: {consistency_label}

---

Please produce a report using EXACTLY these section headers:

### Primary Classification
### Confidence Assessment
### Differential Diagnosis
### Supporting Evidence
### Limitations & Uncertainty
### Clinical Disclaimer
"""


# ─── Data container ───────────────────────────────────────────────────────────

@dataclasses.dataclass
class SubjectClassification:
    """
    Aggregated model output for a single subject.

    Parameters
    ----------
    subject_id        : subject identifier string
    n_windows         : number of EEG windows processed
    window_preds      : integer class prediction per window, shape (N,)
    window_probs      : softmax probabilities per window, shape (N, n_classes)
    class_names       : ordered list of class name strings
    true_label        : ground-truth label string (optional)
    """
    subject_id:   str
    n_windows:    int
    window_preds: list          # List[int]
    window_probs: list          # List[List[float]]  shape (N, C)
    class_names:  List[str]
    true_label:   Optional[str] = None

    def primary_class(self) -> str:
        counts = {c: 0 for c in self.class_names}
        for p in self.window_preds:
            counts[self.class_names[p]] += 1
        return max(counts, key=counts.__getitem__)

    def primary_confidence(self) -> float:
        """Fraction of windows voting for the primary class (0–100)."""
        primary = self.primary_class()
        primary_idx = self.class_names.index(primary)
        return 100.0 * sum(1 for p in self.window_preds if p == primary_idx) / max(self.n_windows, 1)

    def mean_probs(self) -> Dict[str, float]:
        """Mean softmax probability per class across all windows."""
        import numpy as np
        arr = np.array(self.window_probs)   # (N, C)
        means = arr.mean(axis=0)
        return {name: float(means[i]) for i, name in enumerate(self.class_names)}

    def consistency_label(self, pct: float) -> str:
        if pct >= 90:
            return "High — model is strongly consistent across windows"
        if pct >= 70:
            return "Moderate — some variability across windows; interpret with care"
        return "Low — high window-level variability; classification should be treated cautiously"


# ─── Report generator ─────────────────────────────────────────────────────────

def _build_user_message(sc: SubjectClassification) -> str:
    mean_probs = sc.mean_probs()
    primary    = sc.primary_class()
    prim_conf  = sc.primary_confidence()

    class_prob_lines = "\n".join(
        f"  - {name}: {prob * 100:.1f}%"
        for name, prob in sorted(mean_probs.items(), key=lambda x: -x[1])
    )

    import numpy as np
    preds = np.array(sc.window_preds)
    window_dist_lines = "\n".join(
        f"  - {name}: {int((preds == i).sum())} / {sc.n_windows} windows "
        f"({100 * (preds == i).mean():.1f}%)"
        for i, name in enumerate(sc.class_names)
    )

    consistency = prim_conf
    consistency_lbl = sc.consistency_label(consistency)

    return USER_TEMPLATE.format(
        subject_id=sc.subject_id,
        n_windows=sc.n_windows,
        primary_class=primary,
        primary_confidence=prim_conf,
        class_prob_lines=class_prob_lines,
        window_dist_lines=window_dist_lines,
        consistency=consistency,
        consistency_label=consistency_lbl,
    )


def generate_subject_report(
    sc: SubjectClassification,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1024,
) -> str:
    """
    Call the Claude API and return a markdown report string for one subject.

    Parameters
    ----------
    sc        : SubjectClassification with aggregated model output
    api_key   : Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
    model     : Claude model ID
    max_tokens: maximum tokens in the response

    Returns
    -------
    Markdown-formatted report string.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        )

    key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY in .env "
            "or pass api_key= to generate_subject_report()."
        )

    client = anthropic.Anthropic(api_key=key)
    user_message = _build_user_message(sc)

    log.debug("Calling Claude for subject %s", sc.subject_id)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    report_body = message.content[0].text

    # Prepend a header with subject metadata
    true_label_line = (
        f"**Ground-truth label:** {sc.true_label}\n" if sc.true_label else ""
    )
    header = (
        f"# EEG Classification Report — {sc.subject_id}\n\n"
        f"**Primary prediction:** {sc.primary_class()} "
        f"({sc.primary_confidence():.1f}% window agreement)\n"
        f"{true_label_line}"
        f"**Windows analysed:** {sc.n_windows}\n\n"
        "---\n\n"
    )

    return header + report_body
