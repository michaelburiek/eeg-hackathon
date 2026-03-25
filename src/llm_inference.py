"""LLM inference for clinical interpretation of LaBraM outputs.

Loads a Qwen (or compatible) causal-LM via HuggingFace ``transformers``
and generates a clinical interpretation from the structured prompt
produced by ``text_reasoning.build_prompt``.

Typical usage
-------------
>>> from llm_inference import QwenInterpreter
>>> llm = QwenInterpreter("Qwen/Qwen2.5-1.5B-Instruct")
>>> report = llm.interpret(prompt_str)
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_MESSAGE = (
    "You are a clinical neurology assistant. You receive structured outputs "
    "from an EEG classification model (LaBraM) that predicts Alzheimer's "
    "disease, along with SHAP-based feature importance and raw EEG segments. "
    "Provide a concise, evidence-grounded clinical interpretation. "
    "Highlight which EEG channels and time segments drove the prediction, "
    "note uncertainty, and suggest next clinical steps."
)


class QwenInterpreter:
    """Wraps a Qwen (or any HF causal-LM) for clinical EEG interpretation.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID, e.g. ``"Qwen/Qwen2.5-1.5B-Instruct"``.
    device : str
        ``"cuda"``, ``"cpu"``, or ``"auto"``.
    torch_dtype : str
        ``"float16"``, ``"bfloat16"``, or ``"float32"``.
    max_new_tokens : int
        Maximum tokens the LLM may generate.
    system_message : str | None
        System prompt prepended to every request.  Defaults to
        ``SYSTEM_MESSAGE`` defined in this module.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        torch_dtype: str = "float16",
        max_new_tokens: int = 1024,
        system_message: Optional[str] = None,
    ):
        import platform

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dt = dtype_map.get(torch_dtype, torch.float16)

        # MPS (Apple Silicon) hits buffer-size limits with large tensors;
        # fall back to CPU when device is "auto" on macOS.
        if device == "auto" and platform.system() == "Darwin":
            device = "cpu"
            dt = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dt,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.system_message = system_message or SYSTEM_MESSAGE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpret(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        """Generate a clinical interpretation from a structured prompt.

        Parameters
        ----------
        prompt : str
            The output of ``text_reasoning.build_prompt``.
        max_new_tokens : int | None
            Override instance default if provided.
        temperature : float
            Sampling temperature (lower = more deterministic).
        top_p : float
            Nucleus sampling threshold.

        Returns
        -------
        str
            The LLM-generated clinical interpretation.
        """
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
        }

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Slice off the input tokens to get only the generated response
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


# ------------------------------------------------------------------
# CLI quick-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM interpreter quick test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    # Build a demo prompt
    import os
    import sys

    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from src.text_reasoning import build_prompt, load_few_shot_examples

    few_shot = load_few_shot_examples(
        os.path.join(os.path.dirname(__file__), "llm_prompts", "few_shot_examples.json")
    )
    demo_prompt = build_prompt(
        prediction_prob=0.82,
        shap_values=[0.15, -0.08, 0.02, 0.75, -0.25, 0.05],
        top_segments=[[0.1] * 50, [-0.2] * 50],
        segment_ids=["FP1_patch2", "T7_patch0"],
        feature_names=["alpha_power", "beta_power", "delta_power",
                       "theta_power", "frontal_asymmetry", "gamma_power"],
        patient_meta={"age": 74, "notes": "Progressive memory decline over 18 months"},
        few_shot_examples=few_shot,
    )

    print("=== PROMPT ===")
    print(demo_prompt)
    print("\n=== LOADING MODEL ===")

    llm = QwenInterpreter(
        model_name=args.model,
        device=args.device,
        torch_dtype=args.dtype,
        max_new_tokens=args.max_tokens,
    )

    print("=== GENERATING INTERPRETATION ===\n")
    report = llm.interpret(demo_prompt)
    print(report)
