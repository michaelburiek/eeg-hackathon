# EEG-AD: Explainable Alzheimer's Detection from EEG

An end-to-end system that combines a transformer-based EEG classifier (LaBraM) with SHAP explainability and LLM-powered clinical interpretation for Alzheimer's disease detection.

## Architecture

```
EEG Signal -> LaBraM (classification) -> SHAP (explainability) -> Qwen LLM (clinical report)
```

1. **LaBraM** - A pretrained EEG transformer that produces a probabilistic AD prediction.
2. **SHAP Explainer** - Computes per-channel and per-segment feature importance via GradientExplainer.
3. **Prompt Builder** - Assembles the prediction, SHAP values, top EEG segments, patient metadata, and few-shot examples into a structured prompt.
4. **Qwen LLM** - Generates a clinical interpretation with next-step recommendations.

## Project Structure

```
eeg-hackathon/
├── LaBraM/                          # git submodule (upstream, untouched)
│   ├── modeling_finetune.py         # NeuralTransformer model definitions
│   ├── run_class_finetuning.py      # training/eval entry point
│   ├── engine_for_finetuning.py     # train/eval loops
│   ├── utils.py                     # standard_1020 channels, loaders, metrics
│   └── data_processor/              # HDF5 dataset classes
├── src/
│   ├── __init__.py                  # path setup + upstream monkey-patches
│   ├── shap_explainer.py            # LaBraMExplainer, checkpoint loading, data helpers
│   ├── text_reasoning.py            # build_prompt(), few-shot formatting
│   ├── llm_inference.py             # QwenInterpreter (HuggingFace causal-LM wrapper)
│   ├── pipeline.py                  # end-to-end orchestrator
│   └── llm_prompts/
│       └── few_shot_examples.json   # curated healthy/AD examples for in-context learning
├── requirements.txt
└── README.md
```

## Setup

```bash
conda create -n eeg python=3.10 -y
conda activate eeg
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Usage

### Full pipeline (EEG -> clinical report)

```bash
python -m src.pipeline \
    --checkpoint path/to/labram.pth \
    --eeg-file path/to/sample.h5 \
    --llm-model Qwen/Qwen2.5-0.5B-Instruct \
    --patient-meta '{"age": 72, "notes": "Progressive memory decline"}'
```

### SHAP explainer only

```bash
# With a real checkpoint and HDF5 data
python -m src.shap_explainer \
    --checkpoint path/to/labram.pth \
    --data path/to/dataset.hdf5 \
    --patient-age 72 \
    --patient-notes "Mild cognitive complaints"

# Demo mode (random weights, synthetic data)
python -m src.shap_explainer
```

### LLM interpreter only

```bash
python -m src.llm_inference
```

## Datasets

- https://www.kaggle.com/datasets/ucimachinelearning/eeg-alzheimers-dataset
- https://huggingface.co/datasets/Neurazum/Disorders_and_Diagnosis_EEG_Dataset-v4
- [OpenNeuro / LEAD](https://github.com/DL4mHealth/LEAD?tab=readme-ov-file)

## References

- **LaBraM**: [Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://arxiv.org/abs/2405.18765) ([code](https://github.com/935963004/LaBraM))
- **SHAP**: [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
