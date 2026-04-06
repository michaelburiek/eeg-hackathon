# EEG Hackathon

---

Train an [EEGConformer](https://braindecode.org/stable/generated/braindecode.models.EEGConformer.html) from scratch on the [LEAD open-source EEG dataset](https://github.com/DL4mHealth/LEAD), then generate a clinician-style natural language report for every subject using the Claude API.

**Target labels:**

- Alzheimer's Disease (AD)
- Frontotemporal Dementia (FTD)
- Cognitively Normal (CN)

**GPU compute:**
Training runs on the [KOA HPC cluster](https://koa.its.hawaii.edu) via `koa-cli` (bundled in `koa-cli/`).

---

## Full Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — DATA                                                     │
│  LEAD .dat memmaps → .npz window store → subject-level splits       │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — MODEL                                                    │
│  EEGConformer (braindecode) trained from scratch                    │
│  Input: 19 ch × 400 samples (2 s @ 200 Hz)                          │
│  Output: AD / FTD / CN logits + softmax probabilities               │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 3 — REASONING                                                │
│  Per-subject classification aggregated over all EEG windows          │
│  → Claude API → clinician-style natural language report per subject │
└─────────────────────────────────────────────────────────────────────┘
```

### Detailed Data Flow

```
LEAD ADFTD-RS dataset (88 subjects · 121,825 windows · 2 s · 200 Hz · 19 ch)
          │
          ▼  scripts/02_import_lead.py
  .npz window store — one file per subject + index.csv
          │
          ▼  scripts/03_create_splits.py
  Subject-level train / val / test split (70 / 15 / 15, stratified)
          │
          ├──────────────────────────────────────┐
          ▼                                      ▼
  scripts/04_train.py                   scripts/07_inference.py
  ┌─────────────────────┐               ┌──────────────────────────────┐
  │  1. Baseline eval   │               │  Load trained checkpoint     │
  │     (untrained)     │               │  Run ALL subjects through    │
  │  2. Train from      │               │  model (window by window)    │
  │     scratch         │               │  Aggregate per-subject:      │
  │  3. Final eval on   │               │  • primary prediction        │
  │     test set        │               │  • confidence score          │
  │  4. Save best.pt    │               │  • window distribution       │
  └──────────┬──────────┘               └──────────────┬───────────────┘
             │                                         │
             ▼                                         ▼  Claude API
  scripts/05_sync_to_sheets.py          src/reasoning/llm_reasoner.py
  Google Sheets experiment tracker      ┌──────────────────────────────┐
                                        │  Clinician-style report per  │
  scripts/06_push_to_hub.py             │  subject (markdown):         │
  HuggingFace Hub                       │  • Primary classification    │
  michaelburiek/                        │  • Confidence assessment     │
  eeg-hackathon-eegconformer            │  • Differential diagnosis    │
                                        │  • Supporting evidence       │
                                        │  • Limitations & uncertainty │
                                        └──────────────────────────────┘
                                                       │
                                                       ▼
                                        experiments/reports/
                                        ├── sub-0001_report.md
                                        ├── sub-0002_report.md
                                        └── inference_summary.csv
```

---

## Repository Layout

```
configs/
└── lead_train.yaml          # single config for everything

notebooks/
├── LEAD.ipynb               # dataset exploration — splits, class balance, signal peek
└── eeg_hackathon.ipynb      # end-to-end experiment walkthrough

scripts/
├── 01_download_lead.py      # download LEAD from Google Drive
├── 02_import_lead.py        # convert .dat files → .npz window store
├── 03_create_splits.py      # subject-level train / val / test splits
├── 04_train.py              # baseline eval → train → final eval
├── 05_sync_to_sheets.py     # push results to Google Sheets tracker
├── 06_push_to_hub.py        # upload trained model to HuggingFace Hub
├── 07_inference.py          # run all subjects → per-subject LLM reports
└── train.slurm              # KOA GPU job (wraps script 04)

src/
├── config.py                # YAML config loading with deep-merge
├── corpus/
│   ├── lead.py              # LEAD .dat memmap → .npz importer
│   └── splits.py            # train/val/test subject-level splitting
├── data/
│   └── dataset.py           # EEGDataset + window store loader
├── models/
│   └── eeg_conformer.py     # EEGConformer (braindecode, random init)
├── reasoning/
│   └── llm_reasoner.py      # SubjectClassification + Claude API report
├── training/
│   ├── trainer.py           # training loop, early stopping, checkpointing
│   └── losses.py            # label-smoothing and focal loss
├── evaluation/
│   └── metrics.py           # balanced accuracy, macro-F1, Cohen's κ, AUC
└── tracking.py              # optional W&B logging

data/
└── lead/
    ├── L100/                # 1-second windows (100 timestamps)
    │   ├── ADFSU/           # 4,048 segments · 19 ch · 92 subjects
    │   └── ADSZ/            # 1,128 segments · 19 ch · 48 subjects
    ├── L200/                # 2-second windows (200 timestamps)
    │   └── APAVA/           # 9,282 segments · 16 ch · 23 subjects
    └── L400/                # 4-second windows (400 timestamps)  ← primary
        ├── ADFTD-RS/        # 121,825 segments · 19 ch · 88 subjects (resting)
        ├── ADFTD-PS/        # 45,258 segments · 19 ch · 88 subjects (passive)
        └── ADFTD/           # 167,083 segments · 19 ch · 88 subjects (combined)

experiments/
├── checkpoints/             # best.pt (after step 4)
├── results/                 # results.json — baseline vs trained metrics
└── reports/                 # per-subject .md reports + summary.csv (step 7)

koa-cli/                     # bundled HPC tooling — provides the `koa` command
```

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd eeg_hackathon
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable                 | Required for                | Where to get it                                                          |
| ------------------------ | --------------------------- | ------------------------------------------------------------------------ |
| `ANTHROPIC_API_KEY`      | Step 7 — LLM reports        | [console.anthropic.com](https://console.anthropic.com)                   |
| `HF_TOKEN`               | Step 6 — HuggingFace upload | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `WANDB_API_KEY`          | Optional W&B tracking       | [wandb.ai/settings](https://wandb.ai/settings)                           |
| `GSHEETS_SA_JSON`        | Step 5 — Google Sheets      | Google Cloud → IAM → Service Accounts                                    |
| `GSHEETS_SPREADSHEET_ID` | Step 5 — Google Sheets      | From your sheet URL                                                      |

---

## KOA HPC Setup

All GPU-intensive work (step 4) runs as a Slurm job on KOA. `koa-cli` is bundled and installed automatically with `pip install -r requirements.txt`.

### One-time: configure credentials

```bash
koa setup
```

Interactive wizard — you will need your KOA username and SSH key path. KOA uses DUO MFA on every connection. The wizard writes `~/.config/koa-cli/config.yaml`:

```yaml
user: your_koa_username
host: koa.its.hawaii.edu
remote_workdir: ~/koa-jobs
remote_data_dir: /mnt/lustre/koa/scratch/your_koa_username/koa-jobs
identity_file: ~/.ssh/id_ed25519
```

### Verify connectivity

```bash
koa check
```

SSHes to KOA and runs `sinfo` to confirm Slurm and GPU nodes are reachable.

### Push API keys to the cluster

```bash
koa auth sync   # uploads your local .env to $KOA_PROJECT_DIR/.env (chmod 600)
koa auth check  # verify it arrived
```

Re-run `koa auth sync` whenever you rotate keys.

### Upload data to KOA scratch

Run steps 1–3 locally first to build the window store and splits, then upload once:

```bash
rsync -avz --progress \
  data/lead/window_store/ \
  data/lead/splits.json \
  your_koa_username@koa.its.hawaii.edu:/mnt/lustre/koa/scratch/your_koa_username/koa-jobs/eeg_hackathon/data/lead/
```

---

## End-to-End Walkthrough

Steps 1–3 run locally (CPU). Steps 4–7 can run locally or on KOA GPU.

---

### Step 1 — Download LEAD

```bash
python scripts/01_download_lead.py
```

Downloads from [Google Drive](https://drive.google.com/drive/folders/1y66f_Id-kal7q8uu-YYF2qTUHfhbPXOX) using `gdown`. You can also download manually.

Expected layout after download:

```
data/lead/
├── L100/
│   ├── ADFSU/  ─┐
│   └── ADSZ/    ├─ each contains: X.dat · y.dat · meta.json
├── L200/        │
│   └── APAVA/ ─┘
└── L400/
    ├── ADFTD-RS/
    ├── ADFTD-PS/
    └── ADFTD/
```

Each folder's files:

- `X.dat` — EEG windows, float32 memmap, shape `[N, T, C]`
- `y.dat` — float32 memmap, shape `[N, 3]`: `[label, subject_id, sampling_rate]`
- `meta.json` — N, T, C, sampling rates, overlap, step

---

### Step 2 — Import to window store

```bash
python scripts/02_import_lead.py --config configs/lead_train.yaml
```

Reads `.dat` memmaps, groups windows by subject, writes one `.npz` per subject and an `index.csv`.

```
data/window_store/lead/
├── ADFTD-RS/
│   ├── sub-0001.npz  ...  sub-0088.npz
└── index.csv
```

---

### Step 3 — Create subject-level splits

```bash
python scripts/03_create_splits.py --config configs/lead_train.yaml
```

Stratified 70 / 15 / 15 split at the **subject level** — no subject's windows appear in more than one partition. Saves `data/lead/splits.json`.

---

### Step 4 — Baseline → Train → Evaluate

**Locally (CPU / MPS):**

```bash
python scripts/04_train.py --config configs/lead_train.yaml
```

**On KOA GPU (recommended):**

```bash
koa run scripts/train.slurm --watch
```

What this does:

1. Builds EEGConformer with **random weights** — no pretrained checkpoint loaded
2. Evaluates untrained model on test set → **baseline metrics**
3. Trains on the training set with early stopping on val balanced-accuracy
4. Evaluates best checkpoint on test set → **final metrics**
5. Prints side-by-side comparison and saves `experiments/results/results.json`
6. Saves best checkpoint to `experiments/checkpoints/best.pt`

---

### Step 5 — Sync results to Google Sheets

```bash
python scripts/05_sync_to_sheets.py --results experiments/results/results.json
```

Updates the [experiment tracker spreadsheet](https://docs.google.com/spreadsheets/d/153L-tV94HCSjtcL00t9-UOHby5ilSzI1KDF98JNch3A) with baseline vs trained metrics. To reset headers (first run only):

```bash
python scripts/05_sync_to_sheets.py --reset-headers --results experiments/results/results.json
```

---

### Step 6 — Push trained model to HuggingFace

```bash
python scripts/06_push_to_hub.py \
    --checkpoint experiments/checkpoints/best.pt \
    --results    experiments/results/results.json
```

Uploads weights, `config.json`, `results.json`, and a model card to:
**[huggingface.co/michaelburiek/eeg-hackathon-eegconformer](https://huggingface.co/michaelburiek/eeg-hackathon-eegconformer)**

---

### Step 7 — Generate per-subject clinician reports

```bash
# All subjects (every EEG in the dataset gets a report):
python scripts/07_inference.py \
    --config     configs/lead_train.yaml \
    --checkpoint experiments/checkpoints/best.pt

# Test subjects only:
python scripts/07_inference.py \
    --config     configs/lead_train.yaml \
    --checkpoint experiments/checkpoints/best.pt \
    --split      test

# Single subject:
python scripts/07_inference.py \
    --config     configs/lead_train.yaml \
    --checkpoint experiments/checkpoints/best.pt \
    --subject    ADFTD-RS:sub-0001
```

For each subject this script:

1. Runs all EEG windows through the trained model (batch inference)
2. Aggregates window-level predictions into a per-subject classification summary
3. Calls the Claude API (`claude-sonnet-4-6`) with the structured classification data
4. Saves a markdown report to `experiments/reports/<subject_id>_report.md`

**Report sections (generated by Claude):**

- Primary Classification
- Confidence Assessment
- Differential Diagnosis
- Supporting Evidence
- Limitations & Uncertainty
- Clinical Disclaimer

**Output:**

```
experiments/reports/
├── ADFTD-RS_sub-0001_report.md
├── ADFTD-RS_sub-0002_report.md
├── ...
└── inference_summary.csv    ← one row per subject: prediction, confidence, correctness
```

> The LLM is grounded — the system prompt explicitly prohibits Claude from citing findings not present in the structured model output. It synthesises the classification data; it does not diagnose from raw EEG.

> **This is a research tool. Reports are not clinical diagnoses.**

---

## Dataset

**LEAD — Large-Scale EEG Dataset for Alzheimer's Disease and Related Dementias**

- GitHub: [DL4mHealth/LEAD](https://github.com/DL4mHealth/LEAD)
- Download: [Google Drive](https://drive.google.com/drive/folders/1y66f_Id-kal7q8uu-YYF2qTUHfhbPXOX)

### The full LEAD corpus

LEAD (Miltiadous et al., DL4mHealth) curates 18 datasets spanning **2,238 subjects and 427.81 hours** of EEG recordings for Alzheimer's Disease detection. It is divided into two roles:

**Pre-training corpus** (13 datasets · 4,646 subjects · 7,431,484 segments · 1,185.84 hrs)

Used to train the LEAD foundation model. Includes 4 AD datasets and 9 non-AD neurological/healthy datasets:

| Role               | Datasets                                                                                  |
| ------------------ | ----------------------------------------------------------------------------------------- |
| AD (pre-train)     | AD-Auditory, BrainLat, P-ADIC, CAUEEG                                                     |
| Non-AD (pre-train) | BACA-RS, Depression, FEPCR-1, FEPCR-2, MCEF-RS, PD-RS, PEARL-Neuro, SRM-RS, TDBrain, TUEP |

**Downstream fine-tuning corpus** (5 AD datasets · 440 subjects · 303,570 segments · 47.59 hrs)

Reserved for evaluation — not used in pre-training. These are the 5 datasets LEAD makes publicly available for download:

| Dataset | Task                              | Subjects | Available                   |
| ------- | --------------------------------- | -------- | --------------------------- |
| ADFTD   | AD / FTD / CN (resting + passive) | 88       | Public                      |
| ADFSU   | AD / CN                           | 92       | Public                      |
| ADSZ    | AD / CN                           | 48       | Public                      |
| APAVA   | AD / CN                           | 23       | Public                      |
| CNBPM   | AD / CN                           | ~189     | **Private — not available** |

### What this project uses

We use **4 of the 5 publicly available downstream datasets** (251 unique subjects). CNBPM is excluded as it is a private dataset not released by the authors.

Each dataset is preprocessed by LEAD into fixed-length windowed segments at multiple sampling rates and stored as binary memmaps (`X.dat`, `y.dat`, `meta.json`). The ADFTD dataset has two recording paradigms stored separately (RS = resting state, PS = passive stimulation) and also combined:

| Folder          | Source Dataset           | Subjects | Segments | Labels        | Channels | Sampling Rates  |
| --------------- | ------------------------ | -------- | -------- | ------------- | -------- | --------------- |
| `L400/ADFTD-RS` | ADFTD (resting)          | 88       | 121,825  | AD / FTD / CN | 19       | 50, 100, 200 Hz |
| `L400/ADFTD-PS` | ADFTD (passive)          | 88       | 45,258   | AD / FTD / CN | 19       | 50, 100, 200 Hz |
| `L400/ADFTD`    | ADFTD (RS + PS combined) | 88       | 167,083  | AD / FTD / CN | 19       | 50, 100, 200 Hz |
| `L100/ADFSU`    | ADFSU                    | 92       | 4,048    | AD / CN       | 19       | 50, 100 Hz      |
| `L100/ADSZ`     | ADSZ                     | 48       | 1,128    | AD / CN       | 19       | 50, 100 Hz      |
| `L200/APAVA`    | APAVA                    | 23       | 9,282    | AD / CN       | 16       | 50, 100, 200 Hz |

> ADFTD-RS and ADFTD-PS are the **same 88 subjects** recorded under different paradigms. ADFTD is their union. All three are stored separately to allow paradigm-specific training.

**Primary training target — `L400/ADFTD-RS`** (verified against `y.dat`):

| Property                | Value                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| Condition               | Resting state, eyes closed                                                                      |
| Subjects                | 88 — 29 AD · 36 FTD · 23 CN                                                                     |
| Channels                | 19 (10-20 system: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2) |
| Window length           | 400 timestamps (2 s @ 200 Hz · 4 s @ 100 Hz · 8 s @ 50 Hz)                                      |
| Total segments          | 121,825 (50% overlap)                                                                           |
| Segments by class       | AD: 42,067 · FTD: 50,826 · CN: 28,932                                                           |
| Label encoding in y.dat | 0 = AD · 1 = FTD · 2 = CN                                                                       |

---

## Model

**EEGConformer** from [braindecode](https://braindecode.org/stable/generated/braindecode.models.EEGConformer.html) — convolutional transformer trained from random initialization.

**Paper:** Song et al., "EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization", IEEE TNNLS 2023. [https://ieeexplore.ieee.org/document/10178045](https://ieeexplore.ieee.org/document/10178045)

Architecture: temporal conv → depthwise spatial conv → multi-head self-attention → classification head.

Default configuration (19 ch · 400 samples @ 200 Hz):

| Param                | Value   |
| -------------------- | ------- |
| Time filters         | 40      |
| Filter length        | 25      |
| Pool length / stride | 75 / 15 |
| Transformer layers   | 6       |
| Attention heads      | 10      |
| Dropout              | 0.5     |

---

## Training Config

Everything is in one file: [`configs/lead_train.yaml`](configs/lead_train.yaml).

Key sections:

```yaml
project:
  device: "cuda" # cuda | mps | cpu

dataset:
  name: "ADFTD-RS"
  label_map: { CN: 0, AD: 1, FTD: 2 }

splits:
  val_frac: 0.15
  test_frac: 0.15

training:
  epochs: 50
  lr: 1e-4
  optimizer: "adamw"
  lr_scheduler: "cosine"
  patience: 10 # early stopping

wandb:
  enabled: false # set true + WANDB_API_KEY to enable
```

---

## Evaluation Metrics

| Metric            | Why                                      |
| ----------------- | ---------------------------------------- |
| Balanced accuracy | Handles class imbalance — primary metric |
| Macro F1          | Equal weight to minority classes         |
| Cohen's κ         | Agreement above chance                   |
| One-vs-rest AUC   | Probabilistic calibration                |

---

## Citation

> [LEAD repository](https://github.com/DL4mHealth/LEAD)

> Song Y, et al. "EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization." _IEEE TNNLS_, 2023. https://doi.org/10.1109/TNNLS.2022.3230250
