#!/usr/bin/env bash
# ─── scripts/train.slurm ─────────────────────────────────────────────────────
# KOA Slurm job: train EEGConformer from scratch on the LEAD dataset.
#
# Usage (one-shot sync + submit):
#   koa run scripts/train.slurm --watch
#
# Override config or paths at submit time:
#   koa run scripts/train.slurm \
#     --sbatch-arg='--export=CONFIG=configs/lead_train.yaml'
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=eeg-lead-train
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# GPU is selected automatically by koa-cli. Pin explicitly if needed:
# #SBATCH --gres=gpu:a100:1

set -euo pipefail

# ─── Paths (injected by koa-cli at submit time) ───────────────────────────────
PROJECT_DIR="${KOA_PROJECT_DIR:-$HOME/koa-jobs/eeg_hackathon}"
DATA_DIR="${KOA_REMOTE_DATA_DIR:-$PROJECT_DIR/data}"

cd "$PROJECT_DIR"

# ─── Activate venv (bootstrap on first run) ───────────────────────────────────
VENV="$PROJECT_DIR/.venv"
if [ ! -d "$VENV" ]; then
  echo "[bootstrap] Creating venv at $VENV..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r requirements.txt
fi
source "$VENV/bin/activate"

# ─── Load API keys (optional) ─────────────────────────────────────────────────
[ -f "$PROJECT_DIR/.env" ] && set -a && source "$PROJECT_DIR/.env" && set +a

# ─── Arguments ────────────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/lead_train.yaml}"
INDEX_CSV="${INDEX_CSV:-$DATA_DIR/lead/window_store/index.csv}"
SPLITS_JSON="${SPLITS_JSON:-$DATA_DIR/lead/splits.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/experiments/results}"

# ─── Logging ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "Job:     $SLURM_JOB_ID ($SLURM_JOB_NAME)"
echo "Node:    $(hostname)"
echo "GPUs:    ${CUDA_VISIBLE_DEVICES:-none}"
echo "Config:  $CONFIG"
echo "Index:   $INDEX_CSV"
echo "Splits:  $SPLITS_JSON"
echo "Output:  $OUTPUT_DIR"
echo "Started: $(date)"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ─── Run ──────────────────────────────────────────────────────────────────────
python scripts/04_train.py \
  --config     "$CONFIG" \
  --index-csv  "$INDEX_CSV" \
  --splits-json "$SPLITS_JSON" \
  --output-dir "$OUTPUT_DIR"

echo "Finished: $(date)"
