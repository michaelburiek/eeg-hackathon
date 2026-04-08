#!/bin/bash
#SBATCH --job-name=lead-biomarkers
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --error=./logs/slurm/%A_%a.err
#SBATCH --output=./logs/slurm/%A_%a.out
#SBATCH --mail-user=terrica@hawaii.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ── Usage ─────────────────────────────────────────────────────────────────────
# First, count subjects:
#   python -c "
#   import numpy as np, json
#   meta = json.load(open('data/lead/L400/ADFTD-RS/meta.json'))
#   y = np.fromfile('data/lead/L400/ADFTD-RS/y.dat', dtype=np.float32).reshape(meta['N'],3)
#   subs = np.unique(y[y[:,2]==200.0, 1])  # at max sfreq
#   print(f'Submit with: sbatch --array=0-{len(subs)-1} slurm/extract_lead.sbatch')
#   "
#
# Then submit:
#   sbatch --array=0-87 slurm/extract_lead.sbatch
#
# After all jobs finish, gather:
#   python slurm/gather.py --input-dir data/ADFTD-RS_biomarkers --out data/ADFTD-RS_biomarkers.csv

# ── Config ────────────────────────────────────────────────────────────────────
# Edit these for your dataset
LEAD_DATASET="data/lead/L400/ADFTD-RS"
CONDA_ENV="gcm"

# ── Setup ─────────────────────────────────────────────────────────────────────
source /home/terrica/anaconda3/anaconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}

mkdir -p logs/slurm

# ── Run ───────────────────────────────────────────────────────────────────────
python extract_lead.py \
    --dataset ${LEAD_DATASET} \
    --subject-idx ${SLURM_ARRAY_TASK_ID}
