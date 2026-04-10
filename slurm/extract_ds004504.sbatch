#!/bin/bash
#SBATCH --job-name=ds004504-biomarkers
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
# ds004504 has 88 subjects (sub-001 through sub-088):
#   sbatch --array=1-88 slurm/extract_ds004504.sbatch
#
# After all jobs finish, gather:
#   python slurm/gather.py --input-dir data/ds004504_biomarkers --out data/biomarkers_all_subjects.csv

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT="data/ds004504"
CONDA_ENV="gcm"

# ── Setup ─────────────────────────────────────────────────────────────────────
source /home/terrica/anaconda3/anaconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}

mkdir -p logs/slurm data/ds004504_biomarkers

# ── Run ───────────────────────────────────────────────────────────────────────
SUBJECT=$(printf "sub-%03d" ${SLURM_ARRAY_TASK_ID})

python scripts/biomarkers/extract_ds004504.py \
    --dataset ${DATASET_ROOT} \
    --subjects ${SUBJECT} \
    --out data/ds004504_biomarkers/${SUBJECT}.csv
