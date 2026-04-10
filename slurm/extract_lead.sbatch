#!/bin/bash
#SBATCH --job-name=lead-biomarkers
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --error=./logs/slurm/%j.err
#SBATCH --output=./logs/slurm/%j.out
#SBATCH --mail-user=terrica@hawaii.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ── Usage ─────────────────────────────────────────────────────────────────────
# One job per dataset (all subjects processed serially):
#
#   sbatch --export=LEAD_DATASET=data/lead/L400/ADFTD-RS  slurm/extract_lead.sbatch
#   sbatch --export=LEAD_DATASET=data/lead/L400/ADFTD-PS  slurm/extract_lead.sbatch
#   sbatch --export=LEAD_DATASET=data/lead/L400/ADFTD     slurm/extract_lead.sbatch
#   sbatch --export=LEAD_DATASET=data/lead/L200/APAVA     slurm/extract_lead.sbatch
#   sbatch --export=LEAD_DATASET=data/lead/L100/ADSZ      slurm/extract_lead.sbatch
#   sbatch --export=LEAD_DATASET=data/lead/L100/ADFSU     slurm/extract_lead.sbatch

# ── Config ────────────────────────────────────────────────────────────────────
CONDA_ENV="eeg312"

# Validate LEAD_DATASET was passed
if [ -z "${LEAD_DATASET}" ]; then
    echo "ERROR: LEAD_DATASET not set. Submit with --export=LEAD_DATASET=data/lead/L400/ADFTD-RS"
    exit 1
fi

# ── Setup ─────────────────────────────────────────────────────────────────────
source /home/terrica/anaconda3/anaconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}

mkdir -p logs/slurm

# ── Run ───────────────────────────────────────────────────────────────────────
python scripts/biomarkers/extract_lead.py \
    --dataset ${LEAD_DATASET} \
    --sfreq 100