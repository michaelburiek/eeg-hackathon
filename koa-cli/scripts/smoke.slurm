#!/bin/bash
#SBATCH --job-name=koa-smoke
#SBATCH --time=00:05:00

set -euo pipefail

RESULTS_DIR="${KOA_REMOTE_DATA_DIR}/train/results/${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

cd "${KOA_PROJECT_DIR}"
echo "hello from koa-cli" > "${RESULTS_DIR}/smoke.txt"
