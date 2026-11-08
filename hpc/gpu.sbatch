#!/bin/bash
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
# Generic GPU job: sbatch [opts] hpc/gpu.sbatch <command ...>
set -eo pipefail
source "$SLURM_SUBMIT_DIR/hpc/env.sh"
echo "=== $(date) job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0}) on $(hostname): $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
$GTIME -v "$@"
