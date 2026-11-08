#!/bin/bash
#SBATCH -c 8
#SBATCH -t 3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
# Generic CPU job: sbatch [opts] hpc/cpu.sbatch <command ...>
set -eo pipefail
source "$SLURM_SUBMIT_DIR/hpc/env.sh"
echo "=== $(date) job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0}) on $(hostname): $*"
$GTIME -v "$@"
