# Sourced by every job (and by setup). Activates the env and sets offline caches.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/hpc/config.sh"
if type module > /dev/null 2>&1; then module purge > /dev/null 2>&1; module load python > /dev/null 2>&1; fi
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
export GTIME=$CONDA_PREFIX/bin/time
export HF_HOME="$REPO/.cache/hf"               # model weights are pre-downloaded here
export TORCH_HOME="$REPO/.cache/torch"
export CIFAR_DIR="$REPO/data"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
if [ -n "$SLURM_JOB_ID" ]; then export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1; fi   # compute nodes have no internet
cd "$REPO"
