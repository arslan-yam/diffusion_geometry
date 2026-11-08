#!/bin/bash
# Run ONCE on the login node (it has internet; compute nodes do not). Takes ~10-15 min.
#   bash hpc/setup_login.sh            # CUDA 12.4 build of torch (needs NVIDIA driver >= 525)
#   TORCH_CUDA=cu118 bash hpc/setup_login.sh   # if the GPU check reports an old driver
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"; source "$REPO/hpc/config.sh"
module purge; module load python
eval "$(conda shell.bash hook)"
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" --override-channels -c conda-forge python=3.11 time
fi
conda activate "$ENV_NAME"
if [ -n "$TORCH_CUDA" ]; then
  pip install --force-reinstall "torch==2.5.1" "torchvision==0.20.1" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
fi
pip install -r "$REPO/requirements.txt"
[ -d "$DG_REPO" ] || git clone https://github.com/Iolo-Jones/DiffusionGeometry.git "$DG_REPO"
source "$REPO/hpc/env.sh"
mkdir -p results logs data
# pre-download everything the compute nodes need
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("google/ddpm-cifar10-32"))
import torchvision, os
for tr in (True, False): torchvision.datasets.CIFAR10(os.environ["CIFAR_DIR"], train=tr, download=True)
import torch, diffusers, sys
sys.path.insert(0, os.environ["DG_REPO"]); import diffusion_geometry
print("OK  torch", torch.__version__, "| cuda build", torch.version.cuda, "| diffusers", diffusers.__version__)
PY
echo "Setup done. Next: bash hpc/smoke.sh"
