#!/bin/bash
# Two short jobs (<30 min): checks the GPU stack, the pretrained-model path and the original package.
cd "$(dirname "$0")/.." && source hpc/config.sh && mkdir -p logs results/smoke
[ "$ACCOUNT" = proj_XXXX ] && { echo "Run: mp, then: echo proj_NNNN > ~/.dg_account"; exit 1; }
G="-A $ACCOUNT --constraint=$GPU_CONSTRAINT -t 00:30:00"
sbatch -J smoke_gpu -o logs/smoke_gpu.%j.out $G hpc/gpu.sbatch bash -c '
  python -c "import torch; print(\"cuda ok:\", torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  python exp1_synthetic.py --mans sphere --dims 2 --ns 20000 --seeds 0 --steps 4000 --width 256 --depth 4 --lr 1e-3 --m 20 --baselines swh,kernel --out results/smoke/exp1.jsonl
  python exp3_pretrained_cifar.py --per_class 1 --split test --sigmas 0.2 --k 24 --iters 2 --m 4 --clf_epochs 1 --clf_path results/smoke/clf.pt --out results/smoke/exp3.json
  python exp2_images.py --family smooth16 --ns 2000 --noises 0 --arch unet --seeds 0 --steps 500 --m 5 --out results/smoke/exp2.jsonl'
sbatch -J smoke_cpu -o logs/smoke_cpu.%j.out -A $ACCOUNT -p test -t 00:30:00 hpc/cpu.sbatch \
  python exp1_synthetic.py --mans sphere --dims 2 --ns 2000 --seeds 0 --m 20 --baselines mdg_fixed,mdg_shipped --no_neural --device cpu --out results/smoke/mdg.jsonl
squeue -u $USER
