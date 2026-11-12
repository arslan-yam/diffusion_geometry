#!/bin/bash
# Round 2: fixes and additions after the first full run (see README "Round 2").
# Every job writes NEW result files (suffix _r2), so round-1 results are never touched.
# Safe to re-run: finished runs are skipped.
cd "$(dirname "$0")/.." && source hpc/config.sh && mkdir -p logs results
[ "$ACCOUNT" = proj_XXXX ] && { echo "Run: mp, then: echo proj_NNNN > ~/.dg_account"; exit 1; }
G="-A $ACCOUNT --constraint=$GPU_CONSTRAINT"
gpu() { local name=$1 t=$2; shift 2; sbatch --parsable -J $name -o logs/$name.%j.out $G -t $t hpc/gpu.sbatch "$@"; }

PILOT="--width 512 --depth 4 --lr 1e-3 --train_smin 0.01 --train_smax 2.0 --tag pilot"   # recipe that learned T^8 in the CPU pilot
LONG="--steps 60000 --tag long"                                                       # default net, 3x longer training

# R1/R2: training collapse on flat tori and S^16 (core benchmark setting, n=20k, 3 seeds). Baselines already exist.
for man in torus sphere; do
  dims=$([ $man = torus ] && echo 8,16 || echo 16)
  gpu r1_${man}_pilot 1-00:00:00 python exp1_synthetic.py --mans $man --dims $dims --ns 20000 --seeds 0,1,2 --baselines none \
      $PILOT --out results/exp1_core_${man}_pilot_r2.jsonl
  gpu r2_${man}_long  1-00:00:00 python exp1_synthetic.py --mans $man --dims $dims --ns 20000 --seeds 0,1,2 --baselines none \
      $LONG --out results/exp1_core_${man}_long_r2.jsonl
done

# R3: S^16 sample-size sweep with the pilot recipe
gpu r3_nsweep_s16_pilot 2-00:00:00 python exp1_synthetic.py --mans sphere --dims 16 --ns 2000,5000,20000,100000,500000 --seeds 0,1,2 \
    --baselines none $PILOT --out results/exp1_nsweep_sphere16_pilot_r2.jsonl

# R4: first-order "curvature from the Jacobian spectrum" (trace corollary) on the other families, default recipe, 1 seed
gpu r4_trace 1-00:00:00 python exp1_synthetic.py --mans sphere,prod,quadric --dims 4,8,16 --ns 20000 --seeds 0 --baselines none \
    --tag trace --out results/exp1_trace_r2.jsonl

# R5: image manifolds with the UNet, now logging S at every noise level so the stability rule can be applied
gpu r5_smooth_unet 3-00:00:00 python exp2_images.py --family smooth16 --ns 20000,100000 --noises 0,0.03 --arch unet --seeds 0,1,2 \
    --out results/exp2_smooth16_unet_r2.jsonl
gpu r5_sharp_unet  3-00:00:00 python exp2_images.py --family sharp24 --ns 20000,200000 --noises 0 --arch unet --seeds 0,1 \
    --out results/exp2_sharp24_unet_r2.jsonl

# R6: pretrained CIFAR-10 DDPM with a larger eigenvalue budget (k=96 saturated) and the Jacobian trace (LID) logged
for sp in test train; do
  gpu r6_cifar_$sp 3-00:00:00 python exp3_pretrained_cifar.py --split $sp --per_class 50 --sigmas 0.1,0.2,0.4 --k 160 \
      --out results/exp3_${sp}_k160_r2.json
done
squeue -u $USER
