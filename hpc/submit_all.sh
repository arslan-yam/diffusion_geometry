#!/bin/bash
# Submits every experiment as its own job, in priority order. Safe to re-run: finished runs are skipped
# (each script resumes from its results file), so after a crash just resubmit the same line.
cd "$(dirname "$0")/.." && source hpc/config.sh && mkdir -p logs results
[ "$ACCOUNT" = proj_XXXX ] && { echo "Run: mp, then: echo proj_NNNN > ~/.dg_account"; exit 1; }
G="-A $ACCOUNT --constraint=$GPU_CONSTRAINT"
C="-A $ACCOUNT --constraint=$CPU_CONSTRAINT"
gpu() { local name=$1 t=$2; shift 2; sbatch --parsable -J $name -o logs/$name.%j.out $G -t $t "${DEP[@]}" hpc/gpu.sbatch "$@"; }
cpu() { local name=$1 t=$2; shift 2; sbatch --parsable -J $name -o logs/$name.%j.out $C -t $t hpc/cpu.sbatch "$@"; }
DEP=()

# P1 core benchmark: one GPU job per manifold (3 dims x 3 seeds)
for man in sphere prod torus quadric; do
  gpu p1_$man 2-00:00:00 python exp1_synthetic.py --mans $man --dims 4,8,16 --ns 20000 --seeds 0,1,2 \
      --baselines swh,kernel --out results/exp1_core_$man.jsonl
done
# P1 authors' package (shipped + fixed Hessian) on CPU nodes: needs lots of RAM, not a GPU. Same data/queries (same seeds).
for man in sphere prod torus quadric; do
  cpu p1mdg_$man 3-00:00:00 python exp1_synthetic.py --mans $man --dims 4,8,16 --ns 20000 --seeds 0,1,2 \
      --baselines mdg_fixed,mdg_shipped --no_neural --device cpu --out results/exp1_mdg_$man.jsonl
done

# P2 pretrained CIFAR-10 diffusion model: classifier once, then test/train splits in parallel
CLF=$(gpu p2_clf 0-04:00:00 python exp3_pretrained_cifar.py --clf_only --split test)
DEP=(--dependency=afterok:$CLF)
for sp in test train; do
  gpu p2_$sp 3-00:00:00 python exp3_pretrained_cifar.py --split $sp --per_class 50 --sigmas 0.1,0.2,0.4 --out results/exp3_$sp.json
done
DEP=()

# P3 image manifolds with CNN / MLP denoisers
gpu p3_smooth_unet 3-00:00:00 python exp2_images.py --family smooth16 --ns 20000,100000 --noises 0,0.03 --arch unet --seeds 0,1,2 --out results/exp2_smooth16_unet.jsonl
gpu p3_smooth_mlp  2-00:00:00 python exp2_images.py --family smooth16 --ns 20000,100000 --noises 0,0.03 --arch mlp  --seeds 0,1,2 --out results/exp2_smooth16_mlp.jsonl
gpu p3_sharp_unet  3-00:00:00 python exp2_images.py --family sharp24  --ns 20000,200000 --noises 0 --arch unet --seeds 0,1 --out results/exp2_sharp24_unet.jsonl

# P4 sample-size sweep
for man in sphere torus; do for d in 8 16; do
  gpu p4_${man}$d 3-00:00:00 python exp1_synthetic.py --mans $man --dims $d --ns 2000,5000,20000,100000,500000 --seeds 0,1,2 \
      --baselines swh,kernel --out results/exp1_nsweep_${man}$d.jsonl
done; done

# P5 ambient noise in R^128
for nz in 0 0.02 0.05 0.1; do
  gpu p5_noise$nz 2-00:00:00 python exp1_synthetic.py --mans sphere,quadric --dims 4,8 --D 128 --noise $nz --ns 20000 --seeds 0,1,2 \
      --baselines swh,kernel --out results/exp1_noise_$nz.jsonl
done
squeue -u $USER
