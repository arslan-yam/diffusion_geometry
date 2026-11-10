# Curvature from Denoising — GPU experiments

Code for the experiments of *Curvature from Denoising: Second-Order Diffusion Geometry in High Dimensions*.
`tc/` is the PyTorch implementation (Tweedie curvature estimator, manifolds with exact ground truth, EDM denoisers,
baselines including the authors' DiffusionGeometry package). `hpc/` runs everything on the HSE cHARISMa cluster.

## On the HSE cluster
```bash
# login node, once
git clone git@github.com:arslan-yam/diffusion_geometry.git ~/diffusion_geometry && cd ~/diffusion_geometry
mp && echo proj_NNNN > ~/.dg_account   # your project ID
bash hpc/setup_login.sh            # conda env, packages, original package, model + CIFAR-10 download
bash hpc/smoke.sh                  # two <30 min test jobs; check logs/smoke_*.out
bash hpc/submit_all.sh             # 22 jobs (18 GPU, 4 CPU)
bash hpc/progress.sh               # status
```
Every script appends to its results file and skips finished runs, so a failed, timed-out or requeued job is
fixed by resubmitting the same command (the lines are in `hpc/submit_all.sh`).

## Experiments
| Priority | Jobs | Question |
|---|---|---|
| P1 | `p1_<man>` (GPU), `p1mdg_<man>` (CPU) | Does the d=4/8 advantage hold over 3 seeds, against the authors' own code (shipped and Hessian-fixed)? |
| P2 | `p2_clf` → `p2_test`, `p2_train` | Does it work on a pretrained diffusion model (google/ddpm-cifar10-32)? Does curvature predict classifier ambiguity? |
| P3 | `p3_*` | Do convolutional denoisers fix the image-manifold negatives (smooth16 vs SWH, sharp24)? |
| P4 | `p4_<man><d>` | Kernel-to-neural crossover in n; does d=16 become solvable at n=100k–500k? |
| P5 | `p5_noise<level>` | Noise robustness across seeds in R^128 |

## Elsewhere (any single GPU)
```bash
pip install -r requirements.txt
git clone https://github.com/Iolo-Jones/DiffusionGeometry && export DG_REPO=$PWD/DiffusionGeometry
python exp1_synthetic.py --mans sphere --dims 2 --ns 20000 --seeds 0 --steps 4000 --width 256 --depth 4 --lr 1e-3 --m 20 --baselines swh,mdg_fixed,mdg_shipped --out results/sanity.jsonl
```
Expect neural_auto nMAE ≈ 0.004, mdg_fixed ≈ 0.06–0.09, mdg_shipped about 4× worse.

Notes: derivatives are evaluated in float64 and TF32 is disabled (it corrupts second derivatives).
Diffusers' fused attention has no forward-mode AD, so `DDPMAsDenoiser` switches to the classic attention processor.
