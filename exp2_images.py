"""Exp 2 (GPU): rendered image manifolds with exact curvature, CNN (periodic UNet) vs MLP
denoiser vs SWH, sample-size sweep and pixel noise; plus curvature-corrected extrapolation.

Families:
  smooth16: 16x16, widths in [0.12,0.24]  (paper Sec. 4.4; MLP lost to SWH on clean data)
  sharp24 : 24x24, widths in [0.06,0.12]  (paper negative result 1; every method failed)

  python exp2_images.py --family smooth16 --ns 20000,100000 --noises 0,0.03 --arch unet,mlp --seeds 0,1,2
"""
import argparse, json, os, sys, time, math
import numpy as np
import torch
from torch.func import jacfwd, vmap
sys.path.insert(0, os.path.dirname(__file__))
from tc.manifolds import frame_curvature
from tc.models import ResMLP, PeriodicUNet, train_edm
from tc.curvature import tweedie_curvature
from tc.baselines import swh_curvature

p = argparse.ArgumentParser()
p.add_argument("--family", default="smooth16")
p.add_argument("--ns", default="20000,100000")
p.add_argument("--noises", default="0,0.03")
p.add_argument("--arch", default="unet,mlp")
p.add_argument("--seeds", default="0,1,2")
p.add_argument("--steps", type=int, default=40000)
p.add_argument("--m", type=int, default=200)
p.add_argument("--out", default="results/exp2.jsonl")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
dev = a.device
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
FAM = {"smooth16": (16, 0.12, 0.24), "sharp24": (24, 0.06, 0.12)}
res_, smin, smax = FAM[a.family]
u = torch.arange(res_, dtype=torch.float64) / res_ + 0.5 / res_


def render(th):                                   # th = (cx, cy, sx, sy)
    g = lambda c, s: torch.exp((torch.cos(2 * math.pi * (u.to(th) - c)) - 1) / (2 * math.pi * s) ** 2)
    return (g(th[0], th[2])[:, None] * g(th[1], th[3])[None, :]).reshape(-1)


def sample_theta(rng, n, interior=False):
    lo, hi = (smin + 0.15 * (smax - smin), smax - 0.15 * (smax - smin)) if interior else (smin, smax)
    return np.stack([rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(lo, hi, n), rng.uniform(lo, hi, n)], 1)


def render_batch(th):
    return vmap(render)(torch.as_tensor(th, dtype=torch.float64)).numpy()


def gt(th):
    S, II, E = [], [], []
    for t in torch.as_tensor(th, dtype=torch.float64):
        s, ii, e = frame_curvature(jacfwd(render)(t), jacfwd(jacfwd(render))(t)); S.append(s); II.append(ii); E.append(e)
    return torch.stack(S).numpy(), torch.stack(II).numpy(), torch.stack(E).numpy()


def dist_to_M(x, th0, iters=40):
    th = torch.as_tensor(th0, dtype=torch.float64).clone()
    for _ in range(iters):
        r = render(th) - x; J = jacfwd(render)(th)
        th = th - torch.linalg.solve(J.T @ J + 1e-10 * torch.eye(4, dtype=th.dtype), J.T @ r)
    return float((render(th) - x).norm())


SIG = [0.01 * math.sqrt(2) ** k for k in range(12)]
for seed in map(int, a.seeds.split(",")):
    for n in map(int, a.ns.split(",")):
        for noise in map(float, a.noises.split(",")):
            rng = np.random.default_rng(seed)
            X = render_batch(sample_theta(rng, n)); X += noise * rng.standard_normal(X.shape)
            thq = sample_theta(rng, a.m, interior=True); Qc = render_batch(thq)
            Qy = Qc + noise * rng.standard_normal(Qc.shape)
            Sgt, IIgt, Egt = gt(thq)
            Hg = np.einsum("niil->nl", IIgt); scale = float((np.sum(Hg**2, 1) + np.sum(IIgt**2, (1, 2, 3))).mean())
            base = dict(family=a.family, n=n, noise=noise, seed=seed, S_gt_mean=float(Sgt.mean()), S_gt_sd=float(Sgt.std()), scale=scale)
            # SWH (oracle k)
            best = None
            for k in [16, 32, 64, 128, 256, 512]:
                S = swh_curvature(X, Qy, 4, k); r = float(np.corrcoef(S, Sgt)[0, 1])
                o = dict(k=k, mae=float(np.abs(S - Sgt).mean()), r=r)
                if best is None or o["r"] > best["r"]: best = o
            with open(a.out, "a") as f: f.write(json.dumps(dict(base, method="swh_oracle", **best)) + "\n")
            print(base, "SWH", best, flush=True)
            for arch in a.arch.split(","):
                net = PeriodicUNet(ch=64, res=res_) if arch == "unet" else ResMLP(res_ * res_, 1024, 6)
                t = time.time()
                Dn = train_edm(net, X, a.steps, batch=256, lr=3e-4, sigma_range=(0.005, 1.0), device=dev, seed=seed).double()
                ttr = time.time() - t
                Yq = torch.tensor(Qy, dtype=torch.float64, device=dev)
                if noise > 0:
                    with torch.no_grad():
                        for _ in range(3): Yq = Dn(Yq, noise)
                grid = {}
                for s in SIG:
                    outs = [tweedie_curvature(lambda y: Dn(y, s), y, 4) for y in Yq]
                    S = np.array([o["S_sp"].item() for o in outs]); gap = float(np.median([o["gap"].item() for o in outs]))
                    Ef = np.stack([o["E"].cpu().numpy() for o in outs]); IIe = np.stack([o["II"].cpu().numpy() for o in outs])
                    cos2 = float(np.mean(np.sum(np.einsum("nai,naj->nij", Ef, Egt) ** 2, (1, 2)) / 4))
                    He = np.linalg.norm(np.einsum("niil->nl", IIe), axis=1)
                    grid[s] = dict(gap=gap, mae=float(np.abs(S - Sgt).mean()), r=float(np.corrcoef(S, Sgt)[0, 1]),
                                   H_relerr=float(np.median(np.abs(He - np.linalg.norm(Hg, axis=1)) / np.linalg.norm(Hg, axis=1))),
                                   tan_cos2=cos2, S=S.tolist(), _II=IIe, _E=Ef)
                sa = max(grid, key=lambda s: grid[s]["gap"])
                # ground-truth-free stability rule (Sec. 3.9): argmin_s mean|S(s) - S(sqrt2 s)| over valid levels
                ks = sorted(grid)
                stab = {ks[i]: float(np.mean(np.abs(np.array(grid[ks[i]]["S"]) - np.array(grid[ks[i + 1]]["S"]))))
                        for i in range(len(ks) - 1) if grid[ks[i]]["gap"] > 0.25}
                st = min(stab, key=stab.get) if stab else sa
                # curvature-corrected extrapolation along the TRUE tangent direction at the auto level
                ts = np.geomspace(0.05, 1.2, 8); D1, D2 = [], []
                rr = np.random.default_rng(1)
                for q in range(min(60, a.m)):
                    z = rr.standard_normal(4); z /= np.linalg.norm(z)
                    vt = Egt[q] @ z; zE = grid[sa]["_E"][q].T @ vt
                    IIvv = np.einsum("i,j,ijl->l", zE, zE, grid[sa]["_II"][q])
                    x = Qc[q]
                    D1.append([dist_to_M(torch.tensor(x + t * vt), thq[q]) for t in ts])
                    D2.append([dist_to_M(torch.tensor(x + t * vt + 0.5 * t**2 * IIvv), thq[q]) for t in ts])
                clean = {f"{s:.4f}": {k: v for k, v in g.items() if not k.startswith("_")} for s, g in grid.items()}
                rec = dict(base, method=f"neural_{arch}", sigma_auto=sa, sigma_stab=st, train_time=ttr, S_gt=np.asarray(Sgt).tolist(),
                           stab={k: v for k, v in grid[st].items() if not k.startswith("_") and k != "S"},
                           auto={k: v for k, v in grid[sa].items() if not k.startswith("_")}, grid=clean,
                           extrap_t=ts.tolist(), extrap_first=np.median(D1, 0).tolist(), extrap_second=np.median(D2, 0).tolist())
                with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
                print(base, arch, rec["auto"], flush=True)
