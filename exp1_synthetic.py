"""Exp 1 (GPU): synthetic benchmark with seeds and a sample-size sweep.

For each (manifold, d, n, seed): train an EDM-MLP denoiser, evaluate Tweedie curvature
(raw / spectral / Hutchinson) at 200 held-out points over a noise grid, select the noise
level by max spectral gap, and run baselines (SWH, kernel Tweedie, original
DiffusionGeometry package with shipped and fixed Hessian).  One JSON line per run.

Example (full grid ~ 10-20 GPU-hours; see README_GPU.md for subsets):
  python exp1_synthetic.py --mans sphere,prod,torus,quadric --dims 2,4,8,12,16 \
      --ns 2000,5000,20000,100000,500000 --seeds 0,1,2 --out results/exp1.jsonl
"""
import argparse, json, os, sys, time, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from tc.manifolds import make, random_isometry
from tc.models import ResMLP, train_edm
from tc.curvature import tweedie_curvature, full_frame
from tc.baselines import swh_curvature, kernel_denoiser, original_mdg

p = argparse.ArgumentParser()
p.add_argument("--mans", default="sphere,prod,torus,quadric")
p.add_argument("--dims", default="4,8,16")
p.add_argument("--ns", default="20000")
p.add_argument("--seeds", default="0,1,2")
p.add_argument("--D", type=int, default=0, help="ambient dim (0 = native)")
p.add_argument("--noise", type=float, default=0.0)
p.add_argument("--m", type=int, default=200)
p.add_argument("--steps", type=int, default=0, help="0 = auto: max(20k, 4*n/batch)")
p.add_argument("--width", type=int, default=1024)
p.add_argument("--depth", type=int, default=6)
p.add_argument("--train_smax", type=float, default=1.0)
p.add_argument("--train_smin", type=float, default=0.005)
p.add_argument("--tag", default="", help="recipe label stored in every record")
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--baselines", default="swh,kernel,mdg_fixed,mdg_shipped")
p.add_argument("--dtype", default="float64", help="dtype for derivative evaluation")
p.add_argument("--out", default="results/exp1.jsonl")
p.add_argument("--no_neural", action="store_true", help="baselines only (e.g. the original package on a CPU node)")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
dev = a.device; edt = getattr(torch, a.dtype)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
SIG = [0.01 * math.sqrt(2) ** k for k in range(11)]          # 0.01 .. 0.32
done = set()
if os.path.exists(a.out):
    for line in open(a.out):
        r = json.loads(line); done.add((r["man"], r["d"], r["n"], r["seed"], r["D"], r["noise"]))


def summ(S, Sgt, scale):
    err = S - Sgt
    o = {"mae": float(np.abs(err).mean()), "nmae": float(np.abs(err).mean() / scale), "bias": float(err.mean())}
    if Sgt.std() > 1e-6 and S.std() > 0: o["r"] = float(np.corrcoef(S, Sgt)[0, 1])
    if abs(Sgt.mean()) > 1e-6: o["rel"] = float(np.abs(err).mean() / abs(Sgt.mean()))
    return o


def eval_tweedie(Dsig, Y, d, label, Hgt=None):
    """Hgt: true mean-curvature vectors (n x D). Also evaluates the derivative-free mean curvature
    H_df = (2/eps) P_N (D(x) - x) and the log-density gradient |grad log q| = |P_T (D(x) - x)| / eps."""
    res = {}
    for s in SIG:
        Dfn = lambda y: Dsig(y, s)
        Ss, Sr, gaps, Hdf, Hsp, gq, Str, S12 = [], [], [], [], [], [], [], []
        t = time.time()
        for y in Y:
            o = tweedie_curvature(Dfn, y, d)
            Ss.append(o["S_sp"].item()); Sr.append(o["S_raw"].item()); gaps.append(o["gap"].item())
            with torch.no_grad():
                r = Dfn(y) - y; E = o["E"]; rT = E @ (E.T @ r)
            Hdf.append(((2 / s**2) * (r - rT)).cpu().numpy()); gq.append(rT.norm().item() / s**2)
            Hsp.append(torch.einsum("iil->l", o["II"]).detach().cpu().numpy())
            # first-order curvature from the Jacobian spectrum (uniform density): tr G = d - eps*S/2 + ...,
            # and S = 4|N s|^2 - (2/eps) sum_{i>d} lam_i with s = (D(y)-y)/eps  (only the full spectrum is used)
            lam = o["lam"]; e2 = s**2
            Str.append((2 * (d - lam.sum()) / e2).item())
            S12.append((4 * ((r - rT).norm() / e2) ** 2 - 2 * lam[d:].sum() / e2).item())
        res[s] = dict(S_sp=np.array(Ss), S_raw=np.array(Sr), S_tr=np.array(Str), S_12=np.array(S12),
                      gap=float(np.median(gaps)), time=time.time() - t,
                      gradlogq=float(np.median(gq)))
        if Hgt is not None:
            sc = np.linalg.norm(Hgt, axis=1).mean() + 1e-12
            res[s]["H_df_err"] = float(np.linalg.norm(np.array(Hdf) - Hgt, axis=1).mean() / sc)
            res[s]["H_sp_err"] = float(np.linalg.norm(np.array(Hsp) - Hgt, axis=1).mean() / sc)
    return res


for man in a.mans.split(","):
    for d in map(int, a.dims.split(",")):
        M = make(man, d)
        D = a.D or M.D0
        for n in map(int, a.ns.split(",")):
            for seed in map(int, a.seeds.split(",")):
                key = (man, d, n, seed, D, a.noise)
                if key in done: continue
                rng = np.random.default_rng(seed)
                Q = random_isometry(M.D0, D, seed)
                Xn = M.sampler(rng, n); X = Xn @ Q.T + a.noise * rng.standard_normal((n, D))
                mm = int(a.m * 1.5 / 0.6 ** d) + 10 if man == "quadric" else a.m
                Qn = M.sampler(rng, mm)
                if man == "quadric": Qn = Qn[np.linalg.norm(Qn[:, :d], axis=1) < 0.6][: a.m]
                Qc = Qn @ Q.T; Qy = Qc + a.noise * rng.standard_normal(Qc.shape)
                Sgt, IIgt, _ = M.gt(torch.tensor(Qn, dtype=torch.float64))
                Sgt = Sgt.numpy(); IIgt = IIgt.numpy()
                Hg = np.einsum("niil->nl", IIgt)
                HgD = Hg @ Q.T                                    # true mean-curvature vectors in R^D
                scale = float((np.sum(Hg**2, 1) + np.sum(IIgt**2, (1, 2, 3))).mean())
                R = dict(man=man, d=d, n=n, seed=seed, D=D, noise=a.noise, tag=a.tag, width=a.width, depth=a.depth, lr=a.lr, S_gt_mean=float(Sgt.mean()), scale=scale, methods={})
                print(f"== {man} d={d} D={D} n={n} seed={seed}", flush=True)
                # ---------------- baselines
                bl = a.baselines.split(",")
                if "swh" in bl:
                    best = None
                    for k in [16, 32, 64, 128, 256, 512, 1024, 2048]:
                        if k < 3 + d + d * (d + 1) // 2 or k > n // 2: continue
                        o = summ(swh_curvature(X, Qy, d, k), Sgt, scale); o["k"] = k
                        if best is None or o["mae"] < best["mae"]: best = o
                    R["methods"]["swh_oracle"] = best
                for fixed in (True, False):
                    tag = "mdg_fixed" if fixed else "mdg_shipped"
                    if tag in bl and D <= 20 and n <= 20000:   # original package needs ~n*D^3 memory
                        try:
                            S = original_mdg(np.vstack([X, Qy]), np.arange(n, n + len(Qy)), d, fixed=fixed)
                            R["methods"][tag] = summ(S, Sgt, scale)
                        except Exception as e:
                            R["methods"][tag] = {"error": repr(e)[:200]}
                if "kernel" in bl and n <= 200000:
                    Kd = kernel_denoiser(torch.tensor(X, dtype=edt, device=dev))
                    res = eval_tweedie(Kd, torch.tensor(Qy, dtype=edt, device=dev), d, "kernel", HgD)
                    valid = {s: r for s, r in res.items() if r["gap"] > 0.25}
                    if valid:
                        so = min(valid, key=lambda s: np.abs(valid[s]["S_sp"] - Sgt).mean())
                        R["methods"]["kernel_oracle"] = dict(summ(valid[so]["S_sp"], Sgt, scale), sigma=so)
                        ks = sorted(res)
                        stab = {ks[i]: np.mean(np.abs(res[ks[i]]["S_sp"] - res[ks[i + 1]]["S_sp"])) for i in range(len(ks) - 1) if ks[i] in valid}
                        if stab:
                            st = min(stab, key=stab.get); R["methods"]["kernel_stab"] = dict(summ(res[st]["S_sp"], Sgt, scale), sigma=st)
                # ---------------- neural
                if a.no_neural:
                    with open(a.out, "a") as f: f.write(json.dumps(R) + "\n")
                    print("   ", {k: v.get("nmae") for k, v in R["methods"].items()}, flush=True)
                    continue
                t = time.time()
                steps = a.steps or max(20000, 4 * n // 512)
                net = ResMLP(D, a.width, a.depth)
                Dnet = train_edm(net, X, steps, lr=a.lr, sigma_range=(a.train_smin, a.train_smax), device=dev, seed=seed)
                ttrain = time.time() - t
                Dnet = Dnet.to(edt)
                with torch.no_grad():
                    Yq = torch.tensor(Qy, dtype=edt, device=dev)
                    if a.noise > 0:                               # project noisy queries
                        for _ in range(3): Yq = Dnet(Yq, a.noise)
                res = eval_tweedie(lambda y, s: Dnet(y, s), Yq, d, "neural", HgD)
                valid = {s: r for s, r in res.items() if r["gap"] > 0.25}
                if valid:
                    sa = max(valid, key=lambda s: valid[s]["gap"])
                    so = min(valid, key=lambda s: np.abs(valid[s]["S_sp"] - Sgt).mean())
                    R["methods"]["neural_auto"] = dict(summ(valid[sa]["S_sp"], Sgt, scale), sigma=sa, gap=valid[sa]["gap"])
                    R["methods"]["neural_oracle"] = dict(summ(valid[so]["S_sp"], Sgt, scale), sigma=so)
                    R["methods"]["neural_raw_auto"] = dict(summ(valid[sa]["S_raw"], Sgt, scale), sigma=sa)
                    ks = sorted(res)                               # stability rule: argmin_s mean|S(s) - S(sqrt2 s)| (no ground truth)
                    stab = {ks[i]: np.mean(np.abs(res[ks[i]]["S_sp"] - res[ks[i + 1]]["S_sp"])) for i in range(len(ks) - 1) if ks[i] in valid}
                    if stab:
                        st = min(stab, key=stab.get)
                        R["methods"]["neural_stab"] = dict(summ(res[st]["S_sp"], Sgt, scale), sigma=st)
                    R["methods"]["neural_byproducts_auto"] = {k: valid[sa].get(k) for k in ("H_df_err", "H_sp_err", "gradlogq")}
                R["methods"]["neural_grid"] = {f"{s:.4f}": dict(gap=r["gap"], H_df_err=r.get("H_df_err"), H_sp_err=r.get("H_sp_err"),
                                                         gradlogq=r["gradlogq"], **summ(r["S_sp"], Sgt, scale),
                                                         raw_nmae=summ(r["S_raw"], Sgt, scale)["nmae"],
                                                         tr_nmae=summ(r["S_tr"], Sgt, scale)["nmae"],
                                                         s12_nmae=summ(r["S_12"], Sgt, scale)["nmae"]) for s, r in res.items()}
                R["train_time"] = ttrain; R["steps"] = steps
                with open(a.out, "a") as f: f.write(json.dumps(R) + "\n")
                print("   ", {k: (v.get("nmae") if isinstance(v, dict) else None) for k, v in R["methods"].items() if k != "neural_grid"}, flush=True)
