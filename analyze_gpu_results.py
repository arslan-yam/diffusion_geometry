"""Tables and figures from the GPU cluster runs (results in /home/claude/cres)."""
import sys, json, glob, collections
sys.path.insert(0, "/home/claude/ndg/exp")
import numpy as np
from scipy.stats import spearmanr, rankdata, mannwhitneyu
from sklearn.metrics import roc_auc_score
from fig_style import setup, C, METHOD_COLOR, METHOD_LABEL, plt

R = "/home/claude/cres/"
FIG = "/home/claude/ndg/figs/"
PAP = "/home/claude/ndg/paper/"
setup()
load = lambda f: [json.loads(l) for l in open(f)]
OUT = {}


def ms(v):
    v = np.asarray(v, float); return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def fmt(v, bold=False):
    if not v: return "--"
    m, s = ms(v)
    t = (f"{m:.4f}" if m < 0.01 else f"{m:.3f}" if m < 1 else f"{m:.2f}")
    return ("\\textbf{" + t + "}" if bold else t)


# ---------------------------------------------------------------- P1 core benchmark
core = collections.defaultdict(lambda: collections.defaultdict(list))
meta = {}
for f in sorted(glob.glob(R + "exp1_core_*")) + sorted(glob.glob(R + "exp1_mdg_*")):
    for r in load(f):
        k = (r["man"], r["d"]); meta[k] = r
        m = r["methods"]
        for name in ("swh_oracle", "kernel_oracle", "neural_raw_auto", "neural_stab", "neural_oracle", "mdg_fixed", "mdg_shipped"):
            if name in m and "nmae" in m[name]: core[k][name].append(m[name]["nmae"])
        if "_mdg_" in f: continue
        if "neural_grid" in m:
            core[k]["_runs"].append(1)
            if "neural_stab" in m:
                if "rel" in m["neural_stab"]: core[k]["rel"].append(m["neural_stab"]["rel"])
                bp = m["neural_byproducts_auto"]; core[k]["Hsp"].append(bp["H_sp_err"])
            else:
                core[k]["_abstain"].append(1)

names = {"sphere": "S^{%d}", "prod": "S^{%d}\\!\\times\\! S^{%d}", "torus": "T^{%d}", "quadric": "\\text{quadric}_{%d}"}
rows = []
for man in ("sphere", "prod", "torus", "quadric"):
    for d in (4, 8, 16):
        k = (man, d); c = core[k]
        nm = names[man] % ((d // 2, d - d // 2) if man == "prod" else d)
        cols = ["swh_oracle", "mdg_fixed", "mdg_shipped", "kernel_oracle", "neural_raw_auto", "neural_stab"]
        means = {cn: np.mean(c[cn]) for cn in cols if c[cn]}
        best = min(means, key=means.get) if means else None
        cells = []
        for cn in cols:
            if cn == "kernel_oracle" and not c[cn]: cells.append("\\textit{n/a}"); continue
            if cn.startswith("neural") and not c[cn]: cells.append("\\textit{abst.}"); continue
            cells.append(fmt(c[cn], cn == best))
        ab = len(c["_abstain"]); nr = len(c["_runs"])
        okr = f"{nr - ab}/{nr}"
        rel = f"{100 * np.mean(c['rel']):.1f}\\%" if (c["rel"] and man in ("sphere", "prod")) else "--"
        rows.append(f"${nm}$ & {d} & {meta[k]['S_gt_mean']:.1f} & " + " & ".join(cells) + f" & {okr} & {rel} \\\\")
        OUT[f"core_{man}{d}"] = {cn: (float(np.mean(c[cn])) if c[cn] else None) for cn in cols + ["neural_oracle"]}
        OUT[f"core_{man}{d}"]["valid"] = okr
    rows.append("\\midrule")
rows = rows[:-1]
tab = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Scalar-curvature error} (nMAE, mean over 3 seeds; lower is better). $n=20{,}000$ training samples, 200 held-out points, clean data, native ambient space. SWH and kernel use the oracle-best hyper-parameter (chosen with ground truth); MDG is the authors' package with automatic bandwidth, with our Hessian fix and as shipped. \emph{neural spectral} uses the ground-truth-free stability rule of Sec.~\ref{sec:noiserule}. \emph{valid}: runs in which the spectral-gap test found a $d$-dimensional tangent space; the others abstain (\textit{abst.}; trivial denoiser after training collapse, see text) and are excluded from the mean. Best in bold.}
\label{tab:main}
\begin{tabular}{llr|ccc|ccc|cc}
\toprule
 & & & \multicolumn{3}{c|}{baselines} & \multicolumn{3}{c|}{ours: Tweedie curvature} & & \\
manifold & $d$ & $\bar S$ & SWH & MDG fix & MDG ship & kernel & neural raw & neural spectral & valid & rel.\ err \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(PAP + "tab_main.tex", "w").write(tab)

# speed-up factors (neural spectral vs best baseline) for the text
fac = {}
for k, c in core.items():
    if c["neural_stab"]:
        bl = [np.mean(c[x]) for x in ("swh_oracle", "mdg_fixed", "kernel_oracle") if c[x]]
        fac[f"{k[0]}{k[1]}"] = float(min(bl) / np.mean(c["neural_stab"]))
OUT["factor_vs_best_baseline"] = fac

# fig B: error vs d
fams = [("sphere", "spheres $S^d$"), ("prod", r"products $S^{d/2}\!\times S^{d/2}$"), ("torus", "flat tori $T^d$"), ("quadric", "quadrics (codim 2)")]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0), sharey=True)
keys = [("swh", "swh_oracle"), ("mdg", "mdg_fixed"), ("kernel", "kernel_oracle"), ("neural", "neural_stab")]
for ax, (fam, title) in zip(axes, fams):
    for meth, key in keys:
        xs, ys, es = [], [], []
        for d in (4, 8, 16):
            v = core[(fam, d)][key]
            if v: xs.append(d); ys.append(np.mean(v)); es.append(np.std(v))
        if xs:
            ax.errorbar(xs, np.maximum(ys, 1e-4), yerr=es, fmt="-o", color=METHOD_COLOR[meth], ms=3.5, lw=1.4, capsize=2, label=METHOD_LABEL[meth])
        if meth == "neural":
            for d in (8, 16):
                c = core[(fam, d)]
                if c["_abstain"] and not c["neural_stab"]:
                    ax.plot([d], [2e-4], marker="x", color=METHOD_COLOR["neural"], ms=6, mew=1.4, ls="none")
    ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16]); ax.set_xticklabels(["4", "8", "16"])
    ax.set_yscale("log"); ax.set_title(title); ax.set_xlabel("intrinsic dim. $d$")
axes[0].set_ylabel("nMAE of scalar curvature")
axes[2].text(11.3, 3.6e-4, "abstained", fontsize=6.5, color=C["ink2"], ha="center")
axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout(); fig.savefig(FIG + "figB.pdf"); fig.savefig(FIG + "figB.png")

# ---------------------------------------------------------------- P5 noise in R^128
nz = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(R + "exp1_noise_*"):
    for r in load(f):
        for name in ("swh_oracle", "kernel_oracle", "neural_raw_auto", "neural_stab", "neural_oracle"):
            if name in r["methods"]: nz[(r["man"], r["d"], r["noise"])][name].append(r["methods"][name]["nmae"])
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=False)
for ax, (man, d) in zip(axes, [("sphere", 4), ("sphere", 8), ("quadric", 4), ("quadric", 8)]):
    levels = [0.0, 0.02, 0.05, 0.1]
    for key, lab, col, ls in [("swh_oracle", METHOD_LABEL["swh"] + " (oracle $k$)", METHOD_COLOR["swh"], "-"),
                              ("kernel_oracle", METHOD_LABEL["kernel"] + " (oracle)", METHOD_COLOR["kernel"], "-"),
                              ("neural_raw_auto", "neural raw", C["ink2"], "--"),
                              ("neural_stab", "neural spectral (stability rule)", METHOD_COLOR["neural"], "-")]:
        xs = [z for z in levels if len(nz[(man, d, z)][key]) == 3]
        ys = [np.mean(nz[(man, d, z)][key]) for z in xs]
        ax.plot(xs, ys, ls, marker="o", color=col, ms=3.5, lw=1.4, label=lab)
    ax.set_yscale("log"); ax.set_title(("$S^%d$" if man == "sphere" else "quadric$_%d$") % d + r" $\subset\mathbb{R}^{128}$")
    ax.set_xlabel(r"noise s.d. $\varsigma$"); ax.set_ylim(5e-3, 1.5)
axes[0].set_ylabel("nMAE")
axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout(); fig.savefig(FIG + "figD.pdf"); fig.savefig(FIG + "figD.png")
OUT["noise"] = {f"{k[0]}{k[1]}_{k[2]}": {n: float(np.mean(v)) for n, v in c.items()} for k, c in nz.items()}

# ---------------------------------------------------------------- P4 sample size
ns = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(R + "exp1_nsweep_*"):
    for r in load(f):
        k = (r["man"], r["d"], r["n"]); m = r["methods"]
        for name in ("swh_oracle", "kernel_oracle", "neural_stab"):
            if name in m: ns[k][name].append(m[name]["nmae"])
        ns[k]["_runs"].append(1)
        if "neural_stab" not in m: ns[k]["_abst"].append(1)
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.1), sharey=True)
NS = [2000, 5000, 20000, 100000, 500000]
for ax, (man, d) in zip(axes, [("sphere", 8), ("sphere", 16)]):
    for key, meth in [("swh_oracle", "swh"), ("kernel_oracle", "kernel")]:
        xs = [n for n in NS if ns[(man, d, n)][key]]
        ys = [np.median(ns[(man, d, n)][key]) for n in xs]
        ax.plot(xs, ys, "-o", color=METHOD_COLOR[meth], ms=3.5, lw=1.4, label=METHOD_LABEL[meth])
    first = True
    for n in NS:
        v = ns[(man, d, n)]["neural_stab"]
        ax.plot([n] * len(v), v, "o", color=METHOD_COLOR["neural"], ms=3.5, alpha=0.85,
                label=(METHOD_LABEL["neural"] + " (each valid seed)") if first else None); first = False
        a, t = len(ns[(man, d, n)]["_abst"]), len(ns[(man, d, n)]["_runs"])
        if a: ax.plot([n], [1.4e-3], marker="x", color=METHOD_COLOR["neural"], ms=5, mew=1.2); ax.annotate(f"{a}", (n * 1.25, 1.25e-3), fontsize=6, color=C["ink2"])
    n0 = 2e3; ax.plot([n0, 5e5], [0.6, 0.6 * (5e5 / n0) ** (-1 / 3)], ":", color=C["ink2"], lw=0.9, label="slope $-1/3$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"$S^{{{d}}}$"); ax.set_xlabel("training samples $n$")
    ax.set_ylim(1e-3, 6)
axes[0].set_ylabel("nMAE")
axes[-1].plot([], [], "x", color=METHOD_COLOR["neural"], label="abstained runs (count)")
axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout(); fig.savefig(FIG + "figP4.pdf"); fig.savefig(FIG + "figP4.png")
OUT["nsweep"] = {f"{k[0]}{k[1]}_{k[2]}": {n: (float(np.median(v)) if not n.startswith("_") else len(v)) for n, v in c.items()} for k, c in ns.items()}
# empirical slopes of the baselines
for man, d in [("sphere", 8), ("sphere", 16)]:
    for key in ("swh_oracle", "kernel_oracle"):
        xs = [n for n in NS if ns[(man, d, n)][key]]
        if len(xs) >= 3:
            ys = [np.median(ns[(man, d, n)][key]) for n in xs]
            OUT.setdefault("slopes", {})[f"{man}{d}_{key}"] = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])

# ---------------------------------------------------------------- P3 image manifolds
im = collections.defaultdict(list)
for f in glob.glob(R + "exp2_*"):
    for r in load(f):
        im[(r["family"], r["n"], r["noise"], r["method"])].append(r)


def first_gap(r, th=0.5):
    c = [s for s in r["grid"] if r["grid"][s]["gap"] >= th]
    return r["grid"][min(c, key=float)]


imrows = []
for fam, n, noise in [("smooth16", 20000, 0.0), ("smooth16", 100000, 0.0), ("smooth16", 20000, 0.03), ("smooth16", 100000, 0.03),
                      ("sharp24", 20000, 0.0), ("sharp24", 200000, 0.0)]:
    row = {"swh": im[(fam, n, noise, "swh_oracle")]}
    cells = [f"{np.mean([x['r'] for x in row['swh']]):.2f}"]
    rec = {"swh_r": float(np.mean([x["r"] for x in row["swh"]])), "swh_mae": float(np.mean([x["mae"] for x in row["swh"]]))}
    for arch in ("mlp", "unet"):
        rs = im[(fam, n, noise, f"neural_{arch}")]
        if not rs: cells += ["--", "--", "--"]; continue
        au = np.mean([x["auto"]["r"] for x in rs]); fg = np.mean([first_gap(x)["r"] for x in rs])
        orc = np.mean([max(x["grid"].values(), key=lambda g: -g["mae"])["r"] for x in rs])
        cells += [f"{au:.2f}", f"{fg:.2f}", f"{orc:.2f}"]
        rec[arch] = dict(auto=float(au), first=float(fg), oracle=float(orc),
                         first_mae=float(np.mean([first_gap(x)["mae"] for x in rs])),
                         Hrel=float(np.mean([first_gap(x)["H_relerr"] for x in rs])),
                         cos2=float(np.mean([first_gap(x)["tan_cos2"] for x in rs])))
    OUT[f"img_{fam}_{n}_{noise}"] = rec
    lab = {"smooth16": r"smooth $16{\times}16$", "sharp24": r"sharp $24{\times}24$"}[fam]
    imrows.append(f"{lab} & {n // 1000}k & {noise:g} & " + " & ".join(cells) + " \\\\")
open(PAP + "tab_images.tex", "w").write(r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Rendered image manifolds} ($\mathcal M^4\subset\R^{256}$ and $\R^{576}$): Pearson correlation between estimated and exact scalar curvature on 200 held-out images (mean over seeds). SWH uses the oracle-best $k$. For each denoiser we report three noise-level choices: max spectral gap, the smallest level with median gap $\ge\frac12$, both ground-truth-free, and the oracle level.}
\label{tab:images}
\begin{tabular}{llc|c|ccc|ccc}
\toprule
 & & & SWH & \multicolumn{3}{c|}{MLP denoiser} & \multicolumn{3}{c}{periodic UNet denoiser} \\
family & $n$ & pixel noise & oracle & max-gap & first gap$\ge\frac12$ & oracle & max-gap & first gap$\ge\frac12$ & oracle \\
\midrule
""" + "\n".join(imrows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
# extrapolation (UNet, smooth16, n=20k, clean)
rs = im[("smooth16", 20000, 0.0, "neural_unet")]
t = np.array(rs[0]["extrap_t"]); e1 = np.mean([x["extrap_first"] for x in rs], 0); e2 = np.mean([x["extrap_second"] for x in rs], 0)
OUT["extrap_unet"] = dict(t=t.tolist(), first=e1.tolist(), second=e2.tolist())
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.0))
ax = axes[0]
x = np.arange(4); w = 0.26
for j, (key, lab, col) in enumerate([("swh_r", "SWH (oracle $k$)", METHOD_COLOR["swh"]), ("mlp", "MLP denoiser", C["aqua"]), ("unet", "UNet denoiser", METHOD_COLOR["neural"])]):
    vals = []
    for (n, noise) in [(20000, 0.0), (100000, 0.0), (20000, 0.03), (100000, 0.03)]:
        rec = OUT[f"img_smooth16_{n}_{noise}"]; vals.append(rec[key] if key == "swh_r" else rec[key]["first"])
    ax.bar(x + (j - 1) * w, vals, w * 0.92, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels(["20k\nclean", "100k\nclean", "20k\nnoisy", "100k\nnoisy"])
ax.set_ylabel("Pearson $r$ (est. vs exact $S$)"); ax.set_ylim(0, 1.25); ax.set_yticks([0, .2, .4, .6, .8, 1]); ax.legend(loc="upper left", fontsize=5.5, ncol=3, columnspacing=0.6, handlelength=1.0)
ax.set_title("(a) scalar curvature, $16{\\times}16$ images")
ax = axes[1]
ax.loglog(t, e1, "-o", color=C["ink2"], ms=3, label="first order $x+tv$")
ax.loglog(t, e2, "-o", color=METHOD_COLOR["neural"], ms=3, label=r"second order $+\frac{t^2}{2}\widehat{\mathrm{I\!I}}(v,v)$")
ax.set_xlabel("step $t$"); ax.set_ylabel("distance to manifold"); ax.legend(fontsize=6)
ax.set_title("(b) tangent extrapolation (UNet)")
fig.tight_layout(); fig.savefig(FIG + "figE_bump16.pdf"); fig.savefig(FIG + "figE_bump16.png")

# ---------------------------------------------------------------- P2 pretrained CIFAR-10 DDPM
T = json.load(open(R + "exp3_test.json")); Tr = json.load(open(R + "exp3_train.json"))
TE, TR = T["records"], Tr["records"]
OUT["cifar_clf_acc"] = T["clf_acc"]
feats = [("S", "scalar curv. $\\widehat S$", 2), ("II2", "$\\|\\widehat{\\mathrm{I\\!I}}\\|^2$", 2), ("H2", "$\\|\\widehat{\\mathbf H}\\|^2$", 2),
         ("lap_logq", "$\\widehat{\\Delta\\log q}$", 2), ("resid", "denoising residual", 1), ("gradlogq", "$|\\widehat{\\nabla\\log q}|$", 1),
         ("H_df", "$\\|\\mathbf H_{\\rm df}\\|$ (deriv.-free)", 1), ("dloc", "local dimension", 1)]
rng = np.random.default_rng(0)


def within(Rr, name, s):
    y = np.array([r["y"] for r in Rr]); mg = np.array([r["margin"] for r in Rr]); v = np.array([r["sig"][s][name] for r in Rr])
    vr, mr = np.zeros(len(v)), np.zeros(len(v))
    for c in range(10):
        i = y == c; vr[i] = rankdata(v[i]) / i.sum(); mr[i] = rankdata(mg[i]) / i.sum()
    return vr, mr


def boot_auc(lab, v, B=2000):
    a = roc_auc_score(lab, v); sgn = 1 if a >= 0.5 else -1
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(v), len(v))
        if len(set(lab[i])) < 2: continue
        bs.append(roc_auc_score(lab[i], sgn * v[i]))
    return max(a, 1 - a), np.percentile(bs, [2.5, 97.5])


crow = []
for name, lab, order in feats:
    cells = []; rec = {}
    for s in ("0.1", "0.2", "0.4"):
        vr, mr = within(TE, name, s); rho, p = spearmanr(vr, mr)
        bs = []
        for _ in range(1000):
            i = rng.integers(0, len(vr), len(vr)); bs.append(spearmanr(vr[i], mr[i])[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rec[s] = dict(rho=float(rho), p=float(p), lo=float(lo), hi=float(hi))
    s = "0.1"
    wrong = np.array([r["pred"] != r["y"] for r in TE]).astype(int); v = np.array([r["sig"][s][name] for r in TE])
    ae, ae_ci = boot_auc(wrong, v)
    s4 = "0.4"
    v_all = np.array([r["sig"][s4][name] for r in TE] + [r["sig"][s4][name] for r in TR]); mem = np.r_[np.zeros(len(TE)), np.ones(len(TR))].astype(int)
    am, am_ci = boot_auc(mem, v_all)
    rec.update(err_auc=float(ae), err_ci=ae_ci.tolist(), mia_auc=float(am), mia_ci=mia_ci.tolist() if (mia_ci := am_ci) is not None else None)
    OUT[f"cifar_{name}"] = rec
    rr = " & ".join(f"{rec[s]['rho']:+.2f}" for s in ("0.1", "0.2", "0.4"))
    crow.append(f"{lab} & {order} & {rr} & {ae:.3f} & {am:.3f} \\\\")
    if name == "lap_logq": crow.append("\\midrule")
open(PAP + "tab_cifar.tex", "w").write(r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Pretrained CIFAR-10 diffusion model} (\texttt{google/ddpm-cifar10-32}, no retraining). 500 test and 500 training images (50 per class). Columns 3--5: within-class Spearman correlation with the margin of a ResNet-18 (91.3\% test accuracy) at three noise levels; $|\rho|\ge0.12$ is significant at $p<0.01$. Column 6: AUC for detecting misclassified test images ($\sigma=0.1$). Column 7: AUC for telling the diffusion model's training images from test images ($\sigma=0.4$). Second-order quantities (top) versus first-order ones (bottom).}
\label{tab:cifar}
\begin{tabular}{lc|ccc|cc}
\toprule
quantity & order & $\rho_{\sigma=0.1}$ & $\rho_{\sigma=0.2}$ & $\rho_{\sigma=0.4}$ & error AUC & member AUC \\
\midrule
""" + "\n".join(crow) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
# fig: margin by within-class quintile of S and of residual, sigma=0.1
fig, axes = plt.subplots(1, 2, figsize=(5.2, 1.9))
ax = axes[0]
for name, lab, col in [("S", r"$\widehat S$ (2nd order)", METHOD_COLOR["neural"]), ("lap_logq", r"$\widehat{\Delta\log q}$ (2nd order)", C["violet"]), ("resid", "residual (1st order)", C["ink2"])]:
    vr, mr = within(TE, name, "0.1"); mg = np.array([r["margin"] for r in TE])
    q = np.minimum((vr * 5 - 1e-9).astype(int), 4)
    ys = [mg[q == i].mean() for i in range(5)]; es = [mg[q == i].std() / np.sqrt((q == i).sum()) for i in range(5)]
    ax.errorbar(np.arange(1, 6), ys, yerr=es, fmt="-o", color=col, ms=3, lw=1.3, capsize=2, label=lab)
ax.set_xlabel("within-class quintile"); ax.set_ylabel("classifier margin"); ax.set_ylim(0.72, 1.02); ax.legend(fontsize=5.5, loc="lower right", ncol=1)
ax.set_title("(a) margin vs. curvature ($\\sigma=0.1$)")
ax = axes[1]
names_ = ["S", "II2", "lap_logq", "resid", "gradlogq", "dloc"]
labs_ = ["$\\widehat S$", "$\\|\\mathrm{I\\!I}\\|^2$", "$\\Delta\\log q$", "resid.", "$|\\nabla\\log q|$", "dim"]
e = [OUT[f"cifar_{n}"]["err_auc"] for n in names_]; m_ = [OUT[f"cifar_{n}"]["mia_auc"] for n in names_]
x = np.arange(len(names_))
ax.bar(x - 0.2, e, 0.38, color=METHOD_COLOR["neural"], label="misclassification")
ax.bar(x + 0.2, m_, 0.38, color=C["orange"], label="train vs. test")
ax.axhline(0.5, color=C["ink2"], lw=0.6); ax.set_ylim(0.45, 0.72)
ax.set_xticks(x); ax.set_xticklabels(labs_, fontsize=6); ax.set_ylabel("AUC"); ax.legend(fontsize=6, loc="upper right")
ax.axvline(2.5, color=C["grid"], lw=0.8); ax.set_title("(b) detection AUC (2nd | 1st order)")
fig.tight_layout(); fig.savefig(FIG + "figG_cifar.pdf"); fig.savefig(FIG + "figG_cifar.png")
# class-level and dloc saturation facts
dl = {s: float(np.mean([r["sig"][s]["dloc"] == 88 for r in TE])) for s in ("0.1", "0.2", "0.4")}
OUT["cifar_dloc_capped_frac"] = dl
OUT["cifar_dloc_median"] = {s: float(np.median([r["sig"][s]["dloc"] for r in TE])) for s in ("0.1", "0.2", "0.4")}
OUT["cifar_time_per_image_s"] = {s: float(np.median([r["sig"][s]["time"] for r in TE])) for s in ("0.1", "0.2", "0.4")}
json.dump(OUT, open("/home/claude/ndg/results/gpu_summary.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in OUT.items() if not k.startswith(("noise", "nsweep"))}, indent=0, default=float)[:6000])
