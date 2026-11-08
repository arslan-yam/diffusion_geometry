"""Exp 3 (GPU): curvature probing of a PRETRAINED diffusion model (no training of the denoiser).

Model: google/ddpm-cifar10-32 (diffusers), converted to a VE denoiser D(y, sigma) on [-1,1] images.
For CIFAR-10 test (and train) images: matrix-free top-k spectrum of sym(dD) (local dimension),
randomised scalar curvature and |H|^2, |II|^2 (Hutchinson, isotropic spectral normalisation);
then within-class correlation with the margin of a ResNet-18 trained here.

  python exp3_pretrained_cifar.py --per_class 50 --sigmas 0.1,0.2,0.4 --k 96 --out results/exp3.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from tc.models import DDPMAsDenoiser
from tc.curvature import topk_frame, hutchinson_scalar

p = argparse.ArgumentParser()
p.add_argument("--model", default="google/ddpm-cifar10-32")
p.add_argument("--per_class", type=int, default=50)
p.add_argument("--sigmas", default="0.1,0.2,0.4")
p.add_argument("--k", type=int, default=96, help="eigenpairs computed (must exceed local dim)")
p.add_argument("--m", type=int, default=16, help="Hutchinson pairs")
p.add_argument("--iters", type=int, default=6)
p.add_argument("--split", default="test,train")
p.add_argument("--clf_epochs", type=int, default=15)
p.add_argument("--out", default="results/exp3.json")
p.add_argument("--device", default="cuda")
p.add_argument("--data", default=os.environ.get("CIFAR_DIR", "data"), help="folder holding cifar-10-batches-py")
p.add_argument("--clf_path", default="results/clf_resnet18.pt", help="classifier weights are cached here")
p.add_argument("--clf_only", action="store_true", help="train and cache the classifier, then exit")
p.add_argument("--workers", type=int, default=min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - 1))
a = p.parse_args()
dev = a.device
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

import safetensors.torch  # noqa: F401  (diffusers 0.31 expects it pre-imported)
from diffusers import UNet2DModel, DDPMScheduler
import torchvision, torchvision.transforms as T
unet = UNet2DModel.from_pretrained(a.model).to(dev).eval()
sched = DDPMScheduler.from_pretrained(a.model)
Dm = DDPMAsDenoiser(unet, sched).to(dev)
for p_ in Dm.parameters(): p_.requires_grad_(False)

tf = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
DL = not os.path.isdir(os.path.join(a.data, "cifar-10-batches-py"))   # compute nodes are offline: pre-download
sets = {s: torchvision.datasets.CIFAR10(a.data, train=(s == "train"), download=DL, transform=tf) for s in a.split.split(",")}

# ---- classifier (ResNet-18, CIFAR stem) for margins
def train_classifier():
    m = torchvision.models.resnet18(num_classes=10)
    m.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = torch.nn.Identity(); m = m.to(dev)
    if os.path.exists(a.clf_path):
        m.load_state_dict(torch.load(a.clf_path, map_location=dev, weights_only=True)); print("loaded classifier", a.clf_path, flush=True)
        return m.eval()
    tr = torchvision.datasets.CIFAR10(a.data, train=True, download=DL, transform=T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]))
    dl = torch.utils.data.DataLoader(tr, batch_size=256, shuffle=True, num_workers=max(0, a.workers), drop_last=True)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, total_steps=a.clf_epochs * len(dl))
    for ep in range(a.clf_epochs):
        m.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = torch.nn.functional.cross_entropy(m(x), y); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        print(f"classifier epoch {ep + 1}/{a.clf_epochs} loss {loss.item():.3f}", flush=True)
    torch.save(m.state_dict(), a.clf_path)
    return m.eval()

clf = train_classifier()
te = torchvision.datasets.CIFAR10(a.data, train=False, transform=tf)
with torch.no_grad():
    acc = np.mean([(clf(x.to(dev)).argmax(1).cpu() == y).float().mean().item()
                   for x, y in torch.utils.data.DataLoader(te, batch_size=1000)])
print("classifier test acc", acc, flush=True)
if a.clf_only: sys.exit(0)

out = {"model": a.model, "clf_acc": float(acc), "sigmas": a.sigmas, "records": []}
if os.path.exists(a.out):                       # resume (jobs are requeued from the start after maintenance)
    prev = json.load(open(a.out))
    if prev.get("sigmas") == a.sigmas: out["records"] = prev["records"]; print("resuming with", len(out["records"]), "records", flush=True)
done = {(r["split"], r["idx"]) for r in out["records"]}
rng = np.random.default_rng(0)
for split, ds in sets.items():
    labels = np.array(ds.targets)
    idx = np.concatenate([rng.choice(np.where(labels == c)[0], a.per_class, replace=False) for c in range(10)])
    for i in idx:
        if (split, int(i)) in done: continue
        x, y = ds[int(i)]
        xf = x.reshape(-1).to(dev, torch.float64)
        with torch.no_grad():
            pr = torch.softmax(clf(x[None].to(dev)), 1)[0].cpu().numpy()
        ps = np.sort(pr)
        rec = {"split": split, "idx": int(i), "y": int(y), "margin": float(ps[-1] - ps[-2]), "pred": int(pr.argmax()), "sig": {}}
        for s in map(float, a.sigmas.split(",")):
            t = time.time()
            Dfn = lambda z: Dm(z.float(), s).double()
            lam, V = topk_frame(Dfn, xf, a.k, iters=a.iters)
            dloc = int((lam > 0.5).sum().item()); dloc = max(2, min(dloc, a.k - 8))
            E = V[:, :dloc]; lt = lam[:dloc]; nu = lam[dloc:].median()
            h = hutchinson_scalar(Dfn, xf, E, lt, nu, m=a.m)
            resid = float((Dm(xf.float(), s).double() - xf).norm())
            rec["sig"][str(s)] = dict(dloc=dloc, lam=lam[:min(a.k, 128)].cpu().tolist(), S=h["S"].item(), S_se=h["S_se"].item(),
                                 H2=h["H2"].item(), II2=h["II2"].item(), resid=resid, time=time.time() - t)
        out["records"].append(rec)
        if len(out["records"]) % 10 == 0:
            json.dump(out, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
            print(len(out["records"]), {s_: round(v["S"], 2) for s_, v in rec["sig"].items()}, flush=True)
json.dump(out, open(a.out, "w"))

# ---- quick analysis
from scipy.stats import spearmanr, rankdata
for s in map(float, a.sigmas.split(",")):
    R = [r for r in out["records"] if r["split"] == "test"] or out["records"]
    y = np.array([r["y"] for r in R]); mg = np.array([r["margin"] for r in R])
    def within(v):
        vr, mr = np.zeros(len(v)), np.zeros(len(v))
        for c in range(10):
            i = y == c; vr[i] = rankdata(v[i]) / i.sum(); mr[i] = rankdata(mg[i]) / i.sum()
        return spearmanr(vr, mr)
    for name in ("dloc", "S", "H2", "II2", "resid"):
        v = np.array([r["sig"][str(s)][name] for r in R])
        print(f"sigma={s} within-class spearman({name}, margin) = {within(v)}")
