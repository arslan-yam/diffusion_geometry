"""Synthetic manifolds with exact curvature (PyTorch port). Ground truth by autodiff
of local charts + Gauss equation; random isometric embedding into R^D."""
import math
import numpy as np
import torch
from torch.func import jacfwd, vmap


def frame_curvature(J, H):
    g = J.T @ J
    L = torch.linalg.cholesky(g)
    Linv = torch.linalg.inv(L).T
    E = J @ Linv
    Pn = torch.eye(J.shape[0], dtype=J.dtype, device=J.device) - E @ E.T
    IIc = torch.einsum("kab,kl->abl", H, Pn)
    II = torch.einsum("ai,bj,abl->ijl", Linv, Linv, IIc)
    Hm = torch.einsum("iil->l", II)
    return Hm @ Hm - (II * II).sum(), II, E


class Manifold:
    def __init__(self, name, d, D0, sampler, chart, S_true=None):
        self.name, self.d, self.D0, self.sampler, self.chart, self.S_true = name, d, D0, sampler, chart, S_true

    def gt(self, xn):
        """xn: (n, D0) float64 tensor -> S (n,), II (n,d,d,D0), E (n,D0,d)."""
        def one(x0):
            f = lambda th: self.chart(x0, th)
            th0 = torch.zeros(self.d, dtype=x0.dtype, device=x0.device)
            J = jacfwd(f)(th0); H = jacfwd(jacfwd(f))(th0)
            return frame_curvature(J, H)
        out = [one(x) for x in xn]          # loop: charts use data-dependent SVD
        return (torch.stack([o[0] for o in out]), torch.stack([o[1] for o in out]),
                torch.stack([o[2] for o in out]))


def _sphere_chart(R):
    def chart(x0, th):
        _, _, Vh = torch.linalg.svd(x0[None, :])
        B = Vh[1:].T
        y = x0 + B @ th
        return R * y / y.norm()
    return chart


def sphere(d, R=1.0):
    def sample(rng, n):
        z = rng.standard_normal((n, d + 1)); return R * z / np.linalg.norm(z, axis=1, keepdims=True)
    return Manifold(f"S{d}", d, d + 1, sample, _sphere_chart(R), d * (d - 1) / R**2)


def sphere_product(dims, radii):
    D0s = [a + 1 for a in dims]; offs = np.cumsum([0] + D0s); charts = [_sphere_chart(r) for r in radii]

    def sample(rng, n):
        parts = []
        for a, r in zip(dims, radii):
            z = rng.standard_normal((n, a + 1)); parts.append(r * z / np.linalg.norm(z, axis=1, keepdims=True))
        return np.concatenate(parts, 1)

    def chart(x0, th):
        outs, k = [], 0
        for i, a in enumerate(dims):
            outs.append(charts[i](x0[offs[i]:offs[i + 1]], th[k:k + a])); k += a
        return torch.cat(outs)
    S = float(sum(a * (a - 1) / r**2 for a, r in zip(dims, radii)))
    return Manifold("x".join(f"S{a}({r:g})" for a, r in zip(dims, radii)), int(sum(dims)), int(offs[-1]), sample, chart, S)


def flat_torus(d):
    m = sphere_product([1] * d, [1.0] * d); m.name = f"T{d}"; return m


def quadric(d, codim=2, scale=2.0, seed=7, radius=1.0):
    rng = np.random.default_rng(seed); As = []
    for _ in range(codim):
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        As.append(Q @ np.diag(scale * rng.uniform(-1, 1, d)) @ Q.T)
    As_t = torch.tensor(np.stack(As))

    def embed(t):
        A = As_t.to(t)
        return torch.cat([t, 0.5 * torch.einsum("...i,lij,...j->...l", t, A, t)], -1)

    def sample(rng_, n):
        z = rng_.standard_normal((n, d)); z /= np.linalg.norm(z, axis=1, keepdims=True)
        t = z * radius * rng_.uniform(0, 1, (n, 1)) ** (1 / d)
        return embed(torch.tensor(t)).numpy()
    return Manifold(f"Q{d}", d, d + codim, sample, lambda x0, th: embed(x0[:d] + th), None)


def random_isometry(D0, D, seed=0):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, D0)))
    return Q[:, :D0]


def make(name, d):
    return {"sphere": lambda: sphere(d), "torus": lambda: flat_torus(d),
            "prod": lambda: sphere_product([d // 2, d - d // 2], [1.0, 0.7]),
            "quadric": lambda: quadric(d)}[name]()
