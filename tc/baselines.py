"""Baselines: SWH quadratic fit, kernel (empirical) denoiser, and the ORIGINAL
DiffusionGeometry package curvature pipeline (shipped and with the Hessian fix)."""
import os, sys
import numpy as np
import torch


def swh_curvature(X, Q, d, k):
    from sklearn.neighbors import NearestNeighbors
    _, idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(Q)
    iu = np.triu_indices(d); out = np.zeros(len(Q))
    for s in range(0, len(Q), 128):
        nb = X[idx[s:s + 128]] - Q[s:s + 128, None, :]
        _, _, Vt = np.linalg.svd(nb - nb.mean(1, keepdims=True), full_matrices=False)
        T, Nb = Vt[:, :d], Vt[:, d:]
        t = np.einsum("bkD,bjD->bkj", nb, T); nc = np.einsum("bkD,bjD->bkj", nb, Nb)
        quad = (t[:, :, :, None] * t[:, :, None, :])[:, :, iu[0], iu[1]]
        A = np.concatenate([np.ones(t.shape[:2] + (1,)), t, quad], 2)
        cq = (np.linalg.pinv(A) @ nc)[:, 1 + d:, :]
        Hm = np.zeros((len(nb), d, d, cq.shape[-1])); Hm[:, iu[0], iu[1], :] = cq
        Hm = Hm + np.swapaxes(Hm, 1, 2)
        tr = np.einsum("biil->bl", Hm)
        out[s:s + 128] = np.einsum("bl,bl->b", tr, tr) - np.einsum("bijl,bijl->b", Hm, Hm)
    return out


def kernel_denoiser(X):
    """Empirical posterior mean = alpha=0 diffusion-maps kernel applied to coordinates."""
    sq = 0.5 * (X * X).sum(1)

    def D(y, sigma):
        w = torch.softmax((X @ y - sq) / sigma**2, 0)
        return w @ X
    return D


def original_mdg(X, qidx, d, fixed=True, repo=os.environ.get("DG_REPO", "DiffusionGeometry")):
    """Scalar curvature with the authors' package, following
    intro_notebooks/manifold_diffusion_geometry_intro.ipynb verbatim.
    fixed=True replaces the third term permutation in hessian_coords:
      shipped: gamma_composed.transpose((0,3,1,2))  == Γ(x_j, Γ(x_k, x_i))
      fixed:   gamma_composed.transpose((0,2,3,1))  == Γ(x_k, Γ(x_i, x_j))"""
    if repo not in sys.path: sys.path.insert(0, repo)
    from opt_einsum import contract
    from diffusion_geometry import DiffusionGeometry
    import diffusion_geometry.operators as ops
    if not hasattr(ops, "_hc_shipped"): ops._hc_shipped = ops.hessian_coords

    def hc_fixed(coords, gamma_coords, cdc):
        g = cdc(coords, gamma_coords)
        return 0.5 * (g + g.transpose((0, 2, 1, 3)) - g.transpose((0, 2, 3, 1)))
    ops.hessian_coords = hc_fixed if fixed else ops._hc_shipped
    dg = DiffusionGeometry.from_point_cloud(X)
    ev, fr = np.linalg.eigh(dg.cache.gamma_coords); ev, fr = ev[:, ::-1], fr[:, :, ::-1]
    ms = np.median(ev[:, :d].mean(1)); T, N = fr[:, :, :d], fr[:, :, d:]
    hess = dg.cache.hessian_coords.transpose(0, 3, 1, 2) / ms**2
    sff = contract("pstu,psl,pti,puj->plij", hess, N, T, T)
    R = contract("pLik,pLjl->pijkl", sff, sff) - contract("pLjk,pLil->pijkl", sff, sff)
    S = dg._regularise(contract("pii->p", contract("pkikj->pij", R)))
    ops.hessian_coords = ops._hc_shipped
    return np.asarray(S)[qidx]
