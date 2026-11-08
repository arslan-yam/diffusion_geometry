"""Tweedie curvature: second-order diffusion geometry from a denoiser (PyTorch).

Given a denoiser D(y) (any differentiable map R^D -> R^D, e.g. lambda y: model(y, sigma)):
  G = sym(dD(y))                 ~ carré du champ of the coordinates (-> tangent projector)
  E = top-d eigenvectors of G    ~ tangent frame;  nu = normal eigenvalues
  II_ij = P_N D^2D[e_i,e_j] / ((lam_i+lam_j)/2 - nu)     (spectral normalisation)
  S = |sum_i II_ii|^2 - sum_ij |II_ij|^2                 (Gauss equation)
All derivatives are forward-mode (torch.func.jvp), batched with vmap.
"""
import torch
from torch.func import jvp, vjp, vmap, jacfwd


def d2_dir(D, y, u, v):
    """D^2 D(y)[u, v] (forward-over-forward)."""
    g = lambda z: jvp(D, (z,), (u,))[1]
    return jvp(g, (y,), (v,))[1]


def jacobian_chunked(D, y, chunk=64):
    """Forward-mode Jacobian dD(y) (Dn x Dn), computed in column chunks to bound memory."""
    Dn = y.numel(); I = torch.eye(Dn, dtype=y.dtype, device=y.device); cols = []
    for s in range(0, Dn, chunk):
        cols.append(vmap(lambda v: jvp(D, (y,), (v,))[1], in_dims=0, out_dims=1)(I[s:s + chunk]))
    return torch.cat(cols, 1)


def full_frame(D, y, chunk=32):
    """Full symmetric Jacobian eigendecomposition (use for D <= ~3000). Returns lam desc, V."""
    J = jacobian_chunked(D, y, chunk) if y.numel() > chunk else jacfwd(D)(y)
    G = 0.5 * (J + J.T)
    lam, V = torch.linalg.eigh(G)
    return lam.flip(0), V.flip(1)


def topk_frame(D, y, k, iters=6, oversample=8, seed=0):
    """Matrix-free top-k eigenpairs of sym(dD) by block subspace iteration + Rayleigh-Ritz.
    Cost: iters * (k+oversample) JVPs and VJPs. Use for image-sized D."""
    Dn = y.numel()
    gen = torch.Generator(device=y.device).manual_seed(seed)
    Q = torch.randn(Dn, k + oversample, generator=gen, device=y.device, dtype=y.dtype)
    _, f_vjp = vjp(D, y)

    def Gmv(V):  # V: (Dn, m)
        Jv = vmap(lambda v: jvp(D, (y,), (v,))[1], in_dims=1, out_dims=1)(V)
        JTv = vmap(lambda v: f_vjp(v)[0], in_dims=1, out_dims=1)(V)
        return 0.5 * (Jv + JTv)

    for _ in range(iters):
        Q, _ = torch.linalg.qr(Gmv(Q))
    GQ = Gmv(Q)
    T = Q.T @ GQ
    lam, W = torch.linalg.eigh(0.5 * (T + T.T))
    lam, W = lam.flip(0), W.flip(1)
    return lam, Q @ W        # (k+oversample,), (Dn, k+oversample)


def tweedie_curvature(D, y, d, lam=None, V=None, min_gap=0.05, isotropic_normal=None):
    """Full second fundamental form in the top-d frame.
    If V has all D columns, the exact per-direction spectral normalisation is used;
    otherwise (matrix-free frame) the normal spectrum is treated as isotropic with
    nu = median of the available eigenvalues beyond d."""
    if lam is None:
        lam, V = full_frame(D, y)
    E, lt = V[:, :d], lam[:d]
    pair = vmap(lambda u: vmap(lambda v: d2_dir(D, y, u, v))(E.T))(E.T)   # d x d x Dn
    pair = 0.5 * (pair + pair.transpose(0, 1))
    c = 0.5 * (lt[:, None] + lt[None, :])
    full = V.shape[1] == y.numel() and not isotropic_normal
    if full:
        VN, nu = V[:, d:], lam[d:]
        coefN = pair @ VN                                    # normal coordinates
        II_raw = coefN @ VN.T
        II_sp = (coefN / torch.clamp(c[:, :, None] - nu[None, None, :], min=min_gap)) @ VN.T
    else:
        nu = lam[d:].median() if lam.numel() > d else torch.zeros((), device=y.device)
        II_raw = pair - (pair @ E) @ E.T
        II_sp = II_raw / torch.clamp(c - nu, min=min_gap)[:, :, None]
    out = {}
    for name, II in (("raw", II_raw), ("sp", II_sp)):
        H = torch.einsum("iil->l", II)
        out["S_" + name] = H @ H - (II * II).sum()
        out["H_" + name] = H.norm()
        out["II2_" + name] = (II * II).sum()
    out["lam"] = lam
    out["gap"] = lam[d - 1] - lam[d]
    out["II"] = II_sp
    out["E"] = E
    return out


def hutchinson_scalar(D, y, E, lt, nu, m=16, seed=0, min_gap=0.05):
    """Randomised scalar curvature: S = E[R(u,v,v,u)], u,v ~ N(0, E E^T).
    3 second-order JVPs per sample; isotropic spectral normalisation."""
    gen = torch.Generator(device=y.device).manual_seed(seed)
    d = E.shape[1]
    Z = torch.randn(2, m, d, generator=gen, device=y.device, dtype=y.dtype)
    U, W = Z[0] @ E.T, Z[1] @ E.T
    scale = 1.0 / torch.clamp(lt.mean() - nu, min=min_gap)
    P = lambda a: a - E @ (E.T @ a)

    def one(u, w):
        a = P(d2_dir(D, y, u, u)) * scale
        b = P(d2_dir(D, y, w, w)) * scale
        c = P(d2_dir(D, y, u, w)) * scale
        return torch.stack([a @ b, c @ c])
    vals = vmap(one)(U, W)                         # (m, 2): unbiased |H|^2 and |II|^2 terms
    R = vals[:, 0] - vals[:, 1]
    return {"S": R.mean(), "S_se": R.std() / m ** 0.5, "H2": vals[:, 0].mean(), "II2": vals[:, 1].mean()}
