"""Denoisers: EDM-preconditioned residual MLP (vectors) and a small periodic UNet (images),
plus a wrapper turning a pretrained diffusers DDPM eps-model into a VE denoiser D(y, sigma)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEmb(nn.Module):
    def __init__(self, dim=128):
        super().__init__(); self.register_buffer("freq", torch.randn(dim // 2) * 4.0)

    def forward(self, c):                       # c: (B,)
        f = c[:, None] * self.freq[None]
        return torch.cat([f.sin(), f.cos()], -1)


class ResMLP(nn.Module):
    def __init__(self, D, width=512, depth=4, emb=128):
        super().__init__()
        self.emb = FourierEmb(emb); self.e1 = nn.Linear(emb, emb)
        self.inp = nn.Linear(D, width)
        self.film = nn.ModuleList([nn.Linear(emb, 2 * width) for _ in range(depth)])
        self.l1 = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.l2 = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.out = nn.Linear(width, D)
        for m in list(self.film) + list(self.l2) + [self.out]:
            nn.init.normal_(m.weight, std=1e-2 / math.sqrt(m.in_features)); nn.init.zeros_(m.bias)

    def forward(self, x, cnoise):
        e = F.silu(self.e1(self.emb(cnoise)))
        h = self.inp(x)
        for film, l1, l2 in zip(self.film, self.l1, self.l2):
            sc, sh = film(e).chunk(2, -1)
            a = F.silu(h) * (1 + sc) + sh
            h = h + l2(F.silu(l1(a)))
        return self.out(F.silu(h))


class PeriodicUNet(nn.Module):
    """Small UNet with circular padding and GroupNorm-free (smooth) blocks for res x res images."""
    def __init__(self, ch=64, emb=128, res=16):
        super().__init__()
        self.res = res
        self.emb = FourierEmb(emb); self.e1 = nn.Linear(emb, emb)
        conv = lambda i, o: nn.Conv2d(i, o, 3, padding=1, padding_mode="circular")
        self.c0 = conv(1, ch)
        self.d1 = conv(ch, ch); self.d2 = conv(ch, 2 * ch); self.down = nn.Conv2d(2 * ch, 2 * ch, 2, stride=2)
        self.m1 = conv(2 * ch, 2 * ch); self.m2 = conv(2 * ch, 2 * ch)
        self.up = nn.ConvTranspose2d(2 * ch, 2 * ch, 2, stride=2)
        self.u1 = conv(4 * ch, ch); self.u2 = conv(ch, ch); self.o = conv(ch, 1)
        self.films = nn.ModuleList([nn.Linear(emb, 2 * c) for c in (ch, 2 * ch, 2 * ch, ch)])
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)

    def _f(self, h, e, i):
        sc, sh = self.films[i](e).chunk(2, -1)
        return F.silu(h) * (1 + sc[:, :, None, None]) + sh[:, :, None, None]

    def forward(self, x, cnoise):
        B = x.shape[0]; x = x.view(B, 1, self.res, self.res)
        e = F.silu(self.e1(self.emb(cnoise)))
        h0 = self.c0(x)
        h1 = self.d2(self._f(self.d1(F.silu(h0)), e, 0))
        h = self.down(F.silu(h1))
        h = h + self.m2(self._f(self.m1(F.silu(h)), e, 1))
        h = self._f(self.up(h), e, 2)
        h = self.u2(F.silu(self.u1(torch.cat([h, h1], 1))))
        return self.o(self._f(h, e, 3)).view(B, -1)


class EDMDenoiser(nn.Module):
    """D(y, sigma) = mu + s [c_skip y_n + c_out F(c_in y_n, log(sigma_n)/4)] in original units."""
    def __init__(self, net, mu, s, sd):
        super().__init__()
        self.net = net
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.s, self.sd = float(s), float(sd)

    def forward(self, y, sigma):
        single = y.dim() == 1
        if single: y = y[None]
        sn = torch.as_tensor(sigma, dtype=y.dtype, device=y.device) / self.s
        sn = sn.expand(y.shape[0]) if sn.dim() == 0 else sn
        yn = (y - self.mu.to(y.dtype)) / self.s
        sd = self.sd
        c_skip = sd**2 / (sn**2 + sd**2); c_out = sn * sd / torch.sqrt(sn**2 + sd**2); c_in = 1 / torch.sqrt(sn**2 + sd**2)
        out = c_skip[:, None] * yn + c_out[:, None] * self.net(c_in[:, None] * yn, torch.log(sn) / 4)
        out = self.mu.to(y.dtype) + self.s * out
        return out[0] if single else out


def train_edm(net, X, steps, batch=512, lr=1e-3, sigma_range=(0.01, 2.0), ema=0.999, device="cuda",
              log_every=2000, seed=0, fresh_sampler=None):
    """Train with the EDM-weighted denoising loss. If fresh_sampler(n) is given, sample new data
    every step (infinite-data regime) instead of using X."""
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    mu = X.mean(0); Xc = X - mu
    s = float(Xc.pow(2).sum(1).mean().sqrt()); Xn = (Xc / s).to(device)
    sd = float(Xn.pow(2).mean().sqrt())
    net = net.to(device)
    ema_net = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05, final_div_factor=100)
    lo, hi = math.log(sigma_range[0] / s), math.log(sigma_range[1] / s)
    for it in range(steps):
        if fresh_sampler is None:
            xb = Xn[torch.randint(0, Xn.shape[0], (batch,), device=device)]
        else:
            xb = ((torch.as_tensor(fresh_sampler(batch), dtype=torch.float32, device=device) - mu.to(device)) / s)
        sb = torch.exp(torch.empty(batch, device=device).uniform_(lo, hi))
        yb = xb + sb[:, None] * torch.randn_like(xb)
        c_skip = sd**2 / (sb**2 + sd**2); c_out = sb * sd / torch.sqrt(sb**2 + sd**2); c_in = 1 / torch.sqrt(sb**2 + sd**2)
        Fp = net(c_in[:, None] * yb, torch.log(sb) / 4)
        tgt = (xb - c_skip[:, None] * yb) / c_out[:, None]
        loss = (Fp - tgt).pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for e, p in zip(ema_net, net.parameters()):
                e.mul_(ema).add_(p.detach(), alpha=1 - ema)
        if log_every and (it + 1) % log_every == 0:
            print(f"  step {it+1} loss {loss.item():.4f}", flush=True)
    with torch.no_grad():
        for e, p in zip(ema_net, net.parameters()):
            p.copy_(e)
    for p in net.parameters():
        p.requires_grad_(False)          # evaluation uses forward-mode AD only; avoid autograd graphs
    return EDMDenoiser(net.eval(), mu, s, sd).to(device)


class DDPMAsDenoiser(nn.Module):
    """Pretrained diffusers DDPM (eps-prediction, VP) as a VE denoiser on [-1,1] images:
    y = x0 + sigma*eps  <->  x_t = sqrt(abar_t) y,  sigma = sqrt((1-abar_t)/abar_t).
    D(y, sigma) = y - sigma * eps_hat(sqrt(abar_t) y, t), with t the nearest discrete step."""
    def __init__(self, unet, scheduler, shape=(3, 32, 32)):
        super().__init__()
        # fused scaled-dot-product attention has no forward-mode AD -> use the classic processor
        from diffusers.models.attention_processor import Attention, AttnProcessor
        for mod in unet.modules():
            if isinstance(mod, Attention):
                mod.set_processor(AttnProcessor())
        self.unet, self.shape = unet, shape
        abar = scheduler.alphas_cumprod.double()
        self.register_buffer("abar", abar)
        self.register_buffer("sig", ((1 - abar) / abar).sqrt())

    def t_for(self, sigma):
        return int(torch.argmin((self.sig - float(sigma)).abs()))

    def forward(self, y, sigma):
        single = y.dim() == 1
        if single: y = y[None]
        t = self.t_for(sigma); a = self.abar[t].to(y.dtype); s = self.sig[t].to(y.dtype)
        x = (a.sqrt() * y).view(-1, *self.shape)
        tt = torch.full((x.shape[0],), t, device=y.device, dtype=torch.long)
        eps = self.unet(x, tt).sample.reshape(y.shape[0], -1).to(y.dtype)
        out = y - s * eps
        return out[0] if single else out
