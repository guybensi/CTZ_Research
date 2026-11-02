"""PyTorch building blocks for the neural methods (M0 ANNs, M1-ANN, M3, M5, M6).

PyTorch is imported only inside these functions, so `import flux1km` stays NumPy-only and the
core test suite runs without it. Inside the GPU Docker image CTZ_REQUIRE_GPU=1 is set; then
training refuses to fall back silently to the CPU (CLAUDE.md: "mark GPU as required inside the
container").

All map networks predict a per-pixel albedo through a sigmoid and multiply it by the pixel's
insolation S0 cos(SZA), so every map is bounded by S0 mu0 by construction (PLAN.md section 4.3,
"no value exceeds S0 mu0").
"""
from __future__ import annotations

import copy
import os

import numpy as np


def require_torch():
    """Import and return torch, with a clear message if it is missing.

    Raises:
        ImportError: If PyTorch is not installed (pip install -e ".[ml]").
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("this method needs PyTorch: pip install -e '.[ml]' (or use the GPU Docker image)") from e
    return torch


def device():
    """cuda if available; raise if CTZ_REQUIRE_GPU=1 and there is no GPU, else cpu.

    Raises:
        RuntimeError: When a GPU is required but not visible.
    """
    torch = require_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("CTZ_REQUIRE_GPU") == "1":
        raise RuntimeError("CTZ_REQUIRE_GPU=1 but no CUDA device is visible (run docker with --gpus all)")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed torch and numpy for reproducible training."""
    torch = require_torch()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))


# ------------------------------------------------------------------ architectures
def make_mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int = 1, dropout: float = 0.0):
    """Dense ReLU network in_dim -> hidden... -> out_dim (tabular, footprint-level)."""
    nn = require_torch().nn
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def make_pixel_mlp(in_ch: int, hidden: tuple[int, ...] = (64, 64), out_ch: int = 1):
    """Per-pixel MLP as a stack of 1x1 convolutions: no spatial context at all (M3)."""
    nn = require_torch().nn
    layers, d = [], in_ch
    for h in hidden:
        layers += [nn.Conv2d(d, h, 1), nn.ReLU()]
        d = h
    layers.append(nn.Conv2d(d, out_ch, 1))
    return nn.Sequential(*layers)


def make_context_encoder(in_ch: int, width: int = 16, out_ch: int = 1, layers: int = 3):
    """Small 3x3 convolutional encoder (same-size output): the residual context branch of M6b."""
    nn = require_torch().nn
    mods, d = [], in_ch
    for _ in range(layers):
        mods += [nn.Conv2d(d, width, 3, padding=1, padding_mode="replicate"), nn.ReLU()]
        d = width
    mods.append(nn.Conv2d(d, out_ch, 1))
    return nn.Sequential(*mods)


def make_unet(in_ch: int, base: int = 16, depth: int = 2, out_ch: int = 1):
    """A compact UNet (CS-SUNet style) for 27x55 patches; pads to a multiple of 2**depth and crops back."""
    torch = require_torch()
    nn, F = torch.nn, torch.nn.functional

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, padding_mode="replicate"), nn.ReLU(),
                             nn.Conv2d(o, o, 3, padding=1, padding_mode="replicate"), nn.ReLU())

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            chans = [base * 2**k for k in range(depth + 1)]
            self.down = nn.ModuleList([block(in_ch, chans[0])] + [block(chans[k], chans[k + 1]) for k in range(depth)])
            self.up = nn.ModuleList([nn.ConvTranspose2d(chans[k + 1], chans[k], 2, stride=2) for k in reversed(range(depth))])
            self.dec = nn.ModuleList([block(2 * chans[k], chans[k]) for k in reversed(range(depth))])
            self.head = nn.Conv2d(chans[0], out_ch, 1)
            self.m = 2**depth

        def features(self, x):
            h, w = x.shape[-2:]
            ph, pw = (-h) % self.m, (-w) % self.m
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
            skips = []
            for k, d in enumerate(self.down):
                x = d(x)
                if k < len(self.down) - 1:
                    skips.append(x)
                    x = F.max_pool2d(x, 2)
            for up, dec in zip(self.up, self.dec):
                x = dec(torch.cat([up(x), skips.pop()], dim=1))
            return x[..., :h, :w]

        def forward(self, x):
            return self.head(self.features(x))

    return UNet()


# ------------------------------------------------------------------ training: aggregation loss
def _flux(raw, ins):
    return require_torch().sigmoid(raw.squeeze(1)) * ins


def train_map_net(net, X: np.ndarray, ins: np.ndarray, W: np.ndarray, y: np.ndarray, sw: np.ndarray | None = None,
                  val: tuple | None = None, *, epochs: int = 60, lr: float = 1e-3, batch_size: int = 32,
                  patience: int = 8, weight_decay: float = 0.0, extra_loss=None, augment: bool = False,
                  seed: int = 0, forward=None, subset_fraction: float = 1.0) -> dict:
    """Train a map network with the aggregation loss MSE(sum_i w_i f(x_i), F) (PLAN.md section 1).

    f = sigmoid(net(x)) * S0 mu0, so the map is bounded. Early stopping on the validation
    footprint RMSE (val = (X, ins, W, y)); the best state is restored.

    Example:
        >>> # hist = train_map_net(make_pixel_mlp(F), X, ins, W, y, val=(Xv, iv, Wv, yv), epochs=30)

    Args:
        net: torch module mapping (B, F, H, W) -> (B, 1, H, W) raw logits.
        X: (N, F, H, W) standardised inputs.
        ins: (N, H, W) insolation.
        W: (N, H, W) aggregation weights (rows sum to 1).
        y: (N,) footprint labels (W m^-2).
        sw: (N,) loss weights (sampling strategy S7, etc.); default ones.
        val: Optional (X, ins, W, y) for early stopping.
        epochs, lr, batch_size, patience, weight_decay: Optimiser settings.
        extra_loss: Optional callable(xb, flux_b, idx, flipped) -> scalar tensor, added to the loss
            (smoothness, manifold, distillation, ...). idx are the batch's row indices.
        augment: Random left-right flips of inputs, insolation and weights together. A left-right
            flip mirrors the PSF, which is exactly what toggling FOVorientation means (PLAN.md 2.1).
        seed: Random seed.
        forward: Optional callable(net, xb, idx) -> raw logits, for networks with extra inputs.
        subset_fraction: Random-subset aggregation (Fan 2021): keep this fraction of pixels in each
            footprint's weights at every step (1.0 = off).

    Returns:
        {"train_loss": [...], "val_rmse": [...], "best_epoch": int}.

    Raises:
        ValueError: On inconsistent shapes or no finite labels.
    """
    torch = require_torch()
    set_seed(seed)
    dev = device()
    n = len(y)
    if not (X.shape[0] == ins.shape[0] == W.shape[0] == n):
        raise ValueError("X, ins, W and y must have the same number of footprints")
    ok = np.isfinite(y)
    if not ok.any():
        raise ValueError("no finite labels")
    sw = np.ones(n) if sw is None else np.asarray(sw, float)
    sw = np.where(ok, sw, 0.0)
    scale = float(np.nanstd(y[ok])) or 1.0
    t = {k: torch.as_tensor(np.nan_to_num(a), dtype=torch.float32) for k, a in
         (("X", X), ("I", ins), ("W", W), ("y", y), ("s", sw))}
    net = net.to(dev)
    fwd = forward or (lambda m, xb, idx: m(xb))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(int(seed))
    hist = {"train_loss": [], "val_rmse": [], "best_epoch": -1}
    best, best_state, bad = np.inf, copy.deepcopy(net.state_dict()), 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            xb, ib, wb = t["X"][idx].to(dev), t["I"][idx].to(dev), t["W"][idx].to(dev)
            flipped = bool(augment and torch.rand(1, generator=gen).item() < 0.5)
            if flipped:
                xb, ib, wb = xb.flip(-1), ib.flip(-1), wb.flip(-1)
            if subset_fraction < 1.0:
                keep = (torch.rand(wb.shape, generator=gen) < subset_fraction).to(dev)
                wb = wb * keep
                wb = wb / wb.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            flux = _flux(fwd(net, xb, idx), ib)
            pred = (flux * wb).sum(dim=(1, 2))
            loss = (t["s"][idx].to(dev) * (pred - t["y"][idx].to(dev)) ** 2).mean() / scale**2
            if extra_loss is not None:
                loss = loss + extra_loss(xb, flux, idx, flipped)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        hist["train_loss"].append(tot / n)
        if val is not None and len(val[3]):
            pv = (predict_map_net(net, val[0], val[1], forward=forward) * val[2]).sum(axis=(1, 2))
            rmse = float(np.sqrt(np.nanmean((pv - val[3]) ** 2)))
            hist["val_rmse"].append(rmse)
            if rmse < best - 1e-6:
                best, best_state, bad, hist["best_epoch"] = rmse, copy.deepcopy(net.state_dict()), 0, ep
            else:
                bad += 1
                if bad >= patience:
                    break
    if val is not None and hist["best_epoch"] >= 0:
        net.load_state_dict(best_state)
    return hist


def predict_map_net(net, X: np.ndarray, ins: np.ndarray, forward=None, batch_size: int = 64) -> np.ndarray:
    """Per-pixel flux (N, H, W) of a trained map network (same parameterisation as training)."""
    torch = require_torch()
    dev = device()
    fwd = forward or (lambda m, xb, idx: m(xb))
    net.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(X), batch_size):
            idx = torch.arange(b, min(b + batch_size, len(X)))
            xb = torch.as_tensor(np.nan_to_num(X[b:b + batch_size]), dtype=torch.float32, device=dev)
            ib = torch.as_tensor(ins[b:b + batch_size], dtype=torch.float32, device=dev)
            out.append(_flux(fwd(net, xb, idx), ib).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,) + tuple(np.shape(ins)[1:]))


def pretrain_map_net(net, X: np.ndarray, ins: np.ndarray, target: np.ndarray, *, epochs: int = 10, lr: float = 1e-3,
                     batch_size: int = 32, seed: int = 0) -> list[float]:
    """Pixel-wise MSE pretraining towards a target map (M6c: initialise the UNet from M2)."""
    torch = require_torch()
    set_seed(seed)
    dev = device()
    net = net.to(dev)
    tx = torch.as_tensor(np.nan_to_num(X), dtype=torch.float32)
    ti = torch.as_tensor(ins, dtype=torch.float32)
    tt = torch.as_tensor(np.nan_to_num(target), dtype=torch.float32)
    scale = float(np.nanstd(target)) or 1.0
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(int(seed))
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(len(tx), generator=gen)
        tot = 0.0
        for b in range(0, len(tx), batch_size):
            idx = perm[b:b + batch_size]
            flux = _flux(net(tx[idx].to(dev)), ti[idx].to(dev))
            loss = ((flux - tt[idx].to(dev)) ** 2).mean() / scale**2
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        losses.append(tot / len(tx))
    return losses


def edge_aware_smoothness(xb, flux, sigma: float = 1.0):
    """Edge-aware TV: mean |grad f| * exp(-|grad x| / sigma). Smooths only where inputs are smooth (M5)."""
    torch = require_torch()
    fx, fy = (flux[..., :, 1:] - flux[..., :, :-1]).abs(), (flux[..., 1:, :] - flux[..., :-1, :]).abs()
    gx = (xb[..., :, 1:] - xb[..., :, :-1]).abs().mean(1)
    gy = (xb[..., 1:, :] - xb[..., :-1, :]).abs().mean(1)
    scale = flux.detach().abs().mean().clamp_min(1e-6)
    return (fx * torch.exp(-gx / sigma)).mean() / scale + (fy * torch.exp(-gy / sigma)).mean() / scale


def manifold_penalty(xb, flux, n_pairs: int = 2048, bandwidth: float = 1.0, generator=None):
    """Feature-space (manifold) regularisation (Law 2018): penalise output differences of pixels with similar inputs.

    Random pixel pairs within the batch; weight exp(-|x_i - x_j|^2 / (2 F bandwidth^2)). Uses no
    spatial proximity, so it cannot smear cloud signal into neighbouring clear pixels.
    """
    torch = require_torch()
    b, f, h, w = xb.shape
    xs = xb.permute(0, 2, 3, 1).reshape(-1, f)
    fs = flux.reshape(-1)
    i = torch.randint(0, len(fs), (n_pairs,), generator=generator).to(xb.device)
    j = torch.randint(0, len(fs), (n_pairs,), generator=generator).to(xb.device)
    k = torch.exp(-((xs[i] - xs[j]) ** 2).sum(1) / (2 * f * bandwidth**2))
    scale = fs.detach().abs().mean().clamp_min(1e-6)
    return (k * (fs[i] - fs[j]) ** 2).mean() / scale**2


# ------------------------------------------------------------------ training: tabular (footprint-level)
def train_tabular_net(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, val: tuple | None = None, *,
                      hidden: tuple[int, ...] = (64, 32), loss: str = "mse", l1: tuple[float, ...] = (),
                      epochs: int = 200, lr: float = 1e-3, batch_size: int = 256, patience: int = 10,
                      min_delta: float = 0.0, seed: int = 0, dropout: float = 0.0):
    """Train a dense network on footprint features; returns a predict(X) -> (N,) callable.

    X is expected to be standardised already, and y in the training space of the caller (flux,
    albedo or log1p flux). Early stopping on validation RMSE in the same space: training stops when
    the RMSE has not improved by more than min_delta for `patience` epochs (the advisor's
    "change in RMSE" rule with min_delta = 0.001).

    Args:
        X, y, w: Training features (N, F), targets (N,), loss weights (N,).
        val: Optional (Xv, yv).
        hidden: Hidden layer widths.
        loss: "mse", "huber" or "wmae_log1p" (the advisor's weighted MAE: weight log1p|y| + 1).
        l1: L1 coefficients for the weights of each hidden layer (missing = 0).
        epochs, lr, batch_size, patience, min_delta, seed, dropout: Training settings.

    Raises:
        ValueError: For an unknown loss or no finite training rows.
    """
    torch = require_torch()
    if loss not in ("mse", "huber", "wmae_log1p"):
        raise ValueError(f"unknown loss {loss!r}")
    set_seed(seed)
    dev = device()
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ok &= w > 0
    if not ok.any():
        raise ValueError("no finite training rows")
    # standardise the target for mse/huber (the advisor's wmae_log1p works in raw log1p space, as in the original)
    y0, ys = (0.0, 1.0) if loss == "wmae_log1p" else (float(np.mean(y[ok])), float(np.std(y[ok])) or 1.0)
    tx = torch.as_tensor(X[ok], dtype=torch.float32)
    ty = torch.as_tensor((y[ok] - y0) / ys, dtype=torch.float32)
    tw = torch.as_tensor(w[ok], dtype=torch.float32)
    net = make_mlp(X.shape[1], hidden, 1, dropout).to(dev)
    linears = [m for m in net if isinstance(m, torch.nn.Linear)][:-1]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(int(seed))
    yscale = 1.0 if loss != "wmae_log1p" else (float(ty.std()) or 1.0)

    def crit(p, t, wt):
        e = p - t
        if loss == "mse":
            v = e**2 / yscale**2
        elif loss == "huber":
            v = torch.nn.functional.huber_loss(p / yscale, t / yscale, reduction="none")
        else:
            v = (torch.log1p(t.abs()) + 1.0) * e.abs()
        return (wt * v).mean()

    def predict(Z):
        net.eval()
        with torch.no_grad():
            z = torch.as_tensor(np.nan_to_num(np.asarray(Z, float)), dtype=torch.float32, device=dev)
            return y0 + ys * net(z).squeeze(1).cpu().numpy().astype(float)

    best, best_state, bad = np.inf, copy.deepcopy(net.state_dict()), 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(tx), generator=gen)
        for b in range(0, len(tx), batch_size):
            idx = perm[b:b + batch_size]
            p = net(tx[idx].to(dev)).squeeze(1)
            lo = crit(p, ty[idx].to(dev), tw[idx].to(dev))
            for c, lin in zip(l1, linears):
                lo = lo + c * lin.weight.abs().sum()
            opt.zero_grad()
            lo.backward()
            opt.step()
        if val is not None and len(val[1]):
            rmse = float(np.sqrt(np.nanmean((predict(val[0]) - val[1]) ** 2)))
            if rmse < best - min_delta:
                best, best_state, bad = rmse, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if val is not None and np.isfinite(best):
        net.load_state_dict(best_state)
    return predict
