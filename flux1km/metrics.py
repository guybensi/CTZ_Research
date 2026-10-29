"""Scores and statistical tests for the benchmark (PLAN.md sections 4.2-4.3)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage, stats

from .sampling import REGIMES


def _clean(y_true, y_pred, weights=None):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_pred.shape}")
    w = np.ones_like(y_true) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w) & (w > 0)
    return y_true[ok], y_pred[ok], w[ok]


def footprint_scores(y_true, y_pred, weights=None) -> dict:
    """RMSE, MAE, bias (pred - true), R^2 and n over finite pairs, optionally weighted.

    Example:
        >>> footprint_scores([1.0, 2.0], [2.0, 3.0])["bias"]
        1.0

    Raises:
        ValueError: If shapes differ.
    """
    y, p, w = _clean(y_true, y_pred, weights)
    if len(y) == 0:
        return {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan, "n": 0}
    e = p - y
    ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    return {
        "rmse": float(np.sqrt(np.average(e**2, weights=w))),
        "mae": float(np.average(np.abs(e), weights=w)),
        "bias": float(np.average(e, weights=w)),
        "r2": float(1 - np.sum(w * e**2) / ss_tot) if ss_tot > 0 else np.nan,
        "n": len(y),
    }


def scores_by(y_true, y_pred, labels) -> dict:
    """footprint_scores for every distinct label (e.g. regime, SZA bin)."""
    labels = np.asarray(labels)
    return {lab: footprint_scores(np.asarray(y_true)[labels == lab], np.asarray(y_pred)[labels == lab])
            for lab in dict.fromkeys(labels.tolist())}


def macro_rmse(y_true, y_pred, regimes, order=REGIMES) -> float:
    """Mean of the per-regime RMSEs over the regimes present (primary score, PLAN.md section 6).

    Example:
        >>> macro_rmse([0, 0], [1.0, 3.0], ["clear", "overcast"])
        2.0
    """
    by = scores_by(y_true, y_pred, regimes)
    vals = [by[r]["rmse"] for r in order if r in by and by[r]["n"] > 0]
    return float(np.mean(vals)) if vals else np.nan


def binned_residuals(resid, x, bins):
    """Mean/std/count of residuals in bins of x (e.g. SZA, VZA, RAA, glint).

    Returns:
        (bin centres, mean, std, count), each of length len(bins) - 1.
    """
    resid, x = np.asarray(resid, float), np.asarray(x, float)
    idx = np.digitize(x, bins) - 1
    k = len(bins) - 1
    centres = 0.5 * (np.asarray(bins[:-1]) + np.asarray(bins[1:]))
    mean, std, count = np.full(k, np.nan), np.full(k, np.nan), np.zeros(k, int)
    for i in range(k):
        r = resid[(idx == i) & np.isfinite(resid)]
        count[i] = len(r)
        if len(r):
            mean[i], std[i] = r.mean(), r.std()
    return centres, mean, std, count


def residual_trend(resid, x) -> tuple[float, float]:
    """Linear slope of residual vs x and its p-value; a flat model has slope ~ 0.

    Example:
        >>> round(residual_trend([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])[0], 6)
        1.0
    """
    resid, x = np.asarray(resid, float), np.asarray(x, float)
    ok = np.isfinite(resid) & np.isfinite(x)
    r = stats.linregress(x[ok], resid[ok])
    return float(r.slope), float(r.pvalue)


def swath_side_asymmetry(resid, raa, vza, vza_bins=(0, 15, 30, 45, 60, 90)) -> float:
    """Forward-scatter minus backscatter mean residual (RAA < 90 vs >= 90), matched by VZA bin.

    The sun-synchronous analogue of GERB's morning/afternoon test (Bertrand 2006). Bins are
    weighted by their smaller side's count. NaN if no bin contains both sides.
    """
    resid, raa, vza = (np.asarray(a, float) for a in (resid, raa, vza))
    diffs, wts = [], []
    for lo, hi in zip(vza_bins[:-1], vza_bins[1:]):
        b = (vza >= lo) & (vza < hi) & np.isfinite(resid)
        fwd, back = resid[b & (raa < 90)], resid[b & (raa >= 90)]
        if len(fwd) and len(back):
            diffs.append(fwd.mean() - back.mean())
            wts.append(min(len(fwd), len(back)))
    return float(np.average(diffs, weights=wts)) if diffs else np.nan


def unit_losses(y_true, y_pred, units):
    """Mean squared error per dependent unit (granule group or spatial block).

    Returns:
        (sorted unit labels, MSE per unit).
    """
    y, p, u = np.asarray(y_true, float), np.asarray(y_pred, float), np.asarray(units)
    labs = np.array(sorted(set(u.tolist())), dtype=object)
    out = []
    for lab in labs:
        sel = (u == lab) & np.isfinite(y) & np.isfinite(p)
        out.append(np.mean((p[sel] - y[sel]) ** 2) if sel.any() else np.nan)
    return labs, np.array(out)


def paired_bootstrap(loss_a, loss_b, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Bootstrap CI of mean(loss_a - loss_b) over paired units. Negative = A is better.

    Raises:
        ValueError: If the inputs differ in length.
    """
    a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    if a.shape != b.shape:
        raise ValueError("loss arrays must be paired")
    d = a - b
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_units": len(d)}


def wilcoxon_p(loss_a, loss_b) -> float:
    """Two-sided Wilcoxon signed-rank p-value of paired unit losses."""
    return float(stats.wilcoxon(np.asarray(loss_a, float), np.asarray(loss_b, float)).pvalue)


def diebold_mariano(loss_a, loss_b, lag: int | None = None) -> tuple[float, float]:
    """Diebold-Mariano test with a Newey-West (HAC) variance; negative stat = A has lower loss.

    Losses must be ordered (in time or along track) for the HAC correction to mean anything.

    Returns:
        (statistic, two-sided p-value); NaN if the loss difference is constant.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    lag = int(np.floor(n ** (1 / 3))) if lag is None else lag
    dc = d - d.mean()
    var = np.sum(dc**2) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * np.sum(dc[k:] * dc[:-k]) / n
    if var <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input).

    Example:
        >>> holm([0.01, 0.04, 0.03]).round(4).tolist()
        [0.03, 0.06, 0.06]
    """
    p = np.asarray(pvalues, float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def _ssim(a, b, sigma=1.5):
    rng_ = max(np.ptp(a), np.ptp(b), 1e-12)
    c1, c2 = (0.01 * rng_) ** 2, (0.03 * rng_) ** 2
    mu_a, mu_b = ndimage.gaussian_filter(a, sigma), ndimage.gaussian_filter(b, sigma)
    saa = ndimage.gaussian_filter(a * a, sigma) - mu_a**2
    sbb = ndimage.gaussian_filter(b * b, sigma) - mu_b**2
    sab = ndimage.gaussian_filter(a * b, sigma) - mu_a * mu_b
    return float(np.mean(((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (saa + sbb + c2))))


def pixel_scores(truth: np.ndarray, pred: np.ndarray) -> dict:
    """Pixel-level RMSE, MAE, correlation and mean SSIM over maps (synthetic-truth experiment).

    Args:
        truth, pred: Arrays (N, H, W).
    """
    t, p = np.asarray(truth, float), np.asarray(pred, float)
    s = footprint_scores(t.ravel(), p.ravel())
    ok = np.isfinite(t) & np.isfinite(p)
    corr = float(np.corrcoef(t[ok], p[ok])[0, 1]) if np.std(p[ok]) > 0 else np.nan
    return {"rmse": s["rmse"], "mae": s["mae"], "bias": s["bias"], "corr": corr,
            "ssim": float(np.mean([_ssim(a, b) for a, b in zip(t, p)]))}
