"""Physics checks on predicted 1 km maps (PLAN.md section 4.3 and the gates of section 6).

Used during training (epoch selection) and as gates in the decision rule. Distances are computed
inside each 27x55 patch, so clouds just outside a patch are invisible; full-granule mosaics remove
that limitation once pixel geolocation is available.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, stats

from .geometry import toa_insolation


def distance_to_cloud(cloud_mask: np.ndarray, pixel_km: float = 1.0) -> np.ndarray:
    """Euclidean distance (km) from each pixel to the nearest cloudy pixel of its own patch.

    Example:
        >>> m = np.zeros((1, 1, 3), bool); m[0, 0, 0] = True
        >>> distance_to_cloud(m)[0, 0].tolist()
        [0.0, 1.0, 2.0]

    Args:
        cloud_mask: Bool array (N, H, W).
        pixel_km: Pixel size.

    Returns:
        Float array (N, H, W); inf in patches without cloud.
    """
    m = np.asarray(cloud_mask, bool)
    out = np.full(m.shape, np.inf)
    for i in range(m.shape[0]):
        if m[i].any():
            out[i] = ndimage.distance_transform_edt(~m[i]) * pixel_km
    return out


def flux_vs_distance(maps, dist_km, bins):
    """Mean flux of clear pixels (0 < distance < inf) in distance bins.

    Returns:
        (bin centres, mean flux, count).
    """
    maps, d = np.asarray(maps, float), np.asarray(dist_km, float)
    ok = (d > 0) & np.isfinite(d) & np.isfinite(maps)
    idx = np.digitize(d[ok], bins) - 1
    vals = maps[ok]
    k = len(bins) - 1
    mean, n = np.full(k, np.nan), np.zeros(k, int)
    for i in range(k):
        sel = idx == i
        n[i] = sel.sum()
        if n[i]:
            mean[i] = vals[sel].mean()
    return 0.5 * (np.asarray(bins[:-1]) + np.asarray(bins[1:])), mean, n


def decay_score(curve) -> float:
    """Spearman rho of a flux-vs-distance curve against distance; -1 = monotone decay.

    Example:
        >>> decay_score(np.array([3.0, 2.0, 1.0]))
        -1.0
    """
    c = np.asarray(curve, float)
    c = c[np.isfinite(c)]
    if len(c) < 2:
        return np.nan
    return float(stats.spearmanr(np.arange(len(c)), c).statistic)


def bound_violation_fraction(maps, sza) -> float:
    """Fraction of pixels with flux < 0 or > S0*cos(SZA)."""
    maps = np.asarray(maps, float)
    s = toa_insolation(sza)
    return float(np.mean((maps < 0) | (maps > s)))


def percentile_spread(maps, q=(1, 99)) -> tuple[float, float]:
    """Low/high percentiles of all pixels; a steady widening over epochs signals extreme maps."""
    lo, hi = np.nanpercentile(np.asarray(maps, float), q)
    return float(lo), float(hi)


def spacing_peak_ratio(maps, period_px: float = 8.0) -> float:
    """Along-scan power at the footprint spacing relative to neighbouring frequencies.

    A large ratio means footprint-shaped artefacts (e.g. uniform residual redistribution).
    """
    x = np.asarray(maps, float)
    x = x - x.mean(axis=-1, keepdims=True)
    power = (np.abs(np.fft.rfft(x, axis=-1)) ** 2).reshape(-1, x.shape[-1] // 2 + 1).mean(axis=0)
    freqs = np.fft.rfftfreq(x.shape[-1])
    k = int(np.argmin(np.abs(freqs - 1.0 / period_px)))
    neigh = np.r_[power[max(1, k - 4):max(1, k - 1)], power[k + 2:k + 5]]
    return float(power[k] / max(np.median(neigh), 1e-12))


def far_clear_flux(maps, dist_km, min_km: float = 30.0) -> float:
    """Mean flux of clear pixels at least min_km from cloud (pure clear sky); NaN if none.

    Patches without cloud (distance inf) count as far clear sky.
    """
    maps, d = np.asarray(maps, float), np.asarray(dist_km, float)
    sel = (d >= min_km) & np.isfinite(maps)
    return float(maps[sel].mean()) if sel.any() else np.nan


def run_checks(maps, dist_km, sza, pcs_target=(70.0, 90.0), pcs_min_km: float = 20.0, period_px: float = 8.0,
               max_bound_violation: float = 0.001, max_spacing_peak: float = 5.0,
               distance_bins=(0.5, 2, 4, 8, 16, 32)) -> dict:
    """All physics checks with pass/fail gates.

    pcs_target defaults to 80 +- 10 W m^-2 (paper's far-field clear sky, ocean near noon).

    Returns:
        Dict with the individual values, per-check booleans and "passed".
    """
    curve = flux_vs_distance(maps, dist_km, np.asarray(distance_bins))[1]
    pcs = far_clear_flux(maps, dist_km, pcs_min_km)
    res = {
        "pcs_flux": pcs,
        "decay_score": decay_score(curve),
        "bound_violation": bound_violation_fraction(maps, sza),
        "spacing_peak": spacing_peak_ratio(maps, period_px),
        "p1_p99": percentile_spread(maps),
        "flux_vs_distance": curve.tolist(),
    }
    res["ok_pcs"] = bool(np.isnan(pcs) or pcs_target[0] <= pcs <= pcs_target[1])
    res["ok_bounds"] = res["bound_violation"] <= max_bound_violation
    res["ok_spacing"] = res["spacing_peak"] <= max_spacing_peak
    res["passed"] = res["ok_pcs"] and res["ok_bounds"] and res["ok_spacing"]
    return res
