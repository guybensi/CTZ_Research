"""Footprint aggregation operators H: pixel flux map -> footprint flux.

Every method is scored through the same operator (PLAN.md section 4.2). The stored moments are a
weighted average over the footprint ellipse inside the 27x55 patch (advisor, Nov 2026), so the
primary operator is "fov_ellipse": a tilted elliptical Gaussian whose tilt side follows
FOVorientation. Its shape is provisional, fitted to the stored moments of the sample granule with
fit_fov_ellipse; refit it on the full data in E0, or replace it with the real CERES PSF once the
data-building code arrives. "uniform" and "gaussian_psf" are sensitivity checks.
"""
from __future__ import annotations

import numpy as np

# Rough default: CERES 95% energy ellipse ~38 x 31 km at nadir (CERES FAQ) on a 27 x 55 patch.
# FWHM in pixels; replace with the real PSF once known (open question 2 in PLAN.md).
DEFAULT_PSF_FWHM = (14.0, 24.0)  # (y, x)


def uniform_weights(h: int, w: int) -> np.ndarray:
    """Equal weights over an h x w patch, summing to 1.

    Example:
        >>> uniform_weights(2, 2).tolist()
        [[0.25, 0.25], [0.25, 0.25]]

    Raises:
        ValueError: If h or w is not positive.
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"patch shape must be positive, got {(h, w)}")
    return np.full((h, w), 1.0 / (h * w))


def gaussian_psf_weights(h: int, w: int, fwhm_y: float, fwhm_x: float) -> np.ndarray:
    """Normalised elliptical Gaussian centred on the patch, an approximate CERES PSF.

    Example:
        >>> wts = gaussian_psf_weights(3, 3, 1.0, 1.0)
        >>> int(wts.argmax()), round(float(wts.sum()), 6)
        (4, 1.0)

    Args:
        h, w: Patch shape in pixels.
        fwhm_y, fwhm_x: Full width at half maximum along each axis, in pixels.

    Raises:
        ValueError: If a FWHM is not positive.
    """
    if fwhm_y <= 0 or fwhm_x <= 0:
        raise ValueError("FWHM must be positive")
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    sy, sx = fwhm_y / 2.3548, fwhm_x / 2.3548
    g = np.exp(-0.5 * (((yy - (h - 1) / 2) / sy) ** 2 + ((xx - (w - 1) / 2) / sx) ** 2))
    return g / g.sum()


# (sigma_major, sigma_minor, tilt in degrees) in pixels, fitted on data/sample (A2014060.0245):
# relative error of mean/std vs the stored moments drops from 0.23 (uniform) to 0.09.
FOV_ELLIPSE_DEFAULT = (9.6, 5.6, 47.7)
OPERATORS = ("fov_ellipse", "uniform", "gaussian_psf")


def elliptical_gaussian_weights(h: int, w: int, sigma_major: float, sigma_minor: float, angle_deg: float) -> np.ndarray:
    """Normalised tilted elliptical Gaussian centred on the patch.

    The major axis is rotated by angle_deg from the patch x axis (towards +y for a positive angle).

    Example:
        >>> wts = elliptical_gaussian_weights(5, 9, 3.0, 1.0, 0.0)
        >>> int(wts.argmax()), round(float(wts.sum()), 6), bool(wts[2, 0] > wts[0, 4])
        (22, 1.0, True)

    Raises:
        ValueError: If a sigma is not positive.
    """
    if sigma_major <= 0 or sigma_minor <= 0:
        raise ValueError("sigmas must be positive")
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    yy -= (h - 1) / 2
    xx -= (w - 1) / 2
    th = np.radians(angle_deg)
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)
    g = np.exp(-0.5 * ((u / sigma_major) ** 2 + (v / sigma_minor) ** 2))
    return g / g.sum()


def fov_weights(orientation, h: int = 27, w: int = 55, params=FOV_ELLIPSE_DEFAULT) -> np.ndarray:
    """Per-footprint ellipse weights (N, h, w); FOVorientation 1 tilts by +angle, 0 by -angle.

    Which flag value means which side is not verified yet (PLAN.md section 2.1); on the sample
    both mappings fit equally well.

    Example:
        >>> fov_weights(np.array([0, 1])).shape
        (2, 27, 55)
    """
    o = np.asarray(orientation).astype(int).ravel()
    a, b, th = params
    pos = elliptical_gaussian_weights(h, w, a, b, th)
    neg = elliptical_gaussian_weights(h, w, a, b, -th)
    return np.where(o[:, None, None] == 1, pos, neg)


def fit_fov_ellipse(patches: np.ndarray, mean: np.ndarray, std: np.ndarray, orientation,
                    x0=FOV_ELLIPSE_DEFAULT) -> tuple[tuple[float, float, float], float]:
    """Fit (sigma_major, sigma_minor, angle) so weighted patch mean/std reproduce stored moments.

    The E0 tool for recovering the aggregation operator from the data itself.

    Args:
        patches: (N, C, H, W) radiances.
        mean, std: (N, C) stored moments of the same footprints.
        orientation: (N,) FOVorientation flags.
        x0: Starting parameters.

    Returns:
        (fitted parameters, RMS relative error of mean and std).
    """
    from scipy import optimize

    from .moments import moments_from_pixels

    x = np.asarray(patches, float)
    h, w = x.shape[-2:]

    def loss(p):
        if p[0] <= 0.5 or p[1] <= 0.5:
            return 1e9
        wts = fov_weights(orientation, h, w, tuple(p))[:, None]
        m, s, _, _ = moments_from_pixels(x, wts)
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.concatenate([(m / mean - 1).ravel(), (s / std - 1).ravel()])
        return float(np.nanmean(e**2))

    res = optimize.minimize(loss, np.asarray(x0, float), method="Nelder-Mead", options={"maxiter": 800})
    return tuple(float(v) for v in res.x), float(np.sqrt(res.fun))


def aggregate(maps: np.ndarray, weights: np.ndarray, nan_policy: str = "renormalize") -> np.ndarray:
    """Weighted footprint mean sum_i w_i f_i of each pixel map.

    Example:
        >>> float(aggregate(np.ones((1, 2, 2)) * 3, uniform_weights(2, 2))[0])
        3.0

    Args:
        maps: Array (N, H, W) of per-pixel flux.
        weights: Non-negative (H, W) shared weights or (N, H, W) per-footprint weights;
            normalised internally.
        nan_policy: "renormalize" drops NaN pixels and renormalises the remaining weights
            (all-NaN -> NaN); "propagate" returns NaN for any footprint containing NaN.

    Returns:
        Array (N,).

    Raises:
        ValueError: On shape mismatch, negative weights or unknown nan_policy.
    """
    maps = np.asarray(maps, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if nan_policy not in ("renormalize", "propagate"):
        raise ValueError(f"unknown nan_policy {nan_policy!r}")
    if weights.shape[-2:] != maps.shape[-2:] or weights.ndim not in (2, 3):
        raise ValueError(f"weights {weights.shape} do not match maps {maps.shape}")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    w = np.broadcast_to(weights, maps.shape)
    if nan_policy == "propagate":
        return (maps * w).sum(axis=(-2, -1)) / w.sum(axis=(-2, -1))
    valid = np.isfinite(maps)
    wv = np.where(valid, w, 0.0)
    total = wv.sum(axis=(-2, -1))
    num = np.where(valid, maps, 0.0) * wv
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, num.sum(axis=(-2, -1)) / total, np.nan)


def shift_weights(weights: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift (H, W) weights by whole pixels with zero fill, then renormalise.

    Used for the geolocation-jitter sensitivity test (PLAN.md section 4.2).

    Example:
        >>> shift_weights(np.eye(2) / 2, 0, 1).tolist()
        [[0.0, 1.0], [0.0, 0.0]]

    Raises:
        ValueError: If the shift is at least the patch size.
    """
    h, w = weights.shape
    if abs(dy) >= h or abs(dx) >= w:
        raise ValueError(f"shift {(dy, dx)} too large for patch {(h, w)}")
    out = np.zeros_like(weights, dtype=float)
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = weights[ys, xs]
    return out / out.sum()


def random_subset_weights(weights: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Keep a random `fraction` of pixels and renormalise (Fan 2021 random-subset regularisation).

    Example:
        >>> wts = random_subset_weights(uniform_weights(4, 4), 0.5, np.random.default_rng(0))
        >>> round(float(wts.sum()), 6)
        1.0

    Raises:
        ValueError: If fraction is not in (0, 1].
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    keep = rng.random(weights.shape) < fraction
    if not keep.any():
        keep.flat[rng.integers(keep.size)] = True
    out = np.where(keep, weights, 0.0)
    return out / out.sum()


def get_operator(name: str, h: int, w: int, orientation=None) -> np.ndarray:
    """Return the named aggregation weights: "fov_ellipse", "uniform" or "gaussian_psf".

    "fov_ellipse" needs the footprints' FOVorientation flags and returns (N, h, w); without them
    it falls back to the untilted ellipse (h, w). The others return shared (h, w) weights.

    Example:
        >>> get_operator("uniform", 27, 55).shape, get_operator("fov_ellipse", 27, 55, [0, 1, 1]).shape
        ((27, 55), (3, 27, 55))

    Raises:
        KeyError: For an unknown operator name.
    """
    if name == "fov_ellipse":
        if orientation is None:
            a, b, _ = FOV_ELLIPSE_DEFAULT
            return elliptical_gaussian_weights(h, w, a, b, 0.0)
        return fov_weights(orientation, h, w)
    if name == "uniform":
        return uniform_weights(h, w)
    if name == "gaussian_psf":
        return gaussian_psf_weights(h, w, *DEFAULT_PSF_FWHM)
    raise KeyError(f"unknown aggregation operator {name!r}")
