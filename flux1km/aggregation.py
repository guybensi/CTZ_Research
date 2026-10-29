"""Footprint aggregation operators H: pixel flux map -> footprint flux.

Every method is scored through the same operator (PLAN.md section 4.2). Until the CERES PSF used
to build the data is known, the primary operator is the uniform patch mean and an approximate
Gaussian PSF is the sensitivity check.
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


def get_operator(name: str, h: int, w: int) -> np.ndarray:
    """Return the named aggregation weights: "uniform" or "gaussian_psf".

    Example:
        >>> get_operator("uniform", 27, 55).shape
        (27, 55)

    Raises:
        KeyError: For an unknown operator name.
    """
    if name == "uniform":
        return uniform_weights(h, w)
    if name == "gaussian_psf":
        return gaussian_psf_weights(h, w, *DEFAULT_PSF_FWHM)
    raise KeyError(f"unknown aggregation operator {name!r}")
