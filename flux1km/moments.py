"""Per-footprint moments: pixel statistics, raw moments, and the exact polynomial bridge (M2).

The moments files store, per footprint and channel, the mean, std, skewness and kurtosis of the
MODIS pixels. From these the raw moments E[x], ..., E[x^4] are exact, so for any per-pixel model
that is a polynomial of degree <= 4 in each channel the footprint mean of its output is a linear
function of footprint-level features. That is what lets M2 train a pixel model on 5.26M footprints.
"""
from __future__ import annotations

import numpy as np


def moments_from_pixels(patches: np.ndarray, weights: np.ndarray | None = None):
    """Mean, std, skewness and Pearson kurtosis of each channel over the last two (pixel) axes.

    Used to build synthetic moments and, in E0, to check whether the real moments equal the
    (uniform or PSF-weighted) statistics of the image patches.

    Example:
        >>> m, s, sk, ku = moments_from_pixels(np.arange(4.0).reshape(1, 1, 2, 2))
        >>> float(m[0, 0]), round(float(s[0, 0]), 4)
        (1.5, 1.118)

    Args:
        patches: Array (..., H, W).
        weights: Optional non-negative (H, W) or (..., H, W) weights (normalised internally).

    Returns:
        (mean, std, skew, kurt), each of shape patches.shape[:-2]. Skew/kurt are NaN where std == 0.
    """
    x = np.asarray(patches, dtype=float)
    if weights is None:
        w = np.full(x.shape[-2:], 1.0 / (x.shape[-2] * x.shape[-1]))
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum(axis=(-2, -1), keepdims=True)
    mean = (x * w).sum(axis=(-2, -1))
    d = x - mean[..., None, None]
    m2 = (d**2 * w).sum(axis=(-2, -1))
    m3 = (d**3 * w).sum(axis=(-2, -1))
    m4 = (d**4 * w).sum(axis=(-2, -1))
    std = np.sqrt(m2)
    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.where(m2 > 0, m3 / m2**1.5, np.nan)
        kurt = np.where(m2 > 0, m4 / m2**2, np.nan)
    return mean, std, skew, kurt


def raw_moments(mean, std, skew, kurt, excess_kurtosis: bool = False) -> np.ndarray:
    """Raw moments E[x^k], k = 1..4, from mean/std/skew/kurt.

    Example:
        >>> raw_moments(2.0, 0.0, 0.0, 0.0).tolist()
        [2.0, 4.0, 8.0, 16.0]

    Args:
        mean, std, skew, kurt: Arrays of identical shape (NaN skew/kurt with std == 0 is fine).
        excess_kurtosis: True if `kurt` is Fisher (excess) kurtosis rather than Pearson.

    Returns:
        Array of shape mean.shape + (4,).

    Raises:
        ValueError: If the shapes differ.
    """
    arrs = [np.asarray(a, dtype=float) for a in (mean, std, skew, kurt)]
    if len({a.shape for a in arrs}) != 1:
        raise ValueError(f"shape mismatch: {[a.shape for a in arrs]}")
    m, s, sk, ku = arrs
    if excess_kurtosis:
        ku = ku + 3.0
    var = s**2
    mu3 = np.where(s > 0, sk * s**3, 0.0)
    mu4 = np.where(s > 0, ku * var**2, 0.0)
    e1 = m
    e2 = var + m**2
    e3 = mu3 + 3 * m * var + m**3
    e4 = mu4 + 4 * m * mu3 + 6 * m**2 * var + m**4
    return np.stack([e1, e2, e3, e4], axis=-1)


def footprint_polynomial_features(mean, std, skew, kurt, degree: int, excess_kurtosis: bool = False) -> np.ndarray:
    """Footprint-level features [E[x_c^k]] (channel-major, k = 1..degree) for M2.

    Example:
        >>> footprint_polynomial_features([[2.0]], [[0.0]], [[0.0]], [[0.0]], degree=2).tolist()
        [[2.0, 4.0]]

    Args:
        mean, std, skew, kurt: Arrays (N, C).
        degree: Polynomial degree, 1..4.

    Returns:
        Array (N, C * degree), same column order as pixel_polynomial_features.
    """
    _check_degree(degree)
    raw = raw_moments(mean, std, skew, kurt, excess_kurtosis)[..., :degree]  # (N, C, degree)
    return raw.reshape(raw.shape[0], -1)


def pixel_polynomial_features(patches: np.ndarray, degree: int) -> np.ndarray:
    """Per-pixel features [x_c^k] matching footprint_polynomial_features column order.

    Example:
        >>> pixel_polynomial_features(np.full((1, 1, 1, 1), 3.0), degree=2)[0, :, 0, 0].tolist()
        [3.0, 9.0]

    Args:
        patches: Array (N, C, H, W).
        degree: Polynomial degree, 1..4.

    Returns:
        Array (N, C * degree, H, W).
    """
    _check_degree(degree)
    x = np.asarray(patches, dtype=float)
    powers = np.stack([x ** (k + 1) for k in range(degree)], axis=2)  # (N, C, degree, H, W)
    return powers.reshape(x.shape[0], -1, *x.shape[-2:])


def homogeneity_cv(mean, std) -> np.ndarray:
    """Gao et al. (2012) Eq. 1: cv = mean over channels of std/mean, per footprint.

    Smaller is more homogeneous; Gao used cv < 0.2. NaN if any channel has zero mean.

    Example:
        >>> homogeneity_cv([[10.0, 20.0]], [[1.0, 4.0]]).round(6).tolist()
        [0.15]

    Args:
        mean, std: Arrays (N, C).

    Returns:
        Array (N,).
    """
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mean != 0, std / np.abs(mean), np.nan)
    return ratio.mean(axis=-1)


def _check_degree(degree: int) -> None:
    if not 1 <= degree <= 4:
        raise ValueError(f"degree must be in 1..4 (moments go up to kurtosis), got {degree}")
