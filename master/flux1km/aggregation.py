"""Footprint aggregation operators H: pixel flux map -> footprint flux.

Every method is scored through the same operator (PLAN.md section 4.2). The stored moments are a
weighted average over the footprint ellipse inside the 27x55 patch (advisor, Nov 2026), so the
exact operator is "ceres_psf": the advisor's CERES PSF (ceres_psf.csv, resampled to 27x55 by
load_ceres_psf), mirrored left-right where FOVorientation == 0. Until that CSV is in the repo the
primary operator is "fov_ellipse", a tilted elliptical Gaussian fitted to the stored moments
(fit_fov_ellipse); recover_psf_from_moments recovers the PSF itself from enough footprints.
"uniform" and "gaussian_psf" are sensitivity checks.
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
OPERATORS = ("ceres_psf", "fov_ellipse", "uniform", "gaussian_psf")


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
    """Per-footprint ellipse weights (N, h, w); FOVorientation 0 is the left-right mirror of 1.

    In the advisor's code, orientation 0 means the PSF is applied np.fliplr (scan direction); for a
    centred ellipse that is the same as flipping the sign of the tilt.

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
                    x0=FOV_ELLIPSE_DEFAULT, maxiter: int = 800) -> tuple[tuple[float, float, float], float]:
    """Fit (sigma_major, sigma_minor, angle) so weighted patch mean/std reproduce stored moments.

    The E0 tool for recovering the aggregation operator from the data itself.

    Args:
        patches: (N, C, H, W) radiances.
        mean, std: (N, C) stored moments of the same footprints.
        orientation: (N,) FOVorientation flags.
        x0: Starting parameters.
        maxiter: Nelder-Mead iteration limit.

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

    res = optimize.minimize(loss, np.asarray(x0, float), method="Nelder-Mead", options={"maxiter": maxiter})
    return tuple(float(v) for v in res.x), float(np.sqrt(res.fun))


def load_ceres_psf(path, h: int = 27, w: int = 55) -> np.ndarray:
    """Load the CERES PSF CSV exactly as the advisor's Generate_AllSky_2.py does.

    The CSV is the NOAA-20 CERES PSF on a 250 m grid. Steps: drop the last row, append a zero
    column, NaN -> 0, Fourier-resample to w columns and h rows (scipy.signal.resample), normalise to
    sum 1. Fourier resampling can leave small negative lobes; they are kept, to match the stored
    moments exactly.

    Returns:
        (h, w) weights for FOVorientation == 1.
    """
    import pandas as pd
    from scipy import signal

    psf = np.asarray(pd.read_csv(path), dtype=float)[:-1, :]
    psf = np.concatenate([psf, np.zeros((psf.shape[0], 1))], axis=1)
    psf = np.where(np.isnan(psf), 0.0, psf)
    psf = signal.resample(psf, w, axis=1)
    psf = signal.resample(psf, h, axis=0)
    return psf / psf.sum()


def oriented_psf(psf: np.ndarray, orientation) -> np.ndarray:
    """Per-footprint PSF (N, h, w): as stored where FOVorientation == 1, np.fliplr where 0.

    Example:
        >>> p = np.array([[0.0, 1.0]])
        >>> oriented_psf(p, [1, 0])[:, 0].tolist()
        [[0.0, 1.0], [1.0, 0.0]]
    """
    o = np.asarray(orientation).astype(int).ravel()
    return np.where(o[:, None, None] == 1, psf, psf[:, ::-1])


def recover_psf_from_moments(patches: np.ndarray, mean: np.ndarray, orientation, ridge: float = 1e-6) -> np.ndarray:
    """Recover the (h, w) PSF from stored footprint means by linear least squares.

    The stored mean of channel c is sum_p w_p x_cp exactly, so every (footprint, channel) pair is
    one linear equation in the h*w weights (mirrored for FOVorientation == 0). Needs more equations
    than pixels, i.e. N * C > h * w (about 60 footprints for 27 channels on 27x55). An E0 fallback
    if ceres_psf.csv is not available.

    Args:
        patches: (N, C, H, W) radiances as stored in images/.
        mean: (N, C) stored footprint means.
        orientation: (N,) FOVorientation.
        ridge: Tikhonov regularisation (relative to the mean diagonal).

    Returns:
        (H, W) weights for FOVorientation == 1, normalised to sum 1.

    Raises:
        ValueError: If there are fewer equations than unknowns.
    """
    x = np.asarray(patches, float)
    n, c, h, w = x.shape
    if n * c <= h * w:
        raise ValueError(f"{n * c} equations for {h * w} unknowns: need more footprints")
    o = np.asarray(orientation).astype(int).ravel()
    x = np.where(o[:, None, None, None] == 1, x, x[..., ::-1])  # put every patch in orientation-1 frame
    A = x.reshape(n * c, h * w)
    b = np.asarray(mean, float).reshape(n * c)
    ok = np.all(np.isfinite(A), axis=1) & np.isfinite(b)
    A, b = A[ok], b[ok]
    scale = np.abs(b) + 1e-12  # relative residuals: channels have very different magnitudes
    A, b = A / scale[:, None], b / scale
    ata = A.T @ A
    wts = np.linalg.solve(ata + ridge * np.trace(ata) / ata.shape[0] * np.eye(ata.shape[0]), A.T @ b)
    wts = wts.reshape(h, w)
    return wts / wts.sum()


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


def scaled_fov_operator(scale: float, h: int = 27, w: int = 55, orientation=None) -> np.ndarray:
    """The fov_ellipse operator with both sigmas multiplied by `scale` (aggregation-size study).

    Used by the synthetic-truth study (PLAN.md section 4.3, step 4) to repeat the experiment at
    several footprint sizes. Reachable through get_operator as "fov_ellipse:<scale>".

    Example:
        >>> a = scaled_fov_operator(0.5, 27, 55); b = scaled_fov_operator(1.0, 27, 55)
        >>> bool(a.max() > b.max()), a.shape
        (True, (27, 55))

    Args:
        scale: Positive factor applied to sigma_major and sigma_minor.
        h, w: Patch shape.
        orientation: Optional (N,) FOVorientation; then the result is (N, h, w).

    Raises:
        ValueError: If scale is not positive.
    """
    if not scale > 0:
        raise ValueError(f"scale must be positive, got {scale}")
    a, b, th = FOV_ELLIPSE_DEFAULT
    params = (a * scale, b * scale, th)
    if orientation is None:
        return elliptical_gaussian_weights(h, w, params[0], params[1], 0.0)
    return fov_weights(orientation, h, w, params)


def load_psf_file(path, h: int = 27, w: int = 55) -> np.ndarray:
    """Load a PSF for the "ceres_psf" operator: the advisor's CSV, or an (h, w) .npy/.npz array.

    A .npy/.npz file is what e0.fit_aggregation_operator writes after recover_psf_from_moments,
    so the recovered PSF can replace ceres_psf.csv until the advisor sends it.

    Raises:
        ValueError: If a stored array does not have shape (h, w).
    """
    path = str(path)
    if path.endswith(".csv"):
        return load_ceres_psf(path, h, w)
    arr = np.load(path)
    if hasattr(arr, "files"):
        arr = arr["psf"]
    arr = np.asarray(arr, float)
    if arr.shape != (h, w):
        raise ValueError(f"PSF array {arr.shape} does not match patch {(h, w)}")
    return arr / arr.sum()


def get_operator(name: str, h: int, w: int, orientation=None, psf: np.ndarray | None = None) -> np.ndarray:
    """Return the named aggregation weights: "ceres_psf", "fov_ellipse", "uniform" or "gaussian_psf".

    "ceres_psf" is the advisor's PSF (load_ceres_psf) oriented per footprint; it needs `psf` and
    `orientation`, and is the exact operator the stored moments were built with.

    "fov_ellipse" needs the footprints' FOVorientation flags and returns (N, h, w); without them
    it falls back to the untilted ellipse (h, w). The others return shared (h, w) weights.

    Example:
        >>> get_operator("uniform", 27, 55).shape, get_operator("fov_ellipse", 27, 55, [0, 1, 1]).shape
        ((27, 55), (3, 27, 55))

    Raises:
        KeyError: For an unknown operator name.
    """
    if name == "ceres_psf":
        if psf is None:
            raise ValueError("ceres_psf needs the PSF array (aggregation.load_ceres_psf)")
        return psf if orientation is None else oriented_psf(psf, orientation)
    if name.startswith("fov_ellipse:"):
        return scaled_fov_operator(float(name.split(":", 1)[1]), h, w, orientation)
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
