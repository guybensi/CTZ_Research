"""Helpers shared by the method modules: insolation maps, standardisation, context features.

Everything here is NumPy/SciPy only. Context features are built from MODIS radiances alone
(a brightness threshold on the 0.66 um reflectance, never the MYD35 product), so they are legal
model inputs under PLAN.md section 1.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .. import schema
from ..features import Batch, pixel_geometry
from ..geometry import toa_insolation

# Approximate band-averaged solar irradiance of the MODIS SW channels (W m^-2 um^-1), used only to
# turn radiance into a reflectance proxy for the context features. Order = schema.SW_LABELS.
SW_SOLAR_IRRADIANCE: dict[str, float] = {
    "0.47": 2059.0, "0.55": 1863.0, "0.66": 1612.0, "0.85": 988.0, "0.905": 870.0, "0.935": 830.0,
    "0.940": 820.0, "1.24": 469.0, "1.64": 240.0, "2.13": 89.0, "1.375": 362.0,
}


def pixel_insolation(batch: Batch) -> np.ndarray:
    """Per-pixel TOA insolation S0 cos(SZA) (N, H, W) from Imviews.

    Example:
        >>> # I = pixel_insolation(batch); I.shape == batch.images["ImSW"].shape[:1] + (27, 55)

    Raises:
        ValueError: If the batch has no images.
    """
    if not batch.has_images:
        raise ValueError("pixel insolation needs images")
    return toa_insolation(batch.images["Imviews"][:, 0].astype(float))


def footprint_insolation(batch: Batch) -> np.ndarray:
    """Footprint insolation S0 cos(SZA) (N,) from the scalars SZA."""
    return toa_insolation(np.asarray(batch.scalars["SZA"], float))


def albedo_target(batch: Batch) -> np.ndarray:
    """Footprint label as TOA albedo Fsw / (S0 cos SZA); NaN where the sun is down."""
    ins = footprint_insolation(batch)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ins > 0, batch.label / ins, np.nan)


class Standardizer:
    """Per-feature mean/std scaling along a feature axis, robust to NaN and constant features.

    Example:
        >>> s = Standardizer(axis=1).fit(np.array([[0.0, 1.0], [2.0, 1.0]]))
        >>> s.transform(np.array([[1.0, 1.0]])).tolist()
        [[0.0, 0.0]]
    """

    def __init__(self, axis: int = 1):
        self.axis = axis

    def fit(self, X: np.ndarray, max_samples: int = 200_000, seed: int = 0) -> Standardizer:
        """Estimate mean/std per feature (on at most max_samples random entries per feature)."""
        x = np.moveaxis(np.asarray(X, float), self.axis, -1).reshape(-1, np.shape(X)[self.axis])
        if len(x) > max_samples:
            x = x[np.random.default_rng(seed).choice(len(x), max_samples, replace=False)]
        self.mu = np.nanmean(x, axis=0)
        self.sd = np.nanstd(x, axis=0)
        self.mu = np.nan_to_num(self.mu)
        self.sd = np.where(np.isfinite(self.sd) & (self.sd > 0), self.sd, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scaled copy, NaN -> 0 (the training mean)."""
        shape = [1] * np.ndim(X)
        shape[self.axis] = -1
        z = (np.asarray(X, float) - self.mu.reshape(shape)) / self.sd.reshape(shape)
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def reflectance(batch: Batch, label: str = "0.66") -> np.ndarray:
    """Per-pixel reflectance proxy pi L / (E0 cos SZA) of one SW channel, (N, H, W).

    Example:
        >>> # r = reflectance(batch, "0.66"); clouds are bright (r > 0.2), clear ocean dark

    Raises:
        KeyError: If label is not a SW channel.
    """
    band, i = schema.channel_index(label)
    if band != "SW":
        raise KeyError(f"{label} is not a SW channel")
    sza, _, _ = pixel_geometry(batch.images)
    mu0 = np.clip(np.cos(np.radians(sza)), 0.05, None)
    return np.pi * batch.images["ImSW"][:, i].astype(float) / (SW_SOLAR_IRRADIANCE[label] * mu0)


def bright_mask(batch: Batch, threshold: float = 0.2) -> np.ndarray:
    """MODIS-radiance cloud proxy: 0.66 um reflectance above threshold, (N, H, W) bool."""
    return reflectance(batch, "0.66") > threshold


def distance_to_bright(mask: np.ndarray, cap_km: float = 30.0, pixel_km: float = 1.0) -> np.ndarray:
    """Distance (km) of each pixel to the nearest True pixel of its own patch, capped at cap_km.

    Patches with no bright pixel get cap_km everywhere.

    Example:
        >>> m = np.zeros((1, 1, 4), bool); m[0, 0, 0] = True
        >>> distance_to_bright(m)[0, 0].tolist()
        [0.0, 1.0, 2.0, 3.0]
    """
    out = np.full(mask.shape, float(cap_km))
    for k in range(len(mask)):
        if mask[k].any():
            out[k] = np.minimum(ndimage.distance_transform_edt(~mask[k]) * pixel_km, cap_km)
    return out


def _window(a: np.ndarray, size: int, fn) -> np.ndarray:
    return fn(a, size=(1, size, size), mode="nearest")


def context_features(batch: Batch, sizes: tuple[int, ...] = (3, 7, 15), bright_threshold: float = 0.2,
                     psi_km: float = 5.0) -> tuple[np.ndarray, list[str]]:
    """Hand-crafted spatial-context features per pixel (M4, M6b), from MODIS radiances only.

    For every window size s: local mean and std of the 0.66 um reflectance and of the 11 um
    radiance, and the local bright (cloud-proxy) fraction. Plus the distance to the nearest bright
    pixel (km, capped at 30) and a psi-like cloud-influence index (bright mask smoothed with a
    Gaussian of psi_km).

    Example:
        >>> # C, names = context_features(batch); C.shape == (N, 3 * 5 + 2, 27, 55)

    Returns:
        (C (N, K, H, W), names).

    Raises:
        ValueError: If the batch has no images.
    """
    if not batch.has_images:
        raise ValueError("context features need images")
    r066 = reflectance(batch, "0.66")
    t11 = batch.images["ImLW"][:, schema.channel_index("11.0")[1]].astype(float)
    bright = (r066 > bright_threshold).astype(float)
    blocks, names = [], []
    for s in sizes:
        for nm, a in (("r066", r066), ("L11", t11)):
            mean = _window(a, s, ndimage.uniform_filter)
            sq = _window(a * a, s, ndimage.uniform_filter)
            blocks += [mean, np.sqrt(np.clip(sq - mean**2, 0, None))]
            names += [f"{nm}_mean{s}", f"{nm}_std{s}"]
        blocks.append(_window(bright, s, ndimage.uniform_filter))
        names.append(f"bright_frac{s}")
    blocks.append(distance_to_bright(bright > 0) / 30.0)
    names.append("dist_bright")
    blocks.append(ndimage.gaussian_filter(bright, sigma=(0, psi_km, psi_km), mode="nearest"))
    names.append("psi")
    schema.assert_no_ceres_inputs(names)
    return np.stack(blocks, axis=1), names


def tile_masks(h: int, w: int, tile: tuple[int, int]) -> np.ndarray:
    """Boolean masks (T, h, w) of non-overlapping tiles covering an h x w patch (edges included).

    Example:
        >>> tile_masks(2, 4, (2, 2)).shape
        (2, 2, 4)

    Raises:
        ValueError: If a tile dimension is not positive.
    """
    th, tw = tile
    if th <= 0 or tw <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    out = []
    for r in range(0, h, th):
        for c in range(0, w, tw):
            m = np.zeros((h, w), bool)
            m[r:r + th, c:c + tw] = True
            out.append(m)
    return np.stack(out)


def window_moments(x: np.ndarray, mask: np.ndarray):
    """Uniform mean/std/skew/excess-kurtosis of each channel over a pixel mask.

    Args:
        x: (N, C, H, W) radiances.
        mask: (H, W) bool.

    Returns:
        (mean, std, skew, kurt) each (N, C); kurt is excess kurtosis, like the stored moments.
    """
    v = np.asarray(x, float)[..., mask]  # (N, C, P)
    m = v.mean(-1)
    d = v - m[..., None]
    m2 = (d**2).mean(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sk = np.where(m2 > 0, (d**3).mean(-1) / m2**1.5, 0.0)
        ku = np.where(m2 > 0, (d**4).mean(-1) / m2**2 - 3.0, 0.0)
    return m, np.sqrt(m2), sk, ku


def moment_batch(sw_moments: tuple, lw_moments: tuple, sza, vza, raa) -> Batch:
    """A label-free Batch holding only moments and geometry, e.g. for sub-footprint windows.

    Lets a footprint-level teacher (M1) predict the flux of any pixel set: M6d (tiles) and M6f
    (random windows, unlabelled granules) use it to build soft targets.

    Example:
        >>> z = np.zeros((2, 11)); zl = np.zeros((2, 16)); g = np.zeros(2)
        >>> b = moment_batch((z, z, z, z), (zl, zl, zl, zl), g, g, g)
        >>> len(b), b.moments["mSWspec"].shape, bool(np.isnan(b.label).all())
        (2, (2, 11), True)

    Args:
        sw_moments, lw_moments: (mean, std, skew, excess kurtosis), each (N, 11) / (N, 16).
        sza, vza, raa: (N,) geometry in degrees (RAA in the scalars / SSF convention).
    """
    n = len(np.asarray(sza))
    mom = {}
    for band, mm in (("SW", sw_moments), ("LW", lw_moments)):
        for st, arr in zip(("m", "std", "skew", "kurt"), mm):
            mom[f"{st}{band}spec"] = np.asarray(arr, float)
    scal = {"SZA": np.asarray(sza, float), "VZA": np.asarray(vza, float), "RAA": np.asarray(raa, float),
            "Fsw": np.full(n, np.nan)}
    return Batch(rows=np.arange(n), label=scal["Fsw"], scalars=scal, moments=mom, images=None, truth=None,
                 dist_km=None, group=np.full(n, "window", dtype=object), weight=np.ones(n))


def window_batch(images: dict, mask: np.ndarray) -> Batch:
    """moment_batch of one pixel mask applied to every patch of an images dict (uniform weights)."""
    sw = window_moments(images["ImSW"], mask)
    lw = window_moments(images["ImLW"], mask)
    sza, vza, raa = pixel_geometry(images)
    return moment_batch(sw, lw, sza[:, mask].mean(1), vza[:, mask].mean(1), raa[:, mask].mean(1))
