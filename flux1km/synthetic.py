"""Synthetic granules with a known 1 km flux field, in the real file layout.

Used for --demo runs, for tests, and as the core of the synthetic-truth experiment
(PLAN.md section 4.3). The physics is deliberately crude: what matters is that the truth is
known, footprints overlap like CERES footprints, and the CTZ term can be switched off.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from . import aggregation, geometry, moments, schema
from .io import Granule, save_granule

PATCH = (27, 55)
STRIDE = (9, 8)  # footprint spacing in pixels (along-track, along-scan): neighbours overlap
CTZ_AMPLITUDE = 0.02  # albedo increase next to clouds
CTZ_EFOLD_KM = 10.0


def _smooth_field(shape, rng, sigma):
    f = ndimage.gaussian_filter(rng.normal(size=shape), sigma)
    return (f - f.mean()) / (f.std() + 1e-12)


def make_scene(shape: tuple[int, int], rng: np.random.Generator, ctz: bool = True,
               cloud_cover: float = 0.35, sza0: float = 30.0) -> dict[str, np.ndarray]:
    """Make one 1 km scene: geometry, cloud field, radiances and the true SW flux.

    Example:
        >>> s = make_scene((40, 60), np.random.default_rng(0))
        >>> s["flux"].shape, s["sw"].shape, s["lw"].shape
        ((40, 60), (11, 40, 60), (16, 40, 60))

    Args:
        shape: Scene size (rows, cols) in pixels (~1 km).
        rng: Random generator.
        ctz: Add a clear-sky brightening that decays with distance from cloud.
        cloud_cover: Approximate cloudy pixel fraction, in (0, 1).
        sza0: Solar zenith angle at the scene centre, degrees.

    Returns:
        Dict with "sza", "vza", "raa", "glint", "tau", "cloud", "dist_km", "flux" (rows, cols),
        "sw" (11, rows, cols) and "lw" (16, rows, cols).
    """
    h, w = shape
    cols = np.arange(w)[None, :].repeat(h, 0)
    sza = sza0 + 5.0 * (cols / max(w - 1, 1) - 0.5)
    vza = 60.0 * np.abs(cols - (w - 1) / 2) / max((w - 1) / 2, 1)
    raa = np.where(cols < w / 2, 30.0, 150.0)
    glint = geometry.glint_angle(sza, vza, raa)

    field = _smooth_field(shape, rng, sigma=3.0) + 0.5 * _smooth_field(shape, rng, sigma=1.0)
    thr = np.quantile(field, 1 - cloud_cover)
    tau = np.clip(field - thr, 0, None) * 25.0
    cloud = tau > 0.5
    dist_km = ndimage.distance_transform_edt(~cloud) if cloud.any() else np.full(shape, np.inf)

    clear_alb = 0.06 + 0.05 * np.exp(-((glint / 15.0) ** 2))
    if ctz:
        clear_alb = clear_alb + CTZ_AMPLITUDE * np.exp(-dist_km / CTZ_EFOLD_KM) * ~cloud
    cloud_alb = tau / (tau + 7.0)
    albedo = cloud_alb + (1 - cloud_alb) * clear_alb
    flux = geometry.albedo_to_flux(albedo, sza)

    mu0 = np.cos(np.radians(sza))
    wl = np.array(schema.SW_WAVELENGTHS)
    e0 = 1900.0 * np.exp(-0.6 * (wl - 0.47))  # crude solar spectral irradiance, W m^-2 um^-1
    swir = wl > 1.0
    absorb = np.where(swir[:, None, None], 1 - 0.3 * np.minimum(tau / 20.0, 1.0)[None], 1.0)
    refl = albedo[None] * absorb
    if ctz:  # humidified aerosol: weak blue signal near clouds, so MODIS "sees" part of the CTZ
        refl[0] += 0.5 * CTZ_AMPLITUDE * np.exp(-dist_km / CTZ_EFOLD_KM) * ~cloud
    sw = refl * e0[:, None, None] * mu0[None] / np.pi
    sw += rng.normal(scale=0.5, size=sw.shape)

    cth_km = 1.0 + 2.0 * rng.random()
    bt = 290.0 - 6.5 * cth_km * np.minimum(tau / 5.0, 1.0)
    lw = np.stack([bt / 10.0 * (1 + 0.01 * i) for i in range(len(schema.LW_LABELS))])
    lw += rng.normal(scale=0.05, size=lw.shape)

    return {"sza": sza, "vza": vza, "raa": raa, "glint": glint, "tau": tau, "cloud": cloud,
            "dist_km": dist_km, "flux": flux, "sw": sw.astype(np.float32), "lw": lw.astype(np.float32)}


def make_synthetic_granule(granule_id: str, rng: np.random.Generator, n_rows: int = 6, n_cols: int = 20,
                           ctz: bool = True, label_noise: float = 2.0, operator: str = "uniform",
                           lat0: float = 0.0, lon0: float = -150.0) -> Granule:
    """Cut overlapping 27x55 footprints from a synthetic scene and build all four parts.

    Example:
        >>> g = make_synthetic_granule("A2014060.0245", np.random.default_rng(0), n_rows=2, n_cols=3)
        >>> g.n_footprints, g.images["ImSW"].shape
        (6, (6, 11, 27, 55))

    Args:
        granule_id: Name, e.g. "A2014060.0245".
        rng: Random generator.
        n_rows, n_cols: Footprint lattice (along-track x along-scan); spacing STRIDE.
        ctz: Include the CTZ term in the truth.
        label_noise: Std of Gaussian noise added to the footprint labels (W m^-2).
        operator: Aggregation operator used to make the labels ("uniform" / "gaussian_psf").
        lat0, lon0: Coordinates of the scene's first pixel.

    Returns:
        Granule with scalars, moments, images and truth (truth: "flux", "cloud", "dist_km",
        "row0", "col0").
    """
    ph, pw = PATCH
    sy, sx = STRIDE
    scene = make_scene((ph + sy * (n_rows - 1), pw + sx * (n_cols - 1)), rng, ctz=ctz)
    r0, c0 = np.meshgrid(np.arange(n_rows) * sy, np.arange(n_cols) * sx, indexing="ij")
    r0, c0 = r0.ravel(), c0.ravel()

    def cut(a):
        return np.stack([a[..., r:r + ph, c:c + pw] for r, c in zip(r0, c0)])

    views = np.stack([cut(scene["sza"]), cut(scene["vza"]), cut(scene["raa"])], axis=1).astype(np.float32)
    im_sw, im_lw = cut(scene["sw"]), cut(scene["lw"])
    flux, cloud = cut(scene["flux"]), cut(scene["cloud"])
    weights = aggregation.get_operator(operator, ph, pw)
    fsw = aggregation.aggregate(flux, weights) + rng.normal(scale=label_noise, size=len(r0))

    mom = {}
    for band, im in (("SW", im_sw), ("LW", im_lw)):
        m, s, sk, ku = moments.moments_from_pixels(im)
        mom.update({f"m{band}spec": m, f"std{band}spec": s, f"skew{band}spec": sk, f"kurt{band}spec": ku})
    mom["SWwavlngs"] = np.array(schema.SW_WAVELENGTHS)
    mom["LWwavlngs"] = np.array(schema.LW_WAVELENGTHS)

    cf = 100.0 * cloud.mean(axis=(1, 2))
    rc, cc = r0 + ph / 2, c0 + pw / 2
    lat = lat0 + rc * 0.009
    scalars = {
        "Fsw": fsw, "Flw": 260.0 - 0.8 * cf, "Isw": fsw / np.pi, "Ilw": (260.0 - 0.8 * cf) / np.pi,
        "SZA": views[:, 0].mean(axis=(1, 2)), "VZA": views[:, 1].mean(axis=(1, 2)), "RAA": views[:, 2].mean(axis=(1, 2)),
        "Lat": lat, "Lon": lon0 + cc * 0.009 / np.cos(np.radians(lat)),
        "CF": cf, "Clrf": 100.0 - cf, "glintf": 100.0 * (cut(scene["glint"]) < 25).mean(axis=(1, 2)),
        "landf": np.zeros(len(r0)), "AOD": np.full(len(r0), 0.1),
        "meanCTH": 2000.0 * (cf > 0), "stdCTH": np.zeros(len(r0)), "LiqPhsf": np.full(len(r0), 100.0),
    }
    images = {"Imviews": views, "ImSW": im_sw, "ImLW": im_lw, "FOVorientation": np.zeros(len(r0), dtype=np.int8)}
    truth = {"flux": flux.astype(np.float32), "cloud": cloud, "dist_km": cut(scene["dist_km"]).astype(np.float32),
             "row0": r0, "col0": c0}
    return Granule(granule_id, scalars, mom, images, truth)


DEMO_GRANULES = ("A2014060.0245", "A2014060.0415", "A2014060.0530", "A2014090.2240", "A2014090.2350", "A2014090.2355")


def write_demo_dataset(root: str | Path, seed: int = 0, ctz: bool = True, **kwargs) -> list[str]:
    """Write synthetic copies of the six real image granule names under root; return their ids.

    Example:
        >>> import tempfile
        >>> ids = write_demo_dataset(tempfile.mkdtemp(), n_rows=2, n_cols=2)
        >>> len(ids)
        6
    """
    rng = np.random.default_rng(seed)
    for i, gid in enumerate(DEMO_GRANULES):
        g = make_synthetic_granule(gid, rng, ctz=ctz, lat0=-10.0 + 3.0 * i, **kwargs)
        save_granule(root, g)
    return list(DEMO_GRANULES)
