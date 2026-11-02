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
               cloud_cover: float = 0.35, sza0: float = 30.0, hard: bool = False,
               ctz_visible: float | None = None) -> dict[str, np.ndarray]:
    """Make one 1 km scene: geometry, cloud field, radiances and the true SW flux.

    Two difficulty levels:

    * ``hard=False`` (default, the original demo): radiances are the albedo times a crude solar
      spectrum, so the flux is almost a linear function of the channels, and the whole CTZ
      brightening is visible in every channel.
    * ``hard=True`` (PLAN.md section 4.3): see make_hard_radiances. The broadband albedo is a
      nonlinear, SZA-dependent function of the spectral albedos; the radiances carry an
      angle-dependent anisotropy (a sun-glint lobe for clear ocean, limb and forward/backscatter
      terms for clouds); and only a fraction ``ctz_visible`` of the CTZ brightening shows in the
      channels. The rest (think undetected thin cloud or 3D effects) changes the flux but no
      single pixel's radiances, so only spatial context (distance to cloud) can recover it.

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
        hard: Use the nonlinear, angle-dependent radiance model (section 4.3).
        ctz_visible: Fraction of the CTZ brightening visible in the channels, in [0, 1]. Default
            1.0 for the easy scene, 0.5 for the hard one.

    Returns:
        Dict with "sza", "vza", "raa", "glint", "tau", "cloud", "dist_km", "flux", "isw"
        (broadband radiance = flux * anisotropy / pi), "adm" (anisotropy factor; 1 for the easy
        scene) (rows, cols), "sw" (11, rows, cols) and "lw" (16, rows, cols).

    Raises:
        ValueError: If ctz_visible is outside [0, 1].
    """
    if ctz_visible is None:
        ctz_visible = 0.5 if hard else 1.0
    if not 0.0 <= ctz_visible <= 1.0:
        raise ValueError(f"ctz_visible must be in [0, 1], got {ctz_visible}")
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

    if hard:
        sw, lw, flux, adm = make_hard_radiances(sza, vza, raa, glint, tau, cloud, dist_km, rng, ctz, ctz_visible)
        return {"sza": sza, "vza": vza, "raa": raa, "glint": glint, "tau": tau, "cloud": cloud,
                "dist_km": dist_km, "flux": flux, "isw": flux * adm / np.pi, "adm": adm,
                "sw": sw.astype(np.float32), "lw": lw.astype(np.float32)}

    clear_alb = 0.06 + 0.05 * np.exp(-((glint / 15.0) ** 2))
    if ctz:
        clear_alb = clear_alb + CTZ_AMPLITUDE * np.exp(-dist_km / CTZ_EFOLD_KM) * ~cloud
    cloud_alb = tau / (tau + 7.0)
    albedo = cloud_alb + (1 - cloud_alb) * clear_alb
    flux = geometry.albedo_to_flux(albedo, sza)

    mu0 = np.cos(np.radians(sza))
    wl = np.array(schema.SW_WAVELENGTHS)
    e0 = _solar_spectrum()
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
            "dist_km": dist_km, "flux": flux, "isw": flux / np.pi, "adm": np.ones(shape),
            "sw": sw.astype(np.float32), "lw": lw.astype(np.float32)}


def _solar_spectrum() -> np.ndarray:
    wl = np.array(schema.SW_WAVELENGTHS)
    return 1900.0 * np.exp(-0.6 * (wl - 0.47))  # crude solar spectral irradiance, W m^-2 um^-1


def anisotropy(sza, vza, raa, glint, cloud_weight) -> np.ndarray:
    """Angle-dependent radiance anisotropy R = pi L / F used by the hard synthetic scene.

    Clear ocean has a sun-glint lobe (strong near glint angle 0) and limb brightening; clouds have
    a weaker limb term and a forward/backscatter asymmetry in RAA (SSF convention: 0 = forward /
    specular side). The factor is blended by the cloud's share of the reflected light.

    Example:
        >>> float(anisotropy(30.0, 30.0, 0.0, 0.0, 0.0)) > float(anisotropy(30.0, 30.0, 180.0, 60.0, 0.0))
        True

    Args:
        sza, vza, raa, glint: Angles in degrees (broadcastable).
        cloud_weight: Share of the reflected light that comes from cloud, in [0, 1].

    Returns:
        Positive array, the broadcast shape of the inputs.
    """
    muv = np.cos(np.radians(np.asarray(vza, float)))
    clear = (0.75 + 2.5 * np.exp(-((np.asarray(glint, float) / 12.0) ** 2))) * (1.0 + 0.3 * (1.0 - muv))
    cloudy = 0.85 + 0.3 * (1.0 - muv) + 0.15 * np.cos(np.radians(np.asarray(raa, float))) * np.sin(np.radians(np.asarray(sza, float)))
    c = np.clip(np.asarray(cloud_weight, float), 0.0, 1.0)
    return (1.0 - c) * clear + c * cloudy


def broadband_albedo(alpha_vis, alpha_nir, sza) -> np.ndarray:
    """Nonlinear, SZA-dependent narrowband -> broadband albedo of the hard synthetic scene.

    alpha_BB = (0.6 alpha_vis + 0.4 alpha_nir)^1.1 * (1 + 0.2 (1 - mu0)), clipped to [0, 0.95].
    The exponent and the mu0 factor make a linear combination of channels insufficient.

    Example:
        >>> round(float(broadband_albedo(0.5, 0.5, 0.0)), 4)
        0.4665
    """
    mu0 = np.cos(np.radians(np.asarray(sza, float)))
    x = np.clip(0.6 * np.asarray(alpha_vis, float) + 0.4 * np.asarray(alpha_nir, float), 0.0, None)
    return np.clip(x ** 1.1 * (1.0 + 0.2 * (1.0 - mu0)), 0.0, 0.95)


def make_hard_radiances(sza, vza, raa, glint, tau, cloud, dist_km, rng, ctz: bool, ctz_visible: float):
    """Radiances and truth of the hard synthetic scene (PLAN.md section 4.3).

    Spectral (hemispheric) albedos: cloud albedo tau / (tau + 7), lowered in the SWIR by cloud
    absorption and in the 0.9 um water-vapour channels; clear ocean 0.05 + 0.1 (1 - mu0)^3. The CTZ
    brightening delta = CTZ_AMPLITUDE exp(-d / CTZ_EFOLD_KM) on clear pixels always enters the true
    broadband albedo in full; only a fraction ctz_visible of it enters the spectral albedos the
    channels observe (plus a weak blue aerosol signal), so the truth does not depend on ctz_visible.
    Radiances are L_c = alpha_c * anisotropy * E0_c mu0 / pi + noise.

    Returns:
        (sw (11, H, W), lw (16, H, W), flux (H, W), anisotropy (H, W)).
    """
    mu0 = np.cos(np.radians(sza))
    wl = np.array(schema.SW_WAVELENGTHS)
    cloud_alb = tau / (tau + 7.0)
    clear_alb = 0.05 + 0.1 * (1.0 - mu0) ** 3
    delta = CTZ_AMPLITUDE * np.exp(-dist_km / CTZ_EFOLD_KM) * ~cloud if ctz else np.zeros_like(tau)
    vis_delta, hidden_delta = ctz_visible * delta, (1.0 - ctz_visible) * delta
    absorb = 1.0 - 0.35 * np.minimum(tau / 20.0, 1.0)

    def spectral(extra):
        out = []
        for w in wl:
            c_alb = cloud_alb * (absorb if w > 1.0 else 1.0) * (0.8 if 0.9 <= w <= 0.95 else 1.0)
            out.append(c_alb + (1.0 - c_alb) * (clear_alb + extra))
        return np.stack(out)

    i066, i124 = schema.SW_LABELS.index("0.66"), schema.SW_LABELS.index("1.24")
    base = spectral(0.0)
    # truth: the whole CTZ brightening enters the broadband albedo, whatever part of it the channels see
    bb = broadband_albedo(base[i066], base[i124], sza) + (vis_delta + hidden_delta) * (1.0 - cloud_alb)
    flux = geometry.albedo_to_flux(np.clip(bb, 0.0, 0.95), sza)
    alphas = spectral(vis_delta)  # observation: only the visible part
    alphas[0] += 0.5 * vis_delta  # humidified aerosol: blue signal near clouds
    alpha_vis = alphas[i066]
    cloud_weight = np.where(alpha_vis > 0, cloud_alb / np.maximum(alpha_vis, 1e-6), 0.0)
    adm = anisotropy(sza, vza, raa, glint, cloud_weight)
    sw = alphas * adm[None] * _solar_spectrum()[:, None, None] * mu0[None] / np.pi
    sw += rng.normal(scale=0.5, size=sw.shape)
    cth_km = 1.0 + 2.0 * rng.random()
    bt = 290.0 - 6.5 * cth_km * np.minimum(tau / 5.0, 1.0)
    lw = np.stack([bt / 10.0 * (1 + 0.01 * i) for i in range(len(schema.LW_LABELS))])
    lw += rng.normal(scale=0.05, size=lw.shape)
    return sw, lw, flux, adm


def make_synthetic_granule(granule_id: str, rng: np.random.Generator, n_rows: int = 6, n_cols: int = 20,
                           ctz: bool = True, label_noise: float = 2.0, operator: str = "fov_ellipse",
                           lat0: float = 0.0, lon0: float = -150.0, hard: bool = False,
                           ctz_visible: float | None = None) -> Granule:
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
        operator: Aggregation operator used to make the labels and the moments (see aggregation.OPERATORS).
        lat0, lon0: Coordinates of the scene's first pixel.
        hard, ctz_visible: Difficulty of the scene (see make_scene). In the hard scene ``Isw`` is
            the aggregated anisotropic broadband radiance, so Fsw / Isw varies with geometry.

    Returns:
        Granule with scalars, moments, images and truth (truth: "flux", "cloud", "dist_km",
        "row0", "col0").
    """
    ph, pw = PATCH
    sy, sx = STRIDE
    scene = make_scene((ph + sy * (n_rows - 1), pw + sx * (n_cols - 1)), rng, ctz=ctz, hard=hard, ctz_visible=ctz_visible)
    r0, c0 = np.meshgrid(np.arange(n_rows) * sy, np.arange(n_cols) * sx, indexing="ij")
    r0, c0 = r0.ravel(), c0.ravel()

    def cut(a):
        return np.stack([a[..., r:r + ph, c:c + pw] for r, c in zip(r0, c0)])

    raa_img = 180.0 - scene["raa"] if schema.IMVIEWS_RAA_IS_SUPPLEMENT else scene["raa"]  # real-file convention
    views = np.stack([cut(scene["sza"]), cut(scene["vza"]), cut(raa_img)], axis=1).astype(np.float32)
    im_sw, im_lw = cut(scene["sw"]), cut(scene["lw"])
    flux, cloud = cut(scene["flux"]), cut(scene["cloud"])
    orientation = rng.integers(0, 2, size=len(r0)).astype(np.float32)
    weights = aggregation.get_operator(operator, ph, pw, orientation=orientation)
    fsw = aggregation.aggregate(flux, weights) + rng.normal(scale=label_noise, size=len(r0))
    isw = aggregation.aggregate(cut(scene["isw"]), weights) + (fsw - aggregation.aggregate(flux, weights)) / np.pi if hard else fsw / np.pi

    # Like the real files, the moments are a weighted average over the footprint (same operator).
    mw = weights[:, None] if weights.ndim == 3 else weights
    mom = {}
    for band, im in (("SW", im_sw), ("LW", im_lw)):
        m, s, sk, ku = moments.moments_from_pixels(im, mw)
        ku = ku - 3.0 if schema.MOMENTS_EXCESS_KURTOSIS else ku
        mom.update({f"m{band}spec": m, f"std{band}spec": s, f"skew{band}spec": sk, f"kurt{band}spec": ku})
    mom["SWwavlngs"] = np.array(schema.SW_WAVELENGTHS)
    mom["LWwavlngs"] = np.array(schema.LW_WAVELENGTHS)

    cf = 100.0 * cloud.mean(axis=(1, 2))
    rc, cc = r0 + ph / 2, c0 + pw / 2
    lat = lat0 + rc * 0.009
    scalars = {
        "Fsw": fsw, "Flw": 260.0 - 0.8 * cf, "Isw": isw, "Ilw": (260.0 - 0.8 * cf) / np.pi,
        # footprint geometry = the same weighted average as the moments (CERES reports the footprint centre)
        "SZA": aggregation.aggregate(views[:, 0], weights), "VZA": aggregation.aggregate(views[:, 1], weights),
        "RAA": aggregation.aggregate(cut(scene["raa"]), weights),
        "Lat": lat, "Lon": lon0 + cc * 0.009 / np.cos(np.radians(lat)),
        "CF": cf / schema.CF_TO_PERCENT, "Clrf": 100.0 - cf, "glintf": 100.0 * (cut(scene["glint"]) < 25).mean(axis=(1, 2)),
        "landf": np.zeros(len(r0)), "AOD": np.full(len(r0), 0.1),
        "meanCTH": 2000.0 * (cf > 0), "stdCTH": np.zeros(len(r0)), "LiqPhsf": np.full(len(r0), 100.0),
    }
    # row0/col0: optional footprint position (patch corner) in the granule's MODIS grid. Real triplets do not
    # have them yet; an in-house footprint index (PLAN.md section 2.3, phase F) would. Used by M7 (ATPRK).
    images = {"Imviews": views, "ImSW": im_sw, "ImLW": im_lw, "FOVorientation": orientation,
              "row0": r0.astype(np.int32), "col0": c0.astype(np.int32)}
    truth = {"flux": flux.astype(np.float32), "cloud": cloud, "dist_km": cut(scene["dist_km"]).astype(np.float32),
             "row0": r0, "col0": c0}
    return Granule(granule_id, scalars, mom, images, truth)


DEMO_GRANULES = ("A2014060.0245", "A2014060.0415", "A2014060.0530", "A2014090.2240", "A2014090.2350", "A2014090.2355")


def write_demo_dataset(root: str | Path, seed: int = 0, ctz: bool = True, **kwargs) -> list[str]:
    """Write synthetic copies of the six real image granule names under root; return their ids.

    Extra keyword arguments go to make_synthetic_granule (n_rows, n_cols, hard, ctz_visible,
    operator, label_noise, ...).

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
