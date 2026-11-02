"""M8: two-stage narrowband -> broadband -> ADM, the physics route (PLAN.md section 3).

Stage 1, narrowband-to-broadband radiance (Bertrand 2006, Eq. 14 form):
    L_BB = c0 + c1 L0.66 + c2 L0.66^2 + c3 L0.86 + c4 L_swir + c5 theta_s + c6 gamma
fitted by ridge per surface type (water: landf == 0, other: landf > 0, when landf exists) on
homogeneous (Gao cv < cv_max) and glint-free (glint angle gamma > 25 deg) training footprints,
against the CERES SW radiance ``Isw``. Isw is used as a LABEL here, never as an input. The
footprint features use the raw moment E[L0.66^2], so the fitted pixel equation aggregates exactly.
Bertrand's SEVIRI form uses 1.6 um; Aqua band 6 (1.64 um) has dead detectors and is in
schema.BAD_CHANNELS, so the default SWIR channel is 2.13 um (``swir="1.64"`` restores the paper).

Stage 2, an empirical ADM: R = pi Isw / Fsw (both labels) is learned as
    log R = ridge(cos SZA, cos VZA, sin RAA, cos RAA, gamma, glint lobe, scene albedo terms)
where the scene albedo is the stage-1 prediction pi L_BB / (S0 cos SZA), so prediction is
MODIS-only. Flux = pi L_BB / R, clipped to [0, S0 mu0]. Applied per pixel with pixel geometry.

Run alone:
    python3 -m flux1km.methods.m8_nb2bb --demo
"""
from __future__ import annotations

import numpy as np

from .. import geometry, moments, schema
from ..features import Batch, pixel_geometry
from ..models import FluxModel, _Ridge, register

STAGE1_NAMES = ("L0.66", "L0.66^2", "L0.86", "L_swir", "theta_s", "gamma")


def _ch(batch_or_images, label, stat=None):
    band, i = schema.channel_index(label)
    if stat is None:
        return batch_or_images[f"Im{band}"][:, i].astype(float)
    return np.asarray(batch_or_images[f"{stat}{band}spec"][:, i], float)


def stage1_footprint_features(batch: Batch, swir: str = "2.13") -> np.ndarray:
    """Footprint features of the stage-1 regression, (N, 6), in STAGE1_NAMES order.

    E[L0.66^2] comes from the raw moments (mean, std), so fitting on footprints and applying to
    pixels is exact for this polynomial.

    Raises:
        ValueError: If the batch has no moments.
    """
    if batch.moments is None:
        raise ValueError("M8 needs the moments files")
    m066, s066 = _ch(batch.moments, "0.66", "m"), _ch(batch.moments, "0.66", "std")
    e2 = moments.raw_moments(m066, s066, np.zeros_like(m066), np.zeros_like(m066))[..., 1]
    s = batch.scalars
    gam = geometry.glint_angle(s["SZA"], s["VZA"], s["RAA"])
    return np.column_stack([m066, e2, _ch(batch.moments, "0.85", "m"), _ch(batch.moments, swir, "m"),
                            np.asarray(s["SZA"], float), gam])


def stage1_pixel_features(batch: Batch, swir: str = "2.13") -> np.ndarray:
    """Per-pixel stage-1 features (N, 6, H, W), same order as stage1_footprint_features."""
    sza, vza, raa = pixel_geometry(batch.images)
    l066 = _ch(batch.images, "0.66")
    return np.stack([l066, l066**2, _ch(batch.images, "0.85"), _ch(batch.images, swir), sza,
                     geometry.glint_angle(sza, vza, raa)], axis=1)


def adm_features(sza, vza, raa, albedo_proxy) -> np.ndarray:
    """Stage-2 ADM features (..., 9): geometry, glint lobe and scene-albedo terms.

    Example:
        >>> adm_features(np.array([30.0]), np.array([10.0]), np.array([90.0]), np.array([0.3])).shape
        (1, 9)
    """
    enc = geometry.encode_angles(sza, vza, raa)
    gam = geometry.glint_angle(sza, vza, raa)
    lobe = np.exp(-((gam / 12.0) ** 2))
    a = np.clip(np.nan_to_num(np.asarray(albedo_proxy, float)), 0.0, 1.5)
    return np.concatenate([enc, np.stack([gam / 90.0, lobe, a, a * a, a * lobe], axis=-1)], axis=-1)


@register
class NarrowToBroadband(FluxModel):
    """M8: Bertrand-style NB->BB radiance regression (label Isw) + empirical ADM (label Fsw/Isw)."""

    name = "M8"
    complexity = 3
    defaults = {"swir": "2.13", "cv_max": 0.2, "glint_min": 25.0, "alpha1": 1e-3, "alpha2": 1e-2, "min_rows": 20,
                "input_set": "F5", "seed": 0}

    def __init__(self, **params):
        super().__init__(**{**self.defaults, **params})

    @classmethod
    def param_grid(cls):
        return [{"swir": sw, "alpha2": a, "cv_max": cv} for sw in ("2.13", "1.24") for a in (1e-3, 1e-1) for cv in (0.2, 0.5)]

    @staticmethod
    def surface(batch: Batch) -> np.ndarray:
        """Surface class per footprint: "water" (landf == 0 or no landf) or "other"."""
        landf = batch.scalars.get("landf")
        if landf is None:
            return np.full(len(batch), "water", dtype=object)
        return np.where(np.asarray(landf, float) == 0, "water", "other").astype(object)

    def _albedo_proxy(self, lbb, sza):
        ins = geometry.toa_insolation(sza)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ins > 0, np.pi * lbb / ins, 0.0)

    def fit(self, train, val=None):
        if "Isw" not in train.scalars:
            raise ValueError("M8 needs the CERES SW radiance Isw as its stage-1 label")
        p = self.params
        X1 = stage1_footprint_features(train, p["swir"])
        isw = np.asarray(train.scalars["Isw"], float)
        cv = moments.homogeneity_cv(train.moments["mSWspec"], train.moments["stdSWspec"])
        clean = (cv < p["cv_max"]) & (X1[:, 5] > p["glint_min"]) & np.isfinite(isw)
        surf = self.surface(train)
        self.stage1 = {}
        for cls in dict.fromkeys(surf.tolist()):
            sel = (surf == cls) & clean
            if sel.sum() < p["min_rows"]:
                sel = (surf == cls) & np.isfinite(isw)  # too few clean footprints: use them all
            self.stage1[cls] = _Ridge(p["alpha1"]).fit(X1[sel], isw[sel], train.weight[sel])
        self.n_clean = int(clean.sum())
        lbb = self._stage1_predict(X1, surf)
        s = train.scalars
        with np.errstate(divide="ignore", invalid="ignore"):
            logr = np.log(np.pi * isw / train.label)
        X2 = adm_features(s["SZA"], s["VZA"], s["RAA"], self._albedo_proxy(lbb, s["SZA"]))
        ok = np.isfinite(logr)
        self.adm = _Ridge(p["alpha2"]).fit(X2[ok], logr[ok], train.weight[ok])
        return self

    def _stage1_predict(self, X, surf):
        out = np.empty(len(X))
        fallback = next(iter(self.stage1.values()))
        for cls in dict.fromkeys(surf.tolist()):
            sel = surf == cls
            out[sel] = self.stage1.get(cls, fallback).predict(X[sel])
        return out

    def _flux(self, lbb, sza, vza, raa):
        r = np.exp(self.adm.predict(adm_features(sza, vza, raa, self._albedo_proxy(lbb, sza))))
        return np.clip(np.pi * lbb / r, 0.0, geometry.toa_insolation(sza))

    def predict_map(self, batch):
        P = stage1_pixel_features(batch, self.params["swir"])
        n, _, h, w = P.shape
        surf = np.repeat(self.surface(batch), h * w)
        lbb = self._stage1_predict(np.moveaxis(P, 1, -1).reshape(-1, P.shape[1]), surf).reshape(n, h, w)
        sza, vza, raa = pixel_geometry(batch.images)
        return self._flux(lbb, sza, vza, raa)

    def predict_footprint(self, batch, weights):
        if batch.has_images:
            return super().predict_footprint(batch, weights)
        lbb = self._stage1_predict(stage1_footprint_features(batch, self.params["swir"]), self.surface(batch))
        s = batch.scalars
        return self._flux(lbb, np.asarray(s["SZA"], float), np.asarray(s["VZA"], float), np.asarray(s["RAA"], float))


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M8"])
