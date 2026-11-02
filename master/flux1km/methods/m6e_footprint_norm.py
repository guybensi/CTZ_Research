"""M6e: post-hoc per-footprint normalisation of a base map (hybrid, PLAN.md section 3).

GERB-HR style: take a base 1 km map (default M2) and rescale it inside every footprint so that it
aggregates exactly to that footprint's CERES flux. Two modes:
* ``"multiplicative"``: f'_i = f_i * F / sum_j w_j f_j;
* ``"scene"`` (scene-dependent): the residual r = F - sum_j w_j f_j is spread in proportion to how
  far each pixel is above the darkest pixel of the patch, s_i = f_i - min(f) + eps:
  f'_i = f_i + r * s_i / sum_j w_j s_j. Bright (cloudy) pixels absorb more of the correction than
  dark clear-sky pixels, which carry the smallest CERES uncertainty.
Both reproduce the footprint label exactly by construction.

It READS THE TEST FOOTPRINT LABELS, so it is flagged uses_test_labels and never ranked on the
footprint score. PLAN.md: uniform redistribution creates footprint-shaped boxes (Gao, Bertrand),
so this method is scored only on the overlap (interleave) test and on synthetic pixel truth.

Run alone:
    python3 -m flux1km.methods.m6e_footprint_norm --demo
"""
from __future__ import annotations

import numpy as np

from .. import features
from ..models import FluxModel, get_model, register
from ._base import load_psf

MODES = ("multiplicative", "scene")


def normalise_to_footprints(maps: np.ndarray, weights: np.ndarray, labels: np.ndarray, mode: str = "multiplicative",
                            eps: float = 1.0) -> np.ndarray:
    """Rescale each map so that sum_i w_i f'_i equals its footprint label exactly.

    Example:
        >>> m = np.array([[[1.0, 3.0]]]); w = np.array([[[0.5, 0.5]]])
        >>> normalise_to_footprints(m, w, np.array([4.0])).tolist()
        [[[2.0, 6.0]]]

    Args:
        maps: (N, H, W) base maps.
        weights: (N, H, W) or (H, W) aggregation weights (normalised internally).
        labels: (N,) footprint fluxes; NaN labels leave the map unchanged.
        mode: "multiplicative" or "scene".
        eps: Floor of the scene weights s_i (W m^-2).

    Raises:
        ValueError: For an unknown mode.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    f = np.asarray(maps, float)
    w = np.broadcast_to(np.asarray(weights, float), f.shape)
    w = w / w.sum(axis=(-2, -1), keepdims=True)
    agg = (f * w).sum(axis=(-2, -1))
    y = np.asarray(labels, float)
    ok = np.isfinite(y)
    if mode == "multiplicative":
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(ok & (agg != 0), y / agg, 1.0)
        return f * c[:, None, None]
    s = f - f.min(axis=(-2, -1), keepdims=True) + eps
    r = np.where(ok, y - agg, 0.0)
    return f + r[:, None, None] * s / (s * w).sum(axis=(-2, -1), keepdims=True)


@register
class FootprintNormalised(FluxModel):
    """M6e: base map (M2 by default) normalised per footprint to the CERES label (reads test labels)."""

    name = "M6e"
    uses_test_labels = True
    needs_operator = True
    complexity = 3
    defaults = {"base": "M2", "mode": "scene", "operator": "fov_ellipse", "psf_path": None, "input_set": "F5", "seed": 0}

    def __init__(self, **params):
        super().__init__(**{**self.defaults, **params})

    @classmethod
    def param_grid(cls):
        return [{"mode": m} for m in MODES]

    def fit(self, train, val=None):
        p = self.params
        self.base = get_model(p["base"], input_set=p["input_set"]).fit(train, val)
        return self

    def predict_map(self, batch):
        h, w = batch.images["ImSW"].shape[-2:]
        wts = features.footprint_weights(batch, self.params["operator"], load_psf(self.params["psf_path"], h, w))
        return normalise_to_footprints(self.base.predict_map(batch), wts, batch.label, self.params["mode"])


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M2", "M6e"])
