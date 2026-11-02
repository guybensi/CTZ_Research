"""M9: homogeneous-footprint transfer (Gao 2012 DMS), PLAN.md section 3.

On a homogeneous footprint f(mean x) ~ mean f(x), so a footprint model trained ONLY on
homogeneous footprints can be applied to single pixels. Homogeneity is Gao's
cv = mean_b(sigma_b / mu_b) over the SW channels, computed from the moments; footprints with
cv < cv_max (0.2) are used, weighted by 1/cv.

Features are identical at both levels, so the transfer is direct: channel values (footprint
means / pixel values) of the input set + encoded geometry (+ glint for F5).
Learners: ``learner="gbm"`` (histogram gradient boosting) or ``"tree"`` (a single regression
tree, closest to Gao's Cubist model trees). Target: albedo, times S0 mu0 per pixel.

Two variants:
* ``M9``: the global model only. Legal under the protocol; ranked.
* ``M9-local``: Gao Eq. 2-3. A local model per test granule is fitted on that granule's
  homogeneous footprints, and global and local predictions are blended per footprint with
  weights 1 / residual^2. The local model and the residuals use the TEST labels, so this variant
  is flagged uses_test_labels and never ranked (synthetic truth and interleave test only).

Run alone:
    python3 -m flux1km.methods.m9_homogeneous --demo
"""
from __future__ import annotations

import numpy as np

from .. import features, moments
from ..aggregation import aggregate
from ..features import Batch
from ..models import FluxModel, register
from . import _common


def homogeneous_weights(batch: Batch, cv_max: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Gao selection: (mask cv < cv_max, weights 1/cv on the mask, 0 elsewhere).

    Example:
        >>> # mask, w = homogeneous_weights(train, 0.2)
    """
    cv = moments.homogeneity_cv(batch.moments["mSWspec"], batch.moments["stdSWspec"])
    mask = np.isfinite(cv) & (cv < cv_max)
    with np.errstate(divide="ignore"):
        w = np.where(mask, 1.0 / np.maximum(cv, 1e-3), 0.0)
    return mask, w


def _make_learner(kind: str, seed: int, **kw):
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=kw.get("max_iter", 300), learning_rate=kw.get("learning_rate", 0.05),
                                             max_leaf_nodes=kw.get("max_leaf_nodes", 31),
                                             min_samples_leaf=kw.get("min_samples_leaf", 5), random_state=seed)
    if kind == "tree":
        from sklearn.tree import DecisionTreeRegressor

        return DecisionTreeRegressor(max_depth=kw.get("max_depth", 8), min_samples_leaf=kw.get("min_samples_leaf", 5),
                                     random_state=seed)
    raise ValueError(f"unknown learner {kind!r}")


@register
class HomogeneousTransfer(FluxModel):
    """M9: footprint model trained on homogeneous footprints, applied per pixel (global only)."""

    name = "M9"
    complexity = 3
    defaults = {"input_set": "F5", "cv_max": 0.2, "learner": "gbm", "min_rows": 30, "seed": 0}

    def __init__(self, **params):
        super().__init__(**{**self.defaults, **params})

    @classmethod
    def param_grid(cls):
        return [{"learner": lr, "cv_max": cv} for lr in ("gbm", "tree") for cv in (0.2, 0.3, 0.5)]

    def _fp_features(self, batch):
        X, _ = features.footprint_features(batch, self.params["input_set"], ("m",))
        return X

    def _px_features(self, batch):
        X, _ = features.pixel_features(batch, self.params["input_set"])
        n, f, h, w = X.shape
        return np.moveaxis(X, 1, -1).reshape(-1, f), (n, h, w)

    def _fit_on(self, batch, mask, w):
        y = _common.albedo_target(batch)
        sel = mask & np.isfinite(y)
        if sel.sum() < self.params["min_rows"]:   # too few homogeneous footprints: use all, weighted by 1/cv
            sel = np.isfinite(y)
            cv = moments.homogeneity_cv(batch.moments["mSWspec"], batch.moments["stdSWspec"])
            w = 1.0 / np.maximum(np.nan_to_num(cv, nan=1.0), 1e-3)
        model = _make_learner(self.params["learner"], self.params["seed"])
        model.fit(self._fp_features(batch)[sel], y[sel], sample_weight=(w * batch.weight)[sel])
        return model

    def fit(self, train, val=None):
        mask, w = homogeneous_weights(train, self.params["cv_max"])
        self.n_homogeneous = int(mask.sum())
        self.model = self._fit_on(train, mask, w)
        return self

    def _map(self, model, batch):
        T, shape = self._px_features(batch)
        return np.clip(model.predict(T).reshape(shape), 0.0, 1.0) * _common.pixel_insolation(batch)

    def predict_map(self, batch):
        return self._map(self.model, batch)

    def predict_footprint(self, batch, weights):
        if batch.has_images:
            return super().predict_footprint(batch, weights)
        return np.clip(self.model.predict(self._fp_features(batch)), 0, 1) * _common.footprint_insolation(batch)


@register
class HomogeneousTransferLocal(HomogeneousTransfer):
    """M9-local: Gao's global + per-granule local blend, weights 1/residual^2 (reads test labels)."""

    name = "M9-local"
    uses_test_labels = True
    needs_operator = True
    defaults = {**HomogeneousTransfer.defaults, "min_local": 10, "operator": "fov_ellipse", "psf_path": None}

    def predict_map(self, batch):
        from ._base import load_psf

        h, w = batch.images["ImSW"].shape[-2:]
        W = features.footprint_weights(batch, self.params["operator"], load_psf(self.params["psf_path"], h, w))
        glob = self._map(self.model, batch)
        out = glob.copy()
        granules = batch.granule if batch.granule is not None else np.full(len(batch), "all", dtype=object)
        mask, wts = homogeneous_weights(batch, self.params["cv_max"])
        for g in dict.fromkeys(granules.tolist()):
            sel = np.flatnonzero(granules == g)
            if mask[sel].sum() < self.params["min_local"]:
                continue
            sub = _subset(batch, sel)
            local = self._fit_on(sub, mask[sel], wts[sel])
            loc_map = self._map(local, sub)
            e_g = (aggregate(glob[sel], W[sel]) - sub.label) ** 2
            e_l = (aggregate(loc_map, W[sel]) - sub.label) ** 2
            wg, wl = 1.0 / np.maximum(e_g, 1e-6), 1.0 / np.maximum(e_l, 1e-6)
            out[sel] = (wg[:, None, None] * glob[sel] + wl[:, None, None] * loc_map) / (wg + wl)[:, None, None]
        return out


def _subset(batch: Batch, sel: np.ndarray) -> Batch:
    """Rows `sel` of a batch (all per-footprint arrays sliced)."""
    def take(d):
        return None if d is None else {k: v[sel] for k, v in d.items()}

    return Batch(rows=batch.rows[sel], label=batch.label[sel], scalars=take(batch.scalars), moments=take(batch.moments),
                 images=take(batch.images), truth=None if batch.truth is None else batch.truth[sel],
                 dist_km=None if batch.dist_km is None else batch.dist_km[sel], group=batch.group[sel],
                 weight=batch.weight[sel], granule=None if batch.granule is None else batch.granule[sel])


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M9", "M9-local"])
