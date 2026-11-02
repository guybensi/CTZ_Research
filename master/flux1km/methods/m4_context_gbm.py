"""M4: gradient boosting with spatial-context features and the aggregate gradient (PLAN.md section 3).

Inputs per pixel: the MODIS channels, encoded geometry and glint (as M3) plus hand-crafted
context (methods._common.context_features): window mean/std of the 0.66 um reflectance and the
11 um radiance at 3, 7 and 15 px, the local bright (cloud-proxy) fraction, the distance to the
nearest bright pixel and a psi-like cloud-influence index. All MODIS-radiance based.

Training signal: the footprint loss L = sum_k s_k (sum_i w_ik f_i - F_k)^2, whose gradient with
respect to one pixel's output is g_i = 2 w_ik (sum_i w_ik f_i - F_k). The model is the albedo
a(x), f = a(x) * S0 mu0. Each boosting round
  1. computes the footprint residual r_k = F_k - sum_i w_ik f_i,
  2. fits a regression tree to the pixel target r_k / I_k (albedo units) with sample weight
     w_ik * s_k: the weighted least-squares projection of -g onto trees,
  3. picks the step gamma that minimises L along that tree exactly (closed form, the loss is
     quadratic in gamma), and adds learning_rate * gamma * tree.
Trees come from scikit-learn; the pixels used to fit each tree are subsampled per footprint.

Run alone:
    python3 -m flux1km.methods.m4_context_gbm --demo
"""
from __future__ import annotations

import numpy as np

from .. import features
from ..features import Batch
from ..models import FluxModel, register
from . import _common
from ._base import load_psf


def pixel_table(batch: Batch, input_set: str, use_context: bool = True) -> tuple[np.ndarray, list[str]]:
    """Per-pixel features flattened to a table (N * H * W, F), footprint-major.

    Example:
        >>> # T, names = pixel_table(batch, "F5"); T.shape == (len(batch) * 27 * 55, F)
    """
    X, names = features.pixel_features(batch, input_set)
    if use_context:
        C, cnames = _common.context_features(batch)
        X, names = np.concatenate([X, C], axis=1), names + cnames
    n, f, h, w = X.shape
    return np.moveaxis(X, 1, -1).reshape(n * h * w, f), names


@register
class ContextGBM(FluxModel):
    """M4: boosted trees on pixel + context features, trained with the aggregate gradient."""

    name = "M4"
    complexity = 5
    needs_operator = True
    defaults = {"input_set": "F5", "operator": "fov_ellipse", "psf_path": None, "n_rounds": 150, "learning_rate": 0.1,
                "max_depth": 4, "min_samples_leaf": 50, "pixels_per_footprint": 64, "use_context": True, "seed": 0,
                "patience": 15}

    def __init__(self, **params):
        super().__init__(**{**self.defaults, **params})

    @classmethod
    def param_grid(cls):
        return [{"max_depth": d, "learning_rate": lr, "n_rounds": n} for d in (3, 5) for lr in (0.05, 0.1) for n in (100, 200)]

    def _w(self, batch):
        h, w = batch.images["ImSW"].shape[-2:]
        return features.footprint_weights(batch, self.params["operator"], load_psf(self.params["psf_path"], h, w))

    def _predict_albedo(self, T: np.ndarray, n_trees: int | None = None) -> np.ndarray:
        out = np.full(len(T), self.a0)
        for gamma, tree in self.trees[:n_trees]:
            out += gamma * tree.predict(T)
        return out

    def fit(self, train, val=None):
        from sklearn.tree import DecisionTreeRegressor

        if not train.has_images:
            raise ValueError("M4 needs image patches")
        p = self.params
        rng = np.random.default_rng(p["seed"])
        T, self.names = pixel_table(train, p["input_set"], p["use_context"])
        n, h, w = len(train), *train.images["ImSW"].shape[-2:]
        P = h * w
        W = self._w(train).reshape(n, P)
        ins = _common.pixel_insolation(train).reshape(n, P)
        y = train.label.astype(float)
        s = np.where(np.isfinite(y), train.weight, 0.0)
        y = np.nan_to_num(y)
        WI = W * ins  # footprint flux = sum_i (w_i I_i) a_i
        alb = _common.albedo_target(train)
        self.a0 = float(np.nanmedian(alb)) if np.isfinite(alb).any() else 0.3
        A = np.full((n, P), self.a0)
        self.trees = []
        val_state = None
        if val is not None and len(val) and val.has_images:
            Tv, _ = pixel_table(val, p["input_set"], p["use_context"])
            WIv = (self._w(val) * _common.pixel_insolation(val)).reshape(len(val), P)
            val_state = (Tv, WIv, val.label.astype(float), np.full((len(val), P), self.a0))
        best, best_n, bad = np.inf, 0, 0
        k = min(p["pixels_per_footprint"], P)
        for _ in range(p["n_rounds"]):
            r = y - (WI * A).sum(1)                                   # footprint residuals
            ibar = np.maximum(WI.sum(1), 1e-6)                        # sum_i w_i I_i
            pick = np.stack([rng.choice(P, k, replace=False, p=W[j] / W[j].sum()) for j in range(n)])
            rows = (np.arange(n)[:, None] * P + pick).ravel()
            tgt = np.repeat(r / ibar, k)
            sw = np.repeat(s, k)
            if not np.any(sw > 0):
                break
            tree = DecisionTreeRegressor(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                                         random_state=int(rng.integers(1 << 31))).fit(T[rows], tgt, sample_weight=sw)
            hgt = tree.predict(T).reshape(n, P)
            d = (WI * hgt).sum(1)                                      # footprint change per unit step
            denom = float(np.sum(s * d * d))
            if denom <= 0:
                break
            gamma = p["learning_rate"] * float(np.sum(s * d * r)) / denom
            A += gamma * hgt
            self.trees.append((gamma, tree))
            if val_state is not None:
                Tv, WIv, yv, Av = val_state
                Av += gamma * tree.predict(Tv).reshape(Av.shape)
                rm = float(np.sqrt(np.nanmean(((WIv * Av).sum(1) - yv) ** 2)))
                if rm < best - 1e-6:
                    best, best_n, bad = rm, len(self.trees), 0
                else:
                    bad += 1
                    if bad >= p["patience"]:
                        break
        if val_state is not None and best_n:
            self.trees = self.trees[:best_n]
        return self

    def predict_map(self, batch):
        T, _ = pixel_table(batch, self.params["input_set"], self.params["use_context"])
        n, h, w = len(batch), *batch.images["ImSW"].shape[-2:]
        a = np.clip(self._predict_albedo(T).reshape(n, h, w), 0.0, 1.0)
        return a * _common.pixel_insolation(batch)


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M4"])
