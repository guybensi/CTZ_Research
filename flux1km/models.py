"""Model interface and the NumPy baselines (PLAN.md section 3).

Every method implements FluxModel. Map-producing models implement predict_map, and their
footprint predictions come from the same aggregation operator for everyone. Footprint-only
models (M0, M1) override predict_footprint. Neural methods (M3, M5, M6) will live in separate
modules that import TF/PyTorch lazily, and register themselves in MODELS.
"""
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod

import numpy as np

from . import features
from .aggregation import aggregate
from .features import Batch
from .geometry import toa_insolation


class FluxModel(ABC):
    """Common interface. Subclasses set the class attributes and implement fit (+ predict_*)."""

    name = "base"
    produces_map = True
    uses_test_labels = False  # True only for diagnostics that read the footprint's own label

    def __init__(self, **params):
        self.params = params

    @classmethod
    def param_grid(cls) -> list[dict]:
        """Hyper-parameter candidates, tried in order up to the shared tuning budget."""
        return [{}]

    @abstractmethod
    def fit(self, train: Batch, val: Batch | None = None) -> FluxModel:
        """Fit on a training batch; `val` is for early stopping only (never for test)."""

    def predict_map(self, batch: Batch) -> np.ndarray:
        """Per-pixel flux (N, H, W) in W m^-2."""
        raise NotImplementedError(f"{self.name} does not produce 1 km maps")

    def predict_footprint(self, batch: Batch, weights: np.ndarray) -> np.ndarray:
        """Footprint flux (N,); default = aggregation operator applied to predict_map."""
        return aggregate(self.predict_map(batch), weights)


# ------------------------------------------------------------------ ridge helper
class _Ridge:
    """Weighted ridge regression on standardised features (affine, so it commutes with averaging)."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> _Ridge:
        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y) & (w > 0)
        X, y, w = X[ok], y[ok], w[ok]
        if len(y) == 0:
            raise ValueError("no valid training rows")
        self.mu = np.average(X, axis=0, weights=w)
        self.sd = np.sqrt(np.average((X - self.mu) ** 2, axis=0, weights=w))
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        self.y0 = np.average(y, weights=w)
        A = Z.T @ (Z * w[:, None]) + self.alpha * np.eye(Z.shape[1]) * w.sum() / len(w)
        self.coef = np.linalg.solve(A, Z.T @ (w * (y - self.y0)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X (..., F) or (N, F, H, W) with feature axis 1 when `pixel=True` is handled by caller."""
        return self.y0 + ((X - self.mu) / self.sd) @ self.coef

    def predict_pixels(self, P: np.ndarray) -> np.ndarray:
        """P (N, F, H, W) -> (N, H, W)."""
        return self.predict(np.moveaxis(P, 1, -1))


def _target(batch: Batch, kind: str) -> np.ndarray:
    if kind == "flux":
        return batch.label
    with np.errstate(divide="ignore", invalid="ignore"):
        return batch.label / toa_insolation(batch.scalars["SZA"])


def _from_target(pred: np.ndarray, sza, kind: str) -> np.ndarray:
    return pred if kind == "flux" else pred * toa_insolation(sza)


# ------------------------------------------------------------------ null baselines
class Climatology(FluxModel):
    """B-clim: training-mean albedo times each pixel's insolation. The floor every method must beat."""

    name = "B-clim"

    def fit(self, train, val=None):
        a = _target(train, "albedo")
        ok = np.isfinite(a) & (train.weight > 0)
        self.albedo = float(np.average(a[ok], weights=train.weight[ok]))
        return self

    def predict_map(self, batch):
        return self.albedo * toa_insolation(batch.images["Imviews"][:, 0])

    def predict_footprint(self, batch, weights):
        if batch.has_images:
            return super().predict_footprint(batch, weights)
        return self.albedo * toa_insolation(batch.scalars["SZA"])


class UniTR(FluxModel):
    """B-const (Gao's uniTR, Law's constant-in-bag): every pixel gets its footprint's own label.

    Reads test labels, so it is excluded from footprint ranking; it is the reference for
    pixel-level scores on synthetic truth and for the interleave test.
    """

    name = "B-const"
    uses_test_labels = True

    def fit(self, train, val=None):
        return self

    def predict_map(self, batch):
        shape = batch.images["ImSW"].shape
        return np.broadcast_to(batch.label[:, None, None], (shape[0], shape[2], shape[3])).astype(float)


# ------------------------------------------------------------------ ridge models
class _RidgeModel(FluxModel):
    defaults = {"alpha": 1.0, "input_set": "F5", "target": "albedo"}

    def __init__(self, **params):
        super().__init__(**{**self.defaults, **params})

    @classmethod
    def param_grid(cls):
        return [{"alpha": a} for a in (0.01, 0.1, 1.0, 10.0)]


class FootprintRidge(_RidgeModel):
    """M0 (Ridge baseline): footprint mean/std + geometry. Footprint-level only, no map."""

    name = "M0-ridge"
    produces_map = False

    def fit(self, train, val=None):
        X, self.names = features.footprint_features(train, self.params["input_set"], ("m", "std"))
        self.ridge = _Ridge(self.params["alpha"]).fit(X, _target(train, self.params["target"]), train.weight)
        return self

    def predict_footprint(self, batch, weights):
        X, _ = features.footprint_features(batch, self.params["input_set"], ("m", "std"))
        return _from_target(self.ridge.predict(X), batch.scalars["SZA"], self.params["target"])


class BagPixelPoly(_RidgeModel):
    """B-bagpixel (Law 2018): a polynomial model of the footprint MEANS, f(mean x), applied per pixel.

    The naive way to turn a footprint model into a map. Contrast with PixelPoly (M2), which
    learns mean(f(x)) from the raw moments.
    """

    name = "B-bagpixel"
    defaults = {**_RidgeModel.defaults, "degree": 2}

    @classmethod
    def param_grid(cls):
        return [{"alpha": a, "degree": d} for d, a in itertools.product((1, 2, 3), (0.1, 1.0, 10.0))]

    def _fp(self, batch):
        (m, _, _, _), _ = features.footprint_moment_stack(batch, self.params["input_set"])
        geo, _ = features.footprint_geometry(batch, features.schema.input_set(self.params["input_set"]).use_glint)
        return np.concatenate([np.concatenate([m ** (k + 1) for k in range(self.params["degree"])], axis=1), geo], axis=1)

    def fit(self, train, val=None):
        self.ridge = _Ridge(self.params["alpha"]).fit(self._fp(train), _target(train, self.params["target"]), train.weight)
        return self

    def predict_map(self, batch):
        X, _ = features.pixel_features(batch, self.params["input_set"])
        n_ch = len(features.schema.input_set(self.params["input_set"]).sw) + len(features.schema.input_set(self.params["input_set"]).lw)
        ch, geo = X[:, :n_ch], X[:, n_ch:]
        P = np.concatenate([np.concatenate([ch ** (k + 1) for k in range(self.params["degree"])], axis=1), geo], axis=1)
        return _from_target(self.ridge.predict_pixels(P), batch.images["Imviews"][:, 0], self.params["target"])


class PixelPoly(_RidgeModel):
    """M2: per-pixel polynomial model trained exactly from the footprint moments.

    Fitted on footprint_poly_features (raw moments), applied to pixel_poly_features. Because the
    model is linear in those features, mean(f(pixels)) equals f applied to the moments, so this is
    a pixel model trained with the aggregation loss, and it trains on moments-only granules too.
    """

    name = "M2"
    defaults = {**_RidgeModel.defaults, "degree": 2}

    @classmethod
    def param_grid(cls):
        return [{"alpha": a, "degree": d} for d, a in itertools.product((1, 2, 3), (0.1, 1.0, 10.0))]

    def fit(self, train, val=None):
        X, self.names = features.footprint_poly_features(train, self.params["input_set"], self.params["degree"])
        self.ridge = _Ridge(self.params["alpha"]).fit(X, _target(train, self.params["target"]), train.weight)
        return self

    def predict_map(self, batch):
        P, names = features.pixel_poly_features(batch, self.params["input_set"], self.params["degree"])
        assert names == self.names
        return _from_target(self.ridge.predict_pixels(P), batch.images["Imviews"][:, 0], self.params["target"])

    def predict_footprint(self, batch, weights):
        if batch.has_images:
            return super().predict_footprint(batch, weights)
        X, _ = features.footprint_poly_features(batch, self.params["input_set"], self.params["degree"])
        return _from_target(self.ridge.predict(X), batch.scalars["SZA"], self.params["target"])


MODELS: dict[str, type[FluxModel]] = {m.name: m for m in (Climatology, UniTR, FootprintRidge, BagPixelPoly, PixelPoly)}


def get_model(name: str, **params) -> FluxModel:
    """Instantiate a registered model by name.

    Example:
        >>> get_model("M2", degree=1).params["degree"]
        1

    Raises:
        KeyError: For an unknown name.
    """
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name](**params)
