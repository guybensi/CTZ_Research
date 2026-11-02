"""Base classes for the method modules.

* MapNetModel: a PyTorch network that maps per-pixel inputs to a per-pixel flux map and is
  trained only with the footprint aggregation loss (M3, M5, M6a-d, M6f).
* TabularFootprintModel: a footprint-level regressor on tabular features (M0 ANNs, M1).

Both take the shared parameters `input_set` (F1..F5), `seed`, and for map models `operator` and
`psf_path`, which the benchmark passes in so the training operator equals the scoring operator.
"""
from __future__ import annotations

import numpy as np

from .. import aggregation, features
from ..features import Batch
from ..models import FluxModel
from . import _common, _torch


def load_psf(psf_path: str | None, h: int = 27, w: int = 55) -> np.ndarray | None:
    """The PSF array for operator "ceres_psf" (None when no path is given)."""
    return None if psf_path is None else aggregation.load_psf_file(psf_path, h, w)


class MapNetModel(FluxModel):
    """Per-pixel neural map model trained with MSE(sum w f, F). Subclasses implement build_net."""

    needs_operator = True
    needs_torch = True
    defaults: dict = {"input_set": "F5", "operator": "fov_ellipse", "psf_path": None, "epochs": 60, "lr": 1e-3,
                      "batch_size": 32, "patience": 8, "weight_decay": 0.0, "augment": False, "seed": 0}

    def __init__(self, **params):
        super().__init__(**{**self.defaults, **params})

    # -- inputs ------------------------------------------------------------
    def inputs(self, batch: Batch) -> np.ndarray:
        """Per-pixel inputs (N, F, H, W). Default: MODIS channels + encoded geometry (+ glint)."""
        X, self.input_names = features.pixel_features(batch, self.params["input_set"])
        return X

    def build_net(self, n_features: int):
        """Return a torch module (B, F, H, W) -> (B, 1, H, W) logits."""
        raise NotImplementedError

    def extra_loss(self, train: Batch, X: np.ndarray):
        """Optional regulariser callable for _torch.train_map_net (None = none)."""
        return None

    def forward(self):
        """Optional custom forward for networks with extra inputs (None = net(x))."""
        return None

    def _weights(self, batch: Batch) -> np.ndarray:
        h, w = batch.images["ImSW"].shape[-2:]
        return features.footprint_weights(batch, self.params["operator"], load_psf(self.params["psf_path"], h, w))

    def _prepare(self, batch: Batch, fit_scaler: bool = False):
        X = self.inputs(batch)
        if fit_scaler:
            self.scaler = _common.Standardizer(axis=1).fit(X, seed=self.params["seed"])
        return self.scaler.transform(X).astype(np.float32), _common.pixel_insolation(batch)

    def fit(self, train, val=None):
        if not train.has_images:
            raise ValueError(f"{self.name} needs image patches")
        X, ins = self._prepare(train, fit_scaler=True)
        vt = None
        if val is not None and len(val) and val.has_images:
            Xv, iv = self._prepare(val)
            vt = (Xv, iv, self._weights(val), val.label)
        self.net = self.build_net(X.shape[1])
        self.history = self.train(X, ins, self._weights(train), train, vt)
        return self

    def train(self, X, ins, W, train: Batch, vt):
        """Run the aggregation-loss training (subclasses may wrap it, e.g. M6c pretraining)."""
        p = self.params
        return _torch.train_map_net(self.net, X, ins, W, train.label, train.weight, vt, epochs=p["epochs"], lr=p["lr"],
                                    batch_size=p["batch_size"], patience=p["patience"], weight_decay=p["weight_decay"],
                                    extra_loss=self.extra_loss(train, X), augment=p["augment"], seed=p["seed"],
                                    forward=self.forward(), subset_fraction=p.get("subset_fraction", 1.0))

    def predict_map(self, batch):
        X, ins = self._prepare(batch)
        self._predicting = batch
        try:
            return _torch.predict_map_net(self.net, X, ins, forward=self.forward())
        finally:
            self._predicting = None


class TabularFootprintModel(FluxModel):
    """Footprint-level model on tabular MODIS features. Subclasses implement features_of and fit_xy."""

    produces_map = False
    defaults: dict = {"input_set": "F5", "target": "albedo", "seed": 0}

    def __init__(self, **params):
        super().__init__(**{**self.defaults, **params})

    def features_of(self, batch: Batch) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError

    def fit_xy(self, X, y, w, Xv, yv):
        """Fit on standardised features; return predict(X) -> target-space prediction."""
        raise NotImplementedError

    def _to_target(self, batch: Batch) -> np.ndarray:
        return _common.albedo_target(batch) if self.params["target"] == "albedo" else batch.label.astype(float)

    def _from_target(self, pred: np.ndarray, batch: Batch) -> np.ndarray:
        return pred * _common.footprint_insolation(batch) if self.params["target"] == "albedo" else pred

    def fit(self, train, val=None):
        X, self.names = self.features_of(train)
        self.scaler = _common.Standardizer(axis=1).fit(X)
        Xv = yv = None
        if val is not None and len(val):
            Xv, yv = self.scaler.transform(self.features_of(val)[0]), self._to_target(val)
        self._predict = self.fit_xy(self.scaler.transform(X), self._to_target(train), train.weight, Xv, yv)
        return self

    def predict_footprint(self, batch, weights=None):
        X, _ = self.features_of(batch)
        return self._from_target(self._predict(self.scaler.transform(X)), batch)
