"""M10: Gaussian-process aggregate models with per-pixel uncertainty (PLAN.md section 3).

Exact GPs do not scale to ~1,500 pixels per bag, so both variants use random Fourier features
(RFF): f(x) = I(x) * (a0 + phi(x)^T beta), phi(x) = sqrt(2/D) cos(Omega x + b),
Omega ~ N(0, 1 / lengthscale^2), beta ~ N(0, prior_std^2 I), I = S0 mu0 per pixel. This is a
Bayesian linear model, so with a Gaussian likelihood the aggregate posterior is exact:

* ``M10`` (VBAgg, Law 2018, Normal likelihood): a footprint observes
  y_k = sum_i w_ik f(x_i) + noise = a0 sum_i w_ik I_i + phibar_k^T beta + noise, with the bag
  feature phibar_k = sum_i w_ik I_i phi(x_i). The posterior of beta is Gaussian:
  A = PhiBar^T S PhiBar / noise^2 + I / prior^2, mean = A^-1 PhiBar^T S (y - a0 Ibar) / noise^2.
  Pixel mean and std follow directly (predict_map_std).
* ``M10-cmp`` (mediated / deconditional downscaling, Chau 2021): the footprint moments z are the
  mediating variable. A conditional-mean map z -> phibar is learned by ridge on the bags that have
  images, and then the posterior of beta is formed from phibar_hat(z) of ALL labelled footprints,
  including moments-only granules (fit_mediated(image_batch, moments_batch)). fit(train) uses the
  same batch for both, which is the special case where every bag has an image.

Low priority in the plan; it is here so the per-pixel uncertainty can be compared.

Run alone:
    python3 -m flux1km.methods.m10_gp --demo
"""
from __future__ import annotations

import numpy as np

from .. import features
from ..features import Batch
from ..models import FluxModel, register
from . import _common
from ._base import load_psf


class RFF:
    """Random Fourier features of an RBF kernel: phi(x) = sqrt(2/D) cos(x Omega + b).

    Example:
        >>> f = RFF(3, 50, 1.0, np.random.default_rng(0)); f(np.zeros((2, 3))).shape
        (2, 50)
    """

    def __init__(self, n_in: int, n_features: int, lengthscale: float, rng: np.random.Generator):
        if lengthscale <= 0 or n_features < 1:
            raise ValueError("lengthscale must be positive and n_features >= 1")
        self.omega = rng.normal(scale=1.0 / lengthscale, size=(n_in, n_features))
        self.b = rng.uniform(0, 2 * np.pi, size=n_features)
        self.scale = np.sqrt(2.0 / n_features)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.scale * np.cos(X @ self.omega + self.b)


@register
class VBAggRFF(FluxModel):
    """M10: aggregate-likelihood GP (VBAgg, Normal) with RFF; gives a per-pixel posterior std."""

    name = "M10"
    complexity = 4
    needs_operator = True
    defaults = {"input_set": "F5", "operator": "fov_ellipse", "psf_path": None, "n_features": 256, "lengthscale": 3.0,
                "noise": 5.0, "prior_std": 0.2, "seed": 0}

    def __init__(self, **params):
        super().__init__(**{**self.defaults, **params})

    @classmethod
    def param_grid(cls):
        return [{"lengthscale": ls, "noise": nz} for ls in (1.5, 3.0, 6.0) for nz in (2.0, 5.0, 10.0)]

    def _pixels(self, batch):
        X, _ = features.pixel_features(batch, self.params["input_set"])
        return self.scaler.transform(X)

    def _weights(self, batch):
        h, w = batch.images["ImSW"].shape[-2:]
        return features.footprint_weights(batch, self.params["operator"], load_psf(self.params["psf_path"], h, w))

    def bag_features(self, batch: Batch) -> tuple[np.ndarray, np.ndarray]:
        """(PhiBar (N, D), Ibar (N,)): sum_i w_i I_i phi(x_i) and sum_i w_i I_i per footprint."""
        Z = self._pixels(batch)
        WI = self._weights(batch) * _common.pixel_insolation(batch)
        n, f = Z.shape[:2]
        out = np.empty((n, self.rff.omega.shape[1]))
        for k in range(n):
            out[k] = WI[k].ravel() @ self.rff(Z[k].reshape(f, -1).T)
        return out, WI.sum(axis=(1, 2))

    def _posterior(self, Phi, ibar, y, s):
        p = self.params
        ok = np.isfinite(y) & (s > 0)
        Phi, ibar, y, s = Phi[ok], ibar[ok], y[ok], s[ok]
        alb = y / np.maximum(ibar, 1e-6)
        self.a0 = float(np.average(alb, weights=s))
        r = y - self.a0 * ibar
        A = (Phi * s[:, None]).T @ Phi / p["noise"] ** 2 + np.eye(Phi.shape[1]) / p["prior_std"] ** 2
        self.A_inv = np.linalg.inv(A)
        self.beta = self.A_inv @ (Phi.T @ (s * r)) / p["noise"] ** 2

    def _setup(self, train):
        if not train.has_images:
            raise ValueError(f"{self.name} needs image patches")
        X, _ = features.pixel_features(train, self.params["input_set"])
        self.scaler = _common.Standardizer(axis=1).fit(X, seed=self.params["seed"])
        self.rff = RFF(X.shape[1], self.params["n_features"], self.params["lengthscale"], np.random.default_rng(self.params["seed"]))

    def fit(self, train, val=None):
        self._setup(train)
        Phi, ibar = self.bag_features(train)
        self._posterior(Phi, ibar, train.label.astype(float), train.weight.astype(float))
        return self

    def _pixel_phi(self, batch):
        Z = self._pixels(batch)
        n, f, h, w = Z.shape
        return self.rff(np.moveaxis(Z, 1, -1).reshape(-1, f)), (n, h, w)

    def predict_map(self, batch):
        phi, shape = self._pixel_phi(batch)
        alb = self.a0 + phi @ self.beta
        return np.clip(alb.reshape(shape), 0.0, 1.0) * _common.pixel_insolation(batch)

    def predict_map_std(self, batch: Batch) -> np.ndarray:
        """Posterior std of every pixel's flux (N, H, W), W m^-2."""
        phi, shape = self._pixel_phi(batch)
        var = np.einsum("pd,de,pe->p", phi, self.A_inv, phi)
        return np.sqrt(np.clip(var, 0, None)).reshape(shape) * _common.pixel_insolation(batch)


def mediating_features(batch: Batch, input_set: str = "F5") -> np.ndarray:
    """Footprint mediating variable z: raw moments E[x], E[x^2] per channel + geometry, plus a constant."""
    Z, _ = features.footprint_poly_features(batch, input_set, 2)
    return np.column_stack([np.ones(len(Z)), np.nan_to_num(Z)])


@register
class MediatedGP(VBAggRFF):
    """M10-cmp: deconditional (mediated) downscaling with RFF, z = footprint moments."""

    name = "M10-cmp"
    complexity = 4
    defaults = {**VBAggRFF.defaults, "cme_ridge": 1e-2}

    def fit_mediated(self, image_batch: Batch, moments_batch: Batch):
        """Learn z -> phibar on image_batch, then the beta posterior on every labelled footprint of moments_batch."""
        self._setup(image_batch)
        Phi, ibar = self.bag_features(image_batch)
        Zi = mediating_features(image_batch, self.params["input_set"])
        self.z_scaler = _common.Standardizer(axis=1).fit(Zi[:, 1:])
        Zi = np.column_stack([Zi[:, :1], self.z_scaler.transform(Zi[:, 1:])])
        T = np.column_stack([Phi, ibar])
        reg = self.params["cme_ridge"] * len(Zi) * np.eye(Zi.shape[1])
        self.cme = np.linalg.solve(Zi.T @ Zi + reg, Zi.T @ T)
        Zm = mediating_features(moments_batch, self.params["input_set"])
        Zm = np.column_stack([Zm[:, :1], self.z_scaler.transform(Zm[:, 1:])])
        That = Zm @ self.cme
        self._posterior(That[:, :-1], np.maximum(That[:, -1], 1e-6), moments_batch.label.astype(float),
                        moments_batch.weight.astype(float))
        return self

    def fit(self, train, val=None):
        return self.fit_mediated(train, train)


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M10", "M10-cmp"])
