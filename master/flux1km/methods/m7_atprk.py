"""M7: area-to-point regression kriging (ATPRK) of footprint residuals (PLAN.md section 3).

Regression kriging, coherence-preserving:
1. A trend model (default M2) gives a 1 km map f_trend.
2. Each footprint's residual R_k = F_k - sum_i w_ik f_trend,i is downscaled to the pixels by
   area-to-point simple kriging with an exponential point covariance C(d) = exp(-d / range):
       r(x) = c(x)^T (C_AA + nugget I)^-1 R,
   where c_k(x) = sum_j w_kj C(x, x_j) (point-to-area covariance) and
   C_AA[k, l] = sum_i sum_j w_ki w_lj C(x_i, x_j) (area-to-area covariance).
   The covariances are computed with FFT convolutions on the granule grid.
3. Map = f_trend + r. With nugget -> 0 the map aggregates exactly to every footprint label.

When footprint positions are known (images "row0"/"col0", the patch corner in the granule grid,
present in the synthetic data and in a future in-house footprint index), all overlapping
footprints of a granule are kriged jointly. Otherwise each footprint is kriged on its own patch.

It READS THE TEST FOOTPRINT LABELS (that is what downscaling means), so it is flagged
uses_test_labels and never ranked on the footprint score. PLAN.md runs it in the focus phase,
once the exact PSF and the geolocation are available; it is scored on synthetic pixel truth and
on the interleave test.

Run alone:
    python3 -m flux1km.methods.m7_atprk --demo
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .. import features
from ..models import FluxModel, get_model, register
from ._base import load_psf


def exponential_kernel(h: int, w: int, range_px: float) -> np.ndarray:
    """Point covariance exp(-d / range) on all displacements of an h x w grid: (2h-1, 2w-1).

    Example:
        >>> k = exponential_kernel(2, 2, 1.0); k.shape, float(k[1, 1])
        ((3, 3), 1.0)

    Raises:
        ValueError: If range_px is not positive.
    """
    if range_px <= 0:
        raise ValueError("range must be positive")
    dy, dx = np.mgrid[-(h - 1):h, -(w - 1):w]
    return np.exp(-np.hypot(dy, dx) / range_px)


def atpk_residual_maps(weights: np.ndarray, residuals: np.ndarray, range_px: float = 10.0,
                       offsets: np.ndarray | None = None, nugget: float = 1e-6) -> np.ndarray:
    """Area-to-point kriging of footprint residuals onto their patches.

    Example:
        >>> w = np.full((1, 3, 3), 1 / 9); r = atpk_residual_maps(w, np.array([2.0]))
        >>> round(float((r * w).sum()), 4)
        2.0

    Args:
        weights: (N, h, w) aggregation weights of the footprints (normalised internally).
        residuals: (N,) footprint residuals; NaN is treated as 0.
        range_px: Range of the exponential covariance, pixels.
        offsets: Optional (N, 2) patch corners (row, col) in a common grid; then overlapping
            footprints are kriged jointly. Without them each footprint is kriged alone.
        nugget: Relative nugget added to the diagonal of C_AA (numerical stability).

    Returns:
        (N, h, w) residual maps, one per footprint patch.
    """
    W = np.asarray(weights, float)
    W = W / W.sum(axis=(-2, -1), keepdims=True)
    R = np.nan_to_num(np.asarray(residuals, float))
    n, h, w = W.shape
    if offsets is None:
        K = exponential_kernel(h, w, range_px)
        out = np.empty_like(W)
        for k in range(n):
            c = signal.fftconvolve(W[k], K, mode="same")      # point-to-area covariance on the patch
            caa = float((W[k] * c).sum())
            out[k] = c / (caa * (1 + nugget)) * R[k]
        return out
    off = np.asarray(offsets, int)
    off = off - off.min(axis=0)
    H, Wd = int(off[:, 0].max()) + h, int(off[:, 1].max()) + w
    K = exponential_kernel(H, Wd, range_px)
    G = np.zeros((n, H, Wd))
    for k, (r0, c0) in enumerate(off):
        G[k, r0:r0 + h, c0:c0 + w] = W[k]
    Cpa = np.stack([signal.fftconvolve(G[k], K, mode="same") for k in range(n)])   # (n, H, Wd)
    Caa = np.einsum("khw,lhw->kl", G, Cpa)
    Caa = 0.5 * (Caa + Caa.T) + nugget * np.mean(np.diag(Caa)) * np.eye(n)
    lam = np.linalg.solve(Caa, R)
    field = np.einsum("k,khw->hw", lam, Cpa)
    return np.stack([field[r0:r0 + h, c0:c0 + w] for r0, c0 in off])


@register
class ATPRK(FluxModel):
    """M7: trend map (M2) + area-to-point kriging of the footprint residuals (reads test labels)."""

    name = "M7"
    uses_test_labels = True
    needs_operator = True
    complexity = 4
    defaults = {"trend": "M2", "range_px": 10.0, "nugget": 1e-6, "operator": "fov_ellipse", "psf_path": None,
                "input_set": "F5", "seed": 0}

    def __init__(self, **params):
        super().__init__(**{**self.defaults, **params})

    @classmethod
    def param_grid(cls):
        return [{"range_px": r} for r in (5.0, 10.0, 20.0)]

    def fit(self, train, val=None):
        self.trend = get_model(self.params["trend"], input_set=self.params["input_set"]).fit(train, val)
        return self

    def predict_map(self, batch):
        h, w = batch.images["ImSW"].shape[-2:]
        W = features.footprint_weights(batch, self.params["operator"], load_psf(self.params["psf_path"], h, w))
        base = self.trend.predict_map(batch)
        resid = batch.label - (base * W).sum(axis=(-2, -1))
        out = base.copy()
        has_pos = "row0" in batch.images and "col0" in batch.images and batch.granule is not None
        if not has_pos:
            return base + atpk_residual_maps(W, resid, self.params["range_px"], None, self.params["nugget"])
        for g in dict.fromkeys(batch.granule.tolist()):
            sel = np.flatnonzero(batch.granule == g)
            off = np.column_stack([batch.images["row0"][sel], batch.images["col0"][sel]])
            out[sel] += atpk_residual_maps(W[sel], resid[sel], self.params["range_px"], off, self.params["nugget"])
        return out


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M2", "M7"])
