"""M3: per-pixel MLP trained with the aggregation loss (PLAN.md section 3).

f(x_i) uses only the pixel's own MODIS channels, its geometry and its glint angle: no spatial
context at all. The network is a stack of 1x1 convolutions, so the same weights are applied to
every pixel. It is trained only through the footprint loss MSE(sum_i w_i f(x_i), F_CERES), with
w the same aggregation operator H the benchmark scores with.

This is the default method on a tie (PLAN.md section 6, rule 5).

Regularisation (parameter ``reg``, compared in the benchmark as section 3 asks):
* ``"none"``;
* ``"manifold"``: feature-space regularisation (Law 2018): pixels with similar inputs should give
  similar outputs. It uses no spatial proximity, so it cannot create an artificial CTZ;
* ``"subset"``: random-subset aggregation (Fan 2021): each step aggregates a random half of the
  footprint's pixels.
Spatial (TV) smoothness belongs to the spatial models (M5).

Run alone:
    python3 -m flux1km.methods.m3_pixel_mlp --demo
"""
from __future__ import annotations

from ..models import register
from . import _torch
from ._base import MapNetModel

REGULARISERS = ("none", "manifold", "subset")


@register
class PixelMLP(MapNetModel):
    """M3: 1x1-convolution MLP f(pixel, geometry, glint) trained with the aggregation loss."""

    name = "M3"
    complexity = 4
    defaults = {**MapNetModel.defaults, "hidden": (64, 64), "reg": "none", "reg_weight": 0.1, "bandwidth": 1.0,
                "subset_fraction": 1.0}

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["reg"] not in REGULARISERS:
            raise ValueError(f"unknown regulariser {self.params['reg']!r}; choose from {REGULARISERS}")
        if self.params["reg"] == "subset" and self.params["subset_fraction"] >= 1.0:
            self.params["subset_fraction"] = 0.5

    @classmethod
    def param_grid(cls):
        return [{"hidden": h, "lr": lr, "reg": r} for r in REGULARISERS for h in ((64, 64), (128, 64, 32)) for lr in (1e-3, 3e-3)]

    def build_net(self, n_features):
        return _torch.make_pixel_mlp(n_features, tuple(self.params["hidden"]))

    def extra_loss(self, train, X):
        if self.params["reg"] != "manifold":
            return None
        torch = _torch.require_torch()
        gen = torch.Generator().manual_seed(int(self.params["seed"]))
        lam, bw = self.params["reg_weight"], self.params["bandwidth"]
        return lambda xb, flux, idx, flipped: lam * _torch.manifold_penalty(xb, flux, bandwidth=bw, generator=gen)


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M3"])
