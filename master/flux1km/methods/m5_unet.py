"""M5: UNet on 27x55 patches, aggregation loss + edge-aware smoothness (PLAN.md section 3).

Inputs: the per-pixel MODIS channels of the input set (up to 11 SW + 16 LW, minus the bad ones),
encoded geometry and glint: the "3 geometry + 11 SW + 16 LW" stack of the plan. The UNet sees the
whole patch, so it can learn spatial context and 3D effects (e.g. the part of the CTZ that no
single pixel's radiances reveal, see the hard synthetic scene).

Loss: MSE(sum_i w_i f_i, F) + smooth_weight * edge-aware TV, where the TV term
|grad f| * exp(-|grad x| / sigma) smooths the map only where the inputs are smooth, so it does not
blur across cloud edges (PLAN.md warns that plain TV can create an artificial CTZ).

Flip augmentation is on by default: a left-right flip of a patch is applied to the inputs, the
insolation and the aggregation weights together, which is the same as toggling FOVorientation.

Decision rule: the UNet must beat the hybrid M6b, not only M3 (PLAN.md section 6, rule 6).

Run alone:
    python3 -m flux1km.methods.m5_unet --demo
"""
from __future__ import annotations

from ..models import register
from . import _torch
from ._base import MapNetModel


@register
class UNetModel(MapNetModel):
    """M5: compact UNet map model with the aggregation loss and edge-aware smoothness."""

    name = "M5"
    complexity = 6
    defaults = {**MapNetModel.defaults, "base": 16, "depth": 2, "smooth_weight": 0.01, "edge_sigma": 1.0, "augment": True,
                "batch_size": 16}

    @classmethod
    def param_grid(cls):
        return [{"base": b, "smooth_weight": s, "lr": lr} for b in (16, 32) for s in (0.0, 0.01, 0.1) for lr in (1e-3, 3e-4)]

    def build_net(self, n_features):
        return _torch.make_unet(n_features, self.params["base"], self.params["depth"])

    def extra_loss(self, train, X):
        lam, sig = self.params["smooth_weight"], self.params["edge_sigma"]
        if lam <= 0:
            return None
        return lambda xb, flux, idx, flipped: lam * _torch.edge_aware_smoothness(xb, flux, sig)


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M5"])
