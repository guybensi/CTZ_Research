"""M6c: UNet initialised from M2 (hybrid, PLAN.md section 3).

Two stages:
1. Fit M2 (the pixel-additive polynomial trained exactly from the footprint moments) on the
   training footprints and predict its 1 km map on the training patches.
2. Pretrain the M5 UNet pixel-wise towards that M2 map (pretrain_epochs), then fine-tune it with
   the aggregation loss (+ edge-aware smoothness) like M5.

M2 trains on far more footprints (moments-only granules too) than there are image patches, so it
is a cheap, physically sensible starting point; the fine-tuning then adds spatial context.

Run alone:
    python3 -m flux1km.methods.m6c_unet_from_m2 --demo
"""
from __future__ import annotations

from ..models import get_model, register
from . import _torch
from .m5_unet import UNetModel


@register
class UNetFromM2(UNetModel):
    """M6c: M5 UNet, pretrained on the M2 map, then fine-tuned with the aggregation loss."""

    name = "M6c"
    complexity = 6
    defaults = {**UNetModel.defaults, "pretrain_epochs": 10, "m2_degree": 2, "m2_alpha": 1.0}

    @classmethod
    def param_grid(cls):
        return [{"pretrain_epochs": e, "lr": lr, "smooth_weight": s} for e in (5, 20) for lr in (1e-3, 3e-4) for s in (0.0, 0.01)]

    def train(self, X, ins, W, train, vt):
        p = self.params
        self.m2 = get_model("M2", input_set=p["input_set"], degree=p["m2_degree"], alpha=p["m2_alpha"]).fit(train)
        target = self.m2.predict_map(train)
        self.pretrain_loss = _torch.pretrain_map_net(self.net, X, ins, target, epochs=p["pretrain_epochs"], lr=p["lr"],
                                                     batch_size=p["batch_size"], seed=p["seed"])
        return super().train(X, ins, W, train, vt)


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M2", "M6c"])
