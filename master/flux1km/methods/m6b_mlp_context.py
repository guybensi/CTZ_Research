"""M6b: pixel MLP plus a residual context encoder (hybrid, PLAN.md section 3).

logit(x) = MLP(pixel_i) + g(neighbourhood), where MLP is the M3 per-pixel network and g a small
3x3 convolutional encoder over the patch. g is penalised (context_penalty * mean g^2), so the
pixel branch explains what it can and the context branch only the residual that needs
neighbours (e.g. the hidden part of the CTZ).

Decision rule: M5 (the UNet) must beat this model, not only M3; if it does not, spatial context
adds little beyond a residual correction, which is itself a result about the scale of the CTZ
(PLAN.md section 6, rule 6).

Run alone:
    python3 -m flux1km.methods.m6b_mlp_context --demo
"""
from __future__ import annotations

from ..models import register
from . import _torch
from ._base import MapNetModel


def make_mlp_context(n_features: int, hidden: tuple[int, ...] = (64, 64), width: int = 16, layers: int = 3):
    """Network with attributes .pixel (1x1 MLP) and .context (3x3 encoder); forward = sum of both."""
    nn = _torch.require_torch().nn

    class MLPContext(nn.Module):
        def __init__(self):
            super().__init__()
            self.pixel = _torch.make_pixel_mlp(n_features, hidden)
            self.context = _torch.make_context_encoder(n_features, width, 1, layers)

        def forward(self, x):
            return self.pixel(x) + self.context(x)

    return MLPContext()


@register
class MLPContextModel(MapNetModel):
    """M6b: M3 pixel MLP + penalised residual context encoder."""

    name = "M6b"
    complexity = 5
    defaults = {**MapNetModel.defaults, "hidden": (64, 64), "width": 16, "layers": 3, "context_penalty": 0.01}

    @classmethod
    def param_grid(cls):
        return [{"context_penalty": c, "lr": lr, "width": w} for c in (0.001, 0.01, 0.1) for lr in (1e-3, 3e-3) for w in (16,)]

    def build_net(self, n_features):
        return make_mlp_context(n_features, tuple(self.params["hidden"]), self.params["width"], self.params["layers"])

    def extra_loss(self, train, X):
        lam = self.params["context_penalty"]
        if lam <= 0:
            return None
        return lambda xb, flux, idx, flipped: lam * (self.net.context(xb) ** 2).mean()  # self.net is built before training


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M6b"])
