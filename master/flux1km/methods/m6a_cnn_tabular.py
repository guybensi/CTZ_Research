"""M6a: CNN plus a tabular branch (hybrid, PLAN.md section 3).

Two branches merged per pixel:
* a convolutional branch (the UNet trunk of M5) on the per-pixel MODIS channels and geometry;
* a tabular branch: an MLP on the footprint's MODIS moments (mean/std per channel) and footprint
  geometry, the same kind of features M0/M1 use. Its output vector is broadcast over the patch.
A 1x1 head on the concatenation gives the per-pixel albedo; the loss is the aggregation loss.

Question it answers: does footprint-level context (what the Guy model sees) add to what the CNN
learns from the pixels? Implementation note: the tabular features travel as extra constant input
channels, so augmentation and batching treat both branches consistently.

Run alone:
    python3 -m flux1km.methods.m6a_cnn_tabular --demo
"""
from __future__ import annotations

import numpy as np

from .. import features
from ..models import register
from . import _torch
from ._base import MapNetModel


def make_cnn_tabular(n_pixel: int, n_tab: int, base: int = 16, depth: int = 2, tab_hidden: int = 32):
    """Network (B, n_pixel + n_tab, H, W) -> (B, 1, H, W); the last n_tab channels are constant per patch."""
    torch = _torch.require_torch()
    nn = torch.nn

    class CNNTabular(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = _torch.make_unet(n_pixel, base, depth)
            self.tab = _torch.make_mlp(n_tab, (tab_hidden,), tab_hidden)
            self.head = nn.Sequential(nn.Conv2d(base + tab_hidden, base, 1), nn.ReLU(), nn.Conv2d(base, 1, 1))

        def forward(self, x):
            px, tb = x[:, :n_pixel], x[:, n_pixel:, 0, 0]
            fmap = self.cnn.features(px)
            t = self.tab(tb)[:, :, None, None].expand(-1, -1, *fmap.shape[-2:])
            return self.head(torch.cat([fmap, t], dim=1))

    return CNNTabular()


@register
class CNNTabularModel(MapNetModel):
    """M6a: UNet trunk + footprint-moment MLP branch, aggregation loss."""

    name = "M6a"
    complexity = 6
    defaults = {**MapNetModel.defaults, "base": 16, "depth": 2, "tab_hidden": 32, "batch_size": 16, "augment": True}

    @classmethod
    def param_grid(cls):
        return [{"base": b, "lr": lr} for b in (16, 32) for lr in (1e-3, 3e-4)]

    def inputs(self, batch):
        X, names = features.pixel_features(batch, self.params["input_set"])
        T, tnames = features.footprint_features(batch, self.params["input_set"], ("m", "std"))
        self.n_pixel, self.n_tab = X.shape[1], T.shape[1]
        self.input_names = names + [f"tab_{n}" for n in tnames]
        Tb = np.broadcast_to(T[:, :, None, None], (len(T), T.shape[1], *X.shape[-2:]))
        return np.concatenate([X, Tb], axis=1)

    def build_net(self, n_features):
        return make_cnn_tabular(self.n_pixel, self.n_tab, self.params["base"], self.params["depth"], self.params["tab_hidden"])


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M6a"])
