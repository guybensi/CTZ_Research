"""M6d: soft distillation from M1 into a pixel model (hybrid, PLAN.md section 3).

The M1 teacher (the improved Guy model, footprint-level GBM on MODIS-only moments) is fitted on
the training footprints. It is then applied to sub-footprint tiles: every training patch is cut
into non-overlapping tiles (default 9 x 11 px, 15 tiles per 27x55 patch), the tile moments are
computed from the pixels (uniform weights) and the teacher predicts each tile's flux.

The student (the M3 per-pixel MLP) is trained with
    MSE(sum_i w_i f_i, F_CERES) + distill_weight * mean_tiles (mean_tile f - teacher_tile)^2
so the teacher adds supervision at a finer scale than the footprint, softly (distill_weight is
tuned). Caveat, tested by the benchmark itself: the teacher was trained on PSF-weighted 27x55
footprints, so the tile predictions carry a domain shift; the CERES loss keeps the student
anchored.

Run alone:
    python3 -m flux1km.methods.m6d_distill_m1 --demo
"""
from __future__ import annotations

import numpy as np

from ..models import get_model, register
from . import _common, _torch
from .m3_pixel_mlp import PixelMLP


def teacher_tile_targets(teacher, images: dict, masks: np.ndarray) -> np.ndarray:
    """Teacher flux of every tile of every patch: (N, T).

    Example:
        >>> # targets = teacher_tile_targets(m1, batch.images, _common.tile_masks(27, 55, (9, 11)))
    """
    return np.stack([teacher.predict_footprint(_common.window_batch(images, m), None) for m in masks], axis=1)


@register
class DistilledPixelMLP(PixelMLP):
    """M6d: M3 student with tile-level soft targets from an M1 teacher."""

    name = "M6d"
    complexity = 5
    defaults = {**PixelMLP.defaults, "teacher": "M1-gbm", "distill_weight": 0.5, "tile": (9, 11)}

    @classmethod
    def param_grid(cls):
        return [{"distill_weight": d, "lr": lr} for d in (0.1, 0.5, 2.0) for lr in (1e-3, 3e-3)]

    def fit(self, train, val=None):
        p = self.params
        self.teacher = get_model(p["teacher"], input_set=p["input_set"], seed=p["seed"]).fit(train, val)
        h, w = train.images["ImSW"].shape[-2:]
        self.masks = _common.tile_masks(h, w, tuple(p["tile"]))
        self.targets = teacher_tile_targets(self.teacher, train.images, self.masks)
        return super().fit(train, val)

    def extra_loss(self, train, X):
        torch = _torch.require_torch()
        dev = _torch.device()
        M = torch.as_tensor(self.masks, dtype=torch.float32, device=dev)
        T = torch.as_tensor(np.nan_to_num(self.targets), dtype=torch.float32, device=dev)
        scale = float(np.nanstd(train.label)) or 1.0
        lam = self.params["distill_weight"]

        def loss(xb, flux, idx, flipped):
            m = M.flip(-1) if flipped else M
            tile_mean = torch.einsum("bhw,thw->bt", flux, m) / m.sum(dim=(1, 2))
            return lam * ((tile_mean - T[idx.to(dev)]) ** 2).mean() / scale**2

        return loss


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M1-gbm", "M3", "M6d"])
