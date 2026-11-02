"""M6f: mediated pseudo-labels (Chau 2021) for a pixel model (hybrid, PLAN.md section 3).

The footprint moments are the mediating variable. M1 (footprint GBM, MODIS-only) is learned on
the labelled footprints. It then labels new "bags" that have no CERES flux:
* random rectangular windows inside the training patches (different sizes and positions than
  the CERES footprints), and
* windows of extra, unlabelled MODIS image granules (any MODIS L1B granule cut into patches,
  parameter ``unlabeled_root``: a folder whose ``images/*.npz`` hold ImSW, ImLW, Imviews).
For every bag the uniform pixel moments are computed and the teacher predicts its flux. The
student (the M3 per-pixel MLP) is trained with the aggregation loss on the real footprints plus
the pseudo-bags, the latter weighted by ``pseudo_weight``. This lifts the limit of the 6 image
granules with labels. No CERES quantity is ever an input; test labels are never read.

Run alone:
    python3 -m flux1km.methods.m6f_pseudo_labels --demo
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..models import get_model, register
from . import _common
from .m3_pixel_mlp import PixelMLP


def random_windows(h: int, w: int, n: int, rng: np.random.Generator, min_size: tuple[int, int] = (9, 11)) -> np.ndarray:
    """n random rectangular pixel masks (n, h, w), each at least min_size.

    Example:
        >>> m = random_windows(27, 55, 3, np.random.default_rng(0)); m.shape, bool(m.sum((1, 2)).min() >= 99)
        ((3, 27, 55), True)

    Raises:
        ValueError: If min_size does not fit in the patch.
    """
    mh, mw = min_size
    if mh > h or mw > w or mh < 1 or mw < 1:
        raise ValueError(f"min_size {min_size} does not fit in {(h, w)}")
    out = np.zeros((n, h, w), bool)
    for k in range(n):
        hh, ww = int(rng.integers(mh, h + 1)), int(rng.integers(mw, w + 1))
        r, c = int(rng.integers(0, h - hh + 1)), int(rng.integers(0, w - ww + 1))
        out[k, r:r + hh, c:c + ww] = True
    return out


def load_unlabeled_images(root: str | Path, max_patches: int | None = None) -> dict | None:
    """Stack the images/*.npz of an unlabelled folder into one images dict (None if empty).

    Only ImSW, ImLW and Imviews are needed (no scalars, no CERES); FOVorientation is ignored.
    """
    files = sorted(Path(root, "images").glob("*.npz"))
    if not files:
        return None
    parts: dict[str, list] = {"ImSW": [], "ImLW": [], "Imviews": []}
    for f in files:
        with np.load(f) as d:
            for k in parts:
                parts[k].append(d[k])
    out = {k: np.concatenate(v) for k, v in parts.items()}
    if max_patches is not None:
        out = {k: v[:max_patches] for k, v in out.items()}
    return out


@register
class PseudoLabelPixelMLP(PixelMLP):
    """M6f: M3 student trained on real footprints + M1-labelled pseudo-bags."""

    name = "M6f"
    complexity = 5
    defaults = {**PixelMLP.defaults, "teacher": "M1-gbm", "pseudo_per_patch": 2, "pseudo_weight": 0.3,
                "unlabeled_root": None, "max_unlabeled": 5000}

    @classmethod
    def param_grid(cls):
        return [{"pseudo_weight": pw, "pseudo_per_patch": k} for pw in (0.1, 0.3, 1.0) for k in (1, 3)]

    def _pseudo_bags(self, images: dict, rng):
        h, w = images["ImSW"].shape[-2:]
        idx, masks, targets = [], [], []
        for _ in range(self.params["pseudo_per_patch"]):
            m = random_windows(h, w, 1, rng)[0]
            targets.append(self.teacher.predict_footprint(_common.window_batch(images, m), None))
            masks.append(np.broadcast_to(m, (len(images["ImSW"]), h, w)))
            idx.append(np.arange(len(images["ImSW"])))
        return np.concatenate(idx), np.concatenate(masks), np.concatenate(targets)

    def fit(self, train, val=None):
        p = self.params
        self.teacher = get_model(p["teacher"], input_set=p["input_set"], seed=p["seed"]).fit(train, val)
        self.unlabeled = load_unlabeled_images(p["unlabeled_root"], p["max_unlabeled"]) if p["unlabeled_root"] else None
        return super().fit(train, val)

    def train(self, X, ins, W, train, vt):
        rng = np.random.default_rng(self.params["seed"])
        Xs, Is, Ws, ys, ss = [X], [ins], [W], [train.label.astype(float)], [train.weight.astype(float)]
        sources = [(train.images, X, ins)]
        if self.unlabeled is not None:
            ub = _UnlabeledBatch(self.unlabeled)
            Xu = self.scaler.transform(self.inputs(ub)).astype(np.float32)
            sources.append((self.unlabeled, Xu, _common.pixel_insolation(ub)))
        for images, Xsrc, Isrc in sources:
            idx, masks, tgt = self._pseudo_bags(images, rng)
            wm = masks.astype(float)
            Xs.append(Xsrc[idx]); Is.append(Isrc[idx]); Ws.append(wm / wm.sum(axis=(1, 2), keepdims=True))
            ys.append(tgt); ss.append(np.full(len(idx), self.params["pseudo_weight"]))
        self.n_pseudo = sum(len(y) for y in ys[1:])
        p = self.params
        from . import _torch

        return _torch.train_map_net(self.net, np.concatenate(Xs), np.concatenate(Is), np.concatenate(Ws),
                                    np.concatenate(ys), np.concatenate(ss), vt, epochs=p["epochs"], lr=p["lr"],
                                    batch_size=p["batch_size"], patience=p["patience"], weight_decay=p["weight_decay"],
                                    extra_loss=None, augment=p["augment"], seed=p["seed"])


class _UnlabeledBatch:
    """Minimal Batch stand-in for unlabelled images (features.pixel_features only needs .images)."""

    def __init__(self, images: dict):
        self.images = images
        self.has_images = True

    def __len__(self) -> int:
        return len(self.images["ImSW"])


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M1-gbm", "M3", "M6f"])
