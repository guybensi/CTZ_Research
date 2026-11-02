"""M1: the improved Guy model on MODIS-only footprint features (PLAN.md section 3).

Footprint-level, so it gives no map. It is the upper bound on footprint skill and the teacher
for the hybrids M6d (soft distillation) and M6f (mediated pseudo-labels).

Features (guy_features), all from the MODIS moments and the geometry, never from CERES:
* mean and std of every channel of the input set;
* channel ratios: SW means / 0.47 um, 3-5 um means / 4.05 um, thermal means / 11 um
  (only for the reference channels present in the input set);
* trigonometry of the angles (cos SZA, cos VZA, sin RAA, cos RAA) and, for F5, the glint angle;
* a broadband reconstruction: the trapezoid integral of the SW channel means over wavelength,
  and that integral as an albedo proxy pi * L_BB / (S0 cos SZA).

Two learners: ``M1-gbm`` (scikit-learn histogram gradient boosting) and ``M1-ann``
(PyTorch MLP with a Huber loss). Both predict the albedo Fsw / (S0 cos SZA) by default.

Run alone:
    python3 -m flux1km.methods.m1_guy --demo
"""
from __future__ import annotations

import numpy as np

from .. import geometry, schema
from ..features import Batch, footprint_geometry, footprint_moment_stack
from ..models import register
from . import _torch
from ._base import TabularFootprintModel

RATIO_REFERENCES = {"SW": "0.47", "MWIR": "4.05", "TIR": "11.0"}


def _group(label: str) -> str:
    wl = float(label.rstrip("a"))
    return "SW" if wl < 3.0 else ("MWIR" if wl < 5.0 else "TIR")


def broadband_reconstruction(sw_means: np.ndarray, sw_labels: list[str] | tuple[str, ...], sza) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoid integral of SW channel radiances over wavelength, and its albedo proxy.

    Example:
        >>> bb, alb = broadband_reconstruction(np.array([[1.0, 1.0]]), ["0.47", "0.66"], np.array([0.0]))
        >>> round(float(bb[0]), 3)
        0.19

    Args:
        sw_means: (N, C) SW radiances (W m^-2 sr^-1 um^-1).
        sw_labels: Their channel labels.
        sza: (N,) solar zenith angles.

    Returns:
        (L_BB (N,) in W m^-2 sr^-1, pi L_BB / (S0 cos SZA) (N,)).

    Raises:
        ValueError: With fewer than two channels.
    """
    if len(sw_labels) < 2:
        raise ValueError("need at least two SW channels to integrate")
    wl = np.array([float(c) for c in sw_labels])
    order = np.argsort(wl)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 has only trapz
    bb = trapz(np.asarray(sw_means, float)[:, order], wl[order], axis=1)
    ins = geometry.toa_insolation(sza)
    with np.errstate(divide="ignore", invalid="ignore"):
        return bb, np.where(ins > 0, np.pi * bb / ins, np.nan)


def guy_features(batch: Batch, input_set: str = "F5") -> tuple[np.ndarray, list[str]]:
    """M1 feature table (N, F) and names (see the module docstring).

    Example:
        >>> # X, names = guy_features(batch, "F5")

    Raises:
        ValueError: If the batch has no moments.
    """
    (m, s, _, _), ch = footprint_moment_stack(batch, input_set)
    ch = [c.split("_", 1)[1] for c in ch]  # "SW_0.47" -> "0.47"
    blocks, names = [m, s], [f"m_{c}" for c in ch] + [f"std_{c}" for c in ch]
    for grp, ref in RATIO_REFERENCES.items():
        if ref not in ch:
            continue
        r = m[:, ch.index(ref)]
        for i, c in enumerate(ch):
            if _group(c) == grp and c != ref:
                with np.errstate(divide="ignore", invalid="ignore"):
                    blocks.append((m[:, i] / r)[:, None])
                names.append(f"ratio_{c}/{ref}")
    iset = schema.input_set(input_set)
    geo, gnames = footprint_geometry(batch, iset.use_glint)
    blocks.append(geo)
    names += gnames
    sw_ch = [c for c in ch if _group(c) == "SW"]
    bb, alb = broadband_reconstruction(m[:, [ch.index(c) for c in sw_ch]], sw_ch, batch.scalars["SZA"])
    blocks += [bb[:, None], alb[:, None]]
    names += ["L_BB", "albedo_BB"]
    schema.assert_no_ceres_inputs(names)
    X = np.concatenate(blocks, axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


@register
class GuyGBM(TabularFootprintModel):
    """M1-gbm: histogram gradient boosting on guy_features (scikit-learn)."""

    name = "M1-gbm"
    complexity = 3
    defaults = {**TabularFootprintModel.defaults, "max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31,
                "l2_regularization": 0.0, "loss": "squared_error", "min_samples_leaf": 5}

    @classmethod
    def param_grid(cls):
        return [{"learning_rate": lr, "max_leaf_nodes": nl, "max_iter": it}
                for lr in (0.05, 0.1) for nl in (15, 31) for it in (200, 400)]

    def features_of(self, batch):
        return guy_features(batch, self.params["input_set"])

    def fit_xy(self, X, y, w, Xv, yv):
        from sklearn.ensemble import HistGradientBoostingRegressor

        p = self.params
        ok = np.isfinite(y) & (w > 0)
        if not ok.any():
            raise ValueError("no valid training rows")
        reg = HistGradientBoostingRegressor(loss=p["loss"], max_iter=p["max_iter"], learning_rate=p["learning_rate"],
                                            max_leaf_nodes=p["max_leaf_nodes"], l2_regularization=p["l2_regularization"],
                                            min_samples_leaf=p["min_samples_leaf"], random_state=p["seed"])
        reg.fit(X[ok], y[ok], sample_weight=w[ok])
        self.estimator = reg
        return reg.predict


@register
class GuyANN(TabularFootprintModel):
    """M1-ann: dense network on guy_features with a Huber loss (PyTorch)."""

    name = "M1-ann"
    complexity = 3
    needs_torch = True
    defaults = {**TabularFootprintModel.defaults, "hidden": (128, 64, 32), "lr": 1e-3, "epochs": 300, "batch_size": 256,
                "patience": 15, "dropout": 0.0}

    @classmethod
    def param_grid(cls):
        return [{"hidden": h, "lr": lr} for h in ((64, 32), (128, 64, 32)) for lr in (1e-3, 3e-3)]

    def features_of(self, batch):
        return guy_features(batch, self.params["input_set"])

    def fit_xy(self, X, y, w, Xv, yv):
        p = self.params
        return _torch.train_tabular_net(X, y, w, None if Xv is None else (Xv, yv), hidden=tuple(p["hidden"]), loss="huber",
                                        epochs=p["epochs"], lr=p["lr"], batch_size=p["batch_size"], patience=p["patience"],
                                        seed=p["seed"], dropout=p["dropout"])


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M1-gbm", "M1-ann"])
