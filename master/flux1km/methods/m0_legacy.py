"""M0: the legacy footprint ANNs, re-run fairly inside the new protocol (PLAN.md sections 2.3, 3).

Two models, both footprint-level only (no map):

* ``M0-allsky37``: the 37-feature All_Sky_AIflux ANN (pre_master/python/All_Sky_AIflux.py):
  SZA, VZA, RAA + mean and skewness of 17 channels, channel means cross-normalised by the
  0.47 / 4.05 / 11 um means, dense ReLU layers N/2, N/4, N/8, 5, MSE loss.
* ``M0-advisor``: the advisor's current moments ANN (pre_master/advisor_code/
  train_Allsky_moments_narrow_values.py): SZA, VZA, RAA + mean/std/skew/kurt of 13 channels,
  cross-normalised the same way, log1p label with the weighted MAE (weight log1p|y| + 1), dense
  layers N/2, N/4, N/8, 5 with L1 (0.01 on the first, 0.001 on the others), early stopping on a
  change in validation RMSE below 0.001 for 3 epochs.

What changes against the original scripts, on purpose: the split is the frozen group split of
the manifest (not a random 70/30 row split), early stopping uses the validation fold (never the
test set) and the test set is NOT cut to Fsw > 400. The third M0 model, the Ridge baseline, is
``M0-ridge`` in flux1km.models.

Run alone:
    python3 -m flux1km.methods.m0_legacy --demo
    python3 -m flux1km.methods.m0_legacy --data-root <dir> --manifest <json>
"""
from __future__ import annotations

import numpy as np

from .. import schema
from ..features import Batch
from ..models import register
from . import _torch
from ._base import TabularFootprintModel

ALLSKY37_CHANNELS = ("0.47", "0.66", "0.85", "1.24", "1.375", "2.13", "3.96", "4.05", "4.5", "6.7", "7.32",
                     "8.55", "9.7", "11.0", "12.0", "13.33", "14.23")
ADVISOR_CHANNELS = ("0.47", "0.66", "0.85", "1.375", "2.13", "3.96", "4.05", "4.5", "6.7", "7.32", "9.7", "11.0", "12.0")
SOLAR_NORM, MIXED_NORM, THERMAL_NORM = "0.47", "4.05", "11.0"


def _wavelength(label: str) -> float:
    return float(label.rstrip("a"))


def norm_reference(label: str) -> str:
    """The channel a legacy feature is cross-normalised by: 0.47 (< 3 um), 4.05 (3-5 um) or 11 (thermal).

    Example:
        >>> norm_reference("0.66"), norm_reference("3.96"), norm_reference("12.0")
        ('0.47', '4.05', '11.0')
    """
    wl = _wavelength(label)
    return SOLAR_NORM if wl < 3.0 else (MIXED_NORM if wl < 5.0 else THERMAL_NORM)


def legacy_features(batch: Batch, channels: tuple[str, ...], stats: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    """Legacy footprint features: SZA, VZA, RAA + the chosen moments of the chosen channels.

    The mean and std of every channel except the three reference channels are divided by the mean
    of their reference channel (0.47, 4.05 or 11 um), as in the legacy scripts; skew and kurtosis
    are scale-free and kept as they are.

    Example:
        >>> # X, names = legacy_features(batch, ALLSKY37_CHANNELS, ("m", "skew")); X.shape[1] == 37

    Raises:
        ValueError: If the batch has no moments.
        KeyError: For an unknown channel.
    """
    if batch.moments is None:
        raise ValueError("legacy features need the moments files")
    arrays = {}
    for st in ("m", "std", "skew", "kurt"):
        arrays[st] = {}
        for band in ("SW", "LW"):
            key = f"{st}{band}spec"
            for i, lab in enumerate(schema.SW_LABELS if band == "SW" else schema.LW_LABELS):
                arrays[st][lab] = np.asarray(batch.moments[key][:, i], float)
    cols = [np.asarray(batch.scalars[k], float) for k in ("SZA", "VZA", "RAA")]
    names = ["SZA", "VZA", "RAA"]
    for st in stats:
        for c in channels:
            if c not in arrays["m"]:
                raise KeyError(f"unknown channel {c!r}")
            v = arrays[st][c]
            ref = norm_reference(c)
            if st in ("m", "std") and c != ref:
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = v / arrays["m"][ref]
            cols.append(v)
            names.append(f"{st}{c}")
    schema.assert_no_ceres_inputs(names)
    return np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0), names


def legacy_widths(n_features: int) -> tuple[int, ...]:
    """Hidden widths N/2, N/4, N/8, 5 of the legacy ANNs (at least 1 unit each).

    Example:
        >>> legacy_widths(55)
        (28, 14, 7, 5)
    """
    return (max(1, round(n_features / 2)), max(1, round(n_features / 4)), max(1, round(n_features / 8)), 5)


@register
class AllSky37(TabularFootprintModel):
    """M0-allsky37: the All_Sky_AIflux 37-feature ANN, footprint-level, MSE on flux."""

    name = "M0-allsky37"
    complexity = 3
    needs_torch = True
    defaults = {**TabularFootprintModel.defaults, "target": "flux", "lr": 1e-3, "epochs": 200, "batch_size": 5000,
                "patience": 10}

    @classmethod
    def param_grid(cls):
        return [{"lr": lr} for lr in (1e-3, 3e-3, 3e-4)]

    def features_of(self, batch):
        return legacy_features(batch, ALLSKY37_CHANNELS, ("m", "skew"))

    def fit_xy(self, X, y, w, Xv, yv):
        p = self.params
        return _torch.train_tabular_net(X, y, w, None if Xv is None else (Xv, yv), hidden=legacy_widths(X.shape[1]),
                                        loss="mse", epochs=p["epochs"], lr=p["lr"], batch_size=p["batch_size"],
                                        patience=p["patience"], seed=p["seed"])


@register
class AdvisorMomentsANN(TabularFootprintModel):
    """M0-advisor: the advisor's moments ANN (log1p label, weighted MAE, L1, delta-RMSE early stop)."""

    name = "M0-advisor"
    complexity = 3
    needs_torch = True
    defaults = {**TabularFootprintModel.defaults, "target": "flux", "lr": 1e-3, "epochs": 1000, "batch_size": 10000,
                "patience": 3, "min_delta": 0.001}

    @classmethod
    def param_grid(cls):
        return [{"lr": lr} for lr in (1e-3, 3e-3, 3e-4)]

    def features_of(self, batch):
        return legacy_features(batch, ADVISOR_CHANNELS, ("m", "std", "skew", "kurt"))

    def fit_xy(self, X, y, w, Xv, yv):
        p = self.params
        to = np.log1p(np.clip(y, 0, None))
        val = None if Xv is None else (Xv, np.log1p(np.clip(yv, 0, None)))
        pred = _torch.train_tabular_net(X, to, w, val, hidden=legacy_widths(X.shape[1]), loss="wmae_log1p",
                                        l1=(0.01, 0.001, 0.001, 0.001), epochs=p["epochs"], lr=p["lr"],
                                        batch_size=p["batch_size"], patience=p["patience"], min_delta=p["min_delta"],
                                        seed=p["seed"])
        return lambda Z: np.expm1(pred(Z))


if __name__ == "__main__":
    from ..run_single import main_for

    main_for(["M0-ridge", "M0-allsky37", "M0-advisor"])
