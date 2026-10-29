"""Data schema: MODIS band tables, MODIS-only input sets, and the CERES leakage guard.

Band order follows the advisor's script (train_Allsky_moments_narrow_values.py).
To be confirmed against the real files in stage E0 of PLAN.md.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

SW_LABELS: tuple[str, ...] = (
    "0.47", "0.55", "0.66", "0.86", "0.905", "0.935", "0.940", "1.24", "1.64", "2.13", "1.375",
)
LW_LABELS: tuple[str, ...] = (
    "3.75", "3.96a", "3.96", "4.05", "4.46", "4.5", "6.7", "7.3", "8.5", "9.7",
    "11.0", "12.0", "13.33", "13.63", "13.93", "14.23",
)
SW_WAVELENGTHS: tuple[float, ...] = tuple(float(x) for x in SW_LABELS)
LW_WAVELENGTHS: tuple[float, ...] = tuple(float(x.rstrip("a")) for x in LW_LABELS)
VIEW_CHANNELS: tuple[str, ...] = ("SZA", "VZA", "RAA")

# Channels that look like stripes in the sample image (Aqua band 6 dead detectors, 14.23 um).
BAD_CHANNELS: tuple[str, ...] = ("1.64", "14.23")

# Quantities derived from CERES: allowed as labels, never as model inputs.
CERES_DERIVED_KEYS: frozenset[str] = frozenset(
    {"Fsw", "Flw", "Isw", "Ilw", "mSumIsw", "stdSumIsw", "mSumIlw", "stdSumIlw"}
)
# MODIS retrieval products / metadata: allowed for filtering and stratification, not as inputs.
STRATIFICATION_KEYS: frozenset[str] = frozenset(
    {"CF", "Clrf", "meanCTH", "stdCTH", "LiqPhsf", "AOD", "Lat", "Lon", "glintf", "landf", "mDFNC"}
)


class LeakageError(ValueError):
    """Raised when a CERES-derived quantity is about to be used as a model input."""


def channel_index(label: str) -> tuple[str, int]:
    """Return the (band, index) of a MODIS channel label.

    Example:
        >>> channel_index("0.66")
        ('SW', 2)

    Args:
        label: Channel label as in SW_LABELS / LW_LABELS (e.g. "11.0", "3.96a").

    Returns:
        ("SW" or "LW", index into ImSW / ImLW and the moments arrays).

    Raises:
        KeyError: If the label is not a known channel.
    """
    if label in SW_LABELS:
        return "SW", SW_LABELS.index(label)
    if label in LW_LABELS:
        return "LW", LW_LABELS.index(label)
    raise KeyError(f"unknown MODIS channel label {label!r}")


def assert_no_ceres_inputs(feature_names: Iterable[str]) -> None:
    """Fail loudly if any feature name refers to a CERES-derived quantity.

    A name matches if its last path component ("scalars/Isw" -> "Isw") is a CERES key.

    Example:
        >>> assert_no_ceres_inputs(["SZA", "glint"])   # passes silently

    Args:
        feature_names: Names of the columns/channels a model will receive.

    Raises:
        LeakageError: If any name is a CERES-derived key.
    """
    bad = [n for n in feature_names if str(n).split("/")[-1] in CERES_DERIVED_KEYS]
    if bad:
        raise LeakageError(f"CERES-derived quantities used as inputs: {bad}")


@dataclass(frozen=True)
class InputSet:
    """A nested MODIS-only input set for the input ablation (PLAN.md section 3)."""

    name: str
    sw: tuple[str, ...]
    lw: tuple[str, ...]
    use_glint: bool


_VIS = ("0.47", "0.55", "0.66", "0.86")
_SWIR = ("1.24", "1.375", "2.13")
_WV = ("0.905", "0.935", "0.940")
_THERMAL = tuple(c for c in LW_LABELS if c not in BAD_CHANNELS)

_INPUT_SETS = {
    "F1": InputSet("F1", _VIS, (), False),
    "F2": InputSet("F2", _VIS + _SWIR, (), False),
    "F3": InputSet("F3", _VIS + _SWIR + _WV, (), False),
    "F4": InputSet("F4", _VIS + _SWIR + _WV, _THERMAL, False),
    "F5": InputSet("F5", _VIS + _SWIR + _WV, _THERMAL, True),
}


def input_set(name: str) -> InputSet:
    """Return the nested input set F1..F5 (geometry is always included by the caller).

    Example:
        >>> input_set("F1").sw
        ('0.47', '0.55', '0.66', '0.86')

    Raises:
        KeyError: If name is not one of F1..F5.
    """
    if name not in _INPUT_SETS:
        raise KeyError(f"unknown input set {name!r}; expected one of {sorted(_INPUT_SETS)}")
    return _INPUT_SETS[name]
