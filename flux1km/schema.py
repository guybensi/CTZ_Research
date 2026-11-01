"""Data schema: MODIS band tables, MODIS-only input sets, and the CERES leakage guard.

Band order and wavelengths match the real files (SWwavlngs / LWwavlngs in moments/, checked on
the sample granule data/sample/*/A2014060.0245.npz). LW has two 3.96 um channels; the first is
labelled "3.96a".
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

SW_LABELS: tuple[str, ...] = (
    "0.47", "0.55", "0.66", "0.85", "0.905", "0.935", "0.940", "1.24", "1.64", "2.13", "1.375",
)
LW_LABELS: tuple[str, ...] = (
    "3.75", "3.96a", "3.96", "4.05", "4.46", "4.5", "6.7", "7.32", "8.55", "9.7",
    "11.0", "12.0", "13.33", "13.63", "13.93", "14.23",
)
SW_WAVELENGTHS: tuple[float, ...] = tuple(float(x) for x in SW_LABELS)
LW_WAVELENGTHS: tuple[float, ...] = tuple(float(x.rstrip("a")) for x in LW_LABELS)
VIEW_CHANNELS: tuple[str, ...] = ("SZA", "VZA", "RAA")

# MODIS (MYD021KM) band number of each channel. The 27 channels are bands 1-7, 17-19, 26 (SW) and
# 20-25, 27-36 (LW); the ocean-colour bands 8-16 are not used. "3.96a" is band 21 (low-gain fire
# channel), "3.96" is band 22.
SW_MODIS_BANDS: tuple[int, ...] = (3, 4, 1, 2, 17, 19, 18, 5, 6, 7, 26)
LW_MODIS_BANDS: tuple[int, ...] = (20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36)

# Storage conventions seen in the real files (PLAN.md section 2.1).
MOMENTS_EXCESS_KURTOSIS = True   # kurt*spec is Fisher (excess) kurtosis: the files contain values < 1
CF_TO_PERCENT = 100.0            # scalars CF is a fraction 0-1; the table stores it in percent
IMVIEWS_RAA_IS_SUPPLEMENT = True  # Imviews RAA = 180 - scalars RAA (e.g. ~110 vs ~70 in the sample)
# FOVorientation (0/1) says whether the footprint ellipse inside the 27x55 patch is tilted to one
# side or the other (advisor). Which value is which tilt is not yet verified.

# Channels that look like stripes in the sample image (Aqua band 6 dead detectors, 14.23 um).
BAD_CHANNELS: tuple[str, ...] = ("1.64", "14.23")

# Quantities from CERES: allowed as labels (Fsw, Flw) or for stratification (Clrf), never as model
# inputs. Clrf is the CERES clear-area percentage from CERES's own cloud mask (advisor, Nov 2026).
CERES_DERIVED_KEYS: frozenset[str] = frozenset({"Fsw", "Flw", "Isw", "Ilw", "Clrf"})
# Sums of the MODIS narrowband radiances in the footprint, built to mimic the CERES broadband
# integral as a sanity check (advisor). MODIS-derived, so not leakage; a linear combination of
# channels the models already see.
MODIS_BROADBAND_SANITY_KEYS: frozenset[str] = frozenset({"mSumIsw", "stdSumIsw", "mSumIlw", "stdSumIlw"})
# MODIS-derived footprint products / metadata: filtering and stratification only, not inputs.
#   landf    percent land in the footprint, from coordinates
#   glintf   fraction of the footprint area in sun glint (MODIS)
#   LiqPhsf  liquid fraction of the cloud (advisor's sum over MODIS pixels; sample values ~400, units TBC)
#   mDFNC, stdDFNC  mean/std distance of the footprint's MODIS pixels to the nearest cloud pixel
STRATIFICATION_KEYS: frozenset[str] = frozenset(
    {"CF", "meanCTH", "stdCTH", "LiqPhsf", "AOD", "Lat", "Lon", "glintf", "landf", "mDFNC", "stdDFNC"}
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


_VIS = ("0.47", "0.55", "0.66", "0.85")
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
        ('0.47', '0.55', '0.66', '0.85')

    Raises:
        KeyError: If name is not one of F1..F5.
    """
    if name not in _INPUT_SETS:
        raise KeyError(f"unknown input set {name!r}; expected one of {sorted(_INPUT_SETS)}")
    return _INPUT_SETS[name]
