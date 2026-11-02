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

# MODIS (MYD021KM) band number of each channel, as read by the advisor's MOD_CERES_allsky:
# SW = EV_500_Aggr1km bands 3,4 + EV_250_Aggr1km bands 1,2 + EV_1KM_RefSB bands 17,18,19
#      + EV_500_Aggr1km bands 5,6,7 + band 26;  LW = EV_1KM_Emissive bands 20-25, 27-36.
# The ocean-colour bands 8-16 are not used. "0.935" is band 18 (0.936 um), "0.940" band 19.
# "3.96a" is band 21 (low-gain fire channel), "3.96" is band 22. Values are L1B radiances
# (radiance_scales/offsets applied), W m-2 sr-1 um-1.
SW_MODIS_BANDS: tuple[int, ...] = (3, 4, 1, 2, 17, 18, 19, 5, 6, 7, 26)
LW_MODIS_BANDS: tuple[int, ...] = (20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36)

# Storage conventions seen in the real files (PLAN.md section 2.1).
MOMENTS_EXCESS_KURTOSIS = True   # kurt*spec is Fisher (excess) kurtosis: the files contain values < 1
CF_TO_PERCENT = 100.0            # scalars CF is a fraction 0-1; the table stores it in percent
# Imviews RAA is |SolarAzimuth - SensorAzimuth| folded to [0, 180] (MYD03), so the specular
# (glint) direction is at 180. The scalars RAA is the CERES SSF relative azimuth, = 180 - Imviews
# RAA, with the specular direction at 0. features.pixel_geometry converts to the CERES convention,
# and geometry.glint_angle's default ("specular_at_0") then applies to both.
IMVIEWS_RAA_IS_SUPPLEMENT = True
# FOVorientation: 1 = the next CERES footprint lies at larger MODIS column (PSF used as stored),
# 0 = it lies at smaller column (PSF mirrored left-right, np.fliplr). It encodes the scan direction.

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
# Definitions from the advisor's MOD_CERES_allsky (see PLAN.md section 2.3):
#   CF       fraction of valid pixels in the patch flagged cloudy by MYD35 (unweighted, 0-1)
#   glintf   fraction of valid pixels with the MYD35 sun-glint flag set (unweighted, 0-1)
#   landf    patch mean of the MYD35 2-bit land/water code (0 = water ... 3), NOT a percentage;
#            landf == 0 means every pixel is water
#   LiqPhsf  intended: liquid fraction of the cloud. The current code divides a granule-wide count
#            by the footprint's cloudy-pixel count, so the stored values (~400) are not usable
#   mDFNC, stdDFNC  PSF-weighted mean/std over clear pixels of the distance (pixels) to the nearest
#            cloudy pixel; NaN where the distance exceeds the distance to the granule edge
#   meanCTH, stdCTH  PSF-weighted mean/std of MYD06 1 km cloud-top height over cloudy pixels (m)
#   AOD      CERES SSF PSF-weighted MOD04 ocean AOD at 0.55 um
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
