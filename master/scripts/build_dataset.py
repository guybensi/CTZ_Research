#!/usr/bin/env python3
"""Build the training data (scalars/, moments/, images/) from the downloaded MODIS and CERES files.

This is the corrected, resumable version of the advisor's converter (pre_master/advisor_code/
Generate_AllSky_2.py + MOD_CERES_allsky in Twilight_module.py, which stay untouched as the reference).
Same physics and the same output keys, dtypes and shapes as his files, so everything that reads his data
(flux1km, All_Sky_AIflux.py, ...) reads these. What was fixed is listed in the "Fixes" section of
HOW_TO_BUILD_DATASET.md next to this file.

Input (the layout written by download_modis.py and download_ceres.py):
    <data>/AQUA/<year>/<region>/<month>/MOD02/ MOD03/ MOD06/ MOD35/   and one CERES*.nc per month folder
Output:
    <out>/<region>/<year>/<month>/scalars/<granule>.npz     one value per CERES footprint (23 keys)
                                 /moments/<granule>.npz     PSF-weighted mean/std/skew/kurt of 11 SW + 16 LW channels
                                 /images/<granule>.npz      27x55 patches (only without --no-images)
                                 /granules.csv              what happened to every granule
    <out>/<region>/{scalars,moments,images}/                 after --gather: symlinks to every file of the region,
                                                              the three folders the models read (--data-root <out>/<region>)

Usage:
    python3 build_dataset.py --data /archive/GUY --out /archive/GUY/dataset --psf ceres_psf.csv \\
        --start 2008-01-01 --end 2008-12-31 --workers 8
    python3 build_dataset.py --data /archive/GUY --out /archive/GUY/dataset --dry-run
    python3 build_dataset.py --out /archive/GUY/dataset --gather      # the three folders, per region
    python3 build_dataset.py --out /archive/GUY/dataset --verify      # check every triplet

Re-running is safe: granules that are done are skipped, failed ones are tried again.
Needs: numpy, scipy, pandas (for the PSF csv) and netCDF4 built with HDF4 support (conda-forge); the flux1km
package is found next to this folder.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))         # download_modis
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # flux1km, so no pip install is needed
import download_modis as dm  # noqa: E402  (same folder: month names and granule-name parsing)

# ------------------------------- constants (as in the advisor's code) -------------------------------
PATCH = (27, 55)                       # footprint patch (rows x columns), the CERES PSF resampled to 1 km
SW_WAVELENGTHS = np.array([0.47, 0.55, 0.66, 0.85, 0.905, 0.935, 0.940, 1.24, 1.64, 2.13, 1.375])
LW_WAVELENGTHS = np.array([3.75, 3.96, 3.96, 4.05, 4.46, 4.5, 6.7, 7.32, 8.55, 9.7, 11.0, 12.0, 13.33, 13.63,
                           13.93, 14.23])
NAN_PSF_WEIGHT = 0.001                 # a footprint is used if invalid pixels carry less than 0.1% of the PSF
MIN_FSW = 1.0                          # W m-2; smaller = night, the footprint is skipped
MAX_FOV_OFFSET_KM = 10.0               # a footprint farther than this from every pixel is outside the granule
# Channels that must be valid under the PSF for a footprint to be kept: the 13 channels of the advisor's current model
# (train_Allsky_moments_narrow_values.py). Any other channel with invalid pixels (dead detectors, saturation) gets NaN
# moments for that footprint instead of rejecting it. SW index: 0.47=0, 0.66=2, 0.85=3, 2.13=9, 1.375=10;
# LW index: 3.96(B22)=2, 4.05=3, 4.5=5, 6.7=6, 7.3=7, 9.7=9, 11=10, 12=11.
REQUIRED_SW = (0, 2, 3, 9, 10)
REQUIRED_LW = (2, 3, 5, 6, 7, 9, 10, 11)
PSF_COLUMN_OFFSET = 0                  # the advisor's code is equivalent to 1 (see HOW_TO_BUILD_DATASET.md, fix 8)
EARTH_RADIUS_KM = 6371.0
CLOUD_MASK_TYPE = "relaxed"            # "relaxed": cloudy = the most confident cloudy class only

DEFAULT_PSF = Path(__file__).resolve().parents[1] / "resources" / "ceres_psf.csv"   # the advisor's CERES PSF

SCALAR_KEYS = ["Fsw", "Isw", "Flw", "Ilw", "mSumIsw", "stdSumIsw", "mSumIlw", "stdSumIlw", "mDFNC", "stdDFNC", "CF",
               "meanCTH", "stdCTH", "LiqPhsf", "VZA", "SZA", "RAA", "Lat", "Lon", "AOD", "Clrf", "glintf", "landf"]
MOMENT_KEYS = ["mSWspec", "mLWspec", "stdSWspec", "stdLWspec", "skewSWspec", "skewLWspec", "kurtSWspec",
               "kurtLWspec", "SWwavlngs", "LWwavlngs"]
IMAGE_KEYS = ["Imviews", "ImSW", "ImLW", "FOVorientation"]
PARTS = ("scalars", "moments", "images")


# ===================================== time and CERES =====================================
_UNIT_SECONDS = {"day": 86400.0, "days": 86400.0, "hour": 3600.0, "hours": 3600.0, "minute": 60.0,
                 "minutes": 60.0, "second": 1.0, "seconds": 1.0}


def hour_keys(values, units: str) -> np.ndarray:
    """Integer key year*100000 + day_of_year*100 + hour (UTC) of CERES times; -1 where the time is invalid.

    Granules get the same key from their file name, so matching a footprint to a granule is an integer
    comparison (the advisor's code compared floats built from strings).

    Example:
        >>> hour_keys([0.0, 1.5, 31.0], "days since 2008-01-01 00:00:00").tolist()
        [200800100, 200800212, 200803200]
        >>> hour_keys([float("nan")], "days since 2008-01-01").tolist()
        [-1]

    Raises:
        ValueError: if `units` is not "<days|hours|minutes|seconds> since <date[ time]>".
    """
    m = re.match(r"^\s*(\w+)\s+since\s+(\d{4}-\d{1,2}-\d{1,2})(?:[ T](\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?))?",
                 units or "")
    if not m or m[1].lower() not in _UNIT_SECONDS:
        raise ValueError(f"cannot read time units {units!r}; expected e.g. 'days since 2000-01-01 12:00:00'")
    y, mo, d = (int(x) for x in m[2].split("-"))
    hms = (m[3] or "0:00:00").split(".")[0].split(":")
    hms = [int(x) for x in hms] + [0] * (3 - len(hms))
    base = np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}T{hms[0]:02d}:{hms[1]:02d}:{hms[2]:02d}")
    v = np.asarray(values, dtype="float64")
    ok = np.isfinite(v)
    secs = np.zeros(v.shape, dtype="int64")
    secs[ok] = np.round(v[ok] * _UNIT_SECONDS[m[1].lower()]).astype("int64")
    t = base + secs.astype("timedelta64[s]")
    years = t.astype("datetime64[Y]")
    days = t.astype("datetime64[D]")
    doy = (days - years.astype("datetime64[D]")).astype("int64") + 1
    hour = (t - days).astype("timedelta64[h]").astype("int64")
    key = (years.astype("int64") + 1970) * 100000 + doy * 100 + hour
    return np.where(ok, key, -1)


def granule_hour_key(name: str) -> int:
    """Hour key of a MODIS file name, same encoding as `hour_keys`.

    Example:
        >>> granule_hour_key("MYD021KM.A2008032.0305.061.2008032123456.hdf")
        200803203

    Raises:
        ValueError: if the name is not a MODIS granule file name.
    """
    _, key, year, doy = dm.parse_granule_name(name)
    return int(year) * 100000 + int(doy) * 100 + int(key.split(".")[1][:2])


CERES_FIELDS = {          # name in the output dict -> variable in the SSF/Order-Tool file
    "colat": "Colatitude_of_CERES_FOV_at_surface", "lon": "Longitude_of_CERES_FOV_at_surface",
    "sza": "CERES_solar_zenith_at_surface", "vza": "CERES_viewing_zenith_at_surface",
    "raa": "CERES_relative_azimuth_at_surface", "fsw": "CERES_SW_TOA_flux___upwards",
    "flw": "CERES_LW_TOA_flux___upwards", "isw": "CERES_SW_radiance___upwards",
    "ilw": "CERES_LW_radiance___upwards", "clrf": "Cloud_mask_clear_strong_percent_coverage",
    "aod": "PSF_wtd_MOD04_effective_optical_depth_average_ocean__0_550_",
}


def _read_float(var) -> np.ndarray:
    """A netCDF variable as float64 with fill values (masked, or >= 3e38) as NaN."""
    a = np.ma.filled(var[:].astype("float64"), np.nan)
    return np.where(np.abs(a) >= 3.0e38, np.nan, a)


def load_ceres(path) -> dict[str, np.ndarray]:
    """Read one month of CERES SSF footprints: dict of float64 arrays plus the integer `key`, time-ordered.

    Keys: key, lat, lon (-180..180), sza, vza, raa, fsw, isw, flw, ilw, aod, clrf. Footprints keep their
    original order (sorted by time, stable), so footprint i+1 is still the next one along the scan.

    Raises:
        ValueError: if a required variable is missing (the message lists them).
    """
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        missing = [v for v in [*CERES_FIELDS.values(), "time"] if v not in ds.variables]
        if missing:
            raise ValueError(f"{Path(path).name}: missing variables {missing}")
        t = _read_float(ds.variables["time"])
        key = hour_keys(t, ds.variables["time"].getncattr("units"))
        arrays = {k: _read_float(ds.variables[v]) for k, v in CERES_FIELDS.items()}
    order = np.argsort(np.where(np.isfinite(t), t, np.inf), kind="stable")
    out = {k: v[order] for k, v in arrays.items()}
    out["key"] = key[order]
    out["lat"] = 90.0 - out.pop("colat")
    out["lon"] = np.where(out["lon"] > 180, out["lon"] - 360, out["lon"])
    return out


# ===================================== MODIS readers =====================================
def read_radiance(path: str, var: str) -> np.ndarray:
    """MYD021KM scaled radiance (bands, rows, cols) float32, W m-2 sr-1 um-1; invalid and saturated -> NaN.

    Values above 32767 are not radiances (65535 fill, 65533/65534 saturated or unusable): the advisor's code
    removed only 65535.
    """
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        v = ds.variables[var]
        raw = np.ma.filled(v[:].astype("float32"), np.nan)
        raw[raw > 32767] = np.nan
        scales = np.asarray(v.getncattr("radiance_scales"), dtype="float32")
        offsets = np.asarray(v.getncattr("radiance_offsets"), dtype="float32")
    return (raw - offsets[:, None, None]) * scales[:, None, None]


def read_plain(path: str, var: str) -> np.ndarray:
    """A 2-D MODIS science array as float32; fill values (and values outside valid range) -> NaN."""
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        return np.ma.filled(ds.variables[var][:].astype("float32"), np.nan)


def read_cloud_mask_byte0(path: str) -> np.ndarray:
    """First byte of the MYD35 Cloud_Mask as int64 0..255, read raw (auto masking would hide the 0 bytes)."""
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        return np.asarray(ds.variables["Cloud_Mask"][0]).astype("int64") & 0xFF


def decode_cloud_mask(byte0, kind: str = CLOUD_MASK_TYPE) -> dict[str, np.ndarray]:
    """Decode MYD35 byte 0 into boolean `cloud`, boolean `glint` and the 2-bit `land` code (0 water .. 3 land).

    Bits: 0 = mask determined; 1-2 = cloud confidence (0 cloudy, 1 uncertain, 2 probably clear, 3 clear);
    4 = 0 when sun glint is present; 6-7 = 0 water, 1 coastal, 2 desert, 3 land.
    Undetermined pixels count as cloudy, have no glint and are water (as in the advisor's code).
    "relaxed": cloudy = confidence 0; "strict": cloudy = confidence 0 or 1.

    Example:
        >>> b = np.array([0b00000001, 0b00110011, 0b11010111])   # cloudy+glint; uncertain; clear over land
        >>> d = decode_cloud_mask(b)
        >>> d["cloud"].tolist(), d["glint"].tolist(), d["land"].tolist()
        ([True, False, False], [True, False, False], [0, 0, 3])
        >>> decode_cloud_mask(np.array([0b00000011]), "strict")["cloud"].tolist()    # confidence 1
        [True]

    Raises:
        ValueError: if `kind` is not "relaxed" or "strict".
    """
    if kind not in ("relaxed", "strict"):
        raise ValueError(f"cloud mask kind must be 'relaxed' or 'strict', not {kind!r}")
    b = np.asarray(byte0).astype("int64") & 0xFF
    determined = (b & 1) == 1
    conf = np.where(determined, (b >> 1) & 3, 0)
    cloud = conf == 0 if kind == "relaxed" else conf <= 1
    glint = determined & (((b >> 4) & 1) == 0)
    land = np.where(determined, (b >> 6) & 3, 0).astype("int8")
    return {"cloud": cloud, "glint": glint, "land": land}


def relative_azimuth(solar_az, sensor_az) -> np.ndarray:
    """Folded |solar azimuth - sensor azimuth| in [0, 180] degrees (specular = 180), as in the advisor's images.

    The advisor's expression gives values above 180 when the difference is below -180 degrees.

    Example:
        >>> relative_azimuth(np.array([10.0, -170.0, 170.0]), np.array([20.0, 170.0, -170.0])).tolist()
        [10.0, 20.0, 20.0]
    """
    d = np.abs(np.asarray(solar_az, dtype="float64") - np.asarray(sensor_az, dtype="float64")) % 360.0
    return np.where(d > 180.0, 360.0 - d, d).astype("float32")


def distance_to_cloud(cloud: np.ndarray) -> np.ndarray:
    """Distance (pixels) from each clear pixel to the nearest cloudy pixel; 0 in cloud, NaN beyond the edge distance.

    NaN everywhere when the granule has no cloud (the distance is undefined).

    Example:
        >>> c = np.zeros((7, 7), bool); c[3, 0] = True
        >>> distance_to_cloud(c)[3, :3].tolist()
        [0.0, 1.0, 2.0]
    """
    from scipy import ndimage

    if not cloud.any():
        return np.full(cloud.shape, np.nan, dtype="float32")
    dist = ndimage.distance_transform_edt(~cloud).astype("float32")
    h, w = cloud.shape
    ex = np.minimum(np.arange(1, w + 1), np.arange(w, 0, -1))
    ey = np.minimum(np.arange(1, h + 1), np.arange(h, 0, -1))
    edge = np.minimum(ey[:, None], ex[None, :])
    dist[dist > edge] = np.nan
    return dist


# ===================================== footprints =====================================
def _unit_vectors(lat, lon) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def weighted_moments(x: np.ndarray, w: np.ndarray):
    """PSF-weighted mean, std, skewness and excess kurtosis of each channel; x (C, H, W), w (H, W) summing to 1.

    Same formulas as the advisor's code (population std; NaN pixels ignored).

    Example:
        >>> x = np.array([[[1.0, 3.0]]]); w = np.array([[0.5, 0.5]])
        >>> [float(v[0]) for v in weighted_moments(x, w)]
        [2.0, 1.0, 0.0, -2.0]
    """
    x = x.astype("float64")
    mean = np.nansum(x * w, axis=(1, 2))
    d = x - mean[:, None, None]
    std = np.sqrt(np.nansum(w * d ** 2, axis=(1, 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = d / std[:, None, None]
        skew = np.nansum(w * z ** 3, axis=(1, 2))
        kurt = np.nansum(w * z ** 4, axis=(1, 2)) - 3.0
    return mean, std, skew, kurt


def _mean_std_valid(values: np.ndarray, w: np.ndarray, exclude: np.ndarray) -> tuple[float, float]:
    """Weighted mean and std over the pixels that are finite and not excluded; (0, 0) if there are none.

    The advisor's code also returns 0 when no pixel qualifies (clear footprint: no cloud top), and the legacy
    training scripts rely on that, so it is kept.
    """
    ok = np.isfinite(values) & ~exclude
    ws = w[ok].sum()
    if ws == 0:
        return 0.0, 0.0
    mean = float(np.sum(values[ok] * w[ok]) / ws)
    return mean, float(np.sqrt(np.sum(w[ok] / ws * (values[ok] - mean) ** 2)))


def extract_footprints(solar, thermal, lat, lon, szen, vzen, raa, cloud, glint, land, cth, phase,
                       ceres: dict, hour_key: int, psf: np.ndarray, gen_images: bool = True,
                       psf_column_offset: int = PSF_COLUMN_OFFSET, required_sw=REQUIRED_SW, required_lw=REQUIRED_LW,
                       stats: dict | None = None):
    """Cut the CERES footprints of one granule out of the MODIS arrays.

    Args:
        solar, thermal: (11, H, W) and (16, H, W) radiances (NaN = invalid).
        lat, lon, szen, vzen, raa: (H, W) geolocation and geometry (degrees).
        cloud, glint: bool (H, W); land: (H, W) 2-bit code; cth, phase: (H, W) or None (then CTH/LiqPhsf are NaN).
        ceres: the load_ceres arrays of this hour plus ONE footprint after it (orientation needs footprint i+1).
        hour_key: key of the granule's hour; only footprints with that key are used.
        psf: (27, 55) weights for FOVorientation == 1; they are mirrored for 0.
        required_sw, required_lw: channel indices that must be valid (invalid PSF weight <= 0.1%) for a footprint to
            be kept. Every other channel with more invalid weight gets NaN moments (and its pixels stay NaN in the
            images, and out of mSum*). Pass range(11) and range(16) to require every channel.
        stats: optional dict that receives the counts candidates / outside / edge / invalid / accepted.

    Returns:
        None if the granule yields no footprint, else {"scalars": dict, "moments": dict, "images": dict | None}
        with the advisor's keys; footprints are in CERES order.
    """
    from scipy.spatial import cKDTree

    h, w = lat.shape
    ph, pw = psf.shape
    col_sums = psf.sum(axis=0)
    c1 = int(np.nonzero(col_sums == col_sums.max())[0][0])            # PSF peak column
    c2 = pw - c1 - 1                                                   # the same for the mirrored PSF
    half = (ph - 1) // 2

    n_ce = len(ceres["key"])
    cand = np.nonzero(ceres["key"] == hour_key)[0]
    if cand.size == 0:
        return None
    fsw, flw = ceres["fsw"], ceres["flw"]
    keep = ~((np.isnan(flw) & np.isnan(fsw)) | (fsw < MIN_FSW))       # not NaN/night
    cand = cand[keep[cand] & (cand + 1 < n_ce)]                       # footprint i+1 must exist
    cand = cand[np.isfinite(ceres["lat"][cand]) & np.isfinite(ceres["lon"][cand])
                & np.isfinite(ceres["lat"][cand + 1]) & np.isfinite(ceres["lon"][cand + 1])]
    if cand.size == 0:
        return None

    valid_px = np.isfinite(lat) & np.isfinite(lon)
    if not valid_px.any():
        return None
    flat_idx = np.flatnonzero(valid_px.ravel())
    tree = cKDTree(_unit_vectors(lat.ravel()[flat_idx], lon.ravel()[flat_idx]))
    d0, j0 = tree.query(_unit_vectors(ceres["lat"][cand], ceres["lon"][cand]))
    _, j1 = tree.query(_unit_vectors(ceres["lat"][cand + 1], ceres["lon"][cand + 1]))
    row0, col0 = np.divmod(flat_idx[j0], w)
    _, col1 = np.divmod(flat_idx[j1], w)
    dist_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(d0 / 2, 0, 1))

    sol_ok, th_ok = np.isfinite(solar), np.isfinite(thermal)
    req_sw, req_lw = list(required_sw), list(required_lw)
    unwanted = (np.isnan(szen) | np.isnan(vzen) | np.isnan(raa)
                | ~sol_ok[req_sw].all(axis=0) | ~th_ok[req_lw].all(axis=0))
    count = stats if stats is not None else {}
    for k in ("candidates", "outside", "edge", "invalid", "accepted"):
        count.setdefault(k, 0)
    count["candidates"] += len(cand)
    no_cth = cth is None
    dfnc = distance_to_cloud(cloud)
    rows = {k: [] for k in SCALAR_KEYS}
    mom = {k: [] for k in MOMENT_KEYS[:8]}
    ims = {"Imviews": [], "ImSW": [], "ImLW": [], "FOVorientation": []}

    for n, fov in enumerate(cand):
        if dist_km[n] > MAX_FOV_OFFSET_KM:
            count["outside"] += 1
            continue
        cy, cx = int(row0[n]), int(col0[n])
        normal = col1[n] >= cx                                         # next footprint at a larger column
        pc = c1 if normal else c2
        fmask = psf if normal else psf[:, ::-1]
        r0, r1 = cy - half, cy + half
        k0, k1 = cx - pc + psf_column_offset, cx + (pw - 1 - pc) + psf_column_offset   # PSF peak on the centre
        if r0 < 0 or r1 > h - 1 or k0 < 0 or k1 > w - 1:               # patch leaves the granule
            count["edge"] += 1
            continue
        sl = (slice(r0, r1 + 1), slice(k0, k1 + 1))
        unwant = unwanted[sl]
        if fmask[unwant].sum() > NAN_PSF_WEIGHT:                       # too much of the PSF is invalid
            count["invalid"] += 1
            continue
        count["accepted"] += 1
        solarfov, thermalfov = solar[(slice(None),) + sl], thermal[(slice(None),) + sl]
        good_sw = (fmask * ~sol_ok[(slice(None),) + sl]).sum(axis=(1, 2)) <= NAN_PSF_WEIGHT       # channels usable here
        good_lw = (fmask * ~th_ok[(slice(None),) + sl]).sum(axis=(1, 2)) <= NAN_PSF_WEIGHT
        n_valid = max(int((~unwant).sum()), 1)
        cloud_w, phase_w = cloud[sl], (None if phase is None else phase[sl])

        r = {k: np.nan for k in SCALAR_KEYS}
        r["mDFNC"], r["stdDFNC"] = _mean_std_valid(dfnc[sl], fmask, dfnc[sl] < 0.1)
        if no_cth:
            r["meanCTH"] = r["stdCTH"] = np.nan
        else:
            r["meanCTH"], r["stdCTH"] = _mean_std_valid(cth[sl], fmask, cth[sl] < 0.1)
        r["SZA"], r["RAA"], r["VZA"] = ceres["sza"][fov], ceres["raa"][fov], ceres["vza"][fov]
        r["Lon"], r["Lat"] = ceres["lon"][fov], ceres["lat"][fov]
        r["AOD"], r["Clrf"] = ceres["aod"][fov], ceres["clrf"][fov]
        r["CF"] = float(cloud_w.sum()) / n_valid
        r["glintf"] = float(glint[sl].sum()) / n_valid
        r["landf"] = float(land[sl].sum()) / n_valid
        if phase_w is not None:
            n_cloud = int(cloud_w.sum())
            r["LiqPhsf"] = float((cloud_w & (phase_w == 1)).sum()) / n_cloud if n_cloud else np.nan
        if not np.isnan(fsw[fov]):
            r["Fsw"], r["Isw"] = fsw[fov], ceres["isw"][fov]
            s = solarfov[good_sw].sum(axis=0)
            r["mSumIsw"] = float(np.nansum(s * fmask))
            r["stdSumIsw"] = float(np.sqrt(np.nansum(fmask * (s - r["mSumIsw"]) ** 2)))
        if not np.isnan(flw[fov]):
            r["Flw"], r["Ilw"] = flw[fov], ceres["ilw"][fov]
            s = thermalfov[good_lw].sum(axis=0)
            r["mSumIlw"] = float(np.nansum(s * fmask))
            r["stdSumIlw"] = float(np.sqrt(np.nansum(fmask * (s - r["mSumIlw"]) ** 2)))
        for k in SCALAR_KEYS:
            rows[k].append(r[k])
        for tag, arr, good in (("SW", solarfov, good_sw), ("LW", thermalfov, good_lw)):
            m, s, sk, ku = (np.where(good, v, np.nan) for v in weighted_moments(arr, fmask))
            mom[f"m{tag}spec"].append(m)
            mom[f"std{tag}spec"].append(s)
            mom[f"skew{tag}spec"].append(sk)
            mom[f"kurt{tag}spec"].append(ku)
        if gen_images:
            ims["Imviews"].append(np.stack([szen[sl], vzen[sl], raa[sl]]))
            ims["ImSW"].append(solarfov)
            ims["ImLW"].append(thermalfov)
            ims["FOVorientation"].append(float(normal))

    if not rows["Fsw"]:
        return None
    scalars = {k: np.asarray(v, dtype="float32") for k, v in rows.items()}
    moments = {k: np.asarray(v, dtype="float32") for k, v in mom.items()}
    moments["SWwavlngs"], moments["LWwavlngs"] = SW_WAVELENGTHS.copy(), LW_WAVELENGTHS.copy()
    images = {k: np.asarray(v, dtype="float32") for k, v in ims.items()} if gen_images else None
    return {"scalars": scalars, "moments": moments, "images": images}


# ===================================== one granule =====================================
def _atomic_savez(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def granule_files_done(month_out: Path, key: str, gen_images: bool) -> bool:
    """True if scalars (written last), moments and, when wanted, images of the granule exist."""
    parts = ("scalars", "moments") + (("images",) if gen_images else ())
    return all((month_out / p / f"{key}.npz").is_file() for p in parts)


def read_granule(d: dict) -> dict:
    """Read every array of one granule from its files (d: paths of MOD02, MOD03, MOD35 or None, MOD06 or None)."""
    r500 = read_radiance(d["MOD02"], "EV_500_Aggr1km_RefSB")
    r250 = read_radiance(d["MOD02"], "EV_250_Aggr1km_RefSB")
    r1km = read_radiance(d["MOD02"], "EV_1KM_RefSB")
    solar = np.concatenate([r500[0:2], r250, r1km[-4:-1], r500[-3:], r1km[-1:]], axis=0)
    thermal = read_radiance(d["MOD02"], "EV_1KM_Emissive")
    del r500, r250, r1km
    lat, lon = read_plain(d["MOD03"], "Latitude"), read_plain(d["MOD03"], "Longitude")
    szen, vzen = read_plain(d["MOD03"], "SolarZenith"), read_plain(d["MOD03"], "SensorZenith")
    raa = relative_azimuth(read_plain(d["MOD03"], "SolarAzimuth"), read_plain(d["MOD03"], "SensorAzimuth"))
    if d.get("MOD35"):
        mask = decode_cloud_mask(read_cloud_mask_byte0(d["MOD35"]), CLOUD_MASK_TYPE)
    else:                                                               # no cloud mask: flags unknown
        mask = {"cloud": np.zeros(lat.shape, bool), "glint": np.zeros(lat.shape, bool), "land": np.zeros(lat.shape, "int8")}
    cth = phase = None
    if d.get("MOD06"):
        phase = read_plain(d["MOD06"], "Cloud_Phase_Infrared_1km")
        cth = read_plain(d["MOD06"], "cloud_top_height_1km")
    return dict(solar=solar, thermal=thermal, lat=lat, lon=lon, szen=szen, vzen=vzen, raa=raa, cloud=mask["cloud"],
                glint=mask["glint"], land=mask["land"], cth=cth, phase=phase)


def _required(task: dict):
    if task.get("required", "core") == "all":
        return range(len(SW_WAVELENGTHS)), range(len(LW_WAVELENGTHS))
    return REQUIRED_SW, REQUIRED_LW


def _call_extract(m: dict, task: dict, stats: dict):
    req_sw, req_lw = _required(task)
    return extract_footprints(m["solar"], m["thermal"], m["lat"], m["lon"], m["szen"], m["vzen"], m["raa"], m["cloud"],
                              m["glint"], m["land"], m["cth"], m["phase"], task["ceres"], task["hour_key"], task["psf"],
                              task["gen_images"], required_sw=req_sw, required_lw=req_lw, stats=stats)


def process_granule(task: dict) -> dict:
    """Build the three NPZ files of one granule; never raises (errors come back in the result).

    Returns {"key", "status": ok|empty|error, "n": footprints, "sec", "msg"}. For "empty" the message says why
    (how many footprints were outside the granule, near its edge, or had too many invalid pixels).
    """
    t0 = time.time()
    key = task["key"]
    try:
        d = task["dir"]
        stats: dict = {}
        res = _call_extract(read_granule(d), task, stats)
        month_out = Path(task["month_out"])
        if res is None:
            why = " ".join(f"{k}={v}" for k, v in stats.items())
            return {"key": key, "status": "empty", "n": 0, "sec": time.time() - t0, "msg": why}
        if not d.get("MOD35"):                                          # unknown, not "no cloud / no glint / water"
            for k in ("CF", "glintf", "landf", "mDFNC", "stdDFNC"):
                res["scalars"][k][:] = np.nan
        if res["images"] is not None:
            _atomic_savez(month_out / "images" / f"{key}.npz", res["images"])
        _atomic_savez(month_out / "moments" / f"{key}.npz", res["moments"])
        _atomic_savez(month_out / "scalars" / f"{key}.npz", res["scalars"])
        return {"key": key, "status": "ok", "n": len(res["scalars"]["Fsw"]), "sec": time.time() - t0,
                "msg": "" if d.get("MOD35") and d.get("MOD06") else
                       "no " + "/".join(p for p in ("MOD35", "MOD06") if not d.get(p))}
    except Exception as e:  # noqa: BLE001 - one bad granule must not stop the month
        return {"key": key, "status": "error", "n": 0, "sec": time.time() - t0,
                "msg": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"}


def diagnose_granule(task: dict) -> dict:
    """Why does a granule give few or no footprints? Invalid-pixel share per channel and the rejection counts.

    Returns {"key", "invalid_sw": {wavelength: fraction}, "invalid_lw": {...}, "geometry": fraction, "stats": {...},
    "n": footprints kept} or {"key", "error"}.
    """
    try:
        m = read_granule(task["dir"])
        stats: dict = {}
        res = _call_extract(m, task, stats)
        inv_sw = (~np.isfinite(m["solar"])).mean(axis=(1, 2))
        inv_lw = (~np.isfinite(m["thermal"])).mean(axis=(1, 2))
        geo = float((np.isnan(m["szen"]) | np.isnan(m["vzen"]) | np.isnan(m["raa"])).mean())
        return {"key": task["key"], "invalid_sw": [float(x) for x in inv_sw], "invalid_lw": [float(x) for x in inv_lw],
                "geometry": geo, "stats": stats, "n": 0 if res is None else len(res["scalars"]["Fsw"]),
                "ceres_in_hour": int(len(task["ceres"]["key"]))}
    except Exception as e:  # noqa: BLE001
        return {"key": task["key"], "error": f"{type(e).__name__}: {e}"}


def format_diagnosis(d: dict) -> str:
    """One readable block for a diagnose_granule result."""
    if "error" in d:
        return f"{d['key']}: ERROR {d['error']}"
    bad_sw = ", ".join(f"{w:g}um={f:.0%}" for w, f in zip(SW_WAVELENGTHS, d["invalid_sw"]) if f > 0.001) or "none"
    bad_lw = ", ".join(f"{w:g}um={f:.0%}" for w, f in zip(LW_WAVELENGTHS, d["invalid_lw"]) if f > 0.001) or "none"
    st = d["stats"]
    return (f"{d['key']}: CERES footprints in hour+1={d['ceres_in_hour']}, candidates={st.get('candidates', 0)}, "
            f"outside granule={st.get('outside', 0)}, near edge={st.get('edge', 0)}, too many invalid pixels="
            f"{st.get('invalid', 0)}, accepted={st.get('accepted', 0)}, kept={d['n']}\n"
            f"    invalid pixels per SW channel: {bad_sw}\n    invalid pixels per LW channel: {bad_lw}\n"
            f"    invalid geometry: {d['geometry']:.1%}")


def spread(items: list, n: int) -> list:
    """n items evenly spread over the list (the first n of a day-ordered list would all be the same orbit).

    Example:
        >>> spread(list(range(10)), 3)
        [0, 4, 9]
        >>> spread([1, 2], 5)
        [1, 2]
    """
    if n >= len(items):
        return list(items)
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


# ===================================== months =====================================
def find_ceres_file(month_dir: Path) -> Path:
    """The single file with "CERES" in its name in a month folder.

    Raises:
        FileNotFoundError: none found. ValueError: more than one (the advisor's code needs exactly one).
    """
    found = sorted(p for p in Path(month_dir).glob("*CERES*") if p.is_file())
    if not found:
        raise FileNotFoundError(f"no CERES file in {month_dir} (run download_ceres.py first)")
    if len(found) > 1:
        raise ValueError(f"{len(found)} files with 'CERES' in {month_dir}: {[p.name for p in found]}")
    return found[0]


def list_granules(month_dir: Path, start: str, end: str) -> list[dict]:
    """MOD02 granules of a month folder inside [start, end] with the matching MOD03/MOD06/MOD35 file paths.

    Granules without a MOD03 are dropped; MOD06 / MOD35 are optional (their products become NaN).
    Sorted by granule key.
    """
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    by_prod: dict[str, dict[str, str]] = {}
    for sub in ("MOD02", "MOD03", "MOD06", "MOD35"):
        by_prod[sub] = {}
        for f in sorted((Path(month_dir) / sub).glob("*.hdf")):
            try:
                _, key, _, _ = dm.parse_granule_name(f.name)
            except ValueError:
                continue
            by_prod[sub][key] = str(f)
    out = []
    for key, f02 in sorted(by_prod["MOD02"].items()):
        _, _, year, doy = dm.parse_granule_name(f02)
        day = date.fromordinal(date(int(year), 1, 1).toordinal() + int(doy) - 1)
        if not d0 <= day <= d1 or key not in by_prod["MOD03"]:
            continue
        out.append({"key": key, "hour_key": granule_hour_key(f02),
                    "dir": {"MOD02": f02, "MOD03": by_prod["MOD03"][key],
                            "MOD06": by_prod["MOD06"].get(key), "MOD35": by_prod["MOD35"].get(key)}})
    return out


def done_keys(month_out: Path, gen_images: bool, retry_empty: bool = False) -> set[str]:
    """Granules that need no work: files written, or (unless retry_empty) last recorded as having no footprint."""
    done = set()
    log = month_out / "granules.csv"
    if log.is_file() and not retry_empty:
        with open(log, newline="") as f:
            last = {r["granule"]: r["status"] for r in csv.DictReader(f)}     # the latest row of a granule wins
        done |= {k for k, st in last.items() if st == "empty"}
    for p in (month_out / "scalars").glob("*.npz"):
        if granule_files_done(month_out, p.stem, gen_images):
            done.add(p.stem)
    return done


def ceres_slice(ceres: dict, hour_key: int) -> dict | None:
    """Footprints of one hour plus the next one (for the scan direction); None if the hour has none."""
    keys = ceres["key"]
    lo, hi = np.searchsorted(keys, hour_key, "left"), np.searchsorted(keys, hour_key, "right")
    if hi == lo:
        return None
    hi = min(hi + 1, len(keys))
    return {k: v[lo:hi] for k, v in ceres.items()}


def build_month(data_root: Path, out_root: Path, year: int, region: str, month: int, psf: np.ndarray, *,
                start: str, end: str, workers: int, gen_images: bool, limit: int | None, dry_run: bool,
                log, required: str = "core", retry_empty: bool = False, diagnose: int = 0) -> dict[str, int]:
    """Process one month folder; returns counts by status (ok, empty, error, skipped).

    limit: build only that many granules, spread over the month. diagnose: instead of building, report why that
    many granules (spread over the month) give few footprints. required: "core" or "all" (see REQUIRED_SW).
    """
    mdir = dm.MONTH_DIRS[month - 1]
    src = Path(data_root) / dm.SATELLITE_DIR / str(year) / region / mdir
    month_out = Path(out_root) / region / str(year) / mdir
    counts = {"ok": 0, "empty": 0, "error": 0, "skipped": 0}
    if not src.is_dir():
        log(f"{year}/{mdir}: no folder {src}")
        return counts
    granules = list_granules(src, start, end)
    already = set() if diagnose else done_keys(month_out, gen_images, retry_empty)
    todo = [g for g in granules if g["key"] not in already]
    counts["skipped"] = len(granules) - len(todo)
    if limit is not None:
        todo = spread(todo, limit)
    log(f"{year}/{mdir}: {len(granules)} granules, {counts['skipped']} done already, {len(todo)} to build")
    if dry_run or not todo:
        return counts
    ceres = load_ceres(find_ceres_file(src))
    tasks = []
    for g in todo:
        sl = ceres_slice(ceres, g["hour_key"])
        if sl is None:
            if not diagnose:
                counts["empty"] += 1
                _append_log(month_out, {"key": g["key"], "status": "empty", "n": 0, "sec": 0.0,
                                        "msg": "no CERES footprint in this hour"})
            continue
        tasks.append({**g, "ceres": sl, "psf": psf, "gen_images": gen_images, "month_out": str(month_out),
                      "required": required})
    del ceres
    if diagnose:
        for t in spread(tasks, diagnose):
            log(format_diagnosis(diagnose_granule(t)))
        return counts
    def handle(i: int, r: dict) -> None:
        counts[r["status"]] += 1
        _append_log(month_out, r)
        if r["status"] == "error":
            log(f"{r['key']}: ERROR {r['msg'].splitlines()[0]}")
        if i % 25 == 0 or i == len(tasks):
            log(f"{year}/{mdir}: [{i}/{len(tasks)}] ok={counts['ok']} empty={counts['empty']} error={counts['error']}")

    if workers <= 1:
        for i, t in enumerate(tasks, 1):
            handle(i, process_granule(t))
    else:
        with mp.get_context("fork").Pool(workers, maxtasksperchild=20) as pool:   # keep the pool referenced
            for i, r in enumerate(pool.imap_unordered(process_granule, tasks), 1):
                handle(i, r)
    return counts


def _append_log(month_out: Path, r: dict) -> None:
    month_out.mkdir(parents=True, exist_ok=True)
    path = month_out / "granules.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["granule", "status", "n_footprints", "seconds", "message"])
        w.writerow([r["key"], r["status"], r["n"], f"{r['sec']:.1f}", r["msg"].replace("\n", " | ")[:300]])


# ===================================== gather and verify =====================================
def gather(out_root: Path, region: str | None = None) -> dict[str, int]:
    """Symlink every granule file of each region into <out>/<region>/{scalars,moments,images}/ (relative links).

    Returns {region: number of links}. Existing correct links are left alone; stale ones are replaced.
    """
    counts = {}
    regions = [Path(out_root) / region] if region else [p for p in sorted(Path(out_root).iterdir()) if p.is_dir()]
    for rdir in regions:
        n = 0
        for part in PARTS:
            dest = rdir / part
            for f in sorted(rdir.glob(f"*/*/{part}/*.npz")):          # <region>/<year>/<month>/<part>/<file>
                dest.mkdir(exist_ok=True)
                link = dest / f.name
                target = os.path.relpath(f, dest)
                if link.is_symlink() and os.readlink(link) == target:
                    n += 1
                    continue
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(target)
                n += 1
        counts[rdir.name] = n
    return counts


def verify_triplets(region_dir: Path) -> list[str]:
    """Problems in the granule triplets of a region folder (after gather): keys, equal row counts, empty files."""
    problems = []
    root = Path(region_dir)
    names = sorted(p.stem for p in (root / "scalars").glob("*.npz"))
    if not names:
        return [f"no scalars in {root}"]
    for name in names:
        try:
            with np.load(root / "scalars" / f"{name}.npz") as s:
                n = len(s["Fsw"])
                missing = [k for k in SCALAR_KEYS if k not in s.files]
            if missing or n == 0:
                problems.append(f"{name}: scalars missing {missing}" if missing else f"{name}: no rows")
            mpath = root / "moments" / f"{name}.npz"
            if not mpath.exists():
                problems.append(f"{name}: no moments file")
            else:
                with np.load(mpath) as m:
                    miss = [k for k in MOMENT_KEYS if k not in m.files]
                    if miss or len(m["mSWspec"]) != n:
                        problems.append(f"{name}: moments {'missing ' + str(miss) if miss else 'row count differs'}")
            ipath = root / "images" / f"{name}.npz"
            if ipath.exists():
                with np.load(ipath) as im:
                    if len(im["FOVorientation"]) != n:
                        problems.append(f"{name}: images row count differs")
        except Exception as e:  # noqa: BLE001
            problems.append(f"{name}: unreadable ({type(e).__name__}: {e})")
    return problems


# ===================================== command line =====================================
def load_psf(spec: str) -> np.ndarray:
    """PSF weights (27, 55) from ceres_psf.csv / a recovered .npy, or the elliptical-Gaussian approximation.

    spec "ellipse" gives flux1km's fitted ellipse (an approximation that changes the moments; use the advisor's
    ceres_psf.csv for the real data).
    """
    from flux1km import aggregation

    if spec == "ellipse":
        return aggregation.fov_weights(np.array([1]))[0]
    return aggregation.load_psf_file(spec)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", default="/archive/GUY", help="root of the downloaded MODIS/CERES data")
    p.add_argument("--out", default="/archive/GUY/dataset", help="root of the built dataset")
    p.add_argument("--psf", default=str(DEFAULT_PSF),
                   help="the advisor's ceres_psf.csv (default: master/resources/ceres_psf.csv), a recovered PSF .npy, "
                        "or 'ellipse' = an approximation")
    p.add_argument("--start", default=dm.START_DATE)
    p.add_argument("--end", default=dm.END_DATE)
    p.add_argument("--region", default=dm.REGION_NAME)
    p.add_argument("--workers", type=int, default=4, help="parallel granules (each needs about 1.5 GB of memory)")
    p.add_argument("--no-images", action="store_true", help="skip images/ (about 10x less disk)")
    p.add_argument("--limit", type=int, help="build only N granules of each month, spread over the month (a test run)")
    p.add_argument("--diagnose", type=int, metavar="N",
                   help="build nothing: report, for N granules spread over each month, why they give few footprints")
    p.add_argument("--required-channels", choices=["core", "all"], default="core",
                   help="core (default): a footprint needs the advisor's 13 model channels valid, other channels get "
                        "NaN where they have invalid pixels; all: every channel must be valid")
    p.add_argument("--retry-empty", action="store_true", help="also rebuild granules recorded as empty by an earlier run")
    p.add_argument("--dry-run", action="store_true", help="count the granules, build nothing")
    p.add_argument("--gather", action="store_true", help="only make <out>/<region>/{scalars,moments,images} symlinks")
    p.add_argument("--verify", action="store_true", help="only check the triplets of <out>/<region>")
    a = p.parse_args(argv)
    out = Path(a.out).expanduser()

    if a.gather or a.verify:
        if a.gather:
            for r, n in gather(out, a.region if (out / a.region).is_dir() else None).items():
                print(f"{r}: {n} links in {out / r}/{{scalars,moments,images}}")
        if a.verify:
            bad = verify_triplets(out / a.region)
            print("\n".join(bad[:50]) if bad else "all triplets OK")
            return 1 if bad else 0
        return 0

    try:
        chunks = dm.date_chunks(a.start, a.end)
    except ValueError as e:
        p.error(str(e))
    if a.workers < 1:
        p.error("--workers must be at least 1")
    if a.psf != "ellipse" and not Path(a.psf).expanduser().is_file():
        p.error(f"PSF file not found: {a.psf}")
    psf = load_psf(str(Path(a.psf).expanduser()) if a.psf != "ellipse" else a.psf)

    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / ".build.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another build_dataset.py is already running on {out}", file=sys.stderr)
        return 1

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(out / "build_log.txt", "a") as f:
            f.write(line + "\n")

    log(f"start {a.start}..{a.end} region={a.region} psf={a.psf} images={not a.no_images} workers={a.workers}")
    total = {"ok": 0, "empty": 0, "error": 0, "skipped": 0}
    months = sorted({(date.fromisoformat(s).year, date.fromisoformat(s).month) for s, _ in chunks})
    for year, month in months:
        try:
            c = build_month(Path(a.data).expanduser(), out, year, a.region, month, psf, start=a.start, end=a.end,
                            workers=a.workers, gen_images=not a.no_images, limit=a.limit, dry_run=a.dry_run, log=log,
                            required=a.required_channels, retry_empty=a.retry_empty, diagnose=a.diagnose or 0)
        except Exception as e:  # noqa: BLE001 - e.g. a month without its CERES file; go on with the next
            log(f"{year}/{dm.MONTH_DIRS[month - 1]}: FAILED {type(e).__name__}: {e}")
            total["error"] += 1
            continue
        for k, v in c.items():
            total[k] += v
    log(f"done: {total}")
    if not a.dry_run and not a.diagnose:
        gather(out, a.region)
        log(f"gathered: {out / a.region}/{{scalars,moments,images}}")
    return 1 if total["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
