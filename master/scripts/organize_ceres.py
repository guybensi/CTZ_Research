#!/usr/bin/env python3
"""Fetch and organise the CERES SSF (Aqua, Level 2) subset that the advisor's Generate_AllSky_2.py needs.

Normally you do not need this script directly: download_ceres.py downloads the files from NASA's archive
and calls it. Use it for the MANUAL route below (an Order Tool order), or to re-merge / verify.

THE MANUAL ROUTE (Order Tool)
    The subset in the screenshot (12 fields, Pacific box, 2008) is produced by NASA's *CERES Order Tool*
    (https://ceres.larc.nasa.gov/data/ -> "CER SSF Aqua-FM3 Edition4A" -> Order Tool): an interactive web form
    that prepares your subset and e-mails you a link with a list of file URLs (``fileURLs.txt``). It has no
    documented programmatic API, so the ORDER is placed by hand (steps below). This script does everything
    after that: downloads the files, puts them in the advisor's folder layout, merges them into ONE file per
    month (Generate_AllSky_2.py expects exactly one file whose name contains "CERES" per month folder) and
    verifies that the 12 variables the advisor's code reads are present.

HOW TO ORDER (once per month, 12 orders for 2008; or one order if the tool allows a full year)
    Product   : SSF Aqua (FM3), Edition4A.  Dates: the month.  Daytime only if the form offers it.
    Region    : the same box as the MODIS download (150E-120W, 60S-60N). If the form cannot cross the
                dateline, order two boxes (150..180 and -180..-120); this script merges all the parts.
    Parameters: Time/Location/Angles, CERES Observed TOA Fluxes and Radiances, Clear Footprint Area,
                MODIS Ocean Aerosols -> "Effective Optical Depth Ocean, average" (as in the screenshot;
                the GEOS-5 boxes are optional and unused). Output format: netCDF.
    Keep the order e-mail's ``fileURLs.txt`` (save it, one URL per line).

Layout written (same as download_modis.py / the advisor's code):
    <OUT>/AQUA/<year>/<region>/<month>/CERES_SSF_Aqua_<year>_<month>.nc     <- the single merged file
    <OUT>/AQUA/<year>/<region>/<month>/ssf_parts/*.nc                        <- the original pieces
    <OUT>/ceres_incoming/                                                    <- raw downloads, before sorting

Usage:
    python3 organize_ceres.py --urls fileURLs.txt --out /archive/GUY            # download + sort + merge + verify
    python3 organize_ceres.py --src /some/dir/with/ssf_files --out /archive/GUY # files already downloaded by hand
    python3 organize_ceres.py --verify-only --out /archive/GUY                  # just re-check the result

Re-running is safe: finished downloads are skipped, merges are redone only from the parts.
Needs: numpy, netCDF4 (``conda install -c conda-forge netcdf4``).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

# ===================== SETTINGS (same defaults as download_modis.py) =====================
OUT_DIR = "/archive/GUY"
YEAR = "2008"
REGION_NAME = "Pacific"
SATELLITE_DIR = "AQUA"
MONTH_DIRS = ["jan", "feb", "mrch", "apr", "may", "june", "july", "aug", "sep", "oct", "nov", "dec"]
PARTS_DIR = "ssf_parts"   # must NOT contain "CERES": the advisor's code takes the one entry whose name has "CERES"
# The 12 SSF variables Generate_AllSky_2.py reads (names as written by the Order Tool's netCDF output).
REQUIRED_VARS = [
    "time", "Colatitude_of_CERES_FOV_at_surface", "Longitude_of_CERES_FOV_at_surface",
    "CERES_solar_zenith_at_surface", "CERES_viewing_zenith_at_surface", "CERES_relative_azimuth_at_surface",
    "CERES_SW_TOA_flux___upwards", "CERES_LW_TOA_flux___upwards",
    "CERES_SW_radiance___upwards", "CERES_LW_radiance___upwards",
    "Cloud_mask_clear_strong_percent_coverage", "PSF_wtd_MOD04_effective_optical_depth_average_ocean__0_550_",
]
# ==========================================================================================

_DATE_RE = re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?:\d{2})?(?!\d)")


def parse_urls(text: str) -> list[str]:
    """Extract the download URLs from the order e-mail's ``fileURLs.txt``.

    Blank lines, ``#`` comments and lines that are not http/https/ftp URLs are ignored; order is kept and
    duplicates are dropped.

    Example:
        >>> parse_urls("# order 1\\nhttps://x.gov/a.nc\\n\\nhttps://x.gov/a.nc\\nftp://y.gov/b.nc \\n")
        ['https://x.gov/a.nc', 'ftp://y.gov/b.nc']
    """
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("http://", "https://", "ftp://")) and s not in out:
            out.append(s)
    return out


def date_from_name(name: str) -> tuple[int, int] | None:
    """Return (year, month) found in a file name as YYYYMMDD or YYYYMMDDHH, or None.

    Example:
        >>> date_from_name("CER_SSF_Aqua-FM3-MODIS_Edition4A_400405.2008010100.nc")
        (2008, 1)
        >>> date_from_name("subset_order.nc") is None
        True
    """
    m = _DATE_RE.search(Path(name).name)
    return (int(m[1]), int(m[2])) if m else None


def month_folder(out_dir: str | Path, year: str | int, region: str, month: int) -> Path:
    """Folder of one month in the advisor's layout: <out>/AQUA/<year>/<region>/<month name>/.

    Example:
        >>> str(month_folder("/archive/GUY", 2008, "Pacific", 3))
        '/archive/GUY/AQUA/2008/Pacific/mrch'

    Raises:
        ValueError: if month is not in 1..12.
    """
    if not 1 <= int(month) <= 12:
        raise ValueError(f"month must be 1..12, got {month}")
    return Path(out_dir) / SATELLITE_DIR / str(year) / region / MONTH_DIRS[int(month) - 1]


def download(url: str, dest_dir: Path, retries: int = 4) -> Path:
    """Download one URL into dest_dir (skip if a complete non-empty copy exists). Returns the local path.

    Retries with exponential backoff; a transfer shorter than Content-Length counts as a failure and the
    partial file is removed. Raises the last error after `retries` attempts.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(url.split("?")[0]).name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                size = int(r.headers.get("Content-Length") or -1)
                shutil.copyfileobj(r, f, 1 << 20)
            if size >= 0 and dest.stat().st_size != size:
                raise OSError(f"truncated: {dest.stat().st_size} of {size} bytes")
            return dest
        except Exception as e:  # noqa: BLE001 - any network/IO error is retried
            last = e
            dest.unlink(missing_ok=True)
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last}")


def _first_time_month(path: Path) -> tuple[int, int]:
    """(year, month) of the first record of a netCDF file's ``time`` variable."""
    import netCDF4 as nc

    with nc.Dataset(path) as ds:
        t = ds.variables["time"]
        d = nc.num2date(t[0], units=t.getncattr("units"))
        return int(d.year), int(d.month)


def month_of_file(path: str | Path) -> tuple[int, int]:
    """(year, month) of a data file: from its name if it holds a date, else from its first ``time`` record.

    Raises:
        ValueError: if neither the name nor the file content gives a date.
    """
    path = Path(path)
    ym = date_from_name(path.name)
    if ym:
        return ym
    try:
        return _first_time_month(path)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"cannot tell the month of {path.name}: no date in the name and no readable 'time' ({e})") from e


def merge_netcdf(parts: list[Path], dest: Path) -> int:
    """Concatenate netCDF subset files along their record dimension into one file. Returns the record count.

    The record dimension is the first dimension of ``time``. Parts are ordered by their first ``time`` value, so
    consecutive footprints stay consecutive (the advisor's code reads footprint ``i+1`` to find the scan
    direction). Variables that do not have the record dimension are copied from the first part. Attributes and
    fill values are kept.

    Raises:
        ValueError: no parts, or parts with different variables.
    """
    import netCDF4 as nc

    if not parts:
        raise ValueError("nothing to merge")

    def first_t(p: Path) -> float:
        with nc.Dataset(p) as d:
            return float(d.variables["time"][0]) if d.variables["time"].shape[0] else float("inf")

    parts = sorted(parts, key=first_t)
    with nc.Dataset(parts[0]) as d0:
        rec = d0.variables["time"].dimensions[0]
        names = set(d0.variables)
    for p in parts[1:]:
        with nc.Dataset(p) as d:
            if set(d.variables) != names:
                raise ValueError(f"{p.name} has different variables than {parts[0].name}")
    tmp = dest.with_suffix(".tmp")
    total = 0
    with nc.Dataset(tmp, "w", format="NETCDF4") as out, nc.Dataset(parts[0]) as d0:
        for dn, dim in d0.dimensions.items():
            out.createDimension(dn, None if dn == rec else len(dim))
        for vn, v in d0.variables.items():
            fill = v.getncattr("_FillValue") if "_FillValue" in v.ncattrs() else None
            ov = out.createVariable(vn, v.dtype, v.dimensions, fill_value=fill, zlib=True, complevel=1)
            ov.setncatts({a: v.getncattr(a) for a in v.ncattrs() if a != "_FillValue"})
            if rec not in v.dimensions:
                ov[:] = v[:]
        out.setncatts({a: d0.getncattr(a) for a in d0.ncattrs()})
        for p in parts:
            with nc.Dataset(p) as d:
                n = d.dimensions[rec].size
                if n == 0:
                    continue
                for vn, v in d.variables.items():
                    if v.dimensions and v.dimensions[0] == rec:
                        v.set_auto_maskandscale(False)
                        out.variables[vn].set_auto_maskandscale(False)
                        out.variables[vn][total:total + n] = v[:]
                total += n
    tmp.replace(dest)
    return total


def verify_file(path: str | Path, year: int, month: int) -> list[str]:
    """Check one merged CERES file; returns a list of problems (empty = OK).

    Checks: all REQUIRED_VARS exist, there is at least one record, the ``time`` values fall in (year, month),
    and some footprints have a daytime SW flux (Fsw >= 1).
    """
    import netCDF4 as nc
    import numpy as np

    problems: list[str] = []
    with nc.Dataset(path) as ds:
        missing = [v for v in REQUIRED_VARS if v not in ds.variables]
        if missing:
            return [f"missing variables: {', '.join(missing)}"]
        t = ds.variables["time"]
        n = t.shape[0]
        if n == 0:
            return ["no records"]
        units = t.getncattr("units")
        d0, d1 = nc.num2date(t[0], units=units), nc.num2date(t[n - 1], units=units)
        if (d0.year, d0.month) != (year, month) or (d1.year, d1.month) != (year, month):
            problems.append(f"time spans {d0} .. {d1}, expected {year}-{month:02d}")
        fsw = np.ma.filled(ds.variables["CERES_SW_TOA_flux___upwards"][:].astype("float64"), np.nan)
        if not np.any(fsw >= 1):
            problems.append("no footprint with Fsw >= 1 (no daytime data?)")
    return problems


def organize(files: list[Path], out_dir: Path, year: str, region: str, merge: bool = True) -> dict[int, Path]:
    """Sort files into month folders (parts -> ssf_parts/) and merge each month into one CERES_SSF_*.nc.

    Files of another year are skipped with a warning. Returns {month: merged file path}.
    """
    by_month: dict[int, list[Path]] = {}
    for f in files:
        y, m = month_of_file(f)
        if str(y) != str(year):
            print(f"skip (year {y} != {year}): {f.name}")
            continue
        by_month.setdefault(m, []).append(f)
    result: dict[int, Path] = {}
    for m, fs in sorted(by_month.items()):
        mdir = month_folder(out_dir, year, region, m)
        parts_dir = mdir / PARTS_DIR
        parts_dir.mkdir(parents=True, exist_ok=True)
        for f in fs:
            tgt = parts_dir / f.name
            if f.resolve() != tgt.resolve():
                # raw downloads are moved (no second copy on disk); files from --src are copied
                (shutil.move if f.parent.name == "ceres_incoming" else shutil.copy2)(f, tgt)
        parts = sorted(parts_dir.glob("*.nc"))
        dest = mdir / f"CERES_SSF_Aqua_{year}_{MONTH_DIRS[m - 1]}.nc"
        if merge:
            n = merge_netcdf(parts, dest)
            print(f"{MONTH_DIRS[m - 1]}: merged {len(parts)} part(s), {n} footprints -> {dest}")
            result[m] = dest
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--urls", help="fileURLs.txt from the order e-mail (download these first)")
    p.add_argument("--src", help="folder with SSF netCDF files already downloaded by hand")
    p.add_argument("--out", default=OUT_DIR, help=f"root folder (default {OUT_DIR})")
    p.add_argument("--year", default=YEAR)
    p.add_argument("--region", default=REGION_NAME)
    p.add_argument("--verify-only", action="store_true", help="only verify the merged files that exist")
    p.add_argument("--no-merge", action="store_true", help="sort the parts into month folders but do not merge")
    a = p.parse_args(argv)
    out = Path(a.out)

    if not a.verify_only:
        files: list[Path] = []
        if a.urls:
            urls = parse_urls(Path(a.urls).read_text())
            print(f"{len(urls)} URLs")
            for i, u in enumerate(urls, 1):
                print(f"[{i}/{len(urls)}] {u}")
                files.append(download(u, out / "ceres_incoming"))
        if a.src:
            files += sorted(Path(a.src).glob("*.nc"))
        if not files:
            p.error("give --urls and/or --src (or --verify-only)")
        organize(files, out, a.year, a.region, merge=not a.no_merge)

    bad = 0
    for m in range(1, 13):
        mdir = month_folder(out, a.year, a.region, m)
        merged = sorted(mdir.glob("CERES*.nc"))
        if not merged:
            print(f"{MONTH_DIRS[m - 1]:5s} NO CERES FILE")
            bad += 1
            continue
        if len(merged) > 1:
            print(f"{MONTH_DIRS[m - 1]:5s} {len(merged)} files contain 'CERES' (the advisor's code needs exactly one)")
            bad += 1
            continue
        probs = verify_file(merged[0], int(a.year), m)
        print(f"{MONTH_DIRS[m - 1]:5s} {'OK' if not probs else 'PROBLEM: ' + '; '.join(probs)}")
        bad += bool(probs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
