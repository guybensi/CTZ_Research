#!/usr/bin/env python3
"""Download CERES SSF (Aqua, Level 2) from NASA Earthdata and cut it down to what the advisor's code reads.

Like download_modis.py: earthaccess searches NASA's CMR catalogue and downloads from the ASDC archive.
Same Earthdata login (~/.netrc). The full SSF hourly files are global HDF4 files with ~200 variables, so each
file is processed as soon as it is downloaded and the raw file is deleted:
  1. keep only footprints inside the Pacific box (plus one neighbour on each side: Generate_AllSky_2.py reads
     footprint i+1 to find the scan direction, so neighbours must stay consecutive),
  2. keep only the 17 variables of the Order Tool screenshot (organize_ceres.REQUIRED_VARS), renamed the way the CERES Order Tool names them
     (every non-alphanumeric character -> "_": "CERES SW TOA flux - upwards" -> "CERES_SW_TOA_flux___upwards"),
  3. write one small netCDF part per hour into <OUT>/AQUA/<year>/<region>/<month>/ssf_parts/,
  4. when all hours are done, merge each month into ONE <month>/CERES_SSF_Aqua_<year>_<month>.nc and verify it
     (organize_ceres.py does steps 3-4's merge and the check; keep both scripts in the same folder).

Only the hours that have MODIS granules in <OUT>/AQUA/<year>/<region>/<month>/MOD02/ are downloaded
(the advisor's code matches CERES footprints to granules by hour), which roughly halves the download.

Dates, output folder, region name and box are command-line options, the same as in download_modis.py
(--start, --end, --out, --region, --box), so use the same values for both scripts. Several years work too.

Usage:
    python3 download_ceres.py --find                 # list the matching CERES collections
    python3 download_ceres.py --inspect <raw file>   # check that a downloaded raw file has the 17 variables
    python3 download_ceres.py --start 2008-01-01 --end 2008-12-31 --out /archive/GUY --dry-run   # count and GB
    nohup python3 download_ceres.py --start 2008-01-01 --end 2008-12-31 --out /archive/GUY \
        > /archive/GUY/ceres_run.out 2>&1 &          # the real run, in the background
    python3 download_ceres.py --start 2008-01-01 --end 2008-12-31 --out /archive/GUY --merge-only
                                                     # only merge + verify what is already in ssf_parts/

Re-running is safe: hours whose part file exists are skipped.
Needs: earthaccess, numpy, netCDF4 built WITH HDF4 support (conda-forge), the same as for processing MODIS.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_modis as dm  # noqa: E402  (same folder: shared box helper and defaults)
import organize_ceres as oc  # noqa: E402

# =============================== GLOBAL SETTINGS ===============================
# Defaults of the command-line options (--start, --end, --out, --region, --box).
START_DATE = dm.START_DATE
END_DATE = dm.END_DATE
OUT_DIR = oc.OUT_DIR
REGION_NAME = oc.REGION_NAME
# CMR short name and version of the product: SSF of the Aqua FM3 instrument, MODIS-matched, Edition4A
# (checked against the catalogue: 8752 hourly files in 2008, ~60 MB each, no file extension).
SHORT_NAME = "CER_SSF_Aqua-FM3-MODIS"
VERSION = "Edition4A"
# Pacific box (west, east, south, north; same as download_modis.py): 150E eastwards to 120W, 60S-60N.
BOX = (150.0, -120.0, -60.0, 60.0)
LAT_RANGE = (BOX[2], BOX[3])
LON360_RANGE = (BOX[0] % 360, BOX[1] % 360)      # longitude in 0..360 as in the SSF files: (150, 240)
BATCH = 8               # files downloaded together (earthaccess uses threads inside a batch)
ONLY_MODIS_HOURS = True
# ===============================================================================

# SSF names in the HDF4 file (as the Order Tool's names un-sanitised). Matching is done on the sanitised name,
# so it does not matter how the HDF4 layer spells spaces or punctuation.
FILL = 3.4028235e38


def sanitize(name: str) -> str:
    """CERES Order Tool spelling of an SSF variable name: every non-alphanumeric character becomes '_'.

    Example:
        >>> sanitize("CERES SW TOA flux - upwards")
        'CERES_SW_TOA_flux___upwards'
        >>> sanitize("PSF-wtd MOD04 effective optical depth average ocean (0.550)")
        'PSF_wtd_MOD04_effective_optical_depth_average_ocean__0_550_'
    """
    return re.sub(r"[^0-9A-Za-z]", "_", name)


def ssf_hour(name: str) -> tuple[int, int, int] | None:
    """(year, day-of-year, hour) of an SSF hourly file name ending in YYYYMMDDHH, or None.

    Example:
        >>> ssf_hour("CER_SSF_Aqua-FM3-MODIS_Edition4A_400405.2008020103")
        (2008, 32, 3)
        >>> ssf_hour("readme.txt") is None
        True
    """
    m = re.search(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])([01]\d|2[0-3])(?!\d)", Path(name).name)
    if not m:
        return None
    y, mo, d, h = (int(g) for g in m.groups())
    return y, date(y, mo, d).timetuple().tm_yday, h


def needed_hours(out_dir: Path, year: str, region: str) -> set[tuple[int, int, int]]:
    """(year, doy, hour) of every MODIS granule found in <out>/AQUA/<year>/<region>/*/MOD02/.

    Example (a MYD021KM.A2008032.0305... file means day 32, hour 03):
        >>> needed_hours(Path("/no/such/dir"), "2008", "Pacific")
        set()
    """
    out: set[tuple[int, int, int]] = set()
    for f in (Path(out_dir) / oc.SATELLITE_DIR / str(year) / region).glob("*/MOD02/*"):
        m = re.search(r"\.A(\d{4})(\d{3})\.(\d{2})\d{2}\.", f.name)
        if m:
            out.add((int(m[1]), int(m[2]), int(m[3])))
    return out


def box_ranges(west: float, east: float, south: float, north: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """(lat_range, lon360_range) of a box given as in download_modis.py (lon in [-180, 180], west > east wraps).

    Example:
        >>> box_ranges(150, -120, -60, 60)
        ((-60.0, 60.0), (150.0, 240.0))
        >>> box_ranges(-60, -10, -50, 50)
        ((-50.0, 50.0), (300.0, 350.0))

    Raises:
        ValueError: same conditions as download_modis.box_polygons.
    """
    dm.box_polygons(west, east, south, north)                      # validates
    return (float(south), float(north)), (float(west) % 360, float(east) % 360)


def years_in(start: str, end: str) -> list[int]:
    """Calendar years touched by [start, end] (inclusive ISO dates).

    Example:
        >>> years_in("2008-12-30", "2009-01-02")
        [2008, 2009]

    Raises:
        ValueError: malformed date or start > end.
    """
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    if d0 > d1:
        raise ValueError(f"start {start} is after end {end}")
    return list(range(d0.year, d1.year + 1))


def months_in(year: int, start: str, end: str) -> list[int]:
    """Months (1-12) of `year` that overlap [start, end] (inclusive ISO dates).

    Example:
        >>> months_in(2008, "2008-11-15", "2009-02-01")
        [11, 12]
        >>> months_in(2009, "2008-11-15", "2009-02-01")
        [1, 2]
    """
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    return [m for m in range(1, 13)
            if date(year, m, 1) <= d1 and date(year + (m == 12), m % 12 + 1, 1) > d0]


def keep_mask(lat, lon360, lat_range=LAT_RANGE, lon_range=LON360_RANGE):
    """Boolean mask of footprints to keep: inside the box, extended by one footprint on each side.

    The extension keeps the neighbour footprint (index i+1) that the advisor's code reads, for footprints at
    the edge of the box. Invalid (NaN or fill) coordinates are never inside the box. A lon_range with
    lo > hi wraps around 0 (e.g. (350, 10)).

    Example:
        >>> import numpy as np
        >>> keep_mask(np.zeros(7), np.array([10, 10, 160, 170, 300, 310, 310.0])).tolist()
        [False, True, True, True, True, False, False]
        >>> keep_mask(np.zeros(3), np.array([355, 100, 5.0]), lon_range=(350, 10)).tolist()
        [True, True, True]
    """
    import numpy as np

    lat, lon = np.asarray(lat, dtype="float64"), np.asarray(lon360, dtype="float64")
    with np.errstate(invalid="ignore"):
        in_lon = ((lon >= lon_range[0]) & (lon <= lon_range[1]) if lon_range[0] <= lon_range[1]
                  else (lon >= lon_range[0]) | (lon <= lon_range[1]))
        inside = (lat >= lat_range[0]) & (lat <= lat_range[1]) & in_lon
    inside &= np.isfinite(lat) & np.isfinite(lon)
    keep = inside.copy()
    keep[1:] |= inside[:-1]
    keep[:-1] |= inside[1:]
    return keep


def subset_ssf_file(src: Path, dest: Path, lat_range=LAT_RANGE, lon_range=LON360_RANGE) -> int:
    """Cut one raw hourly SSF file down to the box (default Pacific) and the 17 variables; returns footprints kept.

    The output has the Order Tool's variable names, a ``_FillValue`` on every variable (the advisor's code reads
    that attribute) and ``time`` in days since 2000-01-01 12:00 (Julian date - 2451545), which num2date reads.
    A file with no footprint in the box writes nothing and returns 0.

    Raises:
        ValueError: if a required variable is not in the file.
    """
    import netCDF4 as nc
    import numpy as np

    with nc.Dataset(src) as ds:
        by_clean = {sanitize(k): k for k in ds.variables}
        lat_name = by_clean.get("Colatitude_of_CERES_FOV_at_surface")
        lon_name = by_clean.get("Longitude_of_CERES_FOV_at_surface")
        time_name = by_clean.get("Time_of_observation") or by_clean.get("time")
        wanted = [v for v in oc.REQUIRED_VARS if v != "time"]
        missing = [v for v in wanted if v not in by_clean] + ([] if time_name else ["time (Time of observation)"])
        if missing:
            raise ValueError(f"{src.name}: missing variables {missing}; the file has {len(by_clean)} variables, "
                             f"e.g. {sorted(by_clean)[:40]} (run --inspect <file> to list them all)")

        def read(name):
            a = np.ma.filled(ds.variables[name][:].astype("float64"), np.nan)
            return np.where(a >= 3.0e38, np.nan, a)

        lat = 90.0 - read(lat_name)
        lon = read(lon_name)
        keep = keep_mask(lat, lon, lat_range, lon_range)
        n = int(keep.sum())
        if n == 0:
            return 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        with nc.Dataset(tmp, "w", format="NETCDF4") as out:
            out.createDimension("record", n)
            tv = out.createVariable("time", "f8", ("record",), fill_value=FILL, zlib=True)
            tv.units = "days since 2000-01-01 12:00:00"
            tv[:] = np.nan_to_num(read(time_name)[keep] - 2451545.0, nan=FILL)
            for clean in wanted:
                src_var = ds.variables[by_clean[clean]]
                ov = out.createVariable(clean, "f4", ("record",), fill_value=np.float32(FILL), zlib=True)
                if "units" in src_var.ncattrs():
                    ov.units = src_var.getncattr("units")
                data = read(by_clean[clean])[keep]
                ov[:] = np.where(np.isnan(data), np.float32(FILL), data)
        tmp.replace(dest)
        return n


def part_path(out_dir: Path, year: str, region: str, y: int, doy: int, hour: int) -> Path:
    """Path of the part file of one hour: <month folder>/ssf_parts/ssf_<YYYYMMDDHH>.nc.

    Example:
        >>> part_path(Path("/r"), "2008", "Pacific", 2008, 32, 3).as_posix()
        '/r/AQUA/2008/Pacific/feb/ssf_parts/ssf_2008020103.nc'
    """
    d = date.fromordinal(date(y, 1, 1).toordinal() + doy - 1)
    return oc.month_folder(out_dir, year, region, d.month) / oc.PARTS_DIR / f"ssf_{d:%Y%m%d}{hour:02d}.nc"


def search(earthaccess, start: str, end: str, wanted: set | None) -> list:
    """SSF granules in [start, end], restricted to the (year, doy, hour) in `wanted` if given."""
    found = earthaccess.search_data(short_name=SHORT_NAME, version=VERSION, temporal=(start, end))
    out = []
    for g in found:
        links = g.data_links()
        h = ssf_hour(links[0].rstrip("/").split("/")[-1]) if links else None
        if h and (wanted is None or h in wanted):
            out.append((h, g))
    return sorted(out, key=lambda x: x[0])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--start", default=START_DATE, help=f"first day, YYYY-MM-DD (default {START_DATE})")
    p.add_argument("--end", default=END_DATE, help=f"last day, inclusive, YYYY-MM-DD (default {END_DATE})")
    p.add_argument("--out", default=OUT_DIR, help=f"root folder of the data (default {OUT_DIR})")
    p.add_argument("--region", default=REGION_NAME, help=f"region folder name (default {REGION_NAME})")
    p.add_argument("--box", nargs=4, type=float, metavar=("WEST", "EAST", "SOUTH", "NORTH"),
                   help="region box in degrees, lon in [-180,180]; WEST > EAST crosses the dateline "
                        f"(default {' '.join(str(x) for x in BOX)} = Pacific)")
    p.add_argument("--find", action="store_true", help="list CERES SSF Aqua collections in the catalogue")
    p.add_argument("--dry-run", action="store_true", help="search and report counts and size, download nothing")
    p.add_argument("--merge-only", action="store_true", help="skip downloading; merge and verify ssf_parts/")
    p.add_argument("--all-hours", action="store_true", help="do not restrict to hours that have MODIS granules")
    p.add_argument("--inspect", metavar="FILE", help="list which of the 17 needed variables a raw SSF file has")
    a = p.parse_args(argv)
    out, region = Path(a.out).expanduser(), a.region

    if a.inspect:
        import netCDF4 as nc

        with nc.Dataset(a.inspect) as ds:
            have = {sanitize(k) for k in ds.variables}
            print(f"{len(have)} variables")
            for v in ["Time_of_observation"] + [x for x in oc.REQUIRED_VARS if x != "time"]:
                print(("OK      " if v in have else "MISSING ") + v)
        return 0

    try:
        years = years_in(a.start, a.end)
        lat_range, lon_range = box_ranges(*(a.box or BOX))
    except ValueError as e:
        p.error(str(e))

    if not a.merge_only:
        import earthaccess

        earthaccess.login()
        if a.find:
            for c in earthaccess.search_datasets(keyword="CERES SSF Aqua Edition4A", count=20):
                print(c.summary().get("short-name"), "|", c.summary().get("entry-title"))
            return 0
        wanted = None
        if not (a.all_hours or not ONLY_MODIS_HOURS):
            wanted = set().union(*(needed_hours(out, str(y), region) for y in years))
            if not wanted:
                p.error(f"no MODIS granules found under {out}/AQUA/<year>/{region}/*/MOD02/ "
                        "(download MODIS first with the same --out/--region, or use --all-hours)")
        todo = [(h, g) for h, g in search(earthaccess, a.start, a.end, wanted)
                if not part_path(out, str(h[0]), region, *h).exists()]
        print(f"{len(todo)} hourly files to fetch", flush=True)
        if a.dry_run:
            gb = 0.0
            for _, g in todo:
                s = g.size() if callable(g.size) else g.size
                gb += float(s or 0) / 1024
            print(f"total about {gb:.1f} GB (raw, deleted after processing)")
            return 0
        out.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            with tempfile.TemporaryDirectory(dir=out) as tmp:
                files = earthaccess.download([g for _, g in chunk], tmp)
                by_hour = {ssf_hour(Path(f).name): Path(f) for f in files}
                for h, _ in chunk:
                    raw = by_hour.get(h)
                    if raw is None:
                        print(f"FAILED download {h}", flush=True)
                        continue
                    try:
                        n = subset_ssf_file(raw, part_path(out, str(h[0]), region, *h), lat_range, lon_range)
                        print(f"{h}: kept {n} footprints", flush=True)
                    except Exception as e:  # noqa: BLE001 - one bad file must not stop the run
                        print(f"FAILED {h}: {e}", flush=True)
                shutil.rmtree(tmp, ignore_errors=True)
            print(f"[{min(i + BATCH, len(todo))}/{len(todo)}]", flush=True)

    status = 0
    for y in years:
        year = str(y)
        months = months_in(y, a.start, a.end)
        parts = sorted((out / oc.SATELLITE_DIR / year / region).glob(f"*/{oc.PARTS_DIR}/*.nc"))
        if parts:
            oc.organize(parts, out, year, region)
        status |= oc.main(["--verify-only", "--out", str(out), "--year", year, "--region", region,
                           "--months", *map(str, months)])
    return status


if __name__ == "__main__":
    sys.exit(main())
