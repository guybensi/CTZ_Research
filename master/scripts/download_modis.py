#!/usr/bin/env python3
"""Download MODIS/Aqua granules (daytime, one region, one period) from NASA Earthdata.

Source: earthaccess searches NASA's CMR catalogue and downloads the files from
LAADS DAAC (https://ladsweb.modaps.eosdis.nasa.gov), NASA's MODIS archive.
A free Earthdata Login account is required: https://urs.earthdata.nasa.gov

*** Edit the GLOBAL SETTINGS block below (dates, polygons, products, output folder). ***

Only "complete sets" are downloaded: a granule time (e.g. A2020060.0245) is fetched
only if every requested product exists for it.

Layout on disk = the input layout of the advisor's Generate_AllSky_2.py
(pre_master/advisor_code/), which reads datapath = <root>/AQUA/<year>/<region>/<month>/:
    <OUT_DIR>/AQUA/<year>/<REGION_NAME>/<month>/MOD02/MYD021KM.*.hdf
                                               /MOD03/MYD03.*.hdf
                                               /MOD06/   (empty unless MYD06_L2 is in PRODUCTS)
                                               /MOD35/   (empty unless MYD35_L2 is in PRODUCTS)
    (month folders are named as in the advisor's MONTH list: mrch, june, sep, dec, ...;
     the CERES SSF file is NOT downloaded here: put it in each month folder yourself,
     its name must contain "CERES")
    <OUT_DIR>/download_log.txt
    <OUT_DIR>/manifest.csv          (granule, product, file, size_mb, status)

Usage:
    pip install earthaccess
    python3 download_modis.py --dry-run          # search only: counts and total size
    nohup python3 download_modis.py > /archive/GUY/run.out 2>&1 &   # real download, in the background
    python3 download_modis.py --reorganize       # move files of the old <product>/<year>/<doy>/ layout

Earthdata login: ~/.netrc with
    machine urs.earthdata.nasa.gov login <USER> password <PASS>
(chmod 600 ~/.netrc). Without it, earthaccess asks for the password interactively.

Re-running is safe: files that already exist with a non-zero size are skipped.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import re
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

# =============================== GLOBAL SETTINGS ===============================
# Period, inclusive, "YYYY-MM-DD".
START_DATE = "2008-01-01"
END_DATE = "2008-12-31"

# Region: a list of polygons; a granule is kept if it touches ANY of them.
# Each polygon is a list of (lon, lat) points in degrees, lon in [-180, 180].
# Point order does not matter and you don't need to repeat the first point:
# the script closes the ring and orders it counter-clockwise for you.
# A polygon must NOT cross the 180° meridian: split it into two (as below).
POLYGONS = [
    # Pacific, west of the dateline: 150E-180, 60S-60N
    [(150.0, -60.0), (180.0, -60.0), (180.0, 60.0), (150.0, 60.0)],
    # Pacific, east of the dateline: 180-120W, 60S-60N
    [(-180.0, -60.0), (-120.0, -60.0), (-120.0, 60.0), (-180.0, 60.0)],
]

PRODUCTS = ["MYD021KM", "MYD03"]   # add "MYD06_L2", "MYD35_L2" to also fill MOD06/ and MOD35/
VERSION = "6.1"         # MODIS collection 6.1 (CMR spelling; the file names say "061")
DAY_ONLY = True         # daytime granules only
OUT_DIR = "/archive/GUY"
THREADS = 4             # parallel downloads

# Folder layout (matches Generate_AllSky_2.py; change only if that code changes)
SATELLITE_DIR = "AQUA"
REGION_NAME = "Pacific"
PRODUCT_DIRS = {"MYD021KM": "MOD02", "MYD03": "MOD03", "MYD06_L2": "MOD06", "MYD35_L2": "MOD35",
                "MOD021KM": "MOD02", "MOD03": "MOD03", "MOD06_L2": "MOD06", "MOD35_L2": "MOD35"}
# Sub-folders created in every month folder even when empty (Generate_AllSky_2.py lists all four).
MONTH_SUBDIRS = ["MOD02", "MOD03", "MOD06", "MOD35"]
# Advisor's month-folder names (mrch, june, sep, dec come from his MONTH list).
MONTH_DIRS = ["jan", "feb", "mrch", "apr", "may", "june", "july", "aug", "sep", "oct", "nov", "dec"]
# ===============================================================================

GRANULE_RE = re.compile(r"^(?P<product>M[OY]D[0-9A-Z_]+)\.(?P<key>A(?P<year>\d{4})(?P<doy>\d{3})\.(?P<hhmm>\d{4}))\.")


def parse_granule_name(name: str) -> tuple[str, str, str, str]:
    """Split a MODIS file name into (product, granule key, year, day-of-year).

    Example:
        >>> parse_granule_name("MYD021KM.A2020060.0245.061.2020061.hdf")
        ('MYD021KM', 'A2020060.0245', '2020', '060')

    Raises:
        ValueError: if the name is not a MODIS granule file name.
    """
    m = GRANULE_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"not a MODIS granule name: {name!r}")
    return m["product"], m["key"], m["year"], m["doy"]


def date_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end] (inclusive ISO dates) into calendar-month pieces.

    Example:
        >>> date_chunks("2020-01-20", "2020-03-05")
        [('2020-01-20', '2020-01-31'), ('2020-02-01', '2020-02-29'), ('2020-03-01', '2020-03-05')]

    Raises:
        ValueError: if a date is malformed or start > end.
    """
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    if d0 > d1:
        raise ValueError(f"START_DATE {start} is after END_DATE {end}")
    out = []
    cur = d0
    while cur <= d1:
        last = date(cur.year, cur.month, calendar.monthrange(cur.year, cur.month)[1])
        stop = min(last, d1)
        out.append((cur.isoformat(), stop.isoformat()))
        cur = date(stop.year + (stop.month == 12), stop.month % 12 + 1, 1)
    return out


def normalize_polygon(points) -> list[tuple[float, float]]:
    """Validate a (lon, lat) polygon, order it counter-clockwise and close it (as CMR requires).

    Example:
        >>> normalize_polygon([(0, 0), (0, 1), (1, 1), (1, 0)])   # clockwise in, CCW out
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]

    Raises:
        ValueError: fewer than 3 distinct points, coordinates out of range, or zero area.
    """
    pts = [(float(lon), float(lat)) for lon, lat in points]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(set(pts)) < 3:
        raise ValueError("a polygon needs at least 3 distinct (lon, lat) points")
    for lon, lat in pts:
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(f"point out of range (lon in [-180,180], lat in [-90,90]): {(lon, lat)}")
    area2 = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
    if area2 == 0:
        raise ValueError("polygon has zero area (points are collinear)")
    if area2 < 0:
        pts = [pts[0]] + pts[1:][::-1]   # reverse direction, keep the start point
    return pts + [pts[0]]


def target_folder(out_dir: Path, product: str, year: str, doy: str) -> Path:
    """Folder a granule file goes to: <out>/AQUA/<year>/<region>/<month>/<MODxx>.

    Example:
        >>> target_folder(Path("/archive/GUY"), "MYD06_L2", "2008", "060").as_posix()
        '/archive/GUY/AQUA/2008/Pacific/feb/MOD06'

    Raises:
        ValueError: unknown product, or a day-of-year that does not exist in that year.
    """
    if product not in PRODUCT_DIRS:
        raise ValueError(f"no folder defined for product {product!r}; add it to PRODUCT_DIRS")
    y, d = int(year), int(doy)
    if not 1 <= d <= (366 if calendar.isleap(y) else 365):
        raise ValueError(f"day-of-year {doy} does not exist in {year}")
    month = (date(y, 1, 1) + timedelta(days=d - 1)).month
    return Path(out_dir) / SATELLITE_DIR / year / REGION_NAME / MONTH_DIRS[month - 1] / PRODUCT_DIRS[product]


def reorganize(out_dir: Path) -> int:
    """Move files of the old layout <out>/<product>/<year>/<doy>/*.hdf into target_folder; return how many."""
    moved = 0
    for prod in PRODUCT_DIRS:
        for f in sorted(Path(out_dir).glob(f"{prod}/*/*/*.hdf")):
            try:
                product, _, year, doy = parse_granule_name(f.name)
            except ValueError:
                continue
            dest = target_folder(out_dir, product, year, doy)
            dest.mkdir(parents=True, exist_ok=True)
            f.rename(dest / f.name)
            moved += 1
    for prod in PRODUCT_DIRS:                                  # remove the now-empty old folders
        old = Path(out_dir) / prod
        if old.is_dir():
            for d in sorted((x for x in old.rglob("*") if x.is_dir()), key=lambda x: len(x.parts), reverse=True):
                if not any(d.iterdir()):
                    d.rmdir()
            if not any(old.iterdir()):
                old.rmdir()
    return moved


def granule_file_name(granule) -> str:
    """File name of an earthaccess search result (from its first data link)."""
    links = granule.data_links()
    if not links:
        raise ValueError("granule has no data link")
    return links[0].rstrip("/").split("/")[-1]


def search_product(earthaccess, product: str, start: str, end: str, polygons) -> dict:
    """Search one product over all polygons; return {granule key: granule result} (merged)."""
    found = {}
    for poly in polygons:
        kwargs = dict(short_name=product, version=VERSION, temporal=(start, end), polygon=poly)
        if DAY_ONLY:
            kwargs["day_night_flag"] = "day"
        for g in earthaccess.search_data(**kwargs):
            _, key, _, _ = parse_granule_name(granule_file_name(g))
            found[key] = g
    return found


def complete_sets(by_product: dict[str, dict]) -> list[str]:
    """Granule keys present in every product, sorted.

    Example:
        >>> complete_sets({"MYD03": {"A1": 0, "A2": 0}, "MYD06_L2": {"A2": 0}})
        ['A2']
    """
    if not by_product:
        return []
    return sorted(set.intersection(*(set(d) for d in by_product.values())))


def _size_mb(granule) -> float:
    """Granule size in MB; earthaccess has it as a method in some versions and a number in others."""
    size = granule.size
    size = size() if callable(size) else size
    try:
        return float(size or 0)
    except (TypeError, ValueError):
        return 0.0


def log(msg: str, log_path: Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="search and report size, download nothing")
    p.add_argument("--reorganize", action="store_true",
                   help="only move already-downloaded files from the old <product>/<year>/<doy>/ layout, then exit")
    a = p.parse_args(argv)

    if a.reorganize:
        n = reorganize(Path(OUT_DIR).expanduser())
        print(f"moved {n} files into {OUT_DIR}/{SATELLITE_DIR}/<year>/{REGION_NAME}/<month>/MODxx/")
        return 0

    chunks = date_chunks(START_DATE, END_DATE)            # validate settings before logging in
    polygons = [normalize_polygon(poly) for poly in POLYGONS]

    import earthaccess  # imported here so --help works without it

    out = Path(OUT_DIR).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "download_log.txt"

    earthaccess.login(strategy="netrc") or earthaccess.login(strategy="interactive", persist=True)
    log(f"start: {START_DATE}..{END_DATE} products={PRODUCTS} polygons={len(polygons)} "
        f"day_only={DAY_ONLY} dry_run={a.dry_run}", log_path)

    manifest = out / "manifest.csv"
    new_manifest = not manifest.exists()
    total_mb, total_sets = 0.0, 0
    with open(manifest, "a", newline="") as mf:
        w = csv.writer(mf)
        if new_manifest:
            w.writerow(["granule", "product", "file", "size_mb", "status"])

        for start, end in chunks:
            by_product = {}
            for prod in PRODUCTS:
                try:
                    by_product[prod] = search_product(earthaccess, prod, start, end, polygons)
                except Exception as e:  # noqa: BLE001 - keep going with the next product/month
                    log(f"{start}: search failed for {prod}: {e}", log_path)
                    by_product[prod] = {}
            keys = complete_sets(by_product)
            chunk_mb = sum(_size_mb(by_product[pr][k]) for k in keys for pr in PRODUCTS)
            total_mb += chunk_mb
            total_sets += len(keys)
            counts = ", ".join(f"{pr}={len(d)}" for pr, d in by_product.items())
            log(f"{start}..{end}: {len(keys)} complete sets ({counts}), {chunk_mb / 1024:.1f} GB", log_path)
            if a.dry_run or not keys:
                continue

            pending = defaultdict(list)   # target folder -> files to fetch
            for k in keys:
                for prod in PRODUCTS:
                    g = by_product[prod][k]
                    name = granule_file_name(g)
                    _, _, year, doy = parse_granule_name(name)
                    folder = target_folder(out, prod, year, doy)
                    target = folder / name
                    if target.exists() and target.stat().st_size > 0:
                        w.writerow([k, prod, name, f"{_size_mb(g):.1f}", "exists"])
                        continue
                    pending[folder].append((k, prod, name, g))

            for folder, items in sorted(pending.items()):
                for sub in MONTH_SUBDIRS:
                    (folder.parent / sub).mkdir(parents=True, exist_ok=True)
                try:
                    earthaccess.download([it[3] for it in items], local_path=str(folder), threads=THREADS)
                except Exception as e:  # noqa: BLE001
                    log(f"download failed in {folder}: {e}", log_path)
                for k, prod, name, g in items:
                    ok = (folder / name).exists() and (folder / name).stat().st_size > 0
                    w.writerow([k, prod, name, f"{_size_mb(g):.1f}", "ok" if ok else "FAILED"])
                mf.flush()
            log(f"{start}..{end}: done", log_path)

    log(f"total: {total_sets} complete sets, {total_mb / 1024:.1f} GB", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
