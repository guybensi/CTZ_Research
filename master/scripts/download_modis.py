#!/usr/bin/env python3
"""Download MODIS/Aqua granules (daytime, one region, one period) from NASA Earthdata.

Source: earthaccess searches NASA's CMR catalogue and downloads the files from
LAADS DAAC (https://ladsweb.modaps.eosdis.nasa.gov), NASA's MODIS archive.
A free Earthdata Login account is required: https://urs.earthdata.nasa.gov

*** Edit the GLOBAL SETTINGS block below (dates, polygons, products, output folder). ***

Only "complete sets" are downloaded: a granule time (e.g. A2020060.0245) is fetched
only if every requested product exists for it.

Layout on disk:
    <OUT_DIR>/<product>/<year>/<doy>/<file>.hdf
    <OUT_DIR>/download_log.txt
    <OUT_DIR>/manifest.csv          (granule, product, file, size_mb, status)

Usage:
    pip install earthaccess
    python3 download_modis.py --dry-run          # search only: counts and total size
    nohup python3 download_modis.py > ~/GUY/run.out 2>&1 &   # real download, in the background

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
from datetime import date
from pathlib import Path

# =============================== GLOBAL SETTINGS ===============================
# Period, inclusive, "YYYY-MM-DD". Default: 3 days, a small first test.
START_DATE = "2017-03-01"
END_DATE = "2017-03-03"

# Region: a list of polygons; a granule is kept if it touches ANY of them.
# Each polygon is a list of (lon, lat) points in degrees, lon in [-180, 180].
# Point order does not matter and you don't need to repeat the first point:
# the script closes the ring and orders it counter-clockwise for you.
# A polygon must NOT cross the 180° meridian: split it into two (as below).
POLYGONS = [
    # Pacific, west of the dateline: 150E-180, 30S-30N
    [(150.0, -30.0), (180.0, -30.0), (180.0, 30.0), (150.0, 30.0)],
    # Pacific, east of the dateline: 180-120W, 30S-30N
    [(-180.0, -30.0), (-120.0, -30.0), (-120.0, 30.0), (-180.0, 30.0)],
]

PRODUCTS = ["MYD021KM", "MYD03", "MYD06_L2", "MYD35_L2"]
VERSION = "6.1"         # MODIS collection 6.1 (CMR spelling; the file names say "061")
DAY_ONLY = True         # daytime granules only
OUT_DIR = "~/GUY"
THREADS = 4             # parallel downloads
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


def log(msg: str, log_path: Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="search and report size, download nothing")
    a = p.parse_args(argv)

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
            chunk_mb = sum(float(by_product[pr][k].size() or 0) for k in keys for pr in PRODUCTS)
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
                    folder = out / prod / year / doy
                    target = folder / name
                    if target.exists() and target.stat().st_size > 0:
                        w.writerow([k, prod, name, f"{float(g.size() or 0):.1f}", "exists"])
                        continue
                    pending[folder].append((k, prod, name, g))

            for folder, items in sorted(pending.items()):
                folder.mkdir(parents=True, exist_ok=True)
                try:
                    earthaccess.download([it[3] for it in items], local_path=str(folder), threads=THREADS)
                except Exception as e:  # noqa: BLE001
                    log(f"download failed in {folder}: {e}", log_path)
                for k, prod, name, g in items:
                    ok = (folder / name).exists() and (folder / name).stat().st_size > 0
                    w.writerow([k, prod, name, f"{float(g.size() or 0):.1f}", "ok" if ok else "FAILED"])
                mf.flush()
            log(f"{start}..{end}: done", log_path)

    log(f"total: {total_sets} complete sets, {total_mb / 1024:.1f} GB", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
