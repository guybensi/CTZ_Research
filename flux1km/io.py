"""Read and write granule triplets (moments/, scalars/, images/ NPZ files with the same name).

Layout (as in data/paired/<region>/...):
    <root>/moments/A2014060.0245.npz   mSWspec, stdSWspec, skewSWspec, kurtSWspec, (same for LW), SWwavlngs, LWwavlngs
    <root>/scalars/A2014060.0245.npz   Fsw, Flw, Isw, ..., SZA, VZA, RAA, Lat, Lon, CF, ...
    <root>/images/A2014060.0245.npz    Imviews (N,3,H,W), ImSW (N,11,H,W), ImLW (N,16,H,W), FOVorientation (N,)
    <root>/truth/A2014060.0245.npz     synthetic data only: known 1 km truth and helper maps
Footprints are in the same order in every file of a granule.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PARTS = ("moments", "scalars", "images", "truth")
_ID_RE = re.compile(r"^([AT])(\d{4})(\d{3})\.(\d{2})(\d{2})$")


@dataclass(frozen=True)
class GranuleId:
    """Parsed MODIS granule name, e.g. A2014060.0245 -> Aqua, 2014-03-01 02:45 UTC."""

    name: str
    satellite: str
    time: datetime

    @property
    def day(self) -> str:
        """Satellite + date, e.g. "A2014-03-01"; the unit of the day-wise splits."""
        return f"{self.satellite[0]}{self.time:%Y-%m-%d}"


def parse_granule_id(name: str) -> GranuleId:
    """Parse "A2014060.0245" (Aqua) or "T..." (Terra) into a GranuleId.

    Example:
        >>> g = parse_granule_id("A2014060.0245")
        >>> g.satellite, g.time.isoformat(), g.day
        ('Aqua', '2014-03-01T02:45:00', 'A2014-03-01')

    Raises:
        ValueError: If the name does not match the MODIS pattern or the time is invalid.
    """
    stem = Path(name).name.removesuffix(".npz")
    m = _ID_RE.match(stem)
    if not m:
        raise ValueError(f"not a MODIS granule id: {name!r}")
    sat, year, doy, hh, mm = m.groups()
    if not 1 <= int(doy) <= 366 or int(hh) > 23 or int(mm) > 59:
        raise ValueError(f"invalid date/time in granule id {name!r}")
    t = datetime(int(year), 1, 1) + timedelta(days=int(doy) - 1, hours=int(hh), minutes=int(mm))
    return GranuleId(stem, "Aqua" if sat == "A" else "Terra", t)


@dataclass
class Granule:
    """All arrays of one granule. Every array's first axis is the footprint axis."""

    granule_id: str
    scalars: dict[str, np.ndarray]
    moments: dict[str, np.ndarray] | None = None
    images: dict[str, np.ndarray] | None = None
    truth: dict[str, np.ndarray] | None = None

    @property
    def n_footprints(self) -> int:
        return len(next(iter(self.scalars.values())))

    def validate(self) -> None:
        """Check that every per-footprint array has n_footprints rows.

        Raises:
            ValueError: On a row-count mismatch.
        """
        n = self.n_footprints
        for part in ("scalars", "moments", "images", "truth"):
            for key, arr in (getattr(self, part) or {}).items():
                if key in ("SWwavlngs", "LWwavlngs"):
                    continue
                if np.ndim(arr) == 0 or len(arr) != n:
                    raise ValueError(f"{self.granule_id}: {part}/{key} has {np.shape(arr)}, expected {n} rows")


def find_granules(root: str | Path, require: tuple[str, ...] = ("moments", "scalars")) -> list[str]:
    """Sorted granule names present in every required sub-directory of root.

    Example:
        >>> find_granules("/nonexistent")
        []
    """
    root = Path(root)
    sets = [{p.name.removesuffix(".npz") for p in (root / part).glob("*.npz")} for part in require]
    return sorted(set.intersection(*sets)) if sets else []


def load_granule(root: str | Path, granule_id: str, parts: tuple[str, ...] = ("moments", "scalars", "images")) -> Granule:
    """Load the requested parts of one granule and validate row counts.

    Missing optional parts ("images", "truth") are left as None; "scalars" is required.

    Raises:
        FileNotFoundError: If the scalars file is missing.
        ValueError: If row counts differ between parts.
    """
    root = Path(root)
    loaded: dict[str, dict[str, np.ndarray] | None] = {}
    for part in set(parts) | {"scalars"}:
        if part not in PARTS:
            raise ValueError(f"unknown part {part!r}")
        path = root / part / f"{granule_id}.npz"
        if not path.exists():
            if part == "scalars":
                raise FileNotFoundError(path)
            loaded[part] = None
            continue
        with np.load(path, allow_pickle=False) as f:
            loaded[part] = {k: f[k] for k in f.files}
    g = Granule(granule_id, loaded["scalars"], loaded.get("moments"), loaded.get("images"), loaded.get("truth"))
    g.validate()
    return g


def save_granule(root: str | Path, granule: Granule) -> None:
    """Write each non-empty part of a granule to <root>/<part>/<granule_id>.npz."""
    granule.validate()
    for part in PARTS:
        arrays = getattr(granule, part)
        if arrays:
            out = Path(root) / part
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / f"{granule.granule_id}.npz", **arrays)


def load_dataset(root: str | Path, granule_ids: list[str] | None = None,
                 parts: tuple[str, ...] = ("moments", "scalars", "images")) -> list[Granule]:
    """Load several granules (all with moments+scalars under root if granule_ids is None)."""
    ids = granule_ids if granule_ids is not None else find_granules(root)
    return [load_granule(root, gid, parts) for gid in ids]
