"""Read and write granule triplets (moments/, scalars/, images/ NPZ files with the same name).

Layout (as in data/paired/<region>/...):
    <root>/moments/A2014060.0245.npz   mSWspec, stdSWspec, skewSWspec, kurtSWspec, (same for LW), SWwavlngs, LWwavlngs
    <root>/scalars/A2014060.0245.npz   Fsw, Flw, Isw, ..., SZA, VZA, RAA, Lat, Lon, CF, ...
    <root>/images/A2014060.0245.npz    Imviews (N,3,H,W), ImSW (N,11,H,W), ImLW (N,16,H,W), FOVorientation (N,)
                                       optional row0, col0 (N,): patch corner in the granule grid (synthetic data;
                                       an in-house footprint index would provide them for real data; used by M7)
    <root>/truth/A2014060.0245.npz     synthetic data only: known 1 km truth and helper maps
Footprints are in the same order in every file of a granule.
"""
from __future__ import annotations

import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
        ('Aqua', '2014-03-01T02:45:00+00:00', 'A2014-03-01')

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
    t = datetime(int(year), 1, 1, tzinfo=timezone.utc) + timedelta(days=int(doy) - 1, hours=int(hh), minutes=int(mm))
    return GranuleId(stem, "Aqua" if sat == "A" else "Terra", t)


class LazyNpz(Mapping):
    """The arrays of one .npz file, each read from disk only when it is accessed and never kept.

    A granule's images are about 180 KB per footprint, so holding the images of every granule in memory
    limits a run to a few dozen granules. With LazyNpz the images stay on disk; make_batch reads a granule's
    arrays only while it gathers that granule's footprints. Shapes come from the .npy headers, without
    reading the data.

    Example:
        >>> import tempfile, os
        >>> p = os.path.join(tempfile.mkdtemp(), "g.npz"); np.savez(p, a=np.zeros((3, 2)))
        >>> z = LazyNpz(p); z.shapes["a"], z["a"].sum(), list(z)
        ((3, 2), np.float64(0.0), ['a'])

    Args:
        path: An .npz file (np.savez or np.savez_compressed).

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: On access to a key that is not in the file.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, np.dtype] = {}
        with zipfile.ZipFile(self.path) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    version = np.lib.format.read_magic(f)
                    shape, _, dtype = (np.lib.format.read_array_header_1_0(f) if version == (1, 0)
                                       else np.lib.format.read_array_header_2_0(f))
                self.shapes[name.removesuffix(".npy")] = tuple(shape)
                self.dtypes[name.removesuffix(".npy")] = np.dtype(dtype)

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self.shapes:
            raise KeyError(key)
        with np.load(self.path, allow_pickle=False) as f:
            return f[key]

    def __iter__(self):
        return iter(self.shapes)

    def __len__(self) -> int:
        return len(self.shapes)

    def __repr__(self) -> str:
        return f"LazyNpz({str(self.path)!r}, keys={list(self.shapes)})"


def part_shape(part: Mapping, key: str) -> tuple[int, ...]:
    """Shape of part[key] without reading it when the part is a LazyNpz."""
    return part.shapes[key] if isinstance(part, LazyNpz) else np.shape(part[key])


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
            d = getattr(self, part) or {}
            for key in d:
                if key in ("SWwavlngs", "LWwavlngs"):
                    continue
                shape = part_shape(d, key)
                if len(shape) == 0 or shape[0] != n:
                    raise ValueError(f"{self.granule_id}: {part}/{key} has {shape}, expected {n} rows")


def find_granules(root: str | Path, require: tuple[str, ...] = ("moments", "scalars")) -> list[str]:
    """Sorted granule names present in every required sub-directory of root.

    Example:
        >>> find_granules("/nonexistent")
        []
    """
    root = Path(root)
    sets = [{p.name.removesuffix(".npz") for p in (root / part).glob("*.npz")} for part in require]
    return sorted(set.intersection(*sets)) if sets else []


def load_granule(root: str | Path, granule_id: str, parts: tuple[str, ...] = ("moments", "scalars", "images"),
                 lazy_images: bool = False) -> Granule:
    """Load the requested parts of one granule and validate row counts.

    Missing optional parts ("images", "truth") are left as None; "scalars" is required.
    With lazy_images, `images` is a LazyNpz: the arrays stay on disk until a batch reads them.

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
        if part == "images" and lazy_images:
            loaded[part] = LazyNpz(path)
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
                 parts: tuple[str, ...] = ("moments", "scalars", "images"), lazy_images: bool = False) -> list[Granule]:
    """Load several granules (all with moments+scalars under root if granule_ids is None); see load_granule."""
    ids = granule_ids if granule_ids is not None else find_granules(root)
    return [load_granule(root, gid, parts, lazy_images) for gid in ids]
