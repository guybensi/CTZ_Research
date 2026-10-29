"""Leakage-safe train/validation/test splits (PLAN.md section 4.1).

CERES footprints overlap: one MODIS pixel lies in 3-5 neighbouring footprints of the same
overpass. A split is safe only if no test footprint shares pixels with a training or validation
footprint. Pixels can only be shared within one overpass, i.e. within one granule group, so the
leakage check compares footprints of the same group only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .io import parse_granule_id
from .schema import LeakageError
from .table import FootprintTable

EARTH_RADIUS_KM = 6371.0
__all__ = [
    "Fold",
    "LeakageError",
    "check_fold",
    "group_granules",
    "group_kfold",
    "holdout",
    "interleave_split",
    "load_manifest",
    "logo_folds",
    "min_distance_km",
    "pixel_reuse_fraction",
    "save_manifest",
    "spatial_block_folds",
]


@dataclass
class Fold:
    """Row indices into a FootprintTable. `leaky` marks splits that share pixels by design."""

    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    leaky: bool = False
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- granule groups
def group_granules(granule_ids: list[str], max_gap_minutes: float = 10.0,
                   sizes: dict[str, int] | None = None, min_size: int = 0) -> dict[str, str]:
    """Group granules of one overpass, then merge groups smaller than min_size footprints.

    Consecutive granules of the same satellite less than max_gap_minutes apart belong to one
    overpass (e.g. A2014090.2350 + A2014090.2355). A group with fewer than min_size footprints is
    merged into the temporally nearest group of the same satellite and day (e.g. A2014060.0245,
    9 footprints, joins A2014060.0415).

    Example:
        >>> group_granules(["A2014090.2350", "A2014090.2355", "A2014090.2240"])["A2014090.2355"]
        'A2014090.2350'

    Args:
        granule_ids: Granule names.
        max_gap_minutes: Largest start-time gap inside one overpass.
        sizes: Footprints per granule (needed only for min_size).
        min_size: Minimum footprints per group.

    Returns:
        Mapping granule id -> group label (the id of the group's earliest granule).
    """
    parsed = sorted((parse_granule_id(g) for g in granule_ids), key=lambda p: (p.satellite, p.time))
    groups: list[list] = []
    for p in parsed:
        last = groups[-1][-1] if groups else None
        if last and last.satellite == p.satellite and (p.time - last.time).total_seconds() / 60 <= max_gap_minutes:
            groups[-1].append(p)
        else:
            groups.append([p])
    if sizes and min_size > 0:
        changed = True
        while changed:
            changed = False
            for i, grp in enumerate(groups):
                if sum(sizes.get(p.name, 0) for p in grp) >= min_size:
                    continue
                cands = [(min(abs((q.time - p.time).total_seconds()) for p in grp for q in other), j)
                         for j, other in enumerate(groups)
                         if j != i and other[0].satellite == grp[0].satellite and other[0].day == grp[0].day]
                if cands:
                    j = min(cands)[1]
                    groups[j] = sorted(groups[j] + grp, key=lambda p: p.time)
                    del groups[i]
                    changed = True
                    break
    return {p.name: grp[0].name for grp in groups for p in grp}


# ---------------------------------------------------------------- fold builders
def _units_split(units: np.ndarray, rng: np.random.Generator, val_fraction: float) -> tuple[set, set]:
    units = list(units)
    rng.shuffle(units)
    n_val = max(1, round(val_fraction * len(units))) if len(units) > 1 else 0
    return set(units[n_val:]), set(units[:n_val])


def logo_folds(table: FootprintTable, key: str = "group") -> list[Fold]:
    """Leave-one-group-out folds; the next group (cyclically) is the validation set.

    Example:
        >>> # see tests/test_splits.py; with groups G1..G4: test=G1, val=G2, train=G3+G4, ...

    Raises:
        ValueError: With fewer than 3 groups (train, val and test each need one).
    """
    labels = np.asarray(table[key])
    uniq = sorted(set(labels))
    if len(uniq) < 3:
        raise ValueError(f"need at least 3 {key} values for nested LOGO, got {len(uniq)}")
    folds = []
    for i, g in enumerate(uniq):
        v = uniq[(i + 1) % len(uniq)]
        folds.append(Fold(f"logo:{g}", np.flatnonzero((labels != g) & (labels != v)),
                          np.flatnonzero(labels == v), np.flatnonzero(labels == g)))
    return folds


def group_kfold(table: FootprintTable, key: str = "day", k: int = 5, seed: int = 0,
                val_fraction: float = 0.15) -> list[Fold]:
    """K folds over whole groups (e.g. days); validation units are drawn from the training units.

    Example:
        >>> # group_kfold(table, key="day", k=5) -> 5 folds, no day on both sides

    Raises:
        ValueError: If k is larger than the number of distinct keys or smaller than 2.
    """
    labels = np.asarray(table[key])
    uniq = np.array(sorted(set(labels)), dtype=object)
    if not 2 <= k <= len(uniq):
        raise ValueError(f"k={k} invalid for {len(uniq)} distinct {key} values")
    rng = np.random.default_rng(seed)
    chunks = np.array_split(rng.permutation(uniq), k)
    folds = []
    for i, chunk in enumerate(chunks):
        rest = np.array([u for u in uniq if u not in set(chunk)], dtype=object)
        tr, va = _units_split(rest, rng, val_fraction)
        folds.append(Fold(f"{key}-kfold:{i}", np.flatnonzero(np.isin(labels, list(tr))),
                          np.flatnonzero(np.isin(labels, list(va))), np.flatnonzero(np.isin(labels, list(chunk)))))
    return folds


def holdout(table: FootprintTable, test_mask: np.ndarray, name: str = "holdout", key: str = "group",
            val_fraction: float = 0.15, seed: int = 0) -> Fold:
    """One fold with an explicit test set (e.g. year 2015, or region North15).

    Any group touched by the test set is removed from training, and validation groups are drawn
    from the remaining training groups.

    Raises:
        ValueError: If the test or training side is empty.
    """
    test_mask = np.asarray(test_mask, bool)
    labels = np.asarray(table[key])
    if not test_mask.any():
        raise ValueError("empty test set")
    rest_units = np.array(sorted(set(labels[~test_mask]) - set(labels[test_mask])), dtype=object)
    if len(rest_units) == 0:
        raise ValueError("no training groups left after removing the test groups")
    tr, va = _units_split(rest_units, np.random.default_rng(seed), val_fraction)
    return Fold(name, np.flatnonzero(np.isin(labels, list(tr)) & ~test_mask),
                np.flatnonzero(np.isin(labels, list(va)) & ~test_mask), np.flatnonzero(test_mask))


def _xyz(table: FootprintTable, rows: np.ndarray) -> np.ndarray:
    lat = np.radians(np.asarray(table["Lat"], float)[rows])
    lon = np.radians(np.asarray(table["Lon"], float)[rows])
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _chord_to_km(d):
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(d / 2, 0, 1))


def _km_to_chord(km):
    return 2 * np.sin(np.asarray(km, float) / (2 * EARTH_RADIUS_KM))


def _nearest_same_group_km(table: FootprintTable, rows: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Distance from each row to the nearest `ref` row of the same group (inf if none)."""
    out = np.full(len(rows), np.inf)
    groups = np.asarray(table["group"])
    for g in set(groups[rows]):
        r_sel = rows[groups[rows] == g]
        ref_sel = ref[groups[ref] == g]
        if len(ref_sel) == 0:
            continue
        d, _ = cKDTree(_xyz(table, ref_sel)).query(_xyz(table, r_sel))
        out[np.flatnonzero(groups[rows] == g)] = _chord_to_km(d)
    return out


def min_distance_km(table: FootprintTable, a: np.ndarray, b: np.ndarray) -> float:
    """Smallest distance between a row of `a` and a row of `b` in the same group (inf if none)."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    return float(_nearest_same_group_km(table, np.asarray(a), np.asarray(b)).min())


def spatial_block_folds(table: FootprintTable, block_km: float = 100.0, buffer_km: float = 40.0, k: int = 5,
                        seed: int = 0, val_fraction: float = 0.15, scale_with_vza: bool = True) -> list[Fold]:
    """Spatial block CV inside overpasses, with a buffer between test, validation and training.

    Blocks are block_km squares on an equirectangular grid. Training (and validation) rows closer
    than the buffer to a test row of the same group are dropped; with scale_with_vza the buffer of
    a row is buffer_km / cos(VZA), since footprints grow off-nadir. Training rows are also
    buffered from validation rows.

    Raises:
        ValueError: If there are fewer blocks than k.
    """
    lat = np.asarray(table["Lat"], float)
    lon = np.asarray(table["Lon"], float)
    y = lat * 110.57
    x = lon * 111.32 * np.cos(np.radians(np.nanmean(lat)))
    block = np.char.add(np.floor(x / block_km).astype(int).astype(str),
                        np.char.add("_", np.floor(y / block_km).astype(int).astype(str)))
    uniq = np.array(sorted(set(block)))
    if len(uniq) < k:
        raise ValueError(f"only {len(uniq)} blocks for k={k}")
    rng = np.random.default_rng(seed)
    chunks = np.array_split(rng.permutation(uniq), k)
    scale = 1 / np.cos(np.radians(np.asarray(table["VZA"], float))) if scale_with_vza and "VZA" in table else np.ones(len(table))
    folds = []
    for i, chunk in enumerate(chunks):
        test = np.flatnonzero(np.isin(block, chunk))
        rest_blocks = np.array([b for b in uniq if b not in set(chunk)])
        tr_b, va_b = _units_split(rest_blocks, rng, val_fraction)
        val = np.flatnonzero(np.isin(block, list(va_b)))
        train = np.flatnonzero(np.isin(block, list(tr_b)))
        val = val[_nearest_same_group_km(table, val, test) >= buffer_km * scale[val]]
        keep = (_nearest_same_group_km(table, train, test) >= buffer_km * scale[train]) & \
               (_nearest_same_group_km(table, train, val) >= buffer_km * scale[train])
        folds.append(Fold(f"block:{i}", train[keep], val, test, meta={"dropped_by_buffer": int((~keep).sum())}))
    return folds


def interleave_split(table: FootprintTable, period: int = 2) -> Fold:
    """Every `period`-th footprint (by index within its granule) is test: the overlap test.

    Leaky by design (test footprints share pixels with training ones); it measures
    within-scene interpolation and is never used to rank methods.
    """
    idx = np.asarray(table["index"])
    test = np.flatnonzero(idx % period == 1)
    return Fold(f"interleave:{period}", np.flatnonzero(idx % period != 1), np.array([], int), test, leaky=True)


# ---------------------------------------------------------------- leakage check
def pixel_reuse_fraction(table: FootprintTable, train: np.ndarray, test: np.ndarray, radius_km: float = 20.0,
                         scale_with_vza: bool = True, n_ring: int = 6) -> np.ndarray:
    """For each test footprint, the fraction of its area that lies inside any train footprint.

    Footprints are approximated as discs of radius radius_km (/cos VZA when scale_with_vza) around
    their centres, sampled on a polar grid. Only footprints of the same group (overpass) count.

    Example:
        >>> # identical centres -> 1.0; centres 5 radii apart -> 0.0 (see tests)

    Returns:
        Array (len(test),) in [0, 1].
    """
    train, test = np.asarray(train, int), np.asarray(test, int)
    out = np.zeros(len(test))
    if len(train) == 0 or len(test) == 0:
        return out
    vza = np.asarray(table["VZA"], float) if (scale_with_vza and "VZA" in table) else np.zeros(len(table))
    radius = radius_km / np.cos(np.radians(vza))
    rr, tt = np.meshgrid(np.sqrt((np.arange(n_ring) + 0.5) / n_ring), np.linspace(0, 2 * np.pi, 16, endpoint=False))
    offs = np.column_stack([(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel()])  # unit disc, equal-area
    groups = np.asarray(table["group"])
    lat_all = np.asarray(table["Lat"], float)
    lon_all = np.asarray(table["Lon"], float)
    for g in set(groups[test]):
        tr = train[groups[train] == g]
        if len(tr) == 0:
            continue
        tree = cKDTree(_xyz(table, tr))
        rmax = _km_to_chord(radius[tr].max())
        for j in np.flatnonzero(groups[test] == g):
            row = test[j]
            dlat = offs[:, 1] * radius[row] / 110.57
            dlon = offs[:, 0] * radius[row] / (111.32 * np.cos(np.radians(lat_all[row])))
            plat, plon = np.radians(lat_all[row] + dlat), np.radians(lon_all[row] + dlon)
            pts = np.column_stack([np.cos(plat) * np.cos(plon), np.cos(plat) * np.sin(plon), np.sin(plat)])
            covered = 0
            for p, cand in zip(pts, tree.query_ball_point(pts, rmax)):
                if cand and np.any(_chord_to_km(np.linalg.norm(tree.data[cand] - p, axis=1)) < radius[tr[cand]]):
                    covered += 1
            out[j] = covered / len(pts)
    return out


def check_fold(table: FootprintTable, fold: Fold, radius_km: float = 20.0, tol: float = 0.0) -> None:
    """Raise LeakageError if a test footprint shares pixels with a train or validation footprint.

    Folds marked leaky (the interleave test) are skipped.
    """
    if fold.leaky:
        return
    seen = np.concatenate([fold.train, fold.val]).astype(int)
    frac = pixel_reuse_fraction(table, seen, fold.test, radius_km=radius_km)
    if len(frac) and frac.max() > tol:
        raise LeakageError(f"fold {fold.name}: {int((frac > tol).sum())} test footprints share pixels "
                           f"with training/validation (max fraction {frac.max():.2f})")


# ---------------------------------------------------------------- manifest
def _fingerprint(table: FootprintTable) -> list[list]:
    g = np.asarray(table["granule"])
    order = list(dict.fromkeys(g))
    return [[x, int((g == x).sum())] for x in order]


def save_manifest(path: str | Path, table: FootprintTable, folds: list[Fold], meta: dict | None = None) -> None:
    """Freeze folds to JSON together with a fingerprint of the table they index."""
    doc = {"version": 1, "table": _fingerprint(table), "meta": meta or {},
           "folds": [{"name": f.name, "leaky": f.leaky, "meta": f.meta,
                      "train": f.train.tolist(), "val": f.val.tolist(), "test": f.test.tolist()} for f in folds]}
    Path(path).write_text(json.dumps(doc))


def load_manifest(path: str | Path, table: FootprintTable) -> tuple[list[Fold], dict]:
    """Load frozen folds; fail if the table no longer matches the one the manifest was built on.

    Raises:
        ValueError: If the table fingerprint differs.
    """
    doc = json.loads(Path(path).read_text())
    if doc["table"] != _fingerprint(table):
        raise ValueError("manifest was built on a different footprint table")
    folds = [Fold(f["name"], np.array(f["train"], int), np.array(f["val"], int), np.array(f["test"], int),
                  f["leaky"], f.get("meta", {})) for f in doc["folds"]]
    return folds, doc["meta"]
