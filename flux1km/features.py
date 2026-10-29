"""Model inputs: a Batch of footprints and MODIS-only pixel / footprint feature builders.

Everything a model sees goes through here, and every feature list passes
schema.assert_no_ceres_inputs, so CERES-derived quantities cannot reach a model by accident.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry, moments, schema
from .io import Granule
from .table import FootprintTable


@dataclass
class Batch:
    """A set of footprints gathered from one or more granules (rows of a FootprintTable)."""

    rows: np.ndarray
    label: np.ndarray                      # Fsw (W m^-2), the CERES footprint flux
    scalars: dict[str, np.ndarray]
    moments: dict[str, np.ndarray] | None
    images: dict[str, np.ndarray] | None
    truth: np.ndarray | None               # synthetic only: (N, H, W) true flux
    dist_km: np.ndarray | None             # synthetic only: distance to cloud
    group: np.ndarray
    weight: np.ndarray

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def has_images(self) -> bool:
        return self.images is not None


def make_batch(granules: list[Granule], table: FootprintTable, rows: np.ndarray,
               weights: np.ndarray | None = None) -> Batch:
    """Gather the given table rows from their granules into one Batch.

    Image, moment and truth arrays are kept only if every involved granule has them.

    Raises:
        KeyError: If a row refers to a granule that is not in `granules`.
    """
    rows = np.asarray(rows, int)
    if len(rows) == 0:
        return _empty_batch(granules[0], table)
    by_id = {g.granule_id: g for g in granules}
    gids, idx = np.asarray(table["granule"])[rows], np.asarray(table["index"])[rows]
    used = list(dict.fromkeys(gids.tolist()))
    missing = [g for g in used if g not in by_id]
    if missing:
        raise KeyError(f"granules not loaded: {missing}")

    def gather(getter):
        parts = [getter(by_id[g]) for g in used]
        if any(p is None for p in parts):
            return None
        keys = set.intersection(*(set(p) for p in parts))
        out = {}
        for k in keys:
            if k in ("SWwavlngs", "LWwavlngs"):
                continue
            out[k] = np.concatenate([parts[i][k][idx[gids == g]] for i, g in enumerate(used)])
        return out

    order = np.concatenate([np.flatnonzero(gids == g) for g in used]) if len(rows) else np.array([], int)
    inv = np.argsort(order)  # gather() concatenates per granule; restore the requested row order
    scal = {k: v[inv] for k, v in gather(lambda g: g.scalars).items()}
    mom = gather(lambda g: g.moments)
    img = gather(lambda g: g.images)
    tru = gather(lambda g: g.truth)
    return Batch(
        rows=rows, label=np.asarray(scal["Fsw"], float), scalars=scal,
        moments=None if mom is None else {k: v[inv] for k, v in mom.items()},
        images=None if img is None else {k: v[inv] for k, v in img.items()},
        truth=None if tru is None else tru["flux"][inv].astype(float),
        dist_km=None if tru is None else tru["dist_km"][inv].astype(float),
        group=np.asarray(table["group"])[rows],
        weight=np.ones(len(rows)) if weights is None else np.asarray(weights, float),
    )


def _empty_batch(template: Granule, table: FootprintTable) -> Batch:
    """A zero-row batch with the same keys and trailing shapes as `template`."""

    def empty(d):
        return None if d is None else {k: v[:0] for k, v in d.items() if k not in ("SWwavlngs", "LWwavlngs")}

    tru = empty(template.truth)
    scal = empty(template.scalars)
    return Batch(rows=np.array([], int), label=np.asarray(scal["Fsw"], float), scalars=scal,
                 moments=empty(template.moments), images=empty(template.images),
                 truth=None if tru is None else tru["flux"].astype(float),
                 dist_km=None if tru is None else tru["dist_km"].astype(float),
                 group=np.array([], dtype=object), weight=np.array([]))


def _channel_ids(iset: schema.InputSet):
    return ([schema.channel_index(c)[1] for c in iset.sw], [schema.channel_index(c)[1] for c in iset.lw])


def pixel_features(batch: Batch, input_set: str = "F5") -> tuple[np.ndarray, list[str]]:
    """Per-pixel inputs (N, F, H, W): selected SW/LW radiances, encoded geometry, optional glint.

    Example:
        >>> # X, names = pixel_features(batch, "F2"); X.shape == (N, 7 + 4, 27, 55)

    Raises:
        ValueError: If the batch has no images.
    """
    if not batch.has_images:
        raise ValueError("pixel features need images; this batch is moments-only")
    iset = schema.input_set(input_set)
    sw_i, lw_i = _channel_ids(iset)
    views = batch.images["Imviews"].astype(float)
    sza, vza, raa = views[:, 0], views[:, 1], views[:, 2]
    blocks = [batch.images["ImSW"][:, sw_i].astype(float)]
    names = [f"SW_{c}" for c in iset.sw]
    if lw_i:
        blocks.append(batch.images["ImLW"][:, lw_i].astype(float))
        names += [f"LW_{c}" for c in iset.lw]
    blocks.append(np.moveaxis(geometry.encode_angles(sza, vza, raa), -1, 1))
    names += ["cosSZA", "cosVZA", "sinRAA", "cosRAA"]
    if iset.use_glint:
        blocks.append(geometry.glint_angle(sza, vza, raa)[:, None] / 90.0)
        names.append("glint")
    schema.assert_no_ceres_inputs(names)
    return np.concatenate(blocks, axis=1), names


def footprint_geometry(batch: Batch, use_glint: bool) -> tuple[np.ndarray, list[str]]:
    """Footprint-level encoded geometry (N, 4 or 5) from the scalars SZA/VZA/RAA."""
    s = batch.scalars
    enc = geometry.encode_angles(s["SZA"], s["VZA"], s["RAA"])
    names = ["cosSZA", "cosVZA", "sinRAA", "cosRAA"]
    if use_glint:
        enc = np.column_stack([enc, geometry.glint_angle(s["SZA"], s["VZA"], s["RAA"]) / 90.0])
        names.append("glint")
    return enc, names


def footprint_moment_stack(batch: Batch, input_set: str = "F5"):
    """(mean, std, skew, kurt) arrays (N, C) for the channels of an input set, plus channel names.

    Raises:
        ValueError: If the batch has no moments.
    """
    if batch.moments is None:
        raise ValueError("batch has no moments")
    iset = schema.input_set(input_set)
    sw_i, lw_i = _channel_ids(iset)
    out = []
    for stat in ("m", "std", "skew", "kurt"):
        parts = [batch.moments[f"{stat}SWspec"][:, sw_i]]
        if lw_i:
            parts.append(batch.moments[f"{stat}LWspec"][:, lw_i])
        out.append(np.concatenate(parts, axis=1).astype(float))
    names = [f"SW_{c}" for c in iset.sw] + [f"LW_{c}" for c in iset.lw]
    return out, names


def footprint_features(batch: Batch, input_set: str = "F5", stats: tuple[str, ...] = ("m", "std")):
    """Tabular footprint features for M1-style models: chosen moments + geometry (+ glint).

    Returns:
        (X (N, F), names).
    """
    (m, s, sk, ku), ch = footprint_moment_stack(batch, input_set)
    table = {"m": m, "std": s, "skew": sk, "kurt": ku}
    blocks, names = [], []
    for st in stats:
        blocks.append(np.nan_to_num(table[st]))
        names += [f"{st}_{c}" for c in ch]
    geo, gnames = footprint_geometry(batch, schema.input_set(input_set).use_glint)
    blocks.append(geo)
    names += gnames
    schema.assert_no_ceres_inputs(names)
    return np.concatenate(blocks, axis=1), names


def footprint_poly_features(batch: Batch, input_set: str, degree: int, excess_kurtosis: bool = False):
    """M2 footprint features: raw moments E[x^k] (k <= degree) per channel + footprint geometry.

    Column order matches pixel_poly_features, so a linear model fitted on these features and
    applied to pixel_poly_features is exactly a pixel model trained with the aggregation loss.
    """
    (m, s, sk, ku), ch = footprint_moment_stack(batch, input_set)
    poly = moments.footprint_polynomial_features(m, s, np.nan_to_num(sk), np.nan_to_num(ku), degree, excess_kurtosis)
    geo, gnames = footprint_geometry(batch, schema.input_set(input_set).use_glint)
    names = [f"{c}^{k + 1}" for c in ch for k in range(degree)] + gnames
    schema.assert_no_ceres_inputs(names)
    return np.concatenate([poly, geo], axis=1), names


def pixel_poly_features(batch: Batch, input_set: str, degree: int):
    """Per-pixel counterpart of footprint_poly_features: (N, F, H, W)."""
    X, names = pixel_features(batch, input_set)
    n_ch = len(schema.input_set(input_set).sw) + len(schema.input_set(input_set).lw)
    poly = moments.pixel_polynomial_features(X[:, :n_ch], degree)
    geo = X[:, n_ch:]
    pnames = [f"{c}^{k + 1}" for c in names[:n_ch] for k in range(degree)] + names[n_ch:]
    return np.concatenate([poly, geo], axis=1), pnames
