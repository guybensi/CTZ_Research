"""One flat table of all footprints across granules: the unit that splits and sampling act on."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from . import geometry, moments
from .io import Granule, parse_granule_id

_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
_SCALAR_COLUMNS = ("Fsw", "Flw", "SZA", "VZA", "RAA", "Lat", "Lon", "CF", "Clrf", "glintf", "landf", "meanCTH", "stdCTH")


@dataclass
class FootprintTable:
    """Column arrays of equal length N, one row per footprint.

    Columns: granule (str), index (row within its granule), group (granule group label),
    day ("A2014-03-01"), minutes (since 2000-01-01), year, plus the scalar columns present in the
    data, "glint" (deg, from footprint-mean geometry) and "cv" (Gao homogeneity, SW bands) when
    moments are available.
    """

    columns: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.columns["granule"])

    def __getitem__(self, key: str) -> np.ndarray:
        return self.columns[key]

    def __contains__(self, key: str) -> bool:
        return key in self.columns

    def rows_of(self, granule_id: str) -> np.ndarray:
        """Indices of the rows that belong to one granule."""
        return np.flatnonzero(self.columns["granule"] == granule_id)


def build_table(granules: list[Granule], groups: dict[str, str] | None = None) -> FootprintTable:
    """Stack footprints of several granules into one table.

    Example:
        >>> import numpy as np
        >>> from flux1km.synthetic import make_synthetic_granule
        >>> g = make_synthetic_granule("A2014060.0245", np.random.default_rng(0), n_rows=1, n_cols=2)
        >>> t = build_table([g]); len(t), t["granule"][0], t["group"][0]
        (2, 'A2014060.0245', 'A2014060.0245')

    Args:
        granules: Loaded granules.
        groups: Optional granule -> group label (see splits.group_granules); default: own id.

    Returns:
        FootprintTable.
    """
    cols: dict[str, list] = {k: [] for k in ("granule", "index", "group", "day", "minutes", "year")}
    present = [c for c in _SCALAR_COLUMNS if all(c in g.scalars for g in granules)]
    cols.update({c: [] for c in present})
    with_cv = all(g.moments is not None for g in granules)
    if with_cv:
        cols["cv"] = []
    for g in granules:
        n = g.n_footprints
        gid = parse_granule_id(g.granule_id)
        cols["granule"].append(np.full(n, g.granule_id, dtype=object))
        cols["index"].append(np.arange(n))
        cols["group"].append(np.full(n, (groups or {}).get(g.granule_id, g.granule_id), dtype=object))
        cols["day"].append(np.full(n, gid.day, dtype=object))
        cols["minutes"].append(np.full(n, (gid.time - _EPOCH).total_seconds() / 60.0))
        cols["year"].append(np.full(n, gid.time.year))
        for c in present:
            cols[c].append(np.asarray(g.scalars[c], dtype=float))
        if with_cv:
            cols["cv"].append(moments.homogeneity_cv(g.moments["mSWspec"], g.moments["stdSWspec"]))
    out = {k: np.concatenate(v) if v else np.array([]) for k, v in cols.items()}
    if {"SZA", "VZA", "RAA"} <= out.keys():
        out["glint"] = geometry.glint_angle(out["SZA"], out["VZA"], out["RAA"])
    return FootprintTable(out)
