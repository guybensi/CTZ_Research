"""Training-data selection strategies S0-S7 and how similar they are (PLAN.md section 5).

S4 (Fsw >= 400) was dropped: it removes the clear and broken scenes where the CTZ lives.
The other IDs are kept stable.

Sampling changes only the training set; the test population stays fixed. Each strategy returns
(mask, weights) over table rows: mask says which rows are used, weights are loss weights.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import numpy as np
from scipy import stats

from .table import FootprintTable

REGIMES = ("clear", "broken", "overcast")
# Placeholder ADM noise per regime (W m^-2), Su et al. 2015b orders of magnitude; verify in E0.
DEFAULT_SIGMA = {"clear": 2.0, "broken": 10.0, "overcast": 6.0}


def regime(cf, clear_max: float = 5.0, overcast_min: float = 95.0) -> np.ndarray:
    """Cloud regime from cloud fraction in percent: clear (<5), broken, overcast (>=95), unknown (NaN).

    Example:
        >>> regime(np.array([0.0, 50.0, 100.0])).tolist()
        ['clear', 'broken', 'overcast']
    """
    cf = np.asarray(cf, float)
    out = np.full(cf.shape, "unknown", dtype=object)
    out[cf < clear_max] = "clear"
    out[(cf >= clear_max) & (cf < overcast_min)] = "broken"
    out[cf >= overcast_min] = "overcast"
    return out


def _valid(t: FootprintTable) -> np.ndarray:
    return np.isfinite(np.asarray(t["Fsw"], float))


def _s0(t, **_):
    m = _valid(t)
    return m, m.astype(float)


def _s1(t, sza_max: float = 70.0, vza_max: float = 55.0, **_):
    m = _valid(t) & (t["SZA"] < sza_max) & (t["VZA"] < vza_max)
    return m, m.astype(float)


def _s2(t, min_glint: float = 25.0, **_):
    m = _valid(t) & (np.asarray(t["glint"], float) >= min_glint)
    return m, m.astype(float)


def _s3(t, method: str = "cth", cth_max: float = 3000.0, std_cth_max: float = 1000.0, cv_max: float = 0.2, **_):
    if method == "cth":
        m = _valid(t) & (t["meanCTH"] < cth_max) & (t["stdCTH"] < std_cth_max)
        return m, m.astype(float)
    if method == "cv":
        cv = np.asarray(t["cv"], float)
        m = _valid(t) & (cv < cv_max)
        with np.errstate(divide="ignore"):
            w = np.where(m, 1.0 / np.maximum(cv, 1e-6), 0.0)  # Gao: purer samples weigh more
        return m, w
    raise ValueError(f"unknown homogeneity method {method!r}")


def _s5(t, **_):
    m = _valid(t)
    reg = regime(t["CF"])
    w = np.zeros(len(m))
    present = [r for r in REGIMES if np.any(m & (reg == r))]
    for r in present:
        sel = m & (reg == r)
        w[sel] = 1.0 / sel.sum()
    w *= m.sum() / max(w.sum(), 1e-12)
    return m, w


def _s6(t, period: int = 2, **_):
    m = _valid(t) & (np.asarray(t["index"]) % period == 0)
    return m, m.astype(float)


def _s7(t, sigma: dict | None = None, **_):
    sigma = sigma or DEFAULT_SIGMA
    m = _valid(t)
    reg = regime(t["CF"])
    s = np.array([sigma.get(r, np.nan) for r in reg], float)
    w = np.where(m & np.isfinite(s), 1.0 / s**2, 0.0)
    return m, w


STRATEGIES: dict[str, Callable] = {"S0": _s0, "S1": _s1, "S2": _s2, "S3": _s3, "S5": _s5, "S6": _s6, "S7": _s7}


def apply(name: str, table: FootprintTable, rows: np.ndarray | None = None, **params) -> tuple[np.ndarray, np.ndarray]:
    """Apply strategy `name` to the table (or to the given rows only).

    Example:
        >>> # mask, weights = apply("S7", table, rows=fold.train)

    Args:
        name: "S0".."S7" except the dropped "S4" (see PLAN.md section 5).
        table: Footprint table.
        rows: Optional row indices (e.g. a fold's training rows); outputs are aligned with them.
        **params: Strategy parameters (sza_max, min_glint, method, period, sigma, ...).

    Returns:
        (mask, weights): bool and float arrays; weights are 0 where mask is False.

    Raises:
        KeyError: For an unknown strategy.
    """
    if name not in STRATEGIES:
        raise KeyError(f"unknown sampling strategy {name!r}")
    m, w = STRATEGIES[name](table, **params)
    w = np.where(m, w, 0.0)
    if rows is not None:
        return m[rows], w[rows]
    return m, w


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """|a and b| / |a or b| for boolean masks (1.0 if both empty).

    Example:
        >>> jaccard(np.array([True, False]), np.array([True, True]))
        0.5
    """
    union = np.sum(a | b)
    return 1.0 if union == 0 else float(np.sum(a & b) / union)


def ks_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (NaNs dropped).

    Example:
        >>> ks_distance(np.arange(5.0), np.arange(5.0))
        0.0
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(stats.ks_2samp(x[np.isfinite(x)], y[np.isfinite(y)]).statistic)


def mmd(x: np.ndarray, y: np.ndarray, bandwidth: float | None = None, max_n: int = 2000, seed: int = 0) -> float:
    """Squared maximum mean discrepancy with an RBF kernel (biased V-statistic), NaN rows dropped.

    Columns are standardised on the pooled sample; the bandwidth defaults to the median pairwise
    distance. Large samples are subsampled to max_n rows each.

    Example:
        >>> a = np.random.default_rng(0).normal(size=(200, 2))
        >>> mmd(a, a) == 0.0, mmd(a, a + 3.0) > 0.5
        (True, True)

    Args:
        x, y: Arrays (n, d) or (n,).
    """
    x = np.atleast_2d(np.asarray(x, float).T).T
    y = np.atleast_2d(np.asarray(y, float).T).T
    x, y = x[np.all(np.isfinite(x), 1)], y[np.all(np.isfinite(y), 1)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    if len(x) > max_n:
        x = x[rng.choice(len(x), max_n, replace=False)]
    if len(y) > max_n:
        y = y[rng.choice(len(y), max_n, replace=False)]
    pooled = np.vstack([x, y])
    mu, sd = pooled.mean(0), pooled.std(0)
    sd[sd == 0] = 1.0
    x, y, pooled = (x - mu) / sd, (y - mu) / sd, (pooled - mu) / sd

    def sq(a, b):
        return np.maximum((a**2).sum(1)[:, None] + (b**2).sum(1)[None] - 2 * a @ b.T, 0.0)

    if bandwidth is None:
        d = np.sqrt(sq(pooled, pooled))
        bandwidth = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    g = 1.0 / (2 * bandwidth**2)
    return float(np.exp(-g * sq(x, x)).mean() + np.exp(-g * sq(y, y)).mean() - 2 * np.exp(-g * sq(x, y)).mean())


def rank_agreement(scores_a, scores_b) -> float:
    """Kendall tau between two method rankings (e.g. RMSE per method under two strategies).

    Example:
        >>> rank_agreement([1, 2, 3], [1, 2, 3])
        1.0
    """
    return float(stats.kendalltau(scores_a, scores_b).statistic)


def compare_strategies(table: FootprintTable, names: list[str], columns=("Fsw", "CF", "SZA", "VZA", "glint"),
                       reference: str = "S0", **params) -> dict:
    """Pairwise Jaccard of the selected sets, and KS (per column) and MMD (joint) distance to the reference.

    Returns:
        {"jaccard": {(a, b): float}, "ks_vs_reference": {(name, column): float},
         "mmd_vs_reference": {name: float}, "n": {name: int}}
    """
    masks = {n: apply(n, table, **params.get(n, {}))[0] for n in set(names) | {reference}}
    out = {"jaccard": {}, "ks_vs_reference": {}, "mmd_vs_reference": {}, "n": {n: int(masks[n].sum()) for n in names}}
    cols = [c for c in columns if c in table]
    if cols:
        joint = np.column_stack([np.asarray(table[c], float) for c in cols])
        for n in names:
            out["mmd_vs_reference"][n] = mmd(joint[masks[n]], joint[masks[reference]])
    for a, b in combinations(names, 2):
        out["jaccard"][(a, b)] = jaccard(masks[a], masks[b])
    for n in names:
        for c in columns:
            if c in table:
                col = np.asarray(table[c], float)
                out["ks_vs_reference"][(n, c)] = ks_distance(col[masks[n]], col[masks[reference]])
    return out
