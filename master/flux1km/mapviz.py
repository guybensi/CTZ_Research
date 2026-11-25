"""1 km map figures of every map method: the visual check behind the PLAN.md section 6 gates.

    python3 -m flux1km.mapviz --demo
    python3 -m flux1km.mapviz --data-root DIR --manifest split.json --fold day-kfold:0 --psf resources/ceres_psf.csv \\
        --params-from docs/results/2008_Pacific_img6/summary.json --out viz --cache-dir /tmp/ctz_viz
    python3 -m flux1km.mapviz --replot viz/run_<time>          # redraw from the saved maps, no training

What it does:
1. Fit every map method once on one fold's train set (validation for early stopping), with the hyper-parameters
   the benchmark chose for that fold (--params-from summary.json), or the first grid entry.
2. In a few test granules of that fold, choose a window of footprints (default 150 x 500 pixels) that holds clear,
   broken and overcast footprints, predict the 1 km maps there and save them (maps_<granule>.npz).
3. Draw five figures per granule:
   fig1_scene      MODIS 0.66 um with the cloud masks, what CERES sees, and every method's 1 km map (mosaic of the
                   overlapping footprint patches, averaged where they overlap);
   fig2_footprint  one card per regime (clear / broken / overcast): the clouds in the footprint, its cloud fraction
                   from each mask, the PSF, each method's map and its PSF-weighted mean against CERES;
   fig3_scatter    every footprint: PSF-weighted mean of each map against CERES, coloured by cloud fraction;
   fig4_ctz        mean flux of clear pixels against distance from cloud, one panel per method, for each mask;
   fig5_masks      the cloud masks (MYD35 relaxed / strict, 0.66 um brightness proxy) and their footprint cloud
                   fractions against CERES's own (100 - Clrf).

Cloud masks: MYD35 needs the images' ImCloud (scripts/build_dataset.py since 25.11.2026); placing the patches
needs row0/col0 (same). Without them only the brightness proxy is drawn, and without row0/col0 there is no scene.
Matplotlib is imported only when drawing.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import benchmark, io, splits, synthetic
from .features import Batch, make_batch
from .models import MODELS, get_model, predict_map_chunked
from .table import build_table

MASKS = ("MYD35 relaxed", "MYD35 strict", "0.66 um proxy")
REGIME_TARGETS = {"clear": 2.0, "broken": 45.0, "overcast": 98.0}   # CF (%) of the footprint shown per regime
DIST_BINS = (0.5, 1.5, 2.5, 4, 6, 9, 13, 20, 30)                    # km, flux-vs-distance bins
SERIES = ("#257de0", "#f47230")                                     # categorical slots 1-2 (relaxed, strict)


def log(msg: str) -> None:
    """One progress line with time and memory, flushed at once (the run usually writes to a file under nohup)."""
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}  {benchmark.memory_status()}", flush=True)


# ============================================================ data helpers
def cloud_masks(images: dict, vis_reflectance: np.ndarray, proxy_threshold: float = 0.2) -> dict[str, np.ndarray]:
    """Per-pixel cloud masks (N, H, W) bool: MYD35 relaxed / strict from ImCloud when present, and the 0.66 um proxy.

    Example:
        >>> conf = np.array([[[0, 1, 2, 3]]], np.uint8); r = np.array([[[0.5, 0.1, 0.1, 0.1]]])
        >>> {k: v[0, 0].tolist() for k, v in cloud_masks({"ImCloud": conf}, r).items()}
        {'MYD35 relaxed': [True, False, False, False], 'MYD35 strict': [True, True, False, False], '0.66 um proxy': [True, False, False, False]}

    Args:
        images: The footprints' image arrays; ImCloud (uint8 MYD35 confidence, 0 cloudy .. 3 clear) is optional.
        vis_reflectance: (N, H, W) 0.66 um reflectance.
        proxy_threshold: Reflectance above which the proxy calls a pixel cloudy (the benchmark's bright_mask).

    Returns:
        {mask name: (N, H, W) bool}, MYD35 masks only when ImCloud is present.
    """
    out = {}
    if "ImCloud" in images:
        conf = np.asarray(images["ImCloud"])
        out["MYD35 relaxed"] = conf == 0
        out["MYD35 strict"] = conf <= 1
    out["0.66 um proxy"] = np.asarray(vis_reflectance) > proxy_threshold
    return out


def cloud_fraction(mask: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Cloud fraction (%) of each footprint, plain or weighted (PSF weights (N, H, W) or shared (H, W)).

    Example:
        >>> m = np.array([[[True, False], [False, False]]]); w = np.array([[0.7, 0.1], [0.1, 0.1]])
        >>> cloud_fraction(m).tolist(), cloud_fraction(m, w).round(6).tolist()
        ([25.0], [70.0])
    """
    m = np.asarray(mask, float)
    if weights is None:
        return 100.0 * m.mean(axis=(1, 2))
    w = np.broadcast_to(np.clip(weights, 0, None), m.shape)
    return 100.0 * (m * w).sum(axis=(1, 2)) / np.maximum(w.sum(axis=(1, 2)), 1e-12)


def choose_window(row0, col0, cf, patch=(27, 55), size=(150, 500), max_candidates: int = 400) -> tuple[int, int]:
    """Top-left corner of a size window that holds many whole footprints from as many cloud regimes as possible.

    Score of a window: 10000 x (number of regimes among clear < 5 % <= broken < 95 % <= overcast with a footprint
    inside) + number of footprints inside. Candidate corners are footprint corners (at most max_candidates).

    Example:
        >>> r0 = np.array([0, 0, 100]); c0 = np.array([0, 10, 0]); cf = np.array([0.0, 50.0, 99.0])
        >>> choose_window(r0, c0, cf, size=(40, 80))
        (0, 0)

    Args:
        row0, col0: Patch corners of the footprints in the granule grid.
        cf: Cloud fraction (%) per footprint.
        patch: Patch shape (H, W).
        size: Window shape (rows, cols).
        max_candidates: Candidate corners tried (evenly spread over the footprints).

    Returns:
        (row, col) of the window's corner.
    """
    r0, c0, cf = (np.asarray(a) for a in (row0, col0, cf))
    if len(r0) == 0:
        raise ValueError("no footprints to place")
    regime = np.where(cf < 5, 0, np.where(cf < 95, 1, 2))
    idx = np.unique(np.linspace(0, len(r0) - 1, min(max_candidates, len(r0))).round().astype(int))
    best, best_score = (int(r0.min()), int(c0.min())), -1
    for i in idx:
        cr, cc = int(r0[i]), int(c0[i])
        for top, left in ((cr, cc), (cr - size[0] // 2, cc - size[1] // 2)):
            top, left = max(top, int(r0.min())), max(left, int(c0.min()))
            inside = (r0 >= top) & (c0 >= left) & (r0 + patch[0] <= top + size[0]) & (c0 + patch[1] <= left + size[1])
            score = 10000 * len(set(regime[inside].tolist())) + int(inside.sum())
            if score > best_score:
                best, best_score = (top, left), score
    return best


def footprints_in_window(row0, col0, corner, patch=(27, 55), size=(150, 500)) -> np.ndarray:
    """Bool mask of the footprints whose whole patch lies inside the window.

    Example:
        >>> footprints_in_window([0, 130], [0, 0], (0, 0), size=(150, 100)).tolist()
        [True, False]
    """
    r0, c0 = np.asarray(row0), np.asarray(col0)
    top, left = corner
    return (r0 >= top) & (c0 >= left) & (r0 + patch[0] <= top + size[0]) & (c0 + patch[1] <= left + size[1])


def mosaic(patches: np.ndarray, row0, col0, corner=(0, 0), shape=None) -> np.ndarray:
    """Place (N, H, W) patches on a grid at their corners (relative to `corner`); mean where they overlap.

    Pixels no patch covers are NaN. NaN pixels of a patch do not count.

    Example:
        >>> p = np.stack([np.ones((1, 2)), 3 * np.ones((1, 2))])
        >>> mosaic(p, [0, 0], [0, 1]).tolist()
        [[1.0, 2.0, 3.0]]

    Args:
        patches: (N, H, W) values (bool masks give the fraction of overlapping patches that are True).
        row0, col0: Patch corners in the granule grid.
        corner: Grid corner in the granule grid (the window's top-left).
        shape: Grid shape; default just large enough for every patch.

    Returns:
        Float grid.
    """
    p = np.asarray(patches, float)
    r = np.asarray(row0, int) - corner[0]
    c = np.asarray(col0, int) - corner[1]
    h, w = p.shape[1:]
    if shape is None:
        shape = (int(r.max()) + h, int(c.max()) + w) if len(p) else (0, 0)
    if len(p) and (r.min() < 0 or c.min() < 0 or r.max() + h > shape[0] or c.max() + w > shape[1]):
        raise ValueError("a patch falls outside the grid")
    tot, cnt = np.zeros(shape), np.zeros(shape)
    for k in range(len(p)):
        ok = np.isfinite(p[k])
        tot[r[k]:r[k] + h, c[k]:c[k] + w] += np.where(ok, p[k], 0.0)
        cnt[r[k]:r[k] + h, c[k]:c[k] + w] += ok
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def paint_footprints(values, row0, col0, corner, shape, patch=(27, 55)) -> np.ndarray:
    """Each pixel gets the value of the footprint whose patch centre is nearest (what CERES "sees"); NaN off the patches.

    Example:
        >>> paint_footprints([1.0, 5.0], [0, 0], [0, 2], (0, 0), (1, 4), patch=(1, 2)).tolist()
        [[1.0, 1.0, 5.0, 5.0]]
    """
    v = np.asarray(values, float)
    cy = np.asarray(row0) - corner[0] + (patch[0] - 1) / 2
    cx = np.asarray(col0) - corner[1] + (patch[1] - 1) / 2
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    out = np.full(shape, np.nan)
    covered = np.isfinite(mosaic(np.ones((len(v),) + patch), row0, col0, corner, shape))
    pts = np.flatnonzero(covered.ravel())
    for start in range(0, len(pts), 20000):           # chunks keep the distance matrix small
        sel = pts[start:start + 20000]
        d = (yy.ravel()[sel, None] - cy) ** 2 * 4.0 + (xx.ravel()[sel, None] - cx) ** 2   # footprints are 2:1
        out.ravel()[sel] = v[np.argmin(d, axis=1)]
    return out


def clear_flux_vs_distance(flux_grid, cloud_grid, bins=DIST_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Mean flux of clear pixels in distance-from-cloud bins, on a mosaic.

    A pixel counts only if its nearest cloud is closer than the edge of the covered area (otherwise a cloud
    just outside the window could be nearer). Covered = finite flux and finite cloud value.

    Example:
        >>> f = np.array([[500.0, 90.0, 80.0, 70.0, 60.0, 50.0]]); c = np.array([[1.0, 0, 0, 0, 0, 0]])
        >>> clear_flux_vs_distance(f, c, bins=(0.5, 1.5, 2.5))[1].tolist()
        [90.0, 80.0]

    Returns:
        (bin centres in km, mean flux per bin; NaN where a bin is empty).
    """
    f, c = np.asarray(flux_grid, float), np.asarray(cloud_grid, float)
    covered = np.isfinite(f) & np.isfinite(c)
    cloud = covered & (c >= 0.5)
    bins = np.asarray(bins, float)
    centres = 0.5 * (bins[:-1] + bins[1:])
    if not cloud.any():
        return centres, np.full(len(centres), np.nan)
    d_cloud = ndimage.distance_transform_edt(~cloud)
    d_edge = ndimage.distance_transform_edt(covered)
    ok = covered & ~cloud & (d_cloud <= d_edge)
    out = np.full(len(centres), np.nan)
    for i in range(len(centres)):
        sel = ok & (d_cloud >= bins[i]) & (d_cloud < bins[i + 1])
        if sel.any():
            out[i] = f[sel].mean()
    return centres, out


def pick_footprints(cf, targets=REGIME_TARGETS) -> dict[str, int]:
    """Index of the footprint closest to each regime's target cloud fraction (only regimes that are present).

    Example:
        >>> pick_footprints([0.0, 40.0, 60.0])
        {'clear': 0, 'broken': 1}
    """
    cf = np.asarray(cf, float)
    regime = np.where(cf < 5, "clear", np.where(cf < 95, "broken", "overcast"))
    out = {}
    for name, target in targets.items():
        idx = np.flatnonzero((regime == name) & np.isfinite(cf))
        if len(idx):
            out[name] = int(idx[np.argmin(np.abs(cf[idx] - target))])
    return out


def load_params(summary_path: str | Path, fold_name: str) -> dict[str, dict]:
    """The hyper-parameters the benchmark chose for each model on one fold (summary.json "folds" -> "params").

    The training operator and PSF path are dropped: the caller sets them for this run.

    Example:
        >>> import tempfile, os
        >>> p = os.path.join(tempfile.mkdtemp(), "s.json")
        >>> _ = open(p, "w").write('{"folds": [{"name": "f0", "params": {"M5": {"lr": 0.001, "psf_path": "x"}}}]}')
        >>> load_params(p, "f0")
        {'M5': {'lr': 0.001}}

    Raises:
        KeyError: If the summary has no fold of that name.
    """
    doc = json.loads(Path(summary_path).read_text())
    for f in doc.get("folds", []):
        if f.get("name") == fold_name:
            return {m: {k: v for k, v in p.items() if k not in ("operator", "psf_path")}
                    for m, p in f.get("params", {}).items()}
    raise KeyError(f"fold {fold_name!r} not in {summary_path}")


# ============================================================ fit, predict, save
def _vis_reflectance(batch: Batch) -> np.ndarray:
    from .methods import _common

    return _common.reflectance(batch, "0.66")


def granule_window_data(granules, table, rows: np.ndarray, cfg: benchmark.BenchmarkConfig, size=(150, 500),
                        store=None) -> tuple[Batch, dict]:
    """The test footprints of one granule inside the chosen window: their Batch and the arrays the figures need.

    Raises:
        ValueError: If the images have no row0/col0 (patches cannot be placed) or there are no footprints.
    """
    full = make_batch(granules, table, rows, store=store)
    if not full.has_images or "row0" not in full.images:
        raise ValueError("the images have no row0/col0: rebuild them with scripts/build_dataset.py --granules ...")
    patch = full.images["ImSW"].shape[-2:]
    r0, c0 = np.asarray(full.images["row0"]), np.asarray(full.images["col0"])
    cf_all = np.asarray(table["CF"], float)[rows] if "CF" in table else cloud_fraction(_vis_reflectance(full) > 0.2)
    size = (min(size[0], int(r0.max() - r0.min()) + patch[0]), min(size[1], int(c0.max() - c0.min()) + patch[1]))
    corner = choose_window(r0, c0, cf_all, patch, size)
    keep = np.flatnonzero(footprints_in_window(r0, c0, corner, patch, size))
    if len(keep) == 0:
        raise ValueError("no footprint inside the window")
    from .features import take_batch

    batch = take_batch(full, keep)
    vis = _vis_reflectance(batch)
    w = benchmark._weights(cfg.operator, cfg, batch)
    w = np.broadcast_to(w, (len(batch),) + tuple(patch)).astype(np.float32)
    data = {"rows": np.asarray(batch.rows), "row0": r0[keep], "col0": c0[keep], "corner": np.array(corner),
            "size": np.array(size), "label": np.asarray(batch.label, float), "weights": w,
            "vis": vis.astype(np.float32)}
    for k in ("CF", "Clrf", "SZA", "VZA", "RAA", "glint"):
        if k in table:
            data[f"s_{k}"] = np.asarray(table[k], float)[batch.rows]
    for name, m in cloud_masks(batch.images, vis).items():
        data[f"mask__{name}"] = m
    if batch.truth is not None:
        data["truth"] = np.asarray(batch.truth, np.float32)
    return batch, data


def fit_models(models: list[str], train: Batch, val: Batch, cfg: benchmark.BenchmarkConfig,
               params: dict[str, dict] | None = None, log=log) -> dict:
    """Fit each map method once: {name: fitted model}; a method that fails is logged and skipped."""
    out = {}
    for name in models:
        if not MODELS[name].produces_map:
            log(f"  {name}: footprint-only (no map), skipped")
            continue
        p = {**get_model(name).param_grid()[0], **(params or {}).get(name, {}), **benchmark.fixed_params(name, cfg)}
        t0 = datetime.now()
        try:
            out[name] = get_model(name, **p).fit(train, val)
        except (ValueError, RuntimeError, ImportError, KeyError, np.linalg.LinAlgError) as e:
            log(f"  {name}: ERROR {type(e).__name__}: {e}")
            continue
        log(f"  {name}: fitted in {(datetime.now() - t0).total_seconds() / 60:.1f} min  {p}")
    return out


def predict_maps(fitted: dict, batch: Batch, log=log) -> dict[str, np.ndarray]:
    """{name: (N, H, W) float32 map} of the fitted models on one window batch."""
    out = {}
    for name, model in fitted.items():
        try:
            out[name] = predict_map_chunked(model, batch).astype(np.float32)
        except (ValueError, RuntimeError, KeyError) as e:
            log(f"  {name}: prediction ERROR {type(e).__name__}: {e}")
    return out


def save_maps(path: str | Path, data: dict, maps: dict[str, np.ndarray]) -> Path:
    """Write one granule's window data and maps to an .npz (maps under "map__<method>")."""
    path = Path(path)
    np.savez_compressed(path, **data, **{f"map__{k}": v for k, v in maps.items()})
    return path


def load_maps(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read save_maps' file back into (data, maps)."""
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files if not k.startswith("map__")}
        maps = {k[5:]: f[k] for k in f.files if k.startswith("map__")}
    return data, maps


# ============================================================ figures
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold"})
    return plt


def _norm(data, maps):
    from matplotlib.colors import PowerNorm

    vals = [np.asarray(data["label"])] + [m[np.isfinite(m)].ravel()[::7] for m in maps.values()]
    hi = float(np.nanpercentile(np.concatenate(vals), 99.5)) if vals else 1000.0
    return PowerNorm(0.5, vmin=0.0, vmax=max(min(hi, 1400.0), 100.0))   # sqrt scale: clear sky and cloud both readable


def _masks(data):
    return {k[6:]: data[k] for k in data if k.startswith("mask__")}


def _title_tag(name):
    return "  (reads CERES labels)" if name in MODELS and MODELS[name].uses_test_labels else ""


def _rmse(a, b):
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def _footprint_means(data, maps):
    w = data["weights"].astype(float)
    return {k: (np.nan_to_num(m.astype(float)) * w).sum(axis=(1, 2)) / np.maximum(w.sum(axis=(1, 2)), 1e-12)
            for k, m in maps.items()}


def _psf_outline(ax, w, r, c, color, lw=1.4):
    yy, xx = np.mgrid[r:r + w.shape[0], c:c + w.shape[1]]
    ax.contour(xx, yy, w, levels=[float(w.max()) / 4], colors=color, linewidths=lw)


def plot_scene(data: dict, maps: dict, title: str = ""):
    """Figure 1: MODIS + cloud masks, what CERES sees, and every method's mosaic map (one shared colour scale)."""
    plt = _plt()
    corner, size = tuple(data["corner"]), tuple(data["size"])
    r0, c0 = data["row0"], data["col0"]
    grid = lambda a: mosaic(a, r0, c0, corner, size)  # noqa: E731
    masks = _masks(data)
    panels = [("CERES footprint Fsw painted on its pixels", paint_footprints(data["label"], r0, c0, corner, size,
                                                                             data["weights"].shape[1:]))]
    if "truth" in data:
        panels.append(("Truth (synthetic data only)", grid(data["truth"])))
    rm = {k: _rmse(v, data["label"]) for k, v in _footprint_means(data, maps).items()}
    panels += [(f"{k}{_title_tag(k)}   |   RMSE vs CERES {rm[k]:.1f} W/m²", grid(m)) for k, m in maps.items()]
    n = 1 + len(panels)
    ncol = 2
    nrow = (n + 1) // 2
    aspect = size[0] / size[1]
    fig, axs = plt.subplots(nrow, ncol, figsize=(15, 1.0 + nrow * 7.2 * aspect + 0.5 * nrow), constrained_layout=True,
                            squeeze=False)
    ax = axs[0, 0]
    ax.imshow(grid(data["vis"]), cmap="gray", vmin=0, vmax=0.8)
    styles = {"MYD35 relaxed": "#38b6ff", "MYD35 strict": "#f2a600", "0.66 um proxy": "#ee7fac"}
    shown = [k for k in ("MYD35 relaxed", "MYD35 strict") if k in masks] or ["0.66 um proxy"]
    for k in shown:
        ax.contour(grid(masks[k]), levels=[0.5], colors=styles[k], linewidths=0.6)
    pick = pick_footprints(data.get("s_CF", cloud_fraction(next(iter(masks.values())))))
    ax.set_title("MODIS 0.66 µm reflectance; cloud edge: " + ", ".join(f"{k} ({'blue' if 'relaxed' in k else 'yellow' if 'strict' in k else 'pink'})"
                                                                         for k in shown), loc="left")
    norm = _norm(data, maps)
    im = None
    for ax, (name, g) in zip(axs.flat[1:], panels):
        im = ax.imshow(g, cmap="magma", norm=norm)
        ax.set_title(name, loc="left")
    for ax in axs.flat[:1 + len(panels)]:
        for i in pick.values():
            _psf_outline(ax, data["weights"][i], r0[i] - corner[0], c0[i] - corner[1], "#00b894")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axs.flat[1 + len(panels):]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axs, shrink=min(0.5, 3.0 / nrow), label="Fsw [W/m²] (square-root scale)")
    fig.suptitle(f"{title}  -  1 km Fsw per method, window {size[0]} × {size[1]} km; green = footprints of fig2",
                 fontsize=12, fontweight="bold")
    return fig


def plot_footprint(data: dict, maps: dict, i: int, regime: str = "", title: str = ""):
    """Figure 2: one footprint: its clouds and cloud fractions, the PSF, each method's map and weighted mean vs CERES."""
    plt = _plt()
    masks = _masks(data)
    w = data["weights"][i].astype(float)
    label = float(data["label"][i])
    names = list(maps) + (["truth"] if "truth" in data else [])
    get = lambda k: data["truth"][i] if k == "truth" else maps[k][i]  # noqa: E731
    ncol = max(4, 2 + len(masks))
    nrow_maps = (len(names) + ncol - 1) // ncol
    row_h = 15.0 / ncol * w.shape[0] / w.shape[1] + 0.6          # a patch row: image height + two title lines
    dots_h = 0.6 + 0.28 * len(names)
    fig = plt.figure(figsize=(15, row_h * (1 + nrow_maps) + dots_h + 0.6), constrained_layout=True)
    gs = fig.add_gridspec(2 + nrow_maps, ncol, height_ratios=[row_h] * (1 + nrow_maps) + [dots_h])
    a = fig.add_subplot(gs[0, 0])
    a.imshow(data["vis"][i], cmap="gray", vmin=0, vmax=0.8)
    a.contour(w, levels=[w.max() / 4], colors="#00b894", linewidths=1)
    a.set_title("MODIS 0.66 µm in the footprint")
    for j, (k, m) in enumerate(masks.items(), 1):
        a = fig.add_subplot(gs[0, j])
        a.imshow(m[i], cmap="Blues", vmin=-0.3, vmax=1.3)
        a.set_title(f"{k}\nCF {cloud_fraction(m[i:i + 1])[0]:.0f}%  (PSF-weighted {cloud_fraction(m[i:i + 1], w)[0]:.0f}%)")
    a = fig.add_subplot(gs[0, len(masks) + 1])
    a.axis("off")
    s = lambda k: data.get(f"s_{k}", np.full(len(data["label"]), np.nan))[i]  # noqa: E731
    a.text(0.0, 0.98, f"{regime} footprint #{int(data['rows'][i])}", fontsize=12, fontweight="bold", va="top")
    a.text(0.0, 0.78, f"CERES Fsw        {label:6.0f} W/m²\n"
                      f"CF file (MYD35)  {s('CF'):6.0f} %\n"
                      f"CERES cloudy     {100 - s('Clrf'):6.0f} %  (100-Clrf)\n"
                      f"SZA {s('SZA'):4.0f}°  VZA {s('VZA'):4.0f}°\nRAA {s('RAA'):4.0f}°  glint {s('glint'):4.0f}°",
           fontsize=10, va="top", family="monospace")
    for ax in fig.axes:
        if ax.images:
            ax.set_xticks([]); ax.set_yticks([])
    norm = _norm(data, maps)
    means = {}
    for k, name in enumerate(names):
        a = fig.add_subplot(gs[1 + k // ncol, k % ncol])
        m = np.asarray(get(name), float)
        means[name] = float((np.nan_to_num(m) * w).sum() / w.sum())
        a.imshow(m, cmap="magma", norm=norm)
        a.contour(w, levels=[w.max() / 4], colors="#00b894", linewidths=1)
        a.set_title(f"{name}{_title_tag(name)}\nΣw·f {means[name]:.0f}   CERES {label:.0f}   Δ {means[name] - label:+.0f}")
        a.set_xticks([]); a.set_yticks([])
    a = fig.add_subplot(gs[1 + nrow_maps, :])
    order = sorted(means, key=means.get)
    y = np.arange(len(order))
    a.axvspan(label - 10, label + 10, color="#999999", alpha=0.2, lw=0)
    a.axvline(label, color="#191919", lw=2)
    a.scatter([means[k] for k in order], y, s=60, color=SERIES[0], zorder=3)
    a.set_yticks(y, order)
    a.set_xlabel("PSF-weighted mean of the 1 km map [W/m²]   (black: CERES, grey: ±10 W/m² noise floor)")
    a.grid(axis="x", color="#e5e5e5")
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    fig.suptitle(f"{title}  -  one CERES footprint: clouds, the 1 km maps, and their weighted mean", fontsize=12, fontweight="bold")
    return fig


def plot_scatter(data: dict, maps: dict, title: str = ""):
    """Figure 3: every footprint, PSF-weighted mean of each map vs CERES, coloured by cloud fraction."""
    plt = _plt()
    means = _footprint_means(data, maps)
    cf = data.get("s_CF", cloud_fraction(next(iter(_masks(data).values()))))
    n = len(means)
    ncol = min(4, max(n, 1))
    nrow = max((n + ncol - 1) // ncol, 1)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol + 1.5, 3.4 * nrow + 0.6), constrained_layout=True,
                            sharex=True, sharey=True, squeeze=False)
    y = data["label"]
    lim = [float(np.nanmin(y)) - 20, float(np.nanmax(y)) + 20]
    sc = None
    for ax, (k, p) in zip(axs.flat, means.items()):
        sc = ax.scatter(y, p, c=cf, cmap="Blues", vmin=-15, vmax=100, s=8, edgecolors="none")
        ax.plot(lim, lim, color="#666666", lw=1)
        ax.set_title(k + _title_tag(k), loc="left")
        ax.text(0.03, 0.96, f"RMSE {_rmse(p, y):.1f}\nbias {np.nanmean(p - y):+.1f}", transform=ax.transAxes, va="top")
        ax.grid(color="#eeeeee")
    for ax in axs.flat[n:]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("CERES Fsw [W/m²]")
    for ax in axs[:, 0]:
        ax.set_ylabel("Σ w·f of the map [W/m²]")
    if sc is not None:
        fig.colorbar(sc, ax=axs, shrink=0.6, label="cloud fraction in footprint [%] (MYD35, from the file)")
    fig.suptitle(f"{title}  -  every footprint in the window: map aggregated with the PSF vs CERES", fontsize=12, fontweight="bold")
    return fig


def plot_ctz(data: dict, maps: dict, title: str = ""):
    """Figure 4: clear-pixel flux vs distance from cloud, one panel per method, one line per MYD35 mask."""
    plt = _plt()
    corner, size = tuple(data["corner"]), tuple(data["size"])
    grid = lambda a: mosaic(a, data["row0"], data["col0"], corner, size)  # noqa: E731
    masks = _masks(data)
    use = [k for k in ("MYD35 relaxed", "MYD35 strict") if k in masks] or ["0.66 um proxy"]
    mgrid = {k: grid(masks[k]) for k in use}
    series = dict(maps)
    if "truth" in data:
        series = {"truth": data["truth"], **series}
    n = len(series)
    ncol = min(4, max(n, 1))
    nrow = max((n + ncol - 1) // ncol, 1)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol + 0.5, 3.0 * nrow + 0.8), constrained_layout=True,
                            sharex=True, squeeze=False)
    for ax, (name, m) in zip(axs.flat, series.items()):
        g = grid(m)
        for col, k in zip(SERIES, use):
            x, f = clear_flux_vs_distance(g, mgrid[k])
            ax.plot(x, f, color=col, lw=2, marker="o", ms=4, label=k)
        ax.set_title(name + _title_tag(name), loc="left")
        ax.grid(color="#eeeeee")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axs.flat[n:]:
        ax.axis("off")
    axs.flat[0].legend(frameon=False, fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("distance from cloud [km]")
    for ax in axs[:, 0]:
        ax.set_ylabel("mean Fsw of clear pixels [W/m²]")
    fig.suptitle(f"{title}  -  clear-sky flux vs distance from cloud (a CTZ shows as a decay; smoothing can fake one)",
                 fontsize=12, fontweight="bold")
    return fig


def plot_masks(data: dict, title: str = ""):
    """Figure 5: the cloud masks on the window and their footprint cloud fractions vs CERES's (100 - Clrf)."""
    plt = _plt()
    corner, size = tuple(data["corner"]), tuple(data["size"])
    masks = _masks(data)
    img_h = 15.0 * size[0] / size[1] + 0.4
    fig = plt.figure(figsize=(15, img_h * len(masks) + 4.2), constrained_layout=True)
    gs = fig.add_gridspec(len(masks) + 1, 3, height_ratios=[img_h] * len(masks) + [3.8])
    for j, (k, m) in enumerate(masks.items()):
        a = fig.add_subplot(gs[j, :])
        a.imshow(mosaic(m, data["row0"], data["col0"], corner, size), cmap="Blues", vmin=-0.3, vmax=1.3)
        a.set_title(f"{k}: cloudy pixels (dark)", loc="left")
        a.set_xticks([]); a.set_yticks([])
    if "s_Clrf" in data:
        ceres_cloudy = 100.0 - data["s_Clrf"]
        for j, (k, m) in enumerate(masks.items()):
            a = fig.add_subplot(gs[len(masks), j])
            cfw = cloud_fraction(m, data["weights"])
            a.scatter(ceres_cloudy, cfw, s=8, color=SERIES[0], edgecolors="none")
            a.plot([0, 100], [0, 100], color="#666666", lw=1)
            r = np.corrcoef(ceres_cloudy, cfw)[0, 1] if np.std(cfw) > 0 and np.std(ceres_cloudy) > 0 else np.nan
            a.set_title(f"{k}   r = {r:.2f}", loc="left")
            a.set_xlabel("CERES cloudy % (100 - Clrf)")
            a.set_ylabel("PSF-weighted CF [%]")
            a.grid(color="#eeeeee")
    fig.suptitle(f"{title}  -  cloud masks and their footprint cloud fractions", fontsize=12, fontweight="bold")
    return fig


def draw_all(data: dict, maps: dict, out_dir: str | Path, tag: str) -> list[Path]:
    """Write the five figures of one granule window; returns the PNG paths."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        p = out_dir / f"{name}_{tag}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)

    has_place = "row0" in data and "corner" in data
    if has_place:
        save(plot_scene(data, maps, tag), "fig1_scene")
    cf = data.get("s_CF", cloud_fraction(next(iter(_masks(data).values()))))
    for regime, i in pick_footprints(cf).items():
        save(plot_footprint(data, maps, i, regime, tag), f"fig2_footprint_{regime}")
    save(plot_scatter(data, maps, tag), "fig3_scatter")
    if has_place:
        save(plot_ctz(data, maps, tag), "fig4_ctz")
        save(plot_masks(data, tag), "fig5_masks")
    return paths


# ============================================================ command line
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="synthetic data with a known 1 km truth")
    p.add_argument("--demo-hard", action="store_true", help="the harder synthetic scene (PLAN.md 4.3)")
    p.add_argument("--data-root", help="folder with moments/, scalars/, images/ (images with row0/col0, ImCloud)")
    p.add_argument("--manifest", help="frozen split manifest (JSON); without it: day folds as in flux1km.cli")
    p.add_argument("--fold", default=None, help="fold name, e.g. day-kfold:0 (default: the first non-leaky fold)")
    p.add_argument("--models", nargs="+", default=None, help="map methods (default: every registered map method)")
    p.add_argument("--params-from", help="summary.json of a benchmark run: use its chosen hyper-parameters")
    p.add_argument("--granules", nargs="+", help="test granules to draw (default: the --n-granules largest)")
    p.add_argument("--n-granules", type=int, default=2)
    p.add_argument("--window", default="150x500", help="map window ROWSxCOLS in pixels (1 km)")
    p.add_argument("--psf", help="ceres_psf.csv; the operator is then ceres_psf, else fov_ellipse")
    p.add_argument("--input-set", default="F5")
    p.add_argument("--cache-dir", default=None, help="keep images on disk (as in flux1km.cli)")
    p.add_argument("--out", default="benchmark_outputs/flux1km_maps")
    p.add_argument("--replot", help="a run folder of this tool: redraw its saved maps, no training")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if not (a.demo or a.data_root or a.replot):
        p.error("give --data-root, --demo or --replot")
    return a


def main(argv=None) -> list[Path]:
    """Fit, predict, save and draw (or only redraw with --replot); returns the PNG paths."""
    from . import cli

    a = parse_args(argv)
    if a.replot:
        paths = []
        for f in sorted(Path(a.replot).glob("maps_*.npz")):
            data, maps = load_maps(f)
            paths += draw_all(data, maps, a.replot, f.stem[5:])
        print("\n".join(str(p) for p in paths))
        return paths
    root = a.data_root
    if a.demo:
        root = root or tempfile.mkdtemp(prefix="flux1km_viz_demo_")
        synthetic.write_demo_dataset(root, seed=a.seed, n_rows=8, n_cols=30, hard=a.demo_hard)
        log(f"demo data written to {root}")
    granules = io.load_dataset(root, parts=("moments", "scalars", "images", "truth"), lazy_images=bool(a.cache_dir))
    groups = splits.group_granules([g.granule_id for g in granules], sizes={g.granule_id: g.n_footprints for g in granules},
                                   min_size=100)
    table = build_table(granules, groups)
    if a.manifest:
        folds, _ = splits.load_manifest(a.manifest, table)
    else:
        folds = cli.make_folds(table, "day" if not a.demo else "logo", 5, a.seed, 100.0, 40.0)
    folds = [f for f in folds if not f.leaky]
    fold = next((f for f in folds if f.name == a.fold), None) if a.fold else folds[0]
    if fold is None:
        raise SystemExit(f"no fold {a.fold!r}; folds: {[f.name for f in folds]}")
    models = cli.available_models(a.models or [m for m in sorted(MODELS) if MODELS[m].produces_map])
    cfg = benchmark.BenchmarkConfig(models=models, operator="ceres_psf" if a.psf else "fov_ellipse", psf_path=a.psf,
                                    model_params={m: {"input_set": a.input_set} for m in models if m not in ("B-clim", "B-const")})
    params = load_params(a.params_from, fold.name) if a.params_from else {}
    rows, cols = (int(x) for x in a.window.lower().split("x"))
    test_g = np.asarray(table["granule"])[fold.test]
    ids = a.granules or [g for g, _ in sorted(((g, int((test_g == g).sum())) for g in set(test_g.tolist())),
                                              key=lambda t: -t[1])[: a.n_granules]]
    out = Path(a.out) / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    store = tempfile.mkdtemp(prefix="viz_", dir=a.cache_dir) if a.cache_dir else None
    log(f"fold {fold.name}: train {len(fold.train)}, val {len(fold.val)}; granules {ids}; operator {cfg.operator}")
    train = make_batch(granules, table, fold.train, store=store)
    val = make_batch(granules, table, fold.val, store=store)
    log(f"images of train and validation ready; fitting {len(models)} methods")
    fitted = fit_models(models, train, val, cfg, params)
    paths = []
    for gid in ids:
        g_rows = fold.test[test_g == gid]
        if len(g_rows) == 0:
            log(f"{gid}: not in the test set of {fold.name}, skipped")
            continue
        try:
            batch, data = granule_window_data(granules, table, g_rows, cfg, (rows, cols), store=store)
        except ValueError as e:
            log(f"{gid}: {e}")
            continue
        log(f"{gid}: window {data['size'].tolist()} at {data['corner'].tolist()}, {len(batch)} footprints; predicting")
        maps = predict_maps(fitted, batch)
        save_maps(out / f"maps_{gid}.npz", data, maps)
        paths += draw_all(data, maps, out, gid)
    (out / "README.txt").write_text(f"flux1km.mapviz run {datetime.now():%Y-%m-%d %H:%M}\nfold {fold.name}\n"
                                    f"models {list(fitted)}\nparams from {a.params_from}\ngranules {ids}\n"
                                    f"maps_<granule>.npz: the window's data and maps; redraw with --replot {out}\n")
    log(f"figures in {out}:\n" + "\n".join(f"  {p.name}" for p in paths))
    return paths


if __name__ == "__main__":
    main()
