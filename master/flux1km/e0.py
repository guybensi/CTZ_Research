"""Stage E0: data verification and infrastructure (PLAN.md section 2).

Every open E0 item is one function here, and run_e0 runs them all and writes E0_report.txt:

| PLAN.md E0 item                                   | function |
|---------------------------------------------------|----------|
| unpack Data_toGuy.zip and inventory it            | unpack_zip, inventory |
| FOVorientation meaning, ImSW/ImLW units, flags    | value_ranges, check_fov_orientation |
| moments vs image patch statistics                 | moments_vs_patches |
| fit the aggregation weights H                     | fit_aggregation_operator (ellipse fit + PSF recovery, saved as .npz usable by --psf) |
| footprint overlap on a full overpass              | footprint_overlap |
| systematically bad channels                       | stripe_scores |
| one frozen split manifest + leakage check         | build_frozen_manifest (Arena A + Arena B + spatial blocks + interleave) |
| re-run the legacy baselines in the new protocol   | run_e0(..., run_baselines=True) |
| report the LiqPhsf bug to the advisor             | advisor_bug_report |

    python3 -m flux1km.e0 --data-root data/sample --out benchmark_outputs/e0
    python3 -m flux1km.e0 --zip Data_toGuy.zip --data-root data/paired --out benchmark_outputs/e0
    python3 -m flux1km.e0 --demo
"""
from __future__ import annotations

import argparse
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import aggregation, io, moments, schema, splits
from .io import Granule
from .report import format_table, section
from .table import FootprintTable, build_table

# Plausible L1B radiance ranges (W m^-2 sr^-1 um^-1): a units sanity check, not a physical limit.
SW_RADIANCE_RANGE = (-1.0, 900.0)
LW_RADIANCE_RANGE = (-1.0, 25.0)


# ------------------------------------------------------------------ unpack / inventory
def unpack_zip(zip_path: str | Path, dest: str | Path) -> list[str]:
    """Extract a zip into dest, refusing members that would escape it; return the extracted names.

    Example:
        >>> # unpack_zip("Data_toGuy.zip", "data/")

    Raises:
        ValueError: If a member path is absolute or contains "..".
    """
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest)):
                raise ValueError(f"unsafe path in zip: {name!r}")
        z.extractall(dest)
        return z.namelist()


def find_data_roots(base: str | Path) -> list[Path]:
    """Every folder under base (base included) that holds a scalars/ sub-folder with NPZ files.

    Example:
        >>> find_data_roots("/nonexistent")
        []
    """
    base = Path(base)
    if not base.exists():
        return []
    return sorted({p.parent.parent for p in base.rglob("scalars/*.npz")})


def inventory(root: str | Path) -> dict:
    """Granules, footprints, and keys/shapes of moments/, scalars/ and images/ under one root.

    Example:
        >>> inventory("/nonexistent")["n_granules"]
        0

    Returns:
        {"root", "n_granules", "n_footprints", "granules": {id: n}, "parts": {part: n_files},
         "keys": {part: {key: shape-without-N}}, "inconsistent": [messages], "days": [...], "months": [...]}
    """
    root = Path(root)
    out = {"root": str(root), "n_granules": 0, "n_footprints": 0, "granules": {}, "parts": {}, "keys": {},
           "inconsistent": [], "days": [], "months": []}
    for part in io.PARTS:
        files = sorted((root / part).glob("*.npz"))
        out["parts"][part] = len(files)
        shapes: dict[str, tuple] = {}
        for f in files:
            with np.load(f) as d:
                for k in d.files:
                    sh = d[k].shape[1:] if k not in ("SWwavlngs", "LWwavlngs") else d[k].shape
                    if k in shapes and shapes[k] != sh:
                        out["inconsistent"].append(f"{part}/{f.name}: {k} has {sh}, others {shapes[k]}")
                    shapes.setdefault(k, sh)
                if part == "scalars" and d.files:
                    out["granules"][f.stem] = len(d[d.files[0]])
        out["keys"][part] = {k: tuple(int(x) for x in v) for k, v in sorted(shapes.items())}
    out["n_granules"] = len(out["granules"])
    out["n_footprints"] = int(sum(out["granules"].values()))
    parsed = []
    for g in out["granules"]:
        try:
            parsed.append(io.parse_granule_id(g))
        except ValueError:
            out["inconsistent"].append(f"not a MODIS granule id: {g}")
    out["days"] = sorted({p.day for p in parsed})
    out["months"] = sorted({f"{p.time:%Y-%m}" for p in parsed})
    return out


def value_ranges(granules: list[Granule]) -> dict:
    """Range, mean and NaN fraction of every scalar, the FOVorientation values and the radiance ranges.

    Flags radiances outside SW_RADIANCE_RANGE / LW_RADIANCE_RANGE (units check of ImSW / ImLW).

    Returns:
        {"scalars": {key: {min, max, mean, nan_frac}}, "fov_values": [...],
         "radiance": {"SW"/"LW": {label: (min, max)}}, "flags": [messages]}
    """
    out: dict = {"scalars": {}, "fov_values": [], "radiance": {"SW": {}, "LW": {}}, "flags": []}
    keys = sorted(set().union(*(g.scalars.keys() for g in granules))) if granules else []
    for k in keys:
        v = np.concatenate([np.asarray(g.scalars[k], float).ravel() for g in granules if k in g.scalars])
        fin = v[np.isfinite(v)]
        out["scalars"][k] = {"min": float(fin.min()) if fin.size else np.nan, "max": float(fin.max()) if fin.size else np.nan,
                             "mean": float(fin.mean()) if fin.size else np.nan, "nan_frac": float(1 - fin.size / max(v.size, 1))}
    imgs = [g.images for g in granules if g.images is not None]
    if imgs:
        out["fov_values"] = sorted({float(x) for im in imgs for x in np.unique(im["FOVorientation"])})
        for band, labels, (lo, hi) in (("SW", schema.SW_LABELS, SW_RADIANCE_RANGE), ("LW", schema.LW_LABELS, LW_RADIANCE_RANGE)):
            for i, lab in enumerate(labels):
                v = np.concatenate([im[f"Im{band}"][:, i].ravel() for im in imgs]).astype(float)
                v = v[np.isfinite(v)]
                if not v.size:
                    continue
                out["radiance"][band][lab] = (float(v.min()), float(v.max()))
                if v.min() < lo or v.max() > hi:
                    out["flags"].append(f"Im{band} {lab} um outside the expected L1B range {lo}..{hi}: {v.min():.2f}..{v.max():.2f}")
        if not set(out["fov_values"]) <= {0.0, 1.0}:
            out["flags"].append(f"FOVorientation has values other than 0/1: {out['fov_values']}")
    if "CF" in out["scalars"] and out["scalars"]["CF"]["max"] > 1.0 + 1e-6:
        out["flags"].append("CF exceeds 1: stored in percent, not a fraction (check schema.CF_TO_PERCENT)")
    if "landf" in out["scalars"] and out["scalars"]["landf"]["max"] > 3.0:
        out["flags"].append("landf exceeds 3: not the mean 2-bit land/water code assumed in PLAN.md 2.3")
    return out


# ------------------------------------------------------------------ aggregation operator
def _stack(granules: list[Granule]):
    gs = [g for g in granules if g.images is not None and g.moments is not None]
    if not gs:
        raise ValueError("needs granules with both images and moments")
    x = np.concatenate([np.concatenate([g.images["ImSW"], g.images["ImLW"]], axis=1) for g in gs]).astype(float)
    mean = np.concatenate([np.concatenate([g.moments["mSWspec"], g.moments["mLWspec"]], axis=1) for g in gs])
    std = np.concatenate([np.concatenate([g.moments["stdSWspec"], g.moments["stdLWspec"]], axis=1) for g in gs])
    orient = np.concatenate([g.images["FOVorientation"] for g in gs])
    return x, mean, std, orient


def _rel_err(x, mean, std, weights) -> float:
    m, s, _, _ = moments.moments_from_pixels(x, weights[:, None] if weights.ndim == 3 else weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.concatenate([(m / mean - 1).ravel(), (s / std - 1).ravel()])
    return float(np.sqrt(np.nanmean(e**2)))


def moments_vs_patches(granules: list[Granule], psf: np.ndarray | None = None) -> dict[str, float]:
    """RMS relative error of the stored mean/std against patch statistics under each operator.

    Example:
        >>> # moments_vs_patches(gs) -> {"uniform": 0.23, "fov_ellipse": 0.086, "gaussian_psf": ...}
    """
    x, mean, std, orient = _stack(granules)
    h, w = x.shape[-2:]
    out = {}
    for op in aggregation.OPERATORS:
        if op == "ceres_psf" and psf is None:
            continue
        wts = aggregation.get_operator(op, h, w, orientation=orient, psf=psf)
        out[op] = _rel_err(x, mean, std, np.broadcast_to(wts, (len(x), h, w)))
    return out


def check_fov_orientation(granules: list[Granule]) -> dict[str, float]:
    """Moments error with FOVorientation as stored vs inverted (fov_ellipse operator).

    If "as stored" is clearly smaller, the flag means what PLAN.md 2.1 says (0 = PSF mirrored).
    """
    x, mean, std, orient = _stack(granules)
    h, w = x.shape[-2:]
    return {"as_stored": _rel_err(x, mean, std, aggregation.fov_weights(orient, h, w)),
            "inverted": _rel_err(x, mean, std, aggregation.fov_weights(1 - np.asarray(orient), h, w))}


def fit_aggregation_operator(granules: list[Granule], out_path: str | Path | None = None, max_footprints: int = 400,
                             seed: int = 0, maxiter: int = 800) -> dict:
    """Fit H from the data: the tilted-ellipse parameters and, with enough footprints, the full PSF.

    The recovered PSF is saved as an .npz ("psf", "ellipse") that `--psf` accepts, so it can serve
    as the primary operator until the advisor's ceres_psf.csv arrives.

    Returns:
        {"ellipse": (sigma_major, sigma_minor, angle), "ellipse_error": float, "psf_error": float | None,
         "n_footprints": int, "path": str | None}
    """
    x, mean, std, orient = _stack(granules)
    if len(x) > max_footprints:
        sel = np.random.default_rng(seed).choice(len(x), max_footprints, replace=False)
        x, mean, std, orient = x[sel], mean[sel], std[sel], orient[sel]
    params, err = aggregation.fit_fov_ellipse(x, mean, std, orient, maxiter=maxiter)
    out = {"ellipse": params, "ellipse_error": err, "psf_error": None, "n_footprints": len(x), "path": None}
    h, w = x.shape[-2:]
    try:
        psf = aggregation.recover_psf_from_moments(x, mean, orient)
        out["psf_error"] = _rel_err(x, mean, std, aggregation.oriented_psf(np.clip(psf, 0, None), orient))
    except ValueError as e:
        psf = None
        out["psf_note"] = str(e)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        arr = psf if psf is not None else aggregation.fov_weights([1], h, w, params)[0]
        np.savez(out_path, psf=np.clip(arr, 0, None) / np.clip(arr, 0, None).sum(), ellipse=np.array(params))
        out["path"] = str(out_path)
    return out


# ------------------------------------------------------------------ overlap and channels
def footprint_overlap(granule: Granule, k: int = 4, channel: str = "0.66") -> dict:
    """Footprint spacing and pixel sharing within one granule (overpass).

    Spacing: distance (km) from each footprint centre to its nearest neighbour. Sharing: for each
    footprint and its k nearest neighbours, the fraction of the patch's pixel values (of one
    channel) that also occur in the neighbour's patch; identical 1 km values mean shared pixels.

    Returns:
        {"spacing_km_median", "spacing_km_min", "shared_max", "shared_median"} (NaN when undefined).
    """
    lat, lon = np.radians(granule.scalars["Lat"]), np.radians(granule.scalars["Lon"])
    xyz = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    n = len(xyz)
    out = {"spacing_km_median": np.nan, "spacing_km_min": np.nan, "shared_max": np.nan, "shared_median": np.nan}
    if n < 2:
        return out
    kk = min(k + 1, n)
    d, j = cKDTree(xyz).query(xyz, k=kk)
    km = 2 * splits.EARTH_RADIUS_KM * np.arcsin(np.clip(d[:, 1] / 2, 0, 1))
    out["spacing_km_median"], out["spacing_km_min"] = float(np.median(km)), float(km.min())
    if granule.images is None:
        return out
    band, ci = schema.channel_index(channel)
    vals = granule.images[f"Im{band}"][:, ci].reshape(n, -1)
    shared = []
    for a in range(n):
        sa = set(vals[a].tolist())
        for b in j[a, 1:]:
            shared.append(len(sa & set(vals[b].tolist())) / max(len(sa), 1))
    out["shared_max"], out["shared_median"] = float(np.max(shared)), float(np.median(shared))
    return out


def stripe_scores(granules: list[Granule], threshold: float = 0.5) -> dict:
    """Detector striping score per channel and the channels flagged as systematically bad.

    Score = median over patches of std(row means - 3-row running mean) / std(pixels). Striped
    bands (Aqua band 6, 1.64 um) have row-to-row jumps that a smooth scene does not.

    Returns:
        {"scores": {label: float}, "bad": [labels with score > threshold]}
    """
    imgs = [g.images for g in granules if g.images is not None]
    scores = {}
    for band, labels in (("SW", schema.SW_LABELS), ("LW", schema.LW_LABELS)):
        for i, lab in enumerate(labels):
            vals = []
            for im in imgs:
                x = im[f"Im{band}"][:, i].astype(float)
                rows = x.mean(axis=2)
                smooth = np.apply_along_axis(lambda r: np.convolve(r, np.ones(3) / 3, mode="same"), 1, rows)
                resid = (rows - smooth)[:, 1:-1]
                sd = x.reshape(len(x), -1).std(axis=1)
                ok = sd > 0
                vals.extend((resid.std(axis=1)[ok] / sd[ok]).tolist())
            scores[lab] = float(np.median(vals)) if vals else np.nan
    return {"scores": scores, "bad": [k for k, v in scores.items() if np.isfinite(v) and v > threshold]}


# ------------------------------------------------------------------ split manifest
def exclude_days_from_training(table: FootprintTable, folds: list[splits.Fold], days) -> list[splits.Fold]:
    """Remove rows of the given days from the train and validation sides of every fold.

    PLAN.md 4.1: the days of the Arena A image granules are always excluded from M1/M2 training in
    Arena B.

    Example:
        >>> # exclude_days_from_training(table, day_folds, set(table["day"][arena_a_rows]))
    """
    day = np.asarray(table["day"])
    bad = np.isin(day, list(days))
    return [splits.Fold(f.name, f.train[~bad[f.train]], f.val[~bad[f.val]], f.test, f.leaky, dict(f.meta)) for f in folds]


def build_frozen_manifest(table: FootprintTable, path: str | Path | None = None, image_granules=None, k: int = 5,
                          seed: int = 0, block_km: float = 100.0, buffer_km: float = 40.0,
                          leakage_radius_km: float = 20.0) -> tuple[list[splits.Fold], dict]:
    """Build every split of PLAN.md 4.1, check each for pixel reuse, and freeze them to JSON.

    * Arena A: leave-one-group-out over the image granules' overpass groups (+ interleave, leaky).
    * Arena B (moments): group k-fold by day, year holdout (last year), leave-one-month-out; the
      Arena A days are excluded from their training sets.
    * Spatial blocks with a VZA-scaled buffer.
    Splits that cannot be built (too few groups, one year, ...) are skipped with a note.

    Returns:
        (folds, report) where report = {"folds": {name: {"n_train", "n_val", "n_test", "max_reuse", "leaky"}},
        "skipped": [notes], "path": str | None}. Fold names are prefixed "A:", "B:" or "S:".
    """
    report: dict = {"folds": {}, "skipped": [], "path": None}
    folds: list[splits.Fold] = []
    img = set(image_granules or [])
    is_img = np.isin(table["granule"], list(img)) if img else np.ones(len(table), bool)
    a_days = set(np.asarray(table["day"])[is_img]) if img else set()

    def add(prefix, fs):
        for f in fs:
            folds.append(splits.Fold(f"{prefix}:{f.name}", f.train, f.val, f.test, f.leaky, f.meta))

    sub = np.flatnonzero(is_img)
    try:
        sub_table = FootprintTable({c: np.asarray(v)[sub] for c, v in table.columns.items()})
        a = splits.logo_folds(sub_table) + [splits.interleave_split(sub_table)]
        add("A", [splits.Fold(f.name, sub[f.train], sub[f.val], sub[f.test], f.leaky, f.meta) for f in a])
    except ValueError as e:
        report["skipped"].append(f"Arena A: {e}")
    n_days = len(set(table["day"]))
    try:
        add("B", exclude_days_from_training(table, splits.group_kfold(table, "day", min(k, n_days), seed), a_days))
    except ValueError as e:
        report["skipped"].append(f"Arena B day k-fold: {e}")
    years = sorted(set(table["year"]))
    if len(years) > 1:
        add("B", exclude_days_from_training(table, [splits.holdout(table, table["year"] == years[-1], f"year:{years[-1]}", seed=seed)], a_days))
    else:
        report["skipped"].append("Arena B year holdout: only one year")
    months = sorted(set(table["month"]))
    if len(months) > 2:
        add("B", exclude_days_from_training(table, splits.group_kfold(table, "month", len(months), seed), a_days))
    else:
        report["skipped"].append("Arena B leave-one-month-out: fewer than 3 months")
    try:
        add("S", splits.spatial_block_folds(table, block_km=block_km, buffer_km=buffer_km, k=k, seed=seed))
    except ValueError as e:
        report["skipped"].append(f"spatial blocks: {e}")
    for f in folds:
        reuse = splits.pixel_reuse_fraction(table, np.concatenate([f.train, f.val]), f.test, radius_km=leakage_radius_km)
        report["folds"][f.name] = {"n_train": len(f.train), "n_val": len(f.val), "n_test": len(f.test), "leaky": f.leaky,
                                   "max_reuse": float(reuse.max()) if len(reuse) else 0.0}
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        splits.save_manifest(path, table, folds, meta={"built_by": "flux1km.e0", "seed": seed, "image_granules": sorted(img)})
        report["path"] = str(path)
    return folds, report


def arena_folds(folds: list[splits.Fold], arena: str) -> list[splits.Fold]:
    """The folds of one arena ("A", "B" or "S") of a frozen manifest.

    Example:
        >>> arena_folds([splits.Fold("A:logo:x", np.array([]), np.array([]), np.array([]))], "A")[0].name
        'A:logo:x'
    """
    return [f for f in folds if f.name.startswith(f"{arena}:")]


# ------------------------------------------------------------------ advisor report
ADVISOR_BUGS = [
    ("LiqPhsf uses the whole granule's phase array CldPhs instead of the footprint's cphsfov",
     "stored values (~400) are meaningless", "LiqPhsf not used until fixed"),
    ("land/water code parsed from the reversed bit string (bits swapped: coastal <-> desert)",
     "landf is a mean code, not a fraction", "only landf == 0 (all water) is used"),
    ("same bit-order swap in the 2-bit cloudiness code (uncertain <-> probably clear)",
     "none for stored files (relaxed mask uses only 'cloudy'); the 'strict' option would be wrong", "keep 'relaxed'"),
    ("binary2 = binary[::-1] re-reads byte 1 instead of byte 2",
     "thin-cirrus / non-cloud-obstruction bits wrong (not stored)", "none"),
    ("cldmskfov = BWmsk[:subsubRi:, ...] stray colon in the bottom-edge branch",
     "wrong CF for footprints touching the granule's last row", "edge footprints can be dropped"),
    ("RAA fold can exceed 180 deg when SolarAzimuth - SensorAzimuth < -180",
     "rare wrong pixel RAA", "features.pixel_geometry folds it back"),
    ("CERESraa[fov+1] read without a bounds check", "crash on the last footprint of a file", "none"),
]


def advisor_bug_report() -> str:
    """The list of problems in MOD_CERES_allsky to send to the advisor (PLAN.md 2.3), as text.

    Example:
        >>> "LiqPhsf" in advisor_bug_report()
        True
    """
    lines = ["Problems found in MOD_CERES_allsky (pre_master/advisor_code/Twilight_module.py).",
             "None affects the radiances, moments or labels (Fsw, Flw).", ""]
    for i, (what, effect, action) in enumerate(ADVISOR_BUGS, 1):
        lines += [f"{i}. {what}", f"   effect: {effect}", f"   what we do meanwhile: {action}"]
    return "\n".join(lines)


# ------------------------------------------------------------------ driver
def run_e0(root: str | Path, out_dir: str | Path, psf_path: str | None = None, run_baselines: bool = False,
           min_group_size: int = 100, seed: int = 0, baseline_budget: int = 10, max_footprints: int = 400,
           maxiter: int = 800) -> dict:
    """Run every E0 check on one data root and write E0_report.txt, the fitted operator and the manifest.

    Returns:
        Dict with every sub-result and "report_path".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"inventory": inventory(root)}
    ids = io.find_granules(root, require=("scalars",))
    granules = [io.load_granule(root, g, parts=("moments", "scalars", "images")) for g in ids]
    with_img = [g for g in granules if g.images is not None and g.moments is not None]
    res["ranges"] = value_ranges(granules)
    psf = aggregation.load_psf_file(psf_path) if psf_path else None
    if with_img:
        res["moments_vs_patches"] = moments_vs_patches(with_img, psf)
        res["fov_orientation"] = check_fov_orientation(with_img)
        res["operator"] = fit_aggregation_operator(with_img, out_dir / "fitted_operator.npz", max_footprints, seed, maxiter)
        res["overlap"] = {g.granule_id: footprint_overlap(g) for g in with_img}
        res["stripes"] = stripe_scores(with_img)
    have_moments = [g for g in granules if g.moments is not None]
    if have_moments:
        groups = splits.group_granules([g.granule_id for g in have_moments],
                                       sizes={g.granule_id: g.n_footprints for g in have_moments}, min_size=min_group_size)
        table = build_table(have_moments, groups)
        if "landf" in table:
            res["water_only_footprints"] = int(np.sum(np.asarray(table["landf"]) == 0))
        folds, mrep = build_frozen_manifest(table, out_dir / "split_manifest.json",
                                            image_granules=[g.granule_id for g in with_img], seed=seed)
        res["manifest"] = mrep
        if run_baselines and folds:
            from . import benchmark

            cfg = benchmark.BenchmarkConfig(models=["B-clim", "M0-ridge", "M0-allsky37", "M0-advisor"], budget=baseline_budget,
                                            psf_path=psf_path, operator="ceres_psf" if psf_path else "fov_ellipse")
            usable = [f for f in folds if f.name.startswith("B:") and len(f.train) and len(f.test)]
            res["baselines"] = benchmark.run_benchmark(have_moments, table, usable, cfg, out_dir=out_dir)["summary"]
    (out_dir / "advisor_bug_report.txt").write_text(advisor_bug_report() + "\n")
    res["report_path"] = str(out_dir / "E0_report.txt")
    Path(res["report_path"]).write_text(format_e0_report(res))
    return res


def format_e0_report(res: dict) -> str:
    """Plain-text E0 report (see run_e0)."""
    inv = res["inventory"]
    lines = ["Stage E0 report (PLAN.md section 2)", f"written {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             section("1. Inventory"), f"root: {inv['root']}",
             f"granules: {inv['n_granules']}   footprints: {inv['n_footprints']}   files per part: {inv['parts']}",
             f"days: {len(inv['days'])}   months: {inv['months']}"]
    for part, keys in inv["keys"].items():
        if keys:
            lines.append(f"  {part}: " + ", ".join(f"{k}{list(v)}" for k, v in keys.items()))
    lines += [f"  INCONSISTENT: {m}" for m in inv["inconsistent"]]
    if "water_only_footprints" in res:
        lines.append(f"all-water footprints (landf == 0, phase-1 population): {res['water_only_footprints']}")
    r = res["ranges"]
    lines += [section("2. Value ranges and units"),
              format_table(["scalar", "min", "max", "mean", "NaN frac"],
                           [[k, v["min"], v["max"], v["mean"], v["nan_frac"]] for k, v in r["scalars"].items()], [12, 12, 12, 12, 10]),
              f"FOVorientation values: {r['fov_values']}"]
    lines += [f"  FLAG: {f}" for f in r["flags"]] or ["  no range flags"]
    if "moments_vs_patches" in res:
        fo = res["fov_orientation"]
        lines += [section("3. Moments vs patch statistics (RMS relative error of mean and std)")]
        lines += [f"  {op:<14s} {v:.3f}" for op, v in res["moments_vs_patches"].items()]
        lines += [f"FOVorientation check (fov_ellipse): as stored {fo['as_stored']:.3f}, inverted {fo['inverted']:.3f} -> "
                  f"{'meaning confirmed' if fo['as_stored'] < fo['inverted'] else 'CHECK: inverted fits better'}"]
        op = res["operator"]
        lines += [section("4. Fitted aggregation operator H"),
                  f"ellipse (sigma_major, sigma_minor, angle) = {tuple(round(x, 2) for x in op['ellipse'])}, error {op['ellipse_error']:.3f}",
                  f"recovered PSF error: {op['psf_error'] if op['psf_error'] is None else round(op['psf_error'], 3)} "
                  f"{op.get('psf_note', '')}", f"saved to: {op['path']} (use it with --psf until ceres_psf.csv arrives)"]
        lines += [section("5. Footprint overlap per granule"),
                  format_table(["granule", "spacing_med", "spacing_min", "shared_max", "shared_med"],
                               [[g, o["spacing_km_median"], o["spacing_km_min"], o["shared_max"], o["shared_median"]]
                                for g, o in res["overlap"].items()], [16, 12, 12, 12, 12])]
        st = res["stripes"]
        lines += [section("6. Detector striping per channel (bad channels)"),
                  "  " + ", ".join(f"{k}:{v:.2f}" for k, v in st["scores"].items()),
                  f"flagged as systematically bad: {st['bad']} (schema.BAD_CHANNELS = {list(schema.BAD_CHANNELS)})"]
    if "manifest" in res:
        m = res["manifest"]
        rows = [[n, str(v["leaky"]), str(v["n_train"]), str(v["n_val"]), str(v["n_test"]), v["max_reuse"]] for n, v in m["folds"].items()]
        bad = [n for n, v in m["folds"].items() if not v["leaky"] and v["max_reuse"] > 0]
        lines += [section("7. Frozen split manifest and leakage check"), f"manifest: {m['path']}",
                  format_table(["fold", "leaky", "train", "val", "test", "max reuse"], rows, [34, 7, 8, 8, 8, 10]),
                  f"LEAKAGE: {bad}" if bad else "leakage check: every non-leaky fold has zero pixel reuse"]
        lines += [f"  skipped: {s}" for s in m["skipped"]]
    if "baselines" in res:
        from .report import format_report

        lines += [section("8. Legacy baselines re-run in the new protocol (Arena B)"), format_report(res["baselines"], "M0 re-run")]
    lines += [section("9. To report to the advisor"), advisor_bug_report()]
    return "\n".join(lines) + "\n"


def main(argv=None) -> dict:
    """Command line: run E0 on a data root (or on synthetic demo data)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", help="folder with moments/ scalars/ images/ (default with --demo: a temp folder)")
    p.add_argument("--zip", help="unpack this zip (e.g. Data_toGuy.zip) into --data-root first")
    p.add_argument("--out", default="benchmark_outputs/e0")
    p.add_argument("--psf", help="ceres_psf.csv or a recovered PSF .npz")
    p.add_argument("--run-baselines", action="store_true", help="also re-run M0 in Arena B")
    p.add_argument("--min-group-size", type=int, default=100)
    p.add_argument("--demo", action="store_true")
    a = p.parse_args(argv)
    root = a.data_root
    if a.demo:
        from . import synthetic

        root = root or tempfile.mkdtemp(prefix="flux1km_e0_")
        synthetic.write_demo_dataset(root, n_rows=3, n_cols=8)
    if not root:
        p.error("give --data-root or --demo")
    if a.zip:
        unpack_zip(a.zip, root)
    roots = find_data_roots(root) or [Path(root)]
    results = {}
    for r in roots:
        out = Path(a.out) / (r.name if len(roots) > 1 else "")
        results[str(r)] = run_e0(r, out, a.psf, a.run_baselines, a.min_group_size)
        print(f"E0 report: {results[str(r)]['report_path']}")
    return results


if __name__ == "__main__":
    main()
