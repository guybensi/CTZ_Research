"""Run every method through one protocol and decide (PLAN.md sections 4 and 6).

For each fold of a frozen manifest and each model: tune on the validation set within a shared
budget, refit, predict the test footprints through the common aggregation operator (plus the
sensitivity operators), and record everything. Then score, test significance and apply the
lexicographic decision rule.
"""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import aggregation, decision, metrics, physics, report, sampling, splits
from .features import Batch, make_batch
from .io import Granule
from .methods import _common
from .models import MODELS, get_model, predict_map_chunked
from .table import FootprintTable

TREND_COLUMNS = ("SZA", "VZA", "RAA", "glint")


@dataclass
class BenchmarkConfig:
    models: list[str]
    operator: str = "fov_ellipse"
    sensitivity_operators: tuple[str, ...] = ("uniform", "gaussian_psf")
    budget: int = 40
    seeds: tuple[int, ...] = (0,)
    sampling: str = "S0"
    sampling_params: dict = field(default_factory=dict)
    model_params: dict = field(default_factory=dict)  # model name -> fixed params (e.g. input_set)
    check_leakage: bool = True
    leakage_radius_km: float = 20.0
    noise_floor: float = 2.0       # W m^-2: fallback floor when no regime breakdown exists
    # CERES ADM noise floor per regime (Su et al. 2015; to verify): a more complex model must gain more.
    noise_floor_by_regime: dict = field(default_factory=lambda: {"clear": 1.9, "broken": 10.0, "overcast": 6.0})
    flip_tol: float = 0.5          # W m^-2: ranking flips ignore macro-RMSE differences smaller than this
    flip_is_gate: bool = False     # True: a ranking flip between operators drops the method (the rule before 25.11.2026)
    image_data_scarce: bool | None = None  # tie default M2 instead of M3; None = decide from the data (< 10 image granules)
    trend_gate: float = 5.0        # W m^-2: max residual trend effect across the range of an angle
    swath_gate: float = 5.0        # W m^-2: max forward/backscatter residual difference
    pcs_target: tuple[float, float] = (70.0, 90.0)  # reported only since 25.11.2026 (no longer a gate)
    clear_bias_gate: float = 5.0   # W m^-2: max |bias| vs CERES on clear footprints (MYD35 CF < 5 % and Clrf >= clear_min_clrf)
    clear_min_clrf: float = 95.0   # %: CERES clear-strong fraction a clear footprint needs
    clear_min_n: int = 30          # fewer clear footprints: the clear-sky gate is not applied
    patch: tuple[int, int] = (27, 55)
    psf_path: str | None = None    # ceres_psf.csv; required by the "ceres_psf" operator
    cache_dir: str | None = None   # disk cache for the folds' images and network inputs (None = all in memory)
    physics_max_footprints: int = 5000  # maps kept per (model, fold, seed) for the physics checks (random subset)
    progress: bool = False         # print one line per fold and per (model, seed): time, duration, memory


_PSF_CACHE: dict[str, np.ndarray] = {}


def _psf(cfg: BenchmarkConfig) -> np.ndarray | None:
    if cfg.psf_path is None:
        return None
    if cfg.psf_path not in _PSF_CACHE:
        _PSF_CACHE[cfg.psf_path] = aggregation.load_psf_file(cfg.psf_path, *cfg.patch)
    return _PSF_CACHE[cfg.psf_path]


def _weights(name: str, cfg: BenchmarkConfig, batch: Batch) -> np.ndarray:
    orientation = batch.images["FOVorientation"] if batch.has_images and "FOVorientation" in batch.images else None
    return aggregation.get_operator(name, *cfg.patch, orientation=orientation, psf=_psf(cfg))


def _footprint_pred(model, batch: Batch, w: np.ndarray, maps: np.ndarray | None) -> np.ndarray:
    if maps is not None:
        return aggregation.aggregate(maps, w)
    return model.predict_footprint(batch, w)


def fixed_params(name: str, cfg: BenchmarkConfig, seed: int = 0) -> dict:
    """Parameters the protocol imposes on a model: the seed, the training operator, and cfg.model_params.

    Aggregation-loss models (needs_operator) are trained with the operator they are scored with.

    Example:
        >>> fixed_params("M2", BenchmarkConfig(models=["M2"]), seed=3)["seed"]
        3
    """
    out = {"seed": seed}
    if MODELS[name].needs_operator:
        out.update(operator=cfg.operator, psf_path=cfg.psf_path)
    return {**out, **cfg.model_params.get(name, {})}


def tune(name: str, train: Batch, val: Batch, cfg: BenchmarkConfig, seed: int = 0) -> tuple[dict, float]:
    """Pick the best parameters from the model's grid (at most cfg.budget candidates) by val RMSE.

    With an empty validation set the first grid entry is used.
    """
    grid = get_model(name).param_grid()[: cfg.budget]
    fixed = fixed_params(name, cfg, seed)
    if len(val) == 0 or len(grid) == 1:
        return {**grid[0], **fixed}, np.nan
    w = _weights(cfg.operator, cfg, val)
    best, best_score = None, np.inf
    for i, params in enumerate(grid, 1):
        p, t0 = {**params, **fixed}, datetime.now()
        try:
            model = get_model(name, **p).fit(train, val)
            pred = model.predict_footprint(val, w)
        except (ValueError, np.linalg.LinAlgError) as e:
            _progress(cfg, f"    {name} candidate {i}/{len(grid)} {params}: skipped ({type(e).__name__})")
            continue
        score = metrics.footprint_scores(val.label, pred)["rmse"]
        _progress(cfg, f"    {name} candidate {i}/{len(grid)} {params}: val RMSE {score:.2f}"
                       f" in {(datetime.now() - t0).total_seconds() / 60:.1f} min")
        if score < best_score:
            best, best_score = p, score
    if best is None:
        raise RuntimeError(f"no parameter set of {name} could be fitted")
    return best, float(best_score)


def run_benchmark(granules: list[Granule], table: FootprintTable, folds: list[splits.Fold],
                  cfg: BenchmarkConfig, out_dir: str | Path | None = None) -> dict:
    """Run all models on all folds; return {"records": list[dict], "summary": dict}.

    With cfg.cache_dir, each fold's train/val/test images (and the networks' standardised inputs) are
    written to a temporary folder under it and read back as memory maps; the folder is deleted when the
    fold is done. Load the granules with io.load_dataset(..., lazy_images=True) so that their images are not
    held in memory either. The results are the same as without the cache.

    Raises:
        LeakageError: If cfg.check_leakage and a non-leaky fold shares pixels between test and
            train/validation.
    """
    unknown = [m for m in cfg.models if m not in MODELS]
    if unknown:
        raise KeyError(f"unknown models {unknown}")
    records, maps_store, fold_info, errors = [], {}, [], []
    if cfg.cache_dir:
        Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    for k_fold, fold in enumerate(folds, 1):
        store = tempfile.mkdtemp(prefix="fold_", dir=cfg.cache_dir) if cfg.cache_dir else None
        try:
            if cfg.check_leakage:
                splits.check_fold(table, fold, radius_km=cfg.leakage_radius_km)
            tr_mask, tr_w = sampling.apply(cfg.sampling, table, rows=fold.train, **cfg.sampling_params)
            train = make_batch(granules, table, fold.train[tr_mask], tr_w[tr_mask], store=store)
            val = make_batch(granules, table, fold.val, store=store)
            test = make_batch(granules, table, fold.test, store=store)
            fold_info.append({"name": fold.name, "leaky": fold.leaky, "n_train": len(train), "n_val": len(val), "n_test": len(test)})
            _progress(cfg, f"fold {k_fold}/{len(folds)} {fold.name}: train {len(train)}, val {len(val)}, test {len(test)}")
            for name in cfg.models:
                cls = MODELS[name]
                if cls.produces_map and not test.has_images:
                    continue
                for seed in cfg.seeds:
                    t0 = datetime.now()
                    _progress(cfg, f"  {name} seed {seed}: start")
                    try:
                        params, val_rmse = tune(name, train, val, cfg, seed)
                        model = get_model(name, **params).fit(train, val)
                        maps = predict_map_chunked(model, test) if (cls.produces_map and test.has_images) else None
                        preds = {op: _footprint_pred(model, test, _weights(op, cfg, test), maps)
                                 for op in (cfg.operator, *cfg.sensitivity_operators)}
                    except (ValueError, RuntimeError, ImportError, KeyError, np.linalg.LinAlgError) as e:
                        # one failing method must not stop a long real-data run; the error goes to the report
                        errors.append({"model": name, "fold": fold.name, "seed": seed, "error": f"{type(e).__name__}: {e}"})
                        _progress(cfg, f"  {name} seed {seed}: ERROR {type(e).__name__}: {e}")
                        _sweep_cache(store)
                        continue
                    _progress(cfg, f"  {name} seed {seed}: done in {(datetime.now() - t0).total_seconds() / 60:.1f} min")
                    _sweep_cache(store)
                    if maps is not None:
                        keep = _physics_subset(len(test), cfg.physics_max_footprints, seed)
                        maps_store[(name, fold.name, seed)] = (np.asarray(test.rows)[keep], maps[keep])
                    for k, row in enumerate(test.rows):
                        rec = {"model": name, "fold": fold.name, "leaky": fold.leaky, "seed": seed, "row": int(row),
                               "granule": table["granule"][row], "group": table["group"][row],
                               "label": float(test.label[k]), "pred": float(preds[cfg.operator][k]),
                               "regime": sampling.regime(table["CF"][row:row + 1])[0] if "CF" in table else "unknown"}
                        for op in cfg.sensitivity_operators:
                            rec[f"pred_{op}"] = float(preds[op][k])
                        for c in TREND_COLUMNS + ("Clrf",):
                            if c in table:
                                rec[c] = float(table[c][row])
                        records.append(rec)
                    fold_info[-1].setdefault("params", {})[name] = params
                    fold_info[-1].setdefault("val_rmse", {})[name] = val_rmse
        finally:
            if store is not None:
                shutil.rmtree(store, ignore_errors=True)
    pixel = _pixel_and_physics(granules, table, folds, maps_store, cfg)
    summary = summarize(records, cfg, pixel)
    summary["folds"] = fold_info
    summary["errors"] = errors
    summary["config"] = asdict(cfg)
    summary["n_image_granules"] = sum(g.images is not None for g in granules)
    summary["decision"] = decide(summary, cfg)
    if out_dir is not None:
        _write(out_dir, records, summary)
    return {"records": records, "summary": summary}


def memory_status() -> str:
    """This process's real (anonymous) memory and the machine's available memory, from /proc ("" elsewhere).

    Example:
        >>> isinstance(memory_status(), str)
        True
    """
    vals = {}
    for path, keys in (("/proc/self/status", ("RssAnon",)), ("/proc/meminfo", ("MemAvailable",))):
        try:
            with open(path) as f:
                for line in f:
                    k = line.split(":")[0]
                    if k in keys:
                        vals[k] = int(line.split()[1]) / 1048576
        except OSError:
            return ""
    if len(vals) < 2:
        return ""
    return f"mem {vals['RssAnon']:.1f} GB (machine available {vals['MemAvailable']:.1f} GB)"


def _progress(cfg: BenchmarkConfig, msg: str) -> None:
    if cfg.progress:
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}  {memory_status()}", flush=True)


def _sweep_cache(store: str | None) -> int:
    """Delete a model's leftover cache files in a fold folder (inputs X_*, pixel tables rows_*, M6f's X6f_*).

    The fold's batch images (batch_*/<key>.npy) stay until the fold ends. Each method deletes its own files
    when a fit ends; this sweep also catches the files of a fit that raised. Returns the number deleted.

    Example:
        >>> _sweep_cache(None)
        0
    """
    if store is None:
        return 0
    n = 0
    for pattern in ("X_*.npy", "rows_*.npy", "X6f_*.npy"):
        for f in Path(store).glob(f"batch_*/{pattern}"):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    return n


def _physics_subset(n: int, cap: int, seed: int) -> np.ndarray:
    """Positions of the test footprints whose maps are kept for the physics checks: all if n <= cap.

    Example:
        >>> _physics_subset(4, 10, 0).tolist(), len(_physics_subset(100, 10, 0))
        ([0, 1, 2, 3], 10)
    """
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


def _pixel_and_physics(granules, table, folds, maps_store, cfg) -> dict:
    """Pixel scores vs synthetic truth and physics checks, per model (non-leaky folds only)."""
    out: dict[str, dict] = {}
    fold_by_name = {f.name: f for f in folds}
    per_model: dict[str, list] = {}
    for (name, fname, _seed), (rows, maps) in maps_store.items():
        fold = fold_by_name[fname]
        if fold.leaky:
            continue
        per_model.setdefault(name, []).append((rows, maps))
    for name, items in per_model.items():
        truth, pred, dist, sza = [], [], [], []
        for rows, maps in items:
            b = make_batch(granules, table, rows)
            pred.append(maps)
            sza.append(b.images["Imviews"][:, 0])
            if b.truth is not None:
                truth.append(b.truth)
                dist.append(b.dist_km)
            else:  # real data: distance to the nearest cloud of the MYD35 strict mask, or of the 0.66 um proxy
                dist.append(_common.distance_to_bright(cloud_mask(b)))
        res = {}
        if truth and len(truth) == len(pred):
            res["pixel"] = metrics.pixel_scores(np.concatenate(truth), np.concatenate(pred))
        res["physics"] = physics.run_checks(np.concatenate(pred), np.concatenate(dist), np.concatenate(sza),
                                            pcs_target=cfg.pcs_target)
        res["physics"]["distance_source"] = ("synthetic truth" if truth else
                                             "MYD35 strict" if "ImCloud" in b.images else "0.66 um brightness proxy")
        out[name] = res
    return out


def cloud_mask(batch: Batch) -> np.ndarray:
    """Cloudy pixels (N, H, W) for the physics checks: MYD35 strict (confidence cloudy or uncertain) when the images
    carry ImCloud, else the 0.66 um brightness proxy.

    The proxy misses most clouds at low sun (cloud-fraction correlation 0.69 with CERES in img6, MYD35 0.99), so
    "clear sky far from cloud" took in cloudy pixels; the strict MYD35 mask agrees best with CERES's Clrf.

    Example:
        >>> # cloud_mask(test_batch).shape == test_batch.images["ImSW"].shape[:1] + (27, 55)
    """
    if batch.images is not None and "ImCloud" in batch.images:
        return np.asarray(batch.images["ImCloud"]) <= 1
    return _common.bright_mask(batch)


def _arr(records, key, sel):
    return np.array([r[key] for r, s in zip(records, sel) if s], dtype=float)


def summarize(records: list[dict], cfg: BenchmarkConfig, pixel: dict | None = None) -> dict:
    """Scores per model on non-leaky folds (+ interleave scores separately) and pairwise tests."""
    models = list(dict.fromkeys(r["model"] for r in records))
    out: dict = {"models": {}, "comparisons": {}}
    for name in models:
        sel = np.array([r["model"] == name and not r["leaky"] for r in records])
        leak = np.array([r["model"] == name and r["leaky"] for r in records])
        y, p = _arr(records, "label", sel), _arr(records, "pred", sel)
        reg = np.array([r["regime"] for r, s in zip(records, sel) if s], dtype=object)
        res = {"overall": metrics.footprint_scores(y, p) if len(y) else {}, "by_regime": metrics.scores_by(y, p, reg) if len(y) else {},
               "macro_rmse": metrics.macro_rmse(y, p, reg) if len(y) else np.nan, "uses_test_labels": MODELS[name].uses_test_labels,
               "complexity": MODELS[name].complexity, "produces_map": MODELS[name].produces_map}
        for op in cfg.sensitivity_operators:
            ps = _arr(records, f"pred_{op}", sel)
            res[f"macro_rmse_{op}"] = metrics.macro_rmse(y, ps, reg) if len(y) else np.nan
        clrf = _arr(records, "Clrf", sel) if len(y) and "Clrf" in records[0] else None
        res["clear_check"] = metrics.clear_sky_check(y, p, reg, clrf, cfg.clear_min_clrf, cfg.clear_min_n) if len(y) else {}
        resid = p - y
        res["trend_effect"] = {}
        for c in TREND_COLUMNS:
            x = _arr(records, c, sel) if len(y) and c in records[0] else np.array([])
            if len(x) > 2 and np.ptp(x) > 0:
                slope, pval = metrics.residual_trend(resid, x)
                res["trend_effect"][c] = {"effect_wm2": float(abs(slope) * np.ptp(x)), "p": pval}
        if len(y) and "RAA" in records[0]:
            res["swath_asymmetry"] = metrics.swath_side_asymmetry(resid, _arr(records, "RAA", sel), _arr(records, "VZA", sel))
        if leak.any():
            res["interleave"] = metrics.footprint_scores(_arr(records, "label", leak), _arr(records, "pred", leak))
        res.update((pixel or {}).get(name, {}))
        out["models"][name] = res
    ranked = [m for m in models if not MODELS[m].uses_test_labels]
    pairs, pvals = [], []
    for i, a in enumerate(ranked):
        for b in ranked[i + 1:]:
            la, lb = _unit_losses(records, a), _unit_losses(records, b)
            common = sorted(set(la) & set(lb))
            if len(common) < 2:
                continue
            xa, xb = np.array([la[u] for u in common]), np.array([lb[u] for u in common])
            boot = metrics.paired_bootstrap(xa, xb, n_boot=5000)
            try:
                pw = metrics.wilcoxon_p(xa, xb)
            except ValueError:
                pw = np.nan
            dm, pdm = metrics.diebold_mariano(xa, xb)
            pairs.append((a, b, {**boot, "dm_stat": dm, "dm_p": pdm}))
            pvals.append(pw)
    adj = metrics.holm(np.nan_to_num(pvals, nan=1.0)) if pvals else []
    adj_boot = metrics.holm([pb["p_boot"] for _, _, pb in pairs]) if pairs else []
    for (a, b, boot), pw, pa, pab in zip(pairs, pvals, adj, adj_boot):
        rmse_a = out["models"][a]["overall"].get("rmse", np.nan)
        rmse_b = out["models"][b]["overall"].get("rmse", np.nan)
        out["comparisons"][f"{a} vs {b}"] = {**boot, "wilcoxon_p": pw, "p_holm": float(pa), "p_holm_boot": float(pab),
                                             "rmse_diff": rmse_a - rmse_b}
    return out


def _unit_losses(records, model) -> dict:
    sel = [r for r in records if r["model"] == model and not r["leaky"]]
    units, loss = metrics.unit_losses([r["label"] for r in sel], [r["pred"] for r in sel], [r["group"] for r in sel])
    return dict(zip(units.tolist(), loss.tolist()))


def decide(summary: dict, cfg: BenchmarkConfig) -> dict:
    """The decision rule of PLAN.md section 6; see flux1km.decision.decide."""
    return decision.decide(summary, cfg)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _write(out_dir, records, summary) -> Path:
    run = Path(out_dir) / f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    run.mkdir(parents=True, exist_ok=True)
    if records:
        with open(run / "predictions.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
    (run / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    summary["run_dir"] = str(run)
    summary["results_txt"] = str(run / "results.txt")
    (run / "results.txt").write_text(report.format_report(summary))
    return run
