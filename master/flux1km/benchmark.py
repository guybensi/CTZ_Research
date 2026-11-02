"""Run every method through one protocol and decide (PLAN.md sections 4 and 6).

For each fold of a frozen manifest and each model: tune on the validation set within a shared
budget, refit, predict the test footprints through the common aggregation operator (plus the
sensitivity operators), and record everything. Then score, test significance and apply the
lexicographic decision rule.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import aggregation, decision, metrics, physics, report, sampling, splits
from .features import Batch, make_batch
from .io import Granule
from .methods import _common
from .models import MODELS, get_model
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
    flip_tol: float = 0.5          # W m^-2: ranking-flip gate ignores macro-RMSE differences smaller than this
    image_data_scarce: bool | None = None  # tie default M2 instead of M3; None = decide from the data (< 10 image granules)
    trend_gate: float = 5.0        # W m^-2: max residual trend effect across the range of an angle
    swath_gate: float = 5.0        # W m^-2: max forward/backscatter residual difference
    pcs_target: tuple[float, float] = (70.0, 90.0)
    patch: tuple[int, int] = (27, 55)
    psf_path: str | None = None    # ceres_psf.csv; required by the "ceres_psf" operator


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
    for params in grid:
        p = {**params, **fixed}
        try:
            model = get_model(name, **p).fit(train, val)
            pred = model.predict_footprint(val, w)
        except (ValueError, np.linalg.LinAlgError):
            continue
        score = metrics.footprint_scores(val.label, pred)["rmse"]
        if score < best_score:
            best, best_score = p, score
    if best is None:
        raise RuntimeError(f"no parameter set of {name} could be fitted")
    return best, float(best_score)


def run_benchmark(granules: list[Granule], table: FootprintTable, folds: list[splits.Fold],
                  cfg: BenchmarkConfig, out_dir: str | Path | None = None) -> dict:
    """Run all models on all folds; return {"records": list[dict], "summary": dict}.

    Raises:
        LeakageError: If cfg.check_leakage and a non-leaky fold shares pixels between test and
            train/validation.
    """
    unknown = [m for m in cfg.models if m not in MODELS]
    if unknown:
        raise KeyError(f"unknown models {unknown}")
    records, maps_store, fold_info, errors = [], {}, [], []
    for fold in folds:
        if cfg.check_leakage:
            splits.check_fold(table, fold, radius_km=cfg.leakage_radius_km)
        tr_mask, tr_w = sampling.apply(cfg.sampling, table, rows=fold.train, **cfg.sampling_params)
        train = make_batch(granules, table, fold.train[tr_mask], tr_w[tr_mask])
        val = make_batch(granules, table, fold.val)
        test = make_batch(granules, table, fold.test)
        fold_info.append({"name": fold.name, "leaky": fold.leaky, "n_train": len(train), "n_val": len(val), "n_test": len(test)})
        for name in cfg.models:
            cls = MODELS[name]
            if cls.produces_map and not test.has_images:
                continue
            for seed in cfg.seeds:
                try:
                    params, val_rmse = tune(name, train, val, cfg, seed)
                    model = get_model(name, **params).fit(train, val)
                    maps = model.predict_map(test) if (cls.produces_map and test.has_images) else None
                    preds = {op: _footprint_pred(model, test, _weights(op, cfg, test), maps)
                             for op in (cfg.operator, *cfg.sensitivity_operators)}
                except (ValueError, RuntimeError, ImportError, KeyError, np.linalg.LinAlgError) as e:
                    # one failing method must not stop a long real-data run; the error goes to the report
                    errors.append({"model": name, "fold": fold.name, "seed": seed, "error": f"{type(e).__name__}: {e}"})
                    continue
                if maps is not None:
                    maps_store[(name, fold.name, seed)] = maps
                for k, row in enumerate(test.rows):
                    rec = {"model": name, "fold": fold.name, "leaky": fold.leaky, "seed": seed, "row": int(row),
                           "granule": table["granule"][row], "group": table["group"][row],
                           "label": float(test.label[k]), "pred": float(preds[cfg.operator][k]),
                           "regime": sampling.regime(table["CF"][row:row + 1])[0] if "CF" in table else "unknown"}
                    for op in cfg.sensitivity_operators:
                        rec[f"pred_{op}"] = float(preds[op][k])
                    for c in TREND_COLUMNS:
                        if c in table:
                            rec[c] = float(table[c][row])
                    records.append(rec)
                fold_info[-1].setdefault("params", {})[name] = params
                fold_info[-1].setdefault("val_rmse", {})[name] = val_rmse
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


def _pixel_and_physics(granules, table, folds, maps_store, cfg) -> dict:
    """Pixel scores vs synthetic truth and physics checks, per model (non-leaky folds only)."""
    out: dict[str, dict] = {}
    fold_by_name = {f.name: f for f in folds}
    per_model: dict[str, list] = {}
    for (name, fname, _seed), maps in maps_store.items():
        fold = fold_by_name[fname]
        if fold.leaky:
            continue
        per_model.setdefault(name, []).append((fold, maps))
    for name, items in per_model.items():
        truth, pred, dist, sza = [], [], [], []
        for fold, maps in items:
            b = make_batch(granules, table, fold.test)
            pred.append(maps)
            sza.append(b.images["Imviews"][:, 0])
            if b.truth is not None:
                truth.append(b.truth)
                dist.append(b.dist_km)
            else:  # real data: distance to the nearest bright (cloud-proxy) pixel from the MODIS radiances
                dist.append(_common.distance_to_bright(_common.bright_mask(b)))
        res = {}
        if truth and len(truth) == len(pred):
            res["pixel"] = metrics.pixel_scores(np.concatenate(truth), np.concatenate(pred))
        res["physics"] = physics.run_checks(np.concatenate(pred), np.concatenate(dist), np.concatenate(sza),
                                            pcs_target=cfg.pcs_target)
        res["physics"]["distance_source"] = "synthetic truth" if truth else "0.66 um brightness proxy"
        out[name] = res
    return out


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
