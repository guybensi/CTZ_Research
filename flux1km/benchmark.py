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

from . import aggregation, metrics, physics, sampling, splits
from .features import Batch, make_batch
from .io import Granule
from .models import MODELS, get_model
from .table import FootprintTable

TREND_COLUMNS = ("SZA", "VZA", "RAA", "glint")


@dataclass
class BenchmarkConfig:
    models: list[str]
    operator: str = "uniform"
    sensitivity_operators: tuple[str, ...] = ("gaussian_psf",)
    budget: int = 40
    seeds: tuple[int, ...] = (0,)
    sampling: str = "S0"
    sampling_params: dict = field(default_factory=dict)
    model_params: dict = field(default_factory=dict)  # model name -> fixed params (e.g. input_set)
    check_leakage: bool = True
    leakage_radius_km: float = 20.0
    noise_floor: float = 2.0       # W m^-2: a more complex model must gain more than this
    trend_gate: float = 5.0        # W m^-2: max residual trend effect across the range of an angle
    swath_gate: float = 5.0        # W m^-2: max forward/backscatter residual difference
    pcs_target: tuple[float, float] = (70.0, 90.0)
    patch: tuple[int, int] = (27, 55)


def _weights(name: str, cfg: BenchmarkConfig) -> np.ndarray:
    return aggregation.get_operator(name, *cfg.patch)


def _footprint_pred(model, batch: Batch, w: np.ndarray, maps: np.ndarray | None) -> np.ndarray:
    if maps is not None:
        return aggregation.aggregate(maps, w)
    return model.predict_footprint(batch, w)


def tune(name: str, train: Batch, val: Batch, cfg: BenchmarkConfig, seed: int = 0) -> tuple[dict, float]:
    """Pick the best parameters from the model's grid (at most cfg.budget candidates) by val RMSE.

    With an empty validation set the first grid entry is used.
    """
    grid = get_model(name).param_grid()[: cfg.budget]
    fixed = cfg.model_params.get(name, {})
    if len(val) == 0 or len(grid) == 1:
        return {**grid[0], **fixed}, np.nan
    w = _weights(cfg.operator, cfg)
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
    records, maps_store, fold_info = [], {}, []
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
                params, val_rmse = tune(name, train, val, cfg, seed)
                model = get_model(name, **params).fit(train, val)
                maps = model.predict_map(test) if (cls.produces_map and test.has_images) else None
                preds = {op: _footprint_pred(model, test, _weights(op, cfg), maps)
                         for op in (cfg.operator, *cfg.sensitivity_operators)}
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
    summary["config"] = asdict(cfg)
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
        res = {}
        if truth and len(truth) == len(pred):
            res["pixel"] = metrics.pixel_scores(np.concatenate(truth), np.concatenate(pred))
            res["physics"] = physics.run_checks(np.concatenate(pred), np.concatenate(dist), np.concatenate(sza),
                                                pcs_target=cfg.pcs_target)
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
            pairs.append((a, b, boot))
            pvals.append(pw)
    adj = metrics.holm(np.nan_to_num(pvals, nan=1.0)) if pvals else []
    for (a, b, boot), pw, pa in zip(pairs, pvals, adj):
        rmse_a = out["models"][a]["overall"].get("rmse", np.nan)
        rmse_b = out["models"][b]["overall"].get("rmse", np.nan)
        out["comparisons"][f"{a} vs {b}"] = {**boot, "wilcoxon_p": pw, "p_holm": float(pa), "rmse_diff": rmse_a - rmse_b}
    return out


def _unit_losses(records, model) -> dict:
    sel = [r for r in records if r["model"] == model and not r["leaky"]]
    units, loss = metrics.unit_losses([r["label"] for r in sel], [r["pred"] for r in sel], [r["group"] for r in sel])
    return dict(zip(units.tolist(), loss.tolist()))


def decide(summary: dict, cfg: BenchmarkConfig) -> dict:
    """Lexicographic decision rule (PLAN.md section 6).

    1. Gates: physics checks (when available), residual trends and swath asymmetry within gates.
    2. Go through passing models from simple to complex; a model replaces the incumbent only if
       its macro RMSE is lower by more than the noise floor and the paired test (Holm) is
       significant at 0.05, or the incumbent is missing.
    """
    log, passing = [], []
    for name, res in summary["models"].items():
        if res.get("uses_test_labels") or not res.get("overall"):
            continue
        fails = []
        phys = res.get("physics")
        if phys is not None and not phys["passed"]:
            fails.append("physics")
        for c, t in res.get("trend_effect", {}).items():
            if t["effect_wm2"] > cfg.trend_gate and t["p"] < 0.01:
                fails.append(f"trend:{c}")
        if abs(res.get("swath_asymmetry", 0.0) or 0.0) > cfg.swath_gate:
            fails.append("swath_asymmetry")
        if fails:
            log.append(f"{name} fails gates: {fails}")
        else:
            passing.append(name)
    if not passing:
        return {"winner": None, "log": log + ["no model passed the gates"]}
    passing.sort(key=lambda m: (summary["models"][m]["complexity"], summary["models"][m]["macro_rmse"]))
    incumbent = passing[0]
    for cand in passing[1:]:
        inc_s, cand_s = summary["models"][incumbent]["macro_rmse"], summary["models"][cand]["macro_rmse"]
        comp = summary["comparisons"].get(f"{cand} vs {incumbent}") or summary["comparisons"].get(f"{incumbent} vs {cand}")
        significant = comp is not None and comp["p_holm"] < 0.05
        gain = inc_s - cand_s
        if gain > cfg.noise_floor and significant:
            log.append(f"{cand} replaces {incumbent}: macro RMSE gain {gain:.2f} W/m2, p_holm={comp['p_holm']:.3g}")
            incumbent = cand
        else:
            log.append(f"{cand} does not replace {incumbent}: gain {gain:.2f} W/m2, significant={significant}")
    ranking = sorted(passing, key=lambda m: summary["models"][m]["macro_rmse"])
    return {"winner": incumbent, "ranking_by_macro_rmse": ranking, "log": log}


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
    return run
