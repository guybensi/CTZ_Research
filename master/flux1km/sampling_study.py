"""Sampling study: training-data strategies S0-S7 crossed with the leading methods (PLAN.md section 5).

Sampling changes only the TRAINING set; the test footprints of every fold stay the same. For
every strategy the benchmark is run with the 2-3 leading methods, and the study reports:
* how close the strategies are: Jaccard overlap of their footprint sets, KS distance per column
  and joint MMD to S0 on Fsw, CF, SZA, VZA and glint;
* a train-on-Si x test-on-subpopulation RMSE matrix (regimes, glint, SZA and VZA bins);
* the Kendall tau between the method rankings under each Si and under S0;
* the rule: if every tau > 0.8, sampling does not change the winner, so S7 (or S5) is fixed and
  the study moves on; otherwise the strategy is chosen together with the method.

    python3 -m flux1km.sampling_study --demo --models M2 M3 M9
    python3 -m flux1km.sampling_study --data-root DIR --manifest benchmark_outputs/e0/split_manifest.json --models M2 M3
"""
from __future__ import annotations

import argparse
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import benchmark, io, sampling, splits, synthetic
from .report import format_table, section
from .table import FootprintTable, build_table

STRATEGIES = ("S0", "S1", "S2", "S3", "S5", "S6", "S7")
TAU_THRESHOLD = 0.8


def subpopulations(table: FootprintTable) -> dict[str, np.ndarray]:
    """Boolean masks over table rows of the fixed test sub-populations.

    Example:
        >>> # masks = subpopulations(table); masks["regime:broken"].sum()
    """
    out = {}
    if "CF" in table:
        reg = sampling.regime(table["CF"])
        out.update({f"regime:{r}": reg == r for r in sampling.REGIMES})
    if "glint" in table:
        g = np.asarray(table["glint"], float)
        out["glint<25"], out["glint>=25"] = g < 25, g >= 25
    if "SZA" in table:
        s = np.asarray(table["SZA"], float)
        out["SZA<40"], out["SZA40-60"], out["SZA>=60"] = s < 40, (s >= 40) & (s < 60), s >= 60
    if "VZA" in table:
        v = np.asarray(table["VZA"], float)
        out["VZA<30"], out["VZA>=30"] = v < 30, v >= 30
    return out


def subpopulation_rmse(records: list[dict], masks: dict[str, np.ndarray], model: str) -> dict[str, float]:
    """RMSE of one model's non-leaky predictions on each sub-population (NaN if empty)."""
    rec = [r for r in records if r["model"] == model and not r["leaky"]]
    rows = np.array([r["row"] for r in rec], int)
    err = np.array([r["pred"] - r["label"] for r in rec], float)
    out = {}
    for name, m in masks.items():
        sel = m[rows] & np.isfinite(err) if len(rows) else np.zeros(0, bool)
        out[name] = float(np.sqrt(np.mean(err[sel] ** 2))) if sel.any() else np.nan
    return out


def run_sampling_study(granules, table: FootprintTable, folds: list[splits.Fold], cfg: benchmark.BenchmarkConfig,
                       strategies=STRATEGIES, strategy_params: dict | None = None, out_dir: str | Path | None = None) -> dict:
    """Run the benchmark once per strategy and compare (see the module docstring).

    Returns:
        {"per_strategy": {S: {"macro": {model: float}, "winner": str, "subpop": {model: {sub: rmse}}}},
         "tau_vs_S0": {S: float}, "similarity": dict, "recommendation": str, "results_txt": path or None}
    """
    strategy_params = strategy_params or {}
    masks = subpopulations(table)
    per: dict = {}
    for s in strategies:
        c = replace(cfg, sampling=s, sampling_params=strategy_params.get(s, {}))
        res = benchmark.run_benchmark(granules, table, folds, c)
        summ = res["summary"]
        ranked = [m for m in c.models if m in summ["models"] and not summ["models"][m].get("uses_test_labels")]
        per[s] = {"macro": {m: summ["models"][m]["macro_rmse"] for m in ranked},
                  "winner": summ["decision"]["winner"],
                  "subpop": {m: subpopulation_rmse(res["records"], masks, m) for m in ranked},
                  "errors": summ.get("errors", [])}
    ref = per.get("S0") or next(iter(per.values()))
    tau = {}
    for s, r in per.items():
        common = [m for m in ref["macro"] if m in r["macro"] and np.isfinite(ref["macro"][m]) and np.isfinite(r["macro"][m])]
        tau[s] = sampling.rank_agreement([ref["macro"][m] for m in common], [r["macro"][m] for m in common]) if len(common) > 1 else np.nan
    sim = sampling.compare_strategies(table, list(strategies), **strategy_params)
    others = [v for k, v in tau.items() if k != "S0" and np.isfinite(v)]
    if others and min(others) > TAU_THRESHOLD:
        rec = (f"all Kendall tau > {TAU_THRESHOLD}: sampling does not change the method ranking. Fix S7 (or S5) and move on.")
    else:
        winners = ", ".join(f"{k}: {v['winner']}" for k, v in per.items())
        rec = (f"some Kendall tau <= {TAU_THRESHOLD}: sampling changes the ranking, so choose the strategy together "
               f"with the method (winners per strategy: {winners}).")
    out = {"per_strategy": per, "tau_vs_S0": tau, "similarity": sim, "recommendation": rec, "results_txt": None}
    if out_dir is not None:
        run = Path(out_dir) / f"sampling_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        run.mkdir(parents=True, exist_ok=True)
        (run / "sampling_results.txt").write_text(format_sampling_report(out, list(masks)))
        out["results_txt"] = str(run / "sampling_results.txt")
    return out


def format_sampling_report(res: dict, subpops: list[str]) -> str:
    """Plain-text report of run_sampling_study."""
    per, sim = res["per_strategy"], res["similarity"]
    models = list(next(iter(per.values()))["macro"]) if per else []
    lines = ["Sampling study (PLAN.md section 5)", f"written {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             section("RECOMMENDATION"), res["recommendation"],
             section("Macro RMSE per strategy (rows) and method (columns), W m^-2"),
             format_table(["strategy"] + models + ["winner", "tau_vs_S0"],
                          [[s] + [r["macro"].get(m) for m in models] + [str(r["winner"]), res["tau_vs_S0"].get(s)] for s, r in per.items()],
                          [10] + [12] * len(models) + [12, 10])]
    for m in models:
        lines += [section(f"Train-on-Si x test-on-subpopulation RMSE: {m}"),
                  format_table(["strategy"] + [s[:11] for s in subpops],
                               [[s] + [r["subpop"][m].get(sp) for sp in subpops] for s, r in per.items()],
                               [10] + [12] * len(subpops))]
    lines += [section("How close the strategies are"),
              "footprints selected: " + ", ".join(f"{k}={v}" for k, v in sim["n"].items()),
              "Jaccard: " + ", ".join(f"{a}/{b}={v:.2f}" for (a, b), v in sim["jaccard"].items()),
              "MMD^2 to S0: " + ", ".join(f"{k}={v:.3f}" for k, v in sim.get("mmd_vs_reference", {}).items())]
    cols = sorted({c for (_, c) in sim["ks_vs_reference"]})
    lines.append(format_table(["KS vs S0"] + cols, [[s] + [sim["ks_vs_reference"].get((s, c)) for c in cols] for s in sim["n"]],
                              [10] + [9] * len(cols)))
    errs = [e for r in per.values() for e in r["errors"]]
    if errs:
        lines += [section("Errors")] + [f"  {e['model']} / {e['fold']}: {e['error']}" for e in errs]
    return "\n".join(lines) + "\n"


def main(argv=None) -> dict:
    """Command line for the sampling study."""
    from .cli import available_models, make_folds

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--demo-hard", action="store_true")
    p.add_argument("--data-root")
    p.add_argument("--manifest", help="frozen split manifest (e0.build_frozen_manifest); default: LOGO folds")
    p.add_argument("--models", nargs="+", default=["M2", "M9", "M0-ridge"], help="the 2-3 leading methods")
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--out", default="benchmark_outputs/flux1km")
    p.add_argument("--min-group-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    root = a.data_root
    if a.demo:
        root = root or tempfile.mkdtemp(prefix="flux1km_sampling_")
        synthetic.write_demo_dataset(root, seed=a.seed, n_rows=4, n_cols=12, hard=a.demo_hard)
    if not root:
        p.error("give --data-root or --demo")
    granules = io.load_dataset(root, parts=("moments", "scalars", "images", "truth"))
    groups = splits.group_granules([g.granule_id for g in granules], sizes={g.granule_id: g.n_footprints for g in granules},
                                   min_size=1 if a.demo else a.min_group_size)
    table = build_table(granules, groups)
    if a.manifest:
        folds, _ = splits.load_manifest(a.manifest, table)
        folds = [f for f in folds if not f.leaky]
    else:
        folds = make_folds(table, "logo", 5, a.seed, 100.0, 40.0)[:-1]
    cfg = benchmark.BenchmarkConfig(models=available_models(a.models), budget=a.budget)
    res = run_sampling_study(granules, table, folds, cfg, a.strategies, out_dir=a.out)
    print(res["recommendation"])
    print(f"full results (text): {res['results_txt']}")
    return res


if __name__ == "__main__":
    main()
