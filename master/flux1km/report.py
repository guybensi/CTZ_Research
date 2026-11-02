"""Plain-text results report: everything a benchmark run produced, in one readable file.

benchmark.run_benchmark(..., out_dir=...) writes it as ``results.txt`` next to
``predictions.csv`` and ``summary.json`` in ``benchmark_outputs/flux1km/run_<UTC time>/``.
The sampling and synthetic-truth studies append their own sections (format_table, section).
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

import numpy as np

WIDTH = 100

GLOSSARY = """\
How to read this file
  RMSE / MAE / bias / R2  footprint level, predicted vs CERES Fsw, in W m^-2 (bias = pred - CERES).
  macro RMSE              mean of the RMSEs of the clear (CF < 5%), broken and overcast (CF >= 95%)
                          regimes: the PRIMARY score (PLAN.md section 6).
  macro@<operator>        the same score with a sensitivity aggregation operator; the ranking must not
                          flip between operators.
  interleave              RMSE on the leaky interleave test (every 2nd footprint); reported only, never
                          used for ranking. A small gap to the LOGO RMSE is tie-breaker 2.
  pixel RMSE/corr/SSIM    against the known 1 km truth (synthetic data only): tie-breaker 1.
  trend effect            |slope| x range of the residual vs an angle; must stay below the trend gate.
  swath asymmetry         forward- minus backscatter-side residual at matched VZA.
  physics                 pcs = far-field pure clear-sky flux (target 80 +- 10 W m^-2), bound = fraction
                          of pixels above S0 cos(SZA), spacing = power-spectrum peak at the footprint spacing.
  comparisons             paired tests on per-group losses: bootstrap CI of loss(A) - loss(B) (negative =
                          A better), bootstrap p and its Holm correction, Wilcoxon (+Holm), Diebold-Mariano.
  ranked = no             the method reads the test footprints' own CERES labels (B-const, M6e, M7,
                          M9-local): it is a reference for pixel truth and the interleave test only.
"""


def _f(x, nd: int = 2, width: int = 9) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return f"{'-':>{width}s}"
    return f"{'nan':>{width}s}" if not np.isfinite(v) else f"{v:{width}.{nd}f}"


def section(title: str) -> str:
    """A section header line.

    Example:
        >>> section("Scores").splitlines()[2]
        'Scores'
    """
    return "\n" + "=" * WIDTH + f"\n{title}\n" + "=" * WIDTH


def format_table(header: list[str], rows: Iterable[list], widths: list[int] | None = None) -> str:
    """Fixed-width text table; numbers get 2 decimals, NaN prints as nan, None as '-'.

    Example:
        >>> print(format_table(["a", "b"], [["x", 1.0]], [4, 6]))
        a        b
        x     1.00
    """
    rows = [list(r) for r in rows]
    widths = widths or [max(12, len(h) + 1) for h in header]
    out = ["".join(f"{h[:w - 1]:<{w}s}" if i == 0 else f"{h[:w - 1]:>{w}s}" for i, (h, w) in enumerate(zip(header, widths)))]
    for r in rows:
        cells = []
        for i, (v, w) in enumerate(zip(r, widths)):
            if i == 0 or isinstance(v, str):
                cells.append(f"{str(v):<{w}s}" if i == 0 else f"{str(v):>{w}s}")
            else:
                cells.append(_f(v, 2, w))
        out.append("".join(cells))
    return "\n".join(out)


def format_report(summary: dict, title: str = "flux1km benchmark results") -> str:
    """The full results text of one benchmark run (see the module docstring).

    Example:
        >>> s = {"models": {}, "comparisons": {}, "decision": {"winner": None, "log": []}}
        >>> "WINNER" in format_report(s)
        True
    """
    cfg = summary.get("config", {})
    models = summary.get("models", {})
    lines = [title, f"written {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}"]
    if "run_dir" in summary:
        lines.append(f"run folder: {summary['run_dir']}")
    d = summary.get("decision", {})
    lines += [section("WINNER"), f"winner: {d.get('winner')}",
              f"ranking by macro RMSE (gate-passing, ranked methods): {d.get('ranking_by_macro_rmse', [])}"]
    if d.get("tie_set") and len(d["tie_set"]) > 1:
        lines.append(f"tie set: {d['tie_set']}")
    lines += ["decision log:"] + [f"  - {x}" for x in d.get("log", [])]

    if cfg:
        lines += [section("Configuration"),
                  f"models: {cfg.get('models')}",
                  f"primary operator: {cfg.get('operator')}   sensitivity operators: {cfg.get('sensitivity_operators')}",
                  f"PSF file: {cfg.get('psf_path')}   tuning budget: {cfg.get('budget')}   seeds: {cfg.get('seeds')}",
                  f"sampling: {cfg.get('sampling')} {cfg.get('sampling_params') or ''}   leakage check: {cfg.get('check_leakage')} "
                  f"(radius {cfg.get('leakage_radius_km')} km)",
                  f"noise floor by regime: {cfg.get('noise_floor_by_regime')}   trend gate: {cfg.get('trend_gate')}   "
                  f"swath gate: {cfg.get('swath_gate')}   clear-sky target: {cfg.get('pcs_target')}",
                  f"image granules: {summary.get('n_image_granules', '?')}"]

    folds = summary.get("folds", [])
    if folds:
        lines += [section("Folds (frozen split manifest)"),
                  format_table(["fold", "leaky", "n_train", "n_val", "n_test"],
                               [[f["name"], str(f["leaky"]), str(f["n_train"]), str(f["n_val"]), str(f["n_test"])] for f in folds],
                               [28, 8, 10, 10, 10])]
        lines.append("chosen hyper-parameters (validation RMSE):")
        for f in folds:
            if f.get("leaky"):
                continue  # the interleave fold reuses the same tuning procedure; not repeated here
            for m, p in f.get("params", {}).items():
                shown = {k: v for k, v in p.items() if k not in ("operator", "psf_path")}
                lines.append(f"  {f['name']:<24s} {m:<12s} val_rmse={_f(f.get('val_rmse', {}).get(m), 2, 0).strip()}  {shown}")

    if summary.get("errors"):
        lines += [section("Errors (model skipped on that fold)")] + [f"  {e['model']} / {e['fold']} / seed {e['seed']}: {e['error']}"
                                                                   for e in summary["errors"]]

    ops = [k.removeprefix("macro_rmse_") for k in next(iter(models.values()), {}) if k.startswith("macro_rmse_")]
    header = ["model", "ranked", "macroRMSE", "RMSE", "MAE", "bias", "R2", "n"] + [f"macro@{o}" for o in ops] + \
             ["interleave", "pixRMSE", "pixCorr", "SSIM"]
    rows = []
    for n, r in models.items():
        o, px = r.get("overall", {}), r.get("pixel", {})
        rows.append([n, "no" if r.get("uses_test_labels") else "yes", r.get("macro_rmse"), o.get("rmse"), o.get("mae"),
                     o.get("bias"), o.get("r2"), str(o.get("n", 0))] + [r.get(f"macro_rmse_{op}") for op in ops] +
                    [r.get("interleave", {}).get("rmse"), px.get("rmse"), px.get("corr"), px.get("ssim")])
    lines += [section("Scores (non-leaky folds; W m^-2)"),
              format_table(header, rows, [12, 7] + [max(11, len(h) + 2) for h in header[2:]])]

    lines.append(section("Scores by cloud regime (RMSE / bias / n)"))
    rrows = []
    for n, r in models.items():
        by = r.get("by_regime", {})
        row = [n]
        for reg in ("clear", "broken", "overcast"):
            s = by.get(reg, {})
            row += [s.get("rmse"), s.get("bias"), str(s.get("n", 0))]
        rrows.append(row)
    lines.append(format_table(["model", "clear", "bias", "n", "broken", "bias", "n", "overcast", "bias", "n"], rrows,
                              [12] + [10, 9, 7] * 3))

    lines.append(section("Residual flatness (gates: trend effect, swath-side asymmetry)"))
    trows = []
    for n, r in models.items():
        te = r.get("trend_effect", {})
        trows.append([n] + [te.get(c, {}).get("effect_wm2") for c in ("SZA", "VZA", "RAA", "glint")] + [r.get("swath_asymmetry")])
    lines.append(format_table(["model", "SZA", "VZA", "RAA", "glint", "swath"], trows, [12, 10, 10, 10, 10, 10]))

    phys_rows = [[n, r["physics"].get("pcs_flux"), r["physics"].get("bound_violation"), r["physics"].get("spacing_peak"),
                  r["physics"].get("decay_score"), str(r["physics"].get("passed")), r["physics"].get("distance_source", "")]
                 for n, r in models.items() if r.get("physics")]
    if phys_rows:
        lines += [section("Physics checks of the 1 km maps"),
                  format_table(["model", "clear_sky", "bound_frac", "spacing", "decay", "passed", "distance from"], phys_rows,
                               [12, 11, 11, 10, 10, 8, 28])]
        for n, r in models.items():
            if r.get("physics", {}).get("flux_vs_distance") is not None:
                curve = ", ".join(_f(v, 1, 0).strip() for v in r["physics"]["flux_vs_distance"])
                lines.append(f"  {n:<12s} flux vs distance to cloud: [{curve}]")

    comps = summary.get("comparisons", {})
    if comps:
        crow = [[k, c.get("mean_diff"), c.get("ci_low"), c.get("ci_high"), c.get("p_boot"), c.get("p_holm_boot"),
                 c.get("wilcoxon_p"), c.get("p_holm"), c.get("dm_p"), str(c.get("n_units", ""))] for k, c in comps.items()]
        lines += [section("Pairwise comparisons (unit = granule group; negative diff = first model better)"),
                  format_table(["pair", "diff", "ci_low", "ci_high", "p_boot", "p_holm_b", "wilcox", "p_holm_w", "DM p", "units"],
                               crow, [28, 10, 10, 10, 9, 9, 9, 9, 9, 7])]
    gates = d.get("gates", {})
    if gates:
        lines += [section("Gate results")] + [f"  {n:<12s} {'PASS' if not g else 'FAIL: ' + ', '.join(g)}" for n, g in gates.items()]
    lines += [section("Glossary"), GLOSSARY]
    return "\n".join(lines) + "\n"
