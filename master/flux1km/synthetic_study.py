"""Synthetic-truth experiment (PLAN.md section 4.3).

1. Build a known 1 km field (the synthetic generator; ``hard=True`` uses the nonlinear,
   angle-dependent NB->BB conversion and a CTZ only partly visible in the channels).
2. Aggregate it with H and add CERES-level label noise.
3. Retrain each method (the full benchmark protocol, LOGO folds) and measure pixel RMSE, MAE,
   correlation and SSIM, and the flux-vs-distance-to-cloud curve; compare with B-const.
4. Repeat for several numbers of footprints and several aggregation sizes (the fov_ellipse
   operator scaled by 0.5x, 1x, ...), to see how recovery degrades as the resolution gap grows.
5. Include a no-CTZ scene: there the predicted flux-vs-distance curve must come out flat, which
   checks that regularisation does not invent a CTZ.

    python3 -m flux1km.synthetic_study --models M2 M3 M5 M6b --hard
    python3 -m flux1km.synthetic_study --lattices 3x8 6x20 --scales 0.5 1.0 --models M2 M9
"""
from __future__ import annotations

import argparse
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import benchmark, io, physics, splits, synthetic
from .report import format_table, section
from .table import build_table

DISTANCE_BINS = (0.5, 2, 4, 8, 16, 32)


def ctz_signal(curve) -> float:
    """Near-cloud minus far-from-cloud flux of a flux-vs-distance curve (W m^-2); NaN if undefined.

    Mean of the first two finite bins minus the mean of the last two finite bins. Near 0 = flat.

    Example:
        >>> ctz_signal([90.0, 88.0, 82.0, 80.0])
        8.0
    """
    c = np.asarray([v for v in curve if v is not None], float)
    c = c[np.isfinite(c)]
    if len(c) < 3:
        return np.nan
    return float(c[:2].mean() - c[-2:].mean())


def truth_curve(granules) -> list[float]:
    """Flux-vs-distance-to-cloud curve of the synthetic truth itself (all footprints)."""
    flux = np.concatenate([g.truth["flux"] for g in granules]).astype(float)
    dist = np.concatenate([g.truth["dist_km"] for g in granules]).astype(float)
    return physics.flux_vs_distance(flux, dist, np.asarray(DISTANCE_BINS))[1].tolist()


def run_case(models: list[str], n_rows: int, n_cols: int, scale: float, ctz: bool, hard: bool, seed: int = 0,
             budget: int = 5, label_noise: float = 2.0, flat_tol: float = 2.0) -> dict:
    """One synthetic scenario through the benchmark; returns per-model pixel and curve scores."""
    root = tempfile.mkdtemp(prefix="flux1km_truth_")
    op = "fov_ellipse" if scale == 1.0 else f"fov_ellipse:{scale}"
    synthetic.write_demo_dataset(root, seed=seed, n_rows=n_rows, n_cols=n_cols, ctz=ctz, hard=hard, operator=op,
                                 label_noise=label_noise)
    granules = io.load_dataset(root, parts=("moments", "scalars", "images", "truth"))
    table = build_table(granules, splits.group_granules([g.granule_id for g in granules]))
    cfg = benchmark.BenchmarkConfig(models=models, operator=op, sensitivity_operators=("uniform",), budget=budget,
                                    check_leakage=True)
    summ = benchmark.run_benchmark(granules, table, splits.logo_folds(table), cfg)["summary"]
    t_curve = truth_curve(granules)
    out = {"truth_ctz_signal": ctz_signal(t_curve), "truth_curve": t_curve, "models": {}, "errors": summ.get("errors", [])}
    for m, r in summ["models"].items():
        curve = r.get("physics", {}).get("flux_vs_distance")
        sig = ctz_signal(curve) if curve is not None else np.nan
        out["models"][m] = {"pixel": r.get("pixel", {}), "macro_rmse": r.get("macro_rmse"), "curve": curve,
                            "ctz_signal": sig, "flat_ok": (abs(sig) <= flat_tol) if (not ctz and np.isfinite(sig)) else None}
    return out


def run_synthetic_truth_study(models: list[str], lattices=((3, 8), (6, 20)), scales=(0.5, 1.0), hard: bool = True,
                              seed: int = 0, budget: int = 5, out_dir: str | Path | None = None) -> dict:
    """All scenarios: every lattice x every operator scale with CTZ, plus a no-CTZ control per lattice.

    B-const (constant in bag) is always added as the reference that any method must beat.

    Returns:
        {"cases": [{"lattice", "scale", "ctz", **run_case(...)}], "results_txt": path or None}
    """
    models = list(dict.fromkeys(list(models) + ["B-const"]))
    cases = []
    for nr, nc in lattices:
        for sc in scales:
            cases.append({"lattice": f"{nr}x{nc}", "scale": sc, "ctz": True, **run_case(models, nr, nc, sc, True, hard, seed, budget)})
        cases.append({"lattice": f"{nr}x{nc}", "scale": 1.0, "ctz": False, **run_case(models, nr, nc, 1.0, False, hard, seed, budget)})
    out = {"cases": cases, "hard": hard, "results_txt": None}
    if out_dir is not None:
        run = Path(out_dir) / f"synthetic_truth_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        run.mkdir(parents=True, exist_ok=True)
        (run / "synthetic_truth_results.txt").write_text(format_truth_report(out))
        out["results_txt"] = str(run / "synthetic_truth_results.txt")
    return out


def format_truth_report(res: dict) -> str:
    """Plain-text report of run_synthetic_truth_study."""
    lines = ["Synthetic-truth experiment (PLAN.md section 4.3)", f"written {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             f"scene: {'hard (nonlinear angle-dependent NB->BB, CTZ partly hidden)' if res['hard'] else 'easy'}",
             "Pixel scores are against the known 1 km truth of the test granules. 'ctz' = near-cloud minus far-field flux of "
             "the map's flux-vs-distance curve; in the no-CTZ scene it must be ~0 (flat_ok)."]
    for c in res["cases"]:
        lines += [section(f"lattice {c['lattice']} footprints/granule, operator scale {c['scale']}, "
                          f"{'with CTZ' if c['ctz'] else 'NO CTZ (control)'}; truth ctz signal {c['truth_ctz_signal']:.2f} W m^-2")]
        rows = [[m, r["macro_rmse"], r["pixel"].get("rmse"), r["pixel"].get("mae"), r["pixel"].get("corr"), r["pixel"].get("ssim"),
                 r["ctz_signal"], "-" if r["flat_ok"] is None else str(r["flat_ok"])] for m, r in c["models"].items()]
        lines.append(format_table(["model", "macroRMSE", "pixRMSE", "pixMAE", "corr", "SSIM", "ctz", "flat_ok"], rows,
                                  [12, 11, 10, 10, 8, 8, 9, 9]))
        bc = c["models"].get("B-const", {}).get("pixel", {}).get("rmse", np.nan)
        beat = [m for m, r in c["models"].items() if m != "B-const" and r["pixel"].get("rmse", np.inf) < bc]
        lines.append(f"beat B-const on pixel RMSE: {beat}")
        lines += [f"  error: {e['model']} / {e['fold']}: {e['error']}" for e in c["errors"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> dict:
    """Command line for the synthetic-truth study."""
    from .cli import available_models

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["M2", "M9", "M3"])
    p.add_argument("--lattices", nargs="+", default=["3x8", "6x20"], help="footprint lattices ROWSxCOLS per granule")
    p.add_argument("--scales", nargs="+", type=float, default=[0.5, 1.0], help="aggregation sizes (fov_ellipse scale)")
    p.add_argument("--easy", action="store_true", help="use the easy scene instead of the hard one")
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="benchmark_outputs/flux1km")
    a = p.parse_args(argv)
    lattices = [tuple(int(x) for x in s.lower().split("x")) for s in a.lattices]
    res = run_synthetic_truth_study(available_models(a.models), lattices, tuple(a.scales), hard=not a.easy, seed=a.seed,
                                    budget=a.budget, out_dir=a.out)
    print(f"full results (text): {res['results_txt']}")
    return res


if __name__ == "__main__":
    main()
