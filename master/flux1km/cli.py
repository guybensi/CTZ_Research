"""Command line entry point: build splits, run the benchmark, print the decision.

    python3 -m flux1km.cli --demo                      # all registered methods, synthetic data
    python3 -m flux1km.cli --demo --demo-hard          # the harder synthetic scene (PLAN.md 4.3)
    python3 -m flux1km.cli --data-root data/test --split logo --manifest splits/logo.json
    python3 -m flux1km.cli --data-root DIR --models M2 M3 M5 --seeds 0 1 2 3 4

Every run writes benchmark_outputs/flux1km/run_<time>/results.txt (all results, plain text),
summary.json and predictions.csv. Methods that need PyTorch are skipped with a warning when it is
not installed.
"""
from __future__ import annotations

import argparse
import importlib.util
import tempfile
from pathlib import Path

from . import aggregation, benchmark, io, splits, synthetic
from .models import MODELS
from .table import build_table

SPLITS = ("logo", "day", "month", "year", "block")


def make_folds(table, kind: str, k: int, seed: int, block_km: float, buffer_km: float):
    """Folds of the requested kind plus the (leaky, separately reported) interleave test."""
    if kind == "logo":
        folds = splits.logo_folds(table)
    elif kind in ("day", "month"):
        n_units = len(set(table[kind]))
        folds = splits.group_kfold(table, key=kind, k=min(k, n_units), seed=seed)
    elif kind == "year":
        last = max(table["year"])
        folds = [splits.holdout(table, test_mask=table["year"] == last, name=f"year:{last}", seed=seed)]
    elif kind == "block":
        folds = splits.spatial_block_folds(table, block_km=block_km, buffer_km=buffer_km, k=k, seed=seed)
    else:
        raise ValueError(f"unknown split {kind!r}; choose from {SPLITS}")
    return folds + [splits.interleave_split(table)]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="write and use a synthetic dataset with known 1 km truth")
    p.add_argument("--data-root", help="folder with moments/, scalars/ (and images/) NPZ files")
    p.add_argument("--out", default="benchmark_outputs/flux1km", help="output folder")
    p.add_argument("--models", nargs="+", default=sorted(MODELS), help=f"registered: {sorted(MODELS)}")
    p.add_argument("--split", choices=SPLITS, default="logo")
    p.add_argument("--k", type=int, default=5, help="folds for day / month / block splits")
    p.add_argument("--block-km", type=float, default=100.0)
    p.add_argument("--buffer-km", type=float, default=40.0)
    p.add_argument("--manifest", help="load folds from this JSON if it exists, else create and save it")
    p.add_argument("--operator", default=None, choices=aggregation.OPERATORS,
                   help="primary aggregation operator (default: ceres_psf with --psf, else fov_ellipse)")
    p.add_argument("--psf", help="path to ceres_psf.csv (the advisor's CERES PSF) or a recovered PSF .npy (e0)")
    p.add_argument("--budget", type=int, default=40, help="max tuning candidates per model")
    p.add_argument("--sampling", default="S0", help="training-data strategy S0..S7")
    p.add_argument("--input-set", default="F5", help="F1..F5 (PLAN.md section 3)")
    p.add_argument("--min-group-size", type=int, default=100, help="merge granule groups smaller than this")
    p.add_argument("--leakage-radius-km", type=float, default=20.0)
    p.add_argument("--demo-size", default="6x20", help="synthetic footprint lattice per granule, ROWSxCOLS")
    p.add_argument("--demo-hard", action="store_true", help="hard synthetic scene: nonlinear angle-dependent NB->BB, "
                   "anisotropic radiances, CTZ only partly visible in the channels (PLAN.md section 4.3)")
    p.add_argument("--ctz-visible", type=float, default=None, help="fraction of the CTZ visible in the channels (demo)")
    p.add_argument("--no-ctz", action="store_true", help="demo scene without CTZ (the no-CTZ control of section 4.3)")
    p.add_argument("--seed", type=int, default=0, help="seed of the demo data and the split")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="model seeds; the plan retrains the best config with 5")
    args = p.parse_args(argv)
    if not args.demo and not args.data_root:
        p.error("give --data-root or --demo")
    return args


def main(argv=None) -> dict:
    """Run the benchmark from the command line; returns the result dict (records + summary)."""
    args = parse_args(argv)
    root = args.data_root
    if args.demo:
        root = root or tempfile.mkdtemp(prefix="flux1km_demo_")
        n_rows, n_cols = (int(x) for x in args.demo_size.lower().split("x"))
        synthetic.write_demo_dataset(root, seed=args.seed, n_rows=n_rows, n_cols=n_cols, hard=args.demo_hard,
                                     ctz=not args.no_ctz, ctz_visible=args.ctz_visible)
        print(f"demo data written to {root}")
    granules = io.load_dataset(root, parts=("moments", "scalars", "images", "truth"))
    groups = splits.group_granules([g.granule_id for g in granules],
                                   sizes={g.granule_id: g.n_footprints for g in granules}, min_size=args.min_group_size)
    table = build_table(granules, groups)
    if args.manifest and Path(args.manifest).exists():
        folds, _ = splits.load_manifest(args.manifest, table)
    else:
        folds = make_folds(table, args.split, args.k, args.seed, args.block_km, args.buffer_km)
        if args.manifest:
            Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
            splits.save_manifest(args.manifest, table, folds, meta={"split": args.split, "seed": args.seed})
    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; registered: {sorted(MODELS)}")
    map_models = available_models(args.models)
    operator = args.operator or ("ceres_psf" if args.psf else "fov_ellipse")
    available = [o for o in aggregation.OPERATORS if args.psf or o != "ceres_psf"]
    if operator not in available:
        raise SystemExit("--operator ceres_psf needs --psf path/to/ceres_psf.csv")
    cfg = benchmark.BenchmarkConfig(
        models=map_models, operator=operator, psf_path=args.psf,
        sensitivity_operators=tuple(o for o in available if o != operator), budget=args.budget, sampling=args.sampling,
        seeds=tuple(args.seeds) if args.seeds else (args.seed,),
        model_params={m: {"input_set": args.input_set} for m in map_models if m not in ("B-clim", "B-const")},
        leakage_radius_km=args.leakage_radius_km)
    res = benchmark.run_benchmark(granules, table, folds, cfg, out_dir=args.out)
    print(format_summary(res["summary"]))
    return res


def available_models(names: list[str]) -> list[str]:
    """Drop the PyTorch methods (with a printed warning) when PyTorch is not installed.

    Example:
        >>> available_models(["M2"])
        ['M2']
    """
    if importlib.util.find_spec("torch") is not None:
        return list(names)
    skipped = [m for m in names if MODELS[m].needs_torch]
    if skipped:
        print(f"PyTorch not installed: skipping {skipped} (pip install -e '.[ml]')")
    return [m for m in names if not MODELS[m].needs_torch]


def format_summary(summary: dict) -> str:
    """Plain-text table: one line per model, then the decision log."""
    lines = [f"{'model':12s} {'macro_rmse':>10s} {'rmse':>8s} {'bias':>8s} {'pixel_rmse':>10s} {'interleave':>10s}"]
    for name, r in summary["models"].items():
        o = r.get("overall", {})
        lines.append(f"{name:12s} {r.get('macro_rmse', float('nan')):10.2f} {o.get('rmse', float('nan')):8.2f} "
                     f"{o.get('bias', float('nan')):8.2f} {r.get('pixel', {}).get('rmse', float('nan')):10.2f} "
                     f"{r.get('interleave', {}).get('rmse', float('nan')):10.2f}")
    d = summary["decision"]
    lines += ["", f"winner: {d['winner']}"] + [f"  - {x}" for x in d["log"]]
    if "run_dir" in summary:
        lines.append(f"outputs: {summary['run_dir']}")
        lines.append(f"full results (text): {summary['results_txt']}")
    if summary.get("errors"):
        lines.append(f"{len(summary['errors'])} model/fold errors: see results.txt")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
