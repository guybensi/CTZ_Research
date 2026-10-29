# flux1km: new-phase work (1 km all-sky TOA flux map)

This is the shared infrastructure for [PLAN.md](../PLAN.md). Every candidate method plugs into one leakage-safe protocol. The old footprint-level code lives in [legacy/](../legacy/).

The core is NumPy/SciPy only, so it runs and is tested without a GPU. Everything works today on synthetic data that has a known 1 km truth (`--demo`).

## Quick start

```bash
pip install -e ".[dev]"
```

```bash
python3 -m flux1km.cli --demo
```

```bash
python3 -m flux1km.cli --data-root data/test --split logo --manifest benchmark_outputs/flux1km/logo.json
```

```bash
pytest -q flux1km tests
```

The demo writes the six image-granule names as synthetic granules, groups them by overpass and builds leave-one-group-out folds plus the interleave test. It then runs the baselines and prints macro RMSE, pixel RMSE against truth and the decision log. Outputs go to `benchmark_outputs/flux1km/run_*/`, which holds `predictions.csv` and `summary.json`.

## Modules

| Module | What it does | PLAN.md |
|---|---|---|
| `schema` | MODIS band tables, the F1–F5 input sets, and the CERES leakage guard (`assert_no_ceres_inputs`) | §1, §3 |
| `geometry` | glint angle (Bertrand Eq. 14), angle encoding, S0·cos SZA, albedo | §3 |
| `moments` | patch statistics, raw moments from mean/std/skew/kurt, polynomial features that aggregate exactly (M2), Gao cv | §3, §5 |
| `io` | granule triplets `moments/ scalars/ images/` (+ `truth/` for synthetic), granule-id parsing | §2 |
| `synthetic` | synthetic scenes with known flux, overlapping 27×55 footprints, CTZ on/off | §4.3 |
| `aggregation` | operators H: uniform, approximate Gaussian PSF, ±1 px jitter, random-subset | §4.2 |
| `table` | one flat footprint table (granule, group, day, geometry, CF, glint, cv) | §4.1 |
| `splits` | overpass grouping, nested LOGO, day k-fold, holdout, buffered spatial blocks, interleave, **pixel-reuse leakage check**, JSON manifest | §4.1 |
| `sampling` | strategies S0–S7 and their similarity (Jaccard, KS, Kendall τ) | §5 |
| `features` | `Batch` of footprints; MODIS-only pixel and footprint features | §3 |
| `models` | `FluxModel` interface and baselines: B-clim, B-const (uniTR), M0-ridge, B-bagpixel, M2 | §3 |
| `metrics` | scores by regime, macro RMSE, residual trends, swath-side asymmetry, bootstrap / Wilcoxon / DM / Holm, pixel RMSE / corr / SSIM | §4.2 |
| `physics` | distance to cloud, flux vs distance, pure clear-sky flux, bounds, percentile spread, footprint-spacing artefact | §4.3 |
| `benchmark` | tune within a shared budget, then run, score, compare and decide | §4, §6 |
| `cli` | command-line entry point | |

## Adding a method

1. Subclass `models.FluxModel`. Set `name` and `complexity`, then implement:
   - `fit(train, val)`;
   - `predict_map(batch) -> (N, H, W)` for map methods;
   - or override `predict_footprint` for footprint-only methods.
2. Return candidate hyper-parameters from `param_grid()`. The benchmark tries at most `budget` of them on the validation set.
3. Register it in `models.MODELS`. Neural models (M3, M5, M6) go in their own module that imports TF/PyTorch inside `fit`, so the core stays light.
4. Build inputs with `features.*`, which runs the CERES leakage guard.

## Assumptions to verify in stage E0

These are the guesses made before the real data arrived (see PLAN.md §9):
- band order;
- `Imviews` = SZA/VZA/RAA in degrees;
- Pearson kurtosis in the moments files;
- the RAA convention for glint;
- the footprint radius used by the leakage check;
- the PSF width.

Each one is a single constant or argument to change.
