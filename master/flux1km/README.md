# flux1km: new-phase work (1 km all-sky TOA flux map)

This is the shared infrastructure for [PLAN.md](../PLAN.md). Every candidate method plugs into one leakage-safe protocol. The old footprint-level code lives in [pre_master/](../../pre_master/).

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
python3 -m flux1km.methods.m5_unet --demo --demo-hard    # one method alone, harder synthetic scene
python3 -m flux1km.e0 --data-root data/sample             # stage E0 report + frozen split manifest
```

```bash
pytest -q flux1km tests
```

The demo writes the six image-granule names as synthetic granules, groups them by overpass and builds leave-one-group-out folds plus the interleave test. It then runs every registered method and prints macro RMSE, pixel RMSE against truth and the decision log. Outputs go to `benchmark_outputs/flux1km/run_*/`, which holds **`results.txt`** (every result in plain text), `predictions.csv` and `summary.json`. What each method is and how to read `results.txt`: [../MODELS.md](../MODELS.md).

## Modules

| Module | What it does | PLAN.md |
|---|---|---|
| `schema` | MODIS band tables, the F1–F5 input sets, and the CERES leakage guard (`assert_no_ceres_inputs`) | §1, §3 |
| `geometry` | glint angle (Bertrand Eq. 14), angle encoding, S0·cos SZA, albedo | §3 |
| `moments` | patch statistics, raw moments from mean/std/skew/kurt, polynomial features that aggregate exactly (M2), Gao cv | §3, §5 |
| `io` | granule triplets `moments/ scalars/ images/` (+ `truth/` for synthetic), granule-id parsing | §2 |
| `synthetic` | synthetic scenes with known flux, overlapping 27×55 footprints, CTZ on/off | §4.3 |
| `aggregation` | operators H: `ceres_psf` (exact, from the advisor's PSF), `fov_ellipse` (fitted fallback), uniform, Gaussian PSF; PSF recovery from moments; ±1 px jitter; random-subset | §4.2 |
| `table` | one flat footprint table (granule, group, day, geometry, CF, glint, cv) | §4.1 |
| `splits` | overpass grouping, nested LOGO, day / month k-fold, year holdout, buffered spatial blocks, interleave, **pixel-reuse leakage check**, JSON manifest | §4.1 |
| `sampling` | strategies S0–S7 (S4, `Fsw ≥ 400`, dropped) and their similarity (Jaccard, KS, Kendall τ) | §5 |
| `features` | `Batch` of footprints; MODIS-only pixel and footprint features | §3 |
| `models` | `FluxModel` interface and baselines: B-clim, B-const (uniTR), M0-ridge, B-bagpixel, M2 | §3 |
| `metrics` | scores by regime, macro RMSE, residual trends, swath-side asymmetry, bootstrap / Wilcoxon / DM / Holm, pixel RMSE / corr / SSIM | §4.2 |
| `physics` | distance to cloud, flux vs distance, pure clear-sky flux, bounds, percentile spread, footprint-spacing artefact | §4.3 |
| `benchmark` | tune within a shared budget, then run, score, compare and decide; a failing method is logged, not fatal | §4, §6 |
| `decision` | the decision rule: gates (physics, flat residuals, swath asymmetry, ranking flip), bootstrap+Holm and regime noise floor, tie-breakers, M3/M2 default, the UNet-vs-M6b rule | §6 |
| `report` | the plain-text `results.txt` of every run | |
| `e0` | stage E0: inventory/unpack, value ranges and units, FOVorientation check, moments vs patches, fitted operator H, overlap, bad-channel striping, frozen manifest (Arena A/B, blocks) with leakage check, advisor bug report | §2 |
| `sampling_study` | S0–S7 × leading methods: Jaccard/KS/MMD, train-on-Si × test-subpopulation matrix, Kendall τ rule | §5 |
| `synthetic_study` | synthetic-truth experiment: pixel scores vs B-const, footprint counts × aggregation sizes, no-CTZ control | §4.3 |
| `cli`, `run_single` | command-line entry points (all methods / one method module) | |
| `methods/` | the candidate methods, one family per module (table below) | §3 |

### `methods/`

| Module | Registered models | |
|---|---|---|
| `m0_legacy` | M0-allsky37, M0-advisor | legacy footprint ANNs (M0-ridge is in `models`) |
| `m1_guy` | M1-gbm, M1-ann | improved Guy model (footprint-level; teacher of M6d/M6f) |
| `m3_pixel_mlp` | M3 | per-pixel MLP, aggregation loss (+ manifold / random-subset regularisation) |
| `m4_context_gbm` | M4 | boosted trees on pixel + context features, aggregate gradient |
| `m5_unet` | M5 | UNet, aggregation loss + edge-aware smoothness |
| `m6a_cnn_tabular` … `m6f_pseudo_labels` | M6a–M6f | the hybrids |
| `m7_atprk` | M7 | area-to-point regression kriging (reads test labels) |
| `m8_nb2bb` | M8 | narrowband → broadband (label `Isw`) → empirical ADM |
| `m9_homogeneous` | M9, M9-local | homogeneous-footprint transfer (Gao) |
| `m10_gp` | M10, M10-cmp | GP aggregate models with random Fourier features |
| `_common`, `_torch`, `_base` | | shared helpers: context features, PyTorch training with the aggregation loss, base classes |

## Adding a method

1. Subclass `models.FluxModel`. Set `name` and `complexity`, then implement:
   - `fit(train, val)`;
   - `predict_map(batch) -> (N, H, W)` for map methods;
   - or override `predict_footprint` for footprint-only methods.
2. Return candidate hyper-parameters from `param_grid()`. The benchmark tries at most `budget` of them on the validation set.
3. Put it in its own module under `methods/`, decorate the class with `@models.register`, import the module in `methods/__init__.py`, and end the module with `if __name__ == "__main__": from ..run_single import main_for; main_for(["<name>"])` so it runs alone. Neural models subclass `methods._base.MapNetModel` and import PyTorch only inside `fit` (`methods._torch`), so the core stays light. Set `needs_operator = True` to be trained with the scoring operator, and `uses_test_labels = True` if the method reads the test footprints' labels.
4. Build inputs with `features.*`, which runs the CERES leakage guard.

## Assumptions to verify in stage E0

These are the guesses made before the real data arrived (see PLAN.md §9):
- band order (confirmed on the sample);
- `Imviews` = SZA/VZA/(180° − RAA) in degrees (confirmed on the sample);
- excess kurtosis in the moments files (confirmed on the sample);
- the RAA convention for glint (settled by the advisor's code);
- the footprint radius used by the leakage check;
- the aggregation weights: exact `ceres_psf` once `ceres_psf.csv` is available (`--psf`), `fov_ellipse` until then.

Each one is a single constant or argument to change.
