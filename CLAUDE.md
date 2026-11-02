# CLAUDE.md

## Repository Layout

The repo has two top-level work folders:

- **[master/](master/)**: the M.Sc. work. **Read [master/PLAN.md](master/PLAN.md) first**: it is the plan to implement (goal, data format §2, candidate methods §3, benchmark protocol §4, sampling §5, decision rule §6, open questions §7, infrastructure §9).
  - [master/flux1km/](master/flux1km/): the 1 km pipeline package (module map and how to add a method: [master/flux1km/README.md](master/flux1km/README.md)); the candidate methods are in [master/flux1km/methods/](master/flux1km/methods/), documented in [master/MODELS.md](master/MODELS.md); the run guide (Hebrew) is [master/HOW_TO_RUN.md](master/HOW_TO_RUN.md). Keep both in sync when a method, command or output changes.
  - [master/tests/](master/tests/), [master/scripts/](master/scripts/), [master/pyproject.toml](master/pyproject.toml).
  - [master/research/](master/research/): the deep-research report (Hebrew) and notes behind the plan.
- **[pre_master/](pre_master/)**: everything from before the M.Sc. plan (Guy model, benchmarks, All_Sky_AIflux), plus the advisor's data-building and training code in [pre_master/advisor_code/](pre_master/advisor_code/). Frozen: use it as the M0 baseline and as a reference, do not extend it.
- Shared at the root: [Thesis_Goal.md](Thesis_Goal.md), [Documentation/LITERATURE_LESSONS.md](Documentation/LITERATURE_LESSONS.md), `data/` (git-ignored except the real sample granule in `data/sample/`), the Dockerfile and CI.

## New-Phase Rules

- Model inputs are MODIS-only: radiances (channel identity via wavelength), sun/view geometry and glint. Never feed CERES-derived quantities (`Isw`, `Ilw`, `mSum*`) as inputs; CERES is the label and the aggregation constraint only.
- Aggregate predictions in flux (`mean(f(x))`), never in inputs, and score every method through the same footprint aggregation operator.
- Never split train/test by random footprint: overlapping footprints leak. Use the frozen split manifest (granule groups / day / region / time holdouts, buffered spatial blocks) described in PLAN.md §4.
- Give every method the same tuning budget and report the result with several seeds and paired significance tests.
- Keep a `--demo` mode so new code runs before the real data is unpacked.

### New-phase code workflow

- Work from `master/`: `cd master && pip install -e ".[dev]"`, then `pytest -q flux1km tests` (this includes the doctests); lint with `ruff check .`.
- Run: `python3 -m flux1km.cli --demo`, or `python3 -m flux1km.cli --data-root <dir> --manifest <json> [--psf ceres_psf.csv]`. Every run writes `benchmark_outputs/flux1km/run_<time>/results.txt` (all results, plain text).
- One method alone: `python3 -m flux1km.methods.<module> --demo` (e.g. `m3_pixel_mlp`, `m5_unet`, `m9_homogeneous`). The methods of PLAN.md §3 live one family per module in [master/flux1km/methods/](master/flux1km/methods/); what each one is and how to run it: [master/MODELS.md](master/MODELS.md).
- Stage E0 checks and the frozen split manifest: `python3 -m flux1km.e0 --data-root <dir>`. Studies: `python3 -m flux1km.synthetic_study` (§4.3), `python3 -m flux1km.sampling_study` (§5). The §6 decision rule is `flux1km/decision.py`.
- Harder synthetic data (§4.3): `--demo --demo-hard` (nonlinear angle-dependent NB→BB, CTZ only partly visible in the channels).
- Neural methods use PyTorch, imported only inside `fit` (`pip install -e ".[dev,ml]"`); without it they are skipped and their tests are skipped.
- For every function: docstring with an example, then signature, then edge-case tests, then the implementation. Curate the tests into `tests/test_<module>.py` (method modules: `tests/methods/test_<module>.py`), one class per function. CI ([.github/workflows/ci.yml](.github/workflows/ci.yml), run in `master/`) uses that naming to run only the tests of changed modules ([scripts/select_tests.py](master/scripts/select_tests.py)). Core modules (schema, io, table, aggregation, features, models) trigger the full suite.
- Conventional commit messages (`feat(flux1km): ...`, `fix(splits): ...`, `docs: ...`).
- Module map and how to add a method: [flux1km/README.md](master/flux1km/README.md).

## Pre-master Project Rules (apply to `pre_master/` only)

- Keep the original regression logic intact. Improve the training workflow by tuning parameters, cross-validation, and model selection rather than changing the scientific target.
- Prefer changes in [pre_master/python/benchmark_pipeline.py](pre_master/python/benchmark_pipeline.py) for all ML workflow updates.
- For feature combination exploration and model selection workflow, use [pre_master/python/guys_benchmark_models.py](pre_master/python/guys_benchmark_models.py) which systematically benchmarks classical models before training ANN.
- Keep the code runnable before the real dataset is available by preserving demo mode.
- Use the GPU-enabled Docker workflow for ANN training and mark GPU as required inside the container.
- Save model artifacts and metrics in `benchmark_outputs/`.

## How to Run

```bash
docker build -t ctz_research:gpu .
docker run --gpus all --rm -it -v "$PWD":/app ctz_research:gpu
```

Inside the container, the default command runs the `master/` benchmark in demo mode (`python3 -m flux1km.cli --demo`). The pre-master benchmark script with its demo dataset:

```bash
python3 pre_master/python/benchmark_pipeline.py --demo
```

To run against real data:

```bash
python3 pre_master/python/benchmark_pipeline.py --data-path /data --output-dir benchmark_outputs
```

### Guy's Model - Feature Combination Benchmark

Run feature combination analysis with automatic model selection, hyperparameter tuning, and ANN training:

```bash
# Demo mode (synthetic data)
python3 pre_master/python/guys_benchmark_models.py --demo

# With real paired data (default)
python3 pre_master/python/guys_benchmark_models.py

# Custom output directory and epochs
python3 pre_master/python/guys_benchmark_models.py --output-dir my_results --ann-epochs 200

# Skip classical models, go straight to ANN
python3 pre_master/python/guys_benchmark_models.py --skip-classical

# With specific data region
python3 pre_master/python/guys_benchmark_models.py --data-root data/paired/North15_March2014/2014

# Rank features by Random Forest importance and benchmark top-N subsets (5/10/15/20/24) instead of the fixed combinations
python3 pre_master/python/guys_benchmark_models.py --feature-selection importance
```

**What it does:**
1. Loads moments/scalars from paired data and generates feature combinations. The target is `Fsw` (shortwave flux) from the scalars data; wavelength arrays (`SWwavlngs`/`LWwavlngs`) and the unused `Flw` (longwave flux) scalar are excluded from the features, and rows with NaN/inf values are dropped.
2. Benchmarks 7+ classical models (Random Forest, CatBoost, XGBoost, Linear Regression, Gradient Boosting, AdaBoost, SVM) — or, with `--feature-selection importance`, only Ridge and SVM across importance-ranked top-N feature subsets, picking the practical speed/accuracy tradeoff (prefers `top_20`) rather than the single best R².
3. Identifies best model/combo pair
4. Hyperparameter-tunes the best model
5. Builds and trains a TensorFlow ANN using All_Sky_AIflux architecture
6. Outputs: performance heatmaps, feature importance plots, metrics CSV, trained model weights

**Output directory structure:**
```
benchmark_outputs/run_YYYYMMDD_HHMMSS/
  ├── benchmark_results.csv          # All model/combo results
  ├── performance_heatmap.png        # Model performance matrix
  ├── feature_importance.png         # Top feature importances
  ├── ann_model.keras                # Trained ANN weights
  └── summary.json                   # Final metrics & configuration
```

To run the original paired NPZ workflow on specific data regions:

```bash
# North15 March 2014 data
python3 pre_master/python/All_Sky_AIflux.py --data-path data/paired/North15_March2014/2014

# Pacific 2014-2015 data
python3 pre_master/python/All_Sky_AIflux.py --data-path data/paired/Pacific_2014-2015
```

To build a legacy training dataset from paired moments/scalars:

```bash
# Default uses Pacific 2014-2015
python3 pre_master/python/build_legacy_dataset_from_pairs.py

# Or specify a different region
python3 pre_master/python/build_legacy_dataset_from_pairs.py --data-root data/paired/North15_March2014/2014
```

Quick NPZ inspection example:

```bash
python3 pre_master/python/Details_of_npz.py --data-root data/test --file-type images --file-name A2014060.0245.npz
```

To sample a fraction of a large paired data directory into a single NPZ (useful for quick local iteration before running the full benchmark):

```bash
python3 pre_master/python/sample_paired_data.py data/paired/Pacific_2014-2015 benchmark_outputs/sampled_data.npz
```

### Guy Model (Final Locked-Down Recipe)

Once [pre_master/python/guys_benchmark_models.py](pre_master/python/guys_benchmark_models.py) `--feature-selection importance` has identified the best combo/model, [pre_master/python/guy_model_2.0.py](pre_master/python/guy_model_2.0.py) locks that recipe in (currently: top-20 Random Forest importance-selected features, Ridge alpha=1.0 baseline, plus the ANN) and trains it directly without re-running the full sweep:

```bash
python3 pre_master/python/guy_model_2.0.py --demo
python3 pre_master/python/guy_model_2.0.py --data-root data/paired/Pacific_2014-2015

# Restricted 37-feature All_Sky_AIflux-style pool, no RF trimming
python3 pre_master/python/guy_model_2.0.py --feature-source allsky37 --data-root data/paired/Pacific_2014-2015
```

The script is versioned: [pre_master/python/guy_model_1.0.py](pre_master/python/guy_model_1.0.py) is the original recipe (no `--feature-source` flag, always top-20 of 129 features) kept for reference; `guy_model_2.0.py` adds the `--feature-source {full,allsky37}` option, with `full` (default) reproducing 1.0 exactly. See [pre_master/Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt](pre_master/Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt) for the results behind both versions. If a future benchmark run picks a different combo/model as the practical recommendation, bump the version (e.g. add `guy_model_3.0.py`) and update the constants and imports at its top to match.

## Editing Notes

- Update [README.md](README.md) when the workflow or runtime instructions change.
- Update [Dockerfile](Dockerfile) when dependencies or the GPU runtime changes.
- Ignore generated training outputs and local environment files in `.gitignore`.

## Data Structure

```
data/
  paired/                          # Paired moments/scalars data organized by region
    Pacific_2014-2015/             # Pacific region training data (default for legacy scripts)
      moments/
      scalars/
      images/
    North15_March2014/             # North 15° region validation data
      2014/
        moments/
        scalars/
      2015/
        moments/
        scalars/
  test/                            # Quick reference test subset
    moments/
    scalars/
    images/
  consolidated/                    # Pre-built datasets (NPZ/CSV for benchmarking)
```

## Python Folder Migration Notes

- Pre-master Python scripts are under [pre_master/python](pre_master/python); M.Sc. code is under [master/flux1km](master/flux1km).
- Local data is organized under [data/paired](data/paired) by region.
- Test subset is [data/test](data/test).
- Pre-built training data goes in [data/consolidated](data/consolidated).
- New feature combination benchmark script: [pre_master/python/guys_benchmark_models.py](pre_master/python/guys_benchmark_models.py) — compares classical models, hyperparameter-tunes, then trains ANN.
- New sampling helper: [pre_master/python/sample_paired_data.py](pre_master/python/sample_paired_data.py) — samples a fraction of a paired data directory into a single NPZ.
- New locked-down recipe script: [pre_master/python/guy_model_2.0.py](pre_master/python/guy_model_2.0.py) (previously [pre_master/python/guy_model_1.0.py](pre_master/python/guy_model_1.0.py)) — trains the production recipe (Ridge + ANN on top-20 importance-selected features) picked by the benchmark.

