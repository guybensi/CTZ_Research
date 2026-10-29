# CLAUDE.md

## Repository Layout (new phase)

The repo is split into **new work** and **old work**:

- [Thesis_Goal.md](Thesis_Goal.md) and [PLAN.md](PLAN.md): the goal (1 km all-sky TOA flux map) and the explore-then-focus plan for the new phase. Read PLAN.md before starting new-phase work.
- [flux1km/](flux1km/): all NEW code (pixel-level / UNet / hybrid methods, split manifests, aggregation operators, evaluation). See [flux1km/README.md](flux1km/README.md).
- [legacy/](legacy/): the OLD footprint-level work (Guy model, benchmarks, All_Sky_AIflux), frozen. Use it as the M0 baseline and as a source of reusable pieces; do not extend it.
- [Documentation/LITERATURE_LESSONS.md](Documentation/LITERATURE_LESSONS.md), [reports/](reports/) and [research_notes/](research_notes/): literature lessons and the deep-research report the plan is built on.

## New-Phase Rules

- Model inputs are MODIS-only: radiances (channel identity via wavelength), sun/view geometry and glint. Never feed CERES-derived quantities (`Isw`, `Ilw`, `mSum*`) as inputs; CERES is the label and the aggregation constraint only.
- Aggregate predictions in flux (`mean(f(x))`), never in inputs, and score every method through the same footprint aggregation operator.
- Never split train/test by random footprint: overlapping footprints leak. Use the frozen split manifest (granule groups / day / region / time holdouts, buffered spatial blocks) described in PLAN.md §4.
- Give every method the same tuning budget and report the result with several seeds and paired significance tests.
- Keep a `--demo` mode so new code runs before the real data is unpacked.

## Legacy Project Rules (apply to `legacy/` only)

- Keep the original regression logic intact. Improve the training workflow by tuning parameters, cross-validation, and model selection rather than changing the scientific target.
- Prefer changes in [legacy/python/benchmark_pipeline.py](legacy/python/benchmark_pipeline.py) for all ML workflow updates.
- For feature combination exploration and model selection workflow, use [legacy/python/guys_benchmark_models.py](legacy/python/guys_benchmark_models.py) which systematically benchmarks classical models before training ANN.
- Keep the code runnable before the real dataset is available by preserving demo mode.
- Use the GPU-enabled Docker workflow for ANN training and mark GPU as required inside the container.
- Save model artifacts and metrics in `benchmark_outputs/`.

## How to Run

```bash
docker build -t ctz_research:gpu .
docker run --gpus all --rm -it -v "$PWD":/app ctz_research:gpu
```

Inside the container, the default command runs the benchmark script. To use the demo dataset:

```bash
python3 legacy/python/benchmark_pipeline.py --demo
```

To run against real data:

```bash
python3 legacy/python/benchmark_pipeline.py --data-path /data --output-dir benchmark_outputs
```

### Guy's Model - Feature Combination Benchmark

Run feature combination analysis with automatic model selection, hyperparameter tuning, and ANN training:

```bash
# Demo mode (synthetic data)
python3 legacy/python/guys_benchmark_models.py --demo

# With real paired data (default)
python3 legacy/python/guys_benchmark_models.py

# Custom output directory and epochs
python3 legacy/python/guys_benchmark_models.py --output-dir my_results --ann-epochs 200

# Skip classical models, go straight to ANN
python3 legacy/python/guys_benchmark_models.py --skip-classical

# With specific data region
python3 legacy/python/guys_benchmark_models.py --data-root data/paired/North15_March2014/2014

# Rank features by Random Forest importance and benchmark top-N subsets (5/10/15/20/24) instead of the fixed combinations
python3 legacy/python/guys_benchmark_models.py --feature-selection importance
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
python3 legacy/python/All_Sky_AIflux.py --data-path data/paired/North15_March2014/2014

# Pacific 2014-2015 data
python3 legacy/python/All_Sky_AIflux.py --data-path data/paired/Pacific_2014-2015
```

To build a legacy training dataset from paired moments/scalars:

```bash
# Default uses Pacific 2014-2015
python3 legacy/python/build_legacy_dataset_from_pairs.py

# Or specify a different region
python3 legacy/python/build_legacy_dataset_from_pairs.py --data-root data/paired/North15_March2014/2014
```

Quick NPZ inspection example:

```bash
python3 legacy/python/Details_of_npz.py --data-root data/test --file-type images --file-name A2014060.0245.npz
```

To sample a fraction of a large paired data directory into a single NPZ (useful for quick local iteration before running the full benchmark):

```bash
python3 legacy/python/sample_paired_data.py data/paired/Pacific_2014-2015 benchmark_outputs/sampled_data.npz
```

### Guy Model (Final Locked-Down Recipe)

Once [legacy/python/guys_benchmark_models.py](legacy/python/guys_benchmark_models.py) `--feature-selection importance` has identified the best combo/model, [legacy/python/guy_model_2.0.py](legacy/python/guy_model_2.0.py) locks that recipe in (currently: top-20 Random Forest importance-selected features, Ridge alpha=1.0 baseline, plus the ANN) and trains it directly without re-running the full sweep:

```bash
python3 legacy/python/guy_model_2.0.py --demo
python3 legacy/python/guy_model_2.0.py --data-root data/paired/Pacific_2014-2015

# Restricted 37-feature All_Sky_AIflux-style pool, no RF trimming
python3 legacy/python/guy_model_2.0.py --feature-source allsky37 --data-root data/paired/Pacific_2014-2015
```

The script is versioned: [legacy/python/guy_model_1.0.py](legacy/python/guy_model_1.0.py) is the original recipe (no `--feature-source` flag, always top-20 of 129 features) kept for reference; `guy_model_2.0.py` adds the `--feature-source {full,allsky37}` option, with `full` (default) reproducing 1.0 exactly. See [legacy/Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt](legacy/Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt) for the results behind both versions. If a future benchmark run picks a different combo/model as the practical recommendation, bump the version (e.g. add `guy_model_3.0.py`) and update the constants and imports at its top to match.

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

- Python scripts are now under [python](python).
- Local data is organized under [data/paired](data/paired) by region.
- Test subset is [data/test](data/test).
- Pre-built training data goes in [data/consolidated](data/consolidated).
- New feature combination benchmark script: [legacy/python/guys_benchmark_models.py](legacy/python/guys_benchmark_models.py) — compares classical models, hyperparameter-tunes, then trains ANN.
- New sampling helper: [legacy/python/sample_paired_data.py](legacy/python/sample_paired_data.py) — samples a fraction of a paired data directory into a single NPZ.
- New locked-down recipe script: [legacy/python/guy_model_2.0.py](legacy/python/guy_model_2.0.py) (previously [legacy/python/guy_model_1.0.py](legacy/python/guy_model_1.0.py)) — trains the production recipe (Ridge + ANN on top-20 importance-selected features) picked by the benchmark.

