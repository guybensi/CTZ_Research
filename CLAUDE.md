# CLAUDE.md

## Project Rules

- Keep the original regression logic intact. Improve the training workflow by tuning parameters, cross-validation, and model selection rather than changing the scientific target.
- Prefer changes in [python/benchmark_pipeline.py](python/benchmark_pipeline.py) for all ML workflow updates.
- For feature combination exploration and model selection workflow, use [python/guys_benchmark_models.py](python/guys_benchmark_models.py) which systematically benchmarks classical models before training ANN.
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
python3 python/benchmark_pipeline.py --demo
```

To run against real data:

```bash
python3 python/benchmark_pipeline.py --data-path /data --output-dir benchmark_outputs
```

### Guy's Model - Feature Combination Benchmark

Run feature combination analysis with automatic model selection, hyperparameter tuning, and ANN training:

```bash
# Demo mode (synthetic data)
python3 python/guys_benchmark_models.py --demo

# With real paired data (default)
python3 python/guys_benchmark_models.py

# Custom output directory and epochs
python3 python/guys_benchmark_models.py --output-dir my_results --ann-epochs 200

# Skip classical models, go straight to ANN
python3 python/guys_benchmark_models.py --skip-classical

# With specific data region
python3 python/guys_benchmark_models.py --data-root data/paired/North15_March2014/2014

# Rank features by Random Forest importance and benchmark top-N subsets (5/10/15/20/24) instead of the fixed combinations
python3 python/guys_benchmark_models.py --feature-selection importance
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
python3 python/All_Sky_AIflux.py --data-path data/paired/North15_March2014/2014

# Pacific 2014-2015 data
python3 python/All_Sky_AIflux.py --data-path data/paired/Pacific_2014-2015
```

To build a legacy training dataset from paired moments/scalars:

```bash
# Default uses Pacific 2014-2015
python3 python/build_legacy_dataset_from_pairs.py

# Or specify a different region
python3 python/build_legacy_dataset_from_pairs.py --data-root data/paired/North15_March2014/2014
```

Quick NPZ inspection example:

```bash
python3 python/Details_of_npz.py --data-root data/test --file-type images --file-name A2014060.0245.npz
```

To sample a fraction of a large paired data directory into a single NPZ (useful for quick local iteration before running the full benchmark):

```bash
python3 python/sample_paired_data.py data/paired/Pacific_2014-2015 benchmark_outputs/sampled_data.npz
```

### Guy Model (Final Locked-Down Recipe)

Once [python/guys_benchmark_models.py](python/guys_benchmark_models.py) `--feature-selection importance` has identified the best combo/model, [python/guy_model.py](python/guy_model.py) locks that recipe in (currently: top-20 Random Forest importance-selected features, Ridge alpha=1.0 baseline, plus the ANN) and trains it directly without re-running the full sweep:

```bash
python3 python/guy_model.py --demo
python3 python/guy_model.py --data-root data/paired/Pacific_2014-2015
```

If a future benchmark run picks a different combo/model as the practical recommendation, update the constants and imports at the top of `guy_model.py` to match.

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
- New feature combination benchmark script: [python/guys_benchmark_models.py](python/guys_benchmark_models.py) — compares classical models, hyperparameter-tunes, then trains ANN.
- New sampling helper: [python/sample_paired_data.py](python/sample_paired_data.py) — samples a fraction of a paired data directory into a single NPZ.
- New locked-down recipe script: [python/guy_model.py](python/guy_model.py) — trains the production recipe (Ridge + ANN on top-20 importance-selected features) picked by the benchmark.

