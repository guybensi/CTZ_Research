# Benchmarking Guide

This repository now includes a reusable benchmark pipeline at [benchmark_pipeline.py](benchmark_pipeline.py).

## What it does

- Runs a baseline linear regression model and five additional classical regressors.
- Tunes each classical model with grid search and k-fold cross-validation.
- Evaluates the models on multiple train/test splits.
- Trains and tunes an ANN on the same regression task.
- Saves the best classical model, the best ANN, and a JSON/CSV summary of the runs.

## Demo mode

Use demo mode until the final dataset is available:

```bash
python3 benchmark_pipeline.py --demo --output-dir benchmark_outputs
```

## Real data mode

Point the script at a CSV file, an NPZ file, or a directory containing data files:

```bash
python3 benchmark_pipeline.py --data-path /path/to/data --output-dir benchmark_outputs
```

## GPU behavior

The ANN benchmark uses TensorFlow. When the container is launched with GPU access, the script enables GPU memory growth and uses the available device automatically. The Docker image sets `CTZ_REQUIRE_GPU=1` so the ANN benchmark fails fast if a GPU is not present.

## Original model tuning

The copied original pipeline in [All_Sky_AIflux.py](All_Sky_AIflux.py) now supports a pre-load tuning step that searches a small synthetic ANN hyperparameter space before the real dataset is loaded. Use `--preload-tuning` to run the search and `--preload-only` if you only want the tuning summary.

## Outputs

- `benchmark_outputs/classical_results.csv`
- `benchmark_outputs/benchmark_summary.json`
- `benchmark_outputs/best_classical_model.joblib`
- `benchmark_outputs/best_ann_model.keras`
- `benchmark_outputs/best_ann_scaler.joblib`

## After the data arrives

1. Mount or copy the real dataset into the machine or container.
2. Run the original workflow first with preload tuning enabled:

```bash
python3 All_Sky_AIflux.py --data-path /path/to/data --data-file mrch_2014_Pacific.npz --preload-tuning
```

3. If the tuning looks reasonable, run the full original model on the real data without `--preload-only`.
4. Run the broader benchmark pipeline to compare the original model against the additional regressors and ANN variants.
5. Compare the saved metrics and keep the best configuration for the final training run.
