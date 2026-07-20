# CTZ_Research

CTZ_Research is a research-oriented project for improving the quality and predictive modeling of cloud and atmospheric datasets. The repository now includes a GPU-aware benchmarking pipeline that compares a tuned linear baseline, five additional classical regressors, and a configurable ANN.

## What is in this repository

- [benchmark_pipeline.py](benchmark_pipeline.py): main training and benchmarking entry point.
- [BENCHMARKING.md](BENCHMARKING.md): detailed run instructions and output descriptions.
- [CLAUDE.md](CLAUDE.md): project rules for future edits.
- [plan.txt](plan.txt): the current implementation plan.

## Quick start

Demo run:

```bash
python3 benchmark_pipeline.py --demo --output-dir benchmark_outputs
```

Original-model run with pre-load tuning:

```bash
python3 All_Sky_AIflux.py --data-path /path/to/data --data-file mrch_2014_Pacific.npz --preload-tuning
```

To tune first and exit before loading the real dataset:

```bash
python3 All_Sky_AIflux.py --preload-tuning --preload-only
```

Run on real data:

```bash
python3 benchmark_pipeline.py --data-path /path/to/data --output-dir benchmark_outputs
```

## Original model tuning flags

- `--data-path`: overrides the hard-coded data directory used by the original script.
- `--data-file`: overrides the `.npz` file name.
- `--preload-tuning`: runs a synthetic hyperparameter search before the data is loaded.
- `--preload-only`: exits after the tuning step and writes a JSON summary.
- `--tuning-output`: saves the tuning summary to a custom file path.

## GPU container

Build the GPU image and run it on a remote server with NVIDIA Container Toolkit installed:

```bash
docker build -t ctz_research:gpu .
docker run --gpus all --rm -it -v "$PWD":/app ctz_research:gpu
```

The container is configured to run the benchmark script by default and the ANN benchmark will require GPU access inside the container.
