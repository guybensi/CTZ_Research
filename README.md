# CTZ_Research

CTZ_Research is a research-oriented project for improving the quality and predictive modeling of cloud and atmospheric datasets. The repository now includes a GPU-aware benchmarking pipeline that compares a tuned linear baseline, five additional classical regressors, and a configurable ANN.

## What is in this repository

- [python/benchmark_pipeline.py](python/benchmark_pipeline.py): main training and benchmarking entry point.
- [python/All_Sky_AIflux.py](python/All_Sky_AIflux.py): original ANN/linear regression workflow.
- [python/Details_of_npz.py](python/Details_of_npz.py): NPZ inspection and visualization utility.
- [BENCHMARKING.md](BENCHMARKING.md): detailed run instructions and output descriptions.
- [CLAUDE.md](CLAUDE.md): project rules for future edits.
- [plan.txt](plan.txt): the current implementation plan.
- [data](data): consolidated local datasets and legacy copies.

## Quick start

Demo run:

```bash
python3 python/benchmark_pipeline.py --demo --output-dir benchmark_outputs
```

Original-model run with pre-load tuning (paired moments/scalars data):

```bash
python3 python/All_Sky_AIflux.py --data-path data/March_2014N15_momentsNscalars/2014 --preload-tuning
```

To tune first and exit before loading the real dataset:

```bash
python3 python/All_Sky_AIflux.py --preload-tuning --preload-only
```

Run benchmark on real tabular NPZ/CSV data:

```bash
python3 python/benchmark_pipeline.py --data-path /path/to/data --output-dir benchmark_outputs
```

Inspect NPZ files from the consolidated dataset:

```bash
python3 python/Details_of_npz.py --data-root data/test --file-type images --file-name A2014060.0245.npz
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

The ANN benchmark will require GPU access inside the container.

## Notes For Current Data Layout

- The code now lives in [python](python), so all script invocations should use `python/<script>.py`.
- For [python/All_Sky_AIflux.py](python/All_Sky_AIflux.py), pass `--data-path` explicitly. Without it, the script falls back to a Windows-only default path.
- [python/All_Sky_AIflux.py](python/All_Sky_AIflux.py) expects a folder containing `moments/` and `scalars/` (for example [data/test](data/test) or [data/March_2014N15_momentsNscalars/2014](data/March_2014N15_momentsNscalars/2014)).
- [python/benchmark_pipeline.py](python/benchmark_pipeline.py) loads a single CSV/NPZ with a target key like `Flux`/`target`; it does not directly consume the paired `moments/` + `scalars/` folder format used by the original script.
- [python/Details_of_npz.py](python/Details_of_npz.py) currently defaults to `python/Data_toGuy/test`, which does not exist in the reorganized layout, so `--data-root` should be provided.
