# CTZ_Research

CTZ_Research is a research-oriented project for improving the quality and predictive modeling of cloud and atmospheric datasets. The repository now includes a GPU-aware benchmarking pipeline that compares a tuned linear baseline, five additional classical regressors, and a configurable ANN.

## Layout

| Folder | What is in it |
|---|---|
| [master/](master/) | **The M.Sc. work.** [master/PLAN.md](master/PLAN.md) is the plan for the 1 km all-sky TOA flux map ([Thesis_Goal.md](Thesis_Goal.md)): benchmark every candidate direction under one leakage-safe protocol, then focus on the best. [master/flux1km/](master/flux1km/) is its tested infrastructure (data loading, leakage-safe splits, aggregation operators, sampling strategies, metrics, physics checks, baselines, benchmark runner). |
| [pre_master/](pre_master/) | Everything from before the M.Sc. plan: the footprint-level Guy model and benchmarks, All_Sky_AIflux, old docs, and the advisor's data-building code ([pre_master/advisor_code/](pre_master/advisor_code/)). Frozen. |
| [Documentation/](Documentation/) | Shared literature lessons. |
| `data/` | Local data (git-ignored), except one real sample granule in `data/sample/`. |

Quick start for the M.Sc. code:

```bash
cd master
```

```bash
pip install -e ".[dev,ml]"      # ml = PyTorch, needed only by the neural methods
```

```bash
python3 -m flux1km.cli --demo                         # every method of PLAN.md section 3 on synthetic data
python3 -m flux1km.methods.m3_pixel_mlp --demo        # one method alone (each method has its own module)
python3 -m flux1km.e0 --data-root data/sample         # stage E0 checks, frozen split manifest
```

```bash
pytest -q flux1km tests
```

Every run writes all of its results to a new plain-text file, `benchmark_outputs/flux1km/run_<time>/results.txt`.
See [master/MODELS.md](master/MODELS.md) for what every method is, how to run it and how to read the results, and
[master/flux1km/README.md](master/flux1km/README.md) for the module map and how to add a method.

## Pre-master work (`pre_master/`)

- [pre_master/python/benchmark_pipeline.py](pre_master/python/benchmark_pipeline.py): main training and benchmarking entry point.
- [pre_master/python/guys_benchmark_models.py](pre_master/python/guys_benchmark_models.py): feature combination benchmark with model selection, hyperparameter tuning, and ANN training. Target is `Fsw` (shortwave flux) from the scalars data.
- [pre_master/python/guy_model_2.0.py](pre_master/python/guy_model_2.0.py): locked-down production recipe from the benchmark (Ridge + ANN on the top-20 importance-selected features) — trains and saves both models directly, no grid search. Adds a `--feature-source {full,allsky37}` option over [pre_master/python/guy_model_1.0.py](pre_master/python/guy_model_1.0.py) (kept for reference); default `full` reproduces 1.0 exactly.
- [pre_master/python/sample_paired_data.py](pre_master/python/sample_paired_data.py): samples a fraction of a paired moments/scalars data directory and saves it as a single NPZ, for quick local iteration on the full dataset.
- [pre_master/python/All_Sky_AIflux.py](pre_master/python/All_Sky_AIflux.py): original ANN/linear regression workflow.
- [pre_master/python/Details_of_npz.py](pre_master/python/Details_of_npz.py): NPZ inspection and visualization utility.
- [pre_master/BENCHMARKING.md](pre_master/BENCHMARKING.md): detailed run instructions and output descriptions.
- [CLAUDE.md](CLAUDE.md): project rules for future edits.
- [pre_master/Documentation](pre_master/Documentation): old planning docs and result writeups — [pre_master/Documentation/plan_CV.txt](pre_master/Documentation/plan_CV.txt) (earlier CNN plan, superseded by [PLAN.md](master/PLAN.md)), [pre_master/Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt](pre_master/Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt), [pre_master/Documentation/BENCHMARK_RESULTS_1.0.txt](pre_master/Documentation/BENCHMARK_RESULTS_1.0.txt), [pre_master/Documentation/Ideas_For_guy_model_2.0.txt](pre_master/Documentation/Ideas_For_guy_model_2.0.txt), and [Documentation/LITERATURE_LESSONS.md](Documentation/LITERATURE_LESSONS.md) (lessons from the literature for the 1 km CNN model: architecture, overfitting, validation without 1 km truth).
- [data](data): consolidated local datasets and legacy copies.

## Quick start

Demo run:

```bash
python3 pre_master/python/benchmark_pipeline.py --demo --output-dir benchmark_outputs
```

Original-model run with pre-load tuning (paired moments/scalars data):

```bash
python3 pre_master/python/All_Sky_AIflux.py --data-path data/March_2014N15_momentsNscalars/2014 --preload-tuning
```

To tune first and exit before loading the real dataset:

```bash
python3 pre_master/python/All_Sky_AIflux.py --preload-tuning --preload-only
```

Run benchmark on real data (tabular NPZ/CSV or paired moments/scalars folders):

```bash
python3 pre_master/python/benchmark_pipeline.py --data-path /path/to/data --output-dir benchmark_outputs
```

Inspect NPZ files from the consolidated dataset:

```bash
python3 pre_master/python/Details_of_npz.py --data-root data/test --file-type images --file-name A2014060.0245.npz
```

Guy's Model - feature combination benchmark with automatic model selection and ANN training:

```bash
# Demo mode (synthetic data)
python3 pre_master/python/guys_benchmark_models.py --demo

# With real paired data (default), target is Fsw (shortwave flux)
python3 pre_master/python/guys_benchmark_models.py

# Rank features by Random Forest importance and benchmark top-N subsets instead of the fixed combinations
python3 pre_master/python/guys_benchmark_models.py --feature-selection importance
```

Train the final locked-down model (Ridge + ANN on the top-20 importance-selected features, no grid search):

```bash
python3 pre_master/python/guy_model_2.0.py --demo
python3 pre_master/python/guy_model_2.0.py --data-root data/paired/Pacific_2014-2015

# Compare against the restricted 37-feature All_Sky_AIflux-style pool
python3 pre_master/python/guy_model_2.0.py --feature-source allsky37 --data-root data/paired/Pacific_2014-2015
```

Sample a fraction of a large paired data directory into a single NPZ for faster local iteration:

```bash
python3 pre_master/python/sample_paired_data.py data/paired/Pacific_2014-2015 benchmark_outputs/sampled_data.npz
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

- The pre-master code lives in [pre_master/python](pre_master/python), so all its script invocations use `pre_master/python/<script>.py`.
- [pre_master/python/All_Sky_AIflux.py](pre_master/python/All_Sky_AIflux.py) now defaults to [data/test](data/test), and still accepts `--data-path` for explicit dataset selection.
- [pre_master/python/All_Sky_AIflux.py](pre_master/python/All_Sky_AIflux.py) expects a folder containing `moments/` and `scalars/` (for example [data/test](data/test) or [data/paired/North15_March2014/2014](data/paired/North15_March2014/2014)).
- [pre_master/python/benchmark_pipeline.py](pre_master/python/benchmark_pipeline.py) supports both single CSV/NPZ tabular files with a target key (`Flux`/`target`/etc.) and paired folder layouts containing `moments/` and `scalars/` NPZ files (including nested `test/moments` and `test/scalars`).
- [pre_master/python/Details_of_npz.py](pre_master/python/Details_of_npz.py) defaults to [data/test](data/test), and `--data-root` can still be used to inspect other datasets.
