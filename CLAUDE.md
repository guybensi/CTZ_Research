# CLAUDE.md

## Project Rules

- Keep the original regression logic intact. Improve the training workflow by tuning parameters, cross-validation, and model selection rather than changing the scientific target.
- Prefer changes in [python/benchmark_pipeline.py](python/benchmark_pipeline.py) for all ML workflow updates.
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

To run the original paired NPZ workflow on the consolidated local data:

```bash
python3 python/All_Sky_AIflux.py --data-path data/March_2014N15_momentsNscalars/2014
```

Quick NPZ inspection example:

```bash
python3 python/Details_of_npz.py --data-root data/test --file-type images --file-name A2014060.0245.npz
```

## Editing Notes

- Update [README.md](README.md) when the workflow or runtime instructions change.
- Update [Dockerfile](Dockerfile) when dependencies or the GPU runtime changes.
- Ignore generated training outputs and local environment files in `.gitignore`.

## Python Folder Migration Notes

- Python scripts are now under [python](python).
- Local data is consolidated under [data](data).
- Current canonical test subset is [data/test](data/test).

## Required Python-File Changes (Documented Only, Not Applied)

The following changes are still needed in Python files to run more smoothly on the current repository layout without relying on command-line overrides.

1. [python/All_Sky_AIflux.py](python/All_Sky_AIflux.py)
- Replace the Windows-only `DEFAULT_DATAPATH` with a repository-relative default such as `Path(__file__).resolve().parents[1] / "data" / "test"`.
- Optionally extend `_resolve_test_data_root` candidates to include `repo_root/data/test` and `repo_root/data/March_2014N15_momentsNscalars/<year>` automatically.
- Keep support for explicit `--data-path` as highest priority.

2. [python/Details_of_npz.py](python/Details_of_npz.py)
- Change `default_data_root` from `script_dir / "Data_toGuy" / "test"` to a repo-relative location under `data` (for example `repo_root / "data" / "test"`).

3. [python/benchmark_pipeline.py](python/benchmark_pipeline.py)
- Add a loader path for paired `moments/` + `scalars/` folders (the same format used by [python/All_Sky_AIflux.py](python/All_Sky_AIflux.py)).
- Alternatively, document and enforce that this script requires a single prebuilt tabular NPZ/CSV with a target key (`Flux`, `target`, etc.).
