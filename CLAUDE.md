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

