# CLAUDE.md

## Project Rules

- Keep the original regression logic intact. Improve the training workflow by tuning parameters, cross-validation, and model selection rather than changing the scientific target.
- Prefer changes in [benchmark_pipeline.py](benchmark_pipeline.py) for all ML workflow updates.
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
python3 benchmark_pipeline.py --demo
```

To run against real data:

```bash
python3 benchmark_pipeline.py --data-path /data --output-dir benchmark_outputs
```

## Editing Notes

- Update [README.md](README.md) when the workflow or runtime instructions change.
- Update [Dockerfile](Dockerfile) when dependencies or the GPU runtime changes.
- Ignore generated training outputs and local environment files in `.gitignore`.
