# flux1km: new-phase work (1 km all-sky TOA flux map)

New code for [PLAN.md](../PLAN.md) goes here. The old footprint-level code lives in [legacy/](../legacy/) and is only used as a baseline (M0) and as a source of reusable pieces. The legacy pieces worth reusing are:
- the paired-file loader;
- the nearest-wavelength lookup;
- the GPU setup;
- `Details_of_npz.py`.

Planned layout, with each part created when its first code lands:

```
flux1km/
  data/        loaders for moments/scalars/images, per-pixel geometry and glint angle
  splits/      frozen split manifests (granule groups, day/region/time holdouts, spatial blocks) + leakage check
  aggregation/ footprint operators H: uniform patch mean, approximate Gaussian PSF
  models/      M1-M8 from PLAN.md §3 (one module per method, shared fit/predict_map interface)
  evaluation/  footprint metrics by regime, significance tests, overlap/interleave test, synthetic truth, physics checks
  configs/     per-method search spaces and the shared tuning budget
  scripts/     entry points: run_benchmark.py (all methods, one protocol), make_map.py
```

Conventions:
- Every model exposes `predict_map(patch) -> (27, 55)` flux. Footprint-level models (M0, M1) expose `predict_footprint`. All scoring goes through the same aggregation operator.
- Inputs are MODIS radiances, geometry and glint only. A test should fail if a CERES-derived key (`Isw`, `Ilw`, `mSum*`) reaches a model.
- Outputs go to `benchmark_outputs/flux1km/run_YYYYMMDD_HHMMSS/`, which is git-ignored.
- Keep a `--demo` mode with synthetic patches, so everything runs before the real data is unpacked.
