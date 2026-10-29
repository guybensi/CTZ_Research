# Inventory of existing (old) work in CTZ_Research

Scope: everything tracked in the local repo at /home/user/CTZ_Research (git HEAD 93a8d14 "Add literature lessons doc for the 1 km CNN flux model"). Sources are repo-relative file paths. The `data/` directory is **not present** in this checkout (it is git-ignored), so the NPZ data facts below come from the repo docs and the code that reads the files. They were not re-verified against the real files. The only NPZ files that could be opened are the two under `projects/`.

Tracked files: `.gitignore, BENCHMARKING.md, CLAUDE.md, Dockerfile, README.md, Thesis_Goal.md, requirements.txt, Documentation/{BENCHMARK_RESULTS_1.0.txt, BENCHMARK_RESULTS_2.0.txt, GUY_MODEL_VS_ALLSKY_COMPARISON.txt, Ideas_For_guy_model_2.0.txt, LITERATURE_LESSONS.md, plan_CV.txt}, python/{All_Sky_AIflux.py, Details_of_npz.py, benchmark_pipeline.py, build_legacy_dataset_from_pairs.py, consolidate_paired_data.py, guy_model_1.0.py, guy_model_2.0.py, guys_benchmark_models.py, plot_guy_vs_allsky_comparison.py, sample_paired_data.py}, outputs/A2014060.0245_all_wavelengths.png, projects/F1km/ALLSKY_newDat/Pac_allMnth_allCld_400Nup_m4_Ly4_Trns1_Cost0/{...dataHist.png, SW_ANNtestarray_...npz, SW_norm_factors_...npz, figs/...ANN.png, figs/...ANN_weights.png}`.

---

## Q1. Per-script inventory: purpose, I/O, CLI, importers, legacy vs reusable

### Takeaway
Every model script is **tabular and footprint-level**: each CERES footprint is one row of pre-averaged moments and scalars. No code reads `images/*.npz` for training. There is no PSF, CNN, granule split or pixel-level loss anywhere. Reusable pieces for the new phase are limited to:
- the filename-pairing loader pattern;
- the nearest-wavelength channel lookup;
- the CTH homogeneity filter;
- the GPU setup;
- the image plotting utility;
- the Eytan-style scatter plot;
- the metrics conventions.

Everything else is legacy.

### Cited Findings

**python/All_Sky_AIflux.py (1137 lines; author "eeytan", dated Aug and Dec 2024)**
- **Origin and structure:** this is the original supervisor-group script. It runs as a top-level script with no `main()`, uses module-level globals, and runs `argparse.parse_known_args` at import time, so it is not importable as a library. [Source](python/All_Sky_AIflux.py)
- **CLI flags:** `--data-path`, `--data-file` (default `mrch_2014_Pacific.npz`), `--preload-tuning`, `--preload-only`, `--tuning-trials` (8), `--tuning-samples` (1024), `--tuning-output`, `--seed` (42). [Source](python/All_Sky_AIflux.py)
- **Default data path:** `REPO_ROOT/data/test`. `_resolve_test_data_root()` also searches these hard-coded fallbacks: `<cwd>/test`, `<cwd>/Data_toGuy/test`, `data/test`, `data/`, `data/March_2014N15_momentsNscalars/<year>`. [Source](python/All_Sky_AIflux.py)
- **Two input paths:**
  1. A legacy `.npz` plus `.pkl` bundle with keys `column_names, label_normalization_factor, SWnorm, MIXnorm, LWnorm, Norms, df_mean, df_std` (as produced by `build_legacy_dataset_from_pairs.py`).
  2. Otherwise it builds a DataFrame from the `moments/` and `scalars/` pairs.

  [Source](python/All_Sky_AIflux.py)
- **Features:** 37 in total.
  - SZA, VZA, RAA.
  - `m` (mean) and `sk` (skew) at 17 wavelengths: 0.47, 0.66, 0.86, 1.24, 1.375, 2.13, 3.96, 4.05, 4.5, 6.7, 7.3, 8.5, 9.7, 11.0, 12.0, 13.33, 14.23 µm.
  - Each wavelength is matched to the nearest real channel via argmin over `SWwavlngs`+`LWwavlngs`.

  [Source](python/All_Sky_AIflux.py)
- **Target:** `Fsw` when `SWf=True`, otherwise `Flw`.
- **Homogeneity filter:** keeps rows with `meanCTH < 3000` and `stdCTH < 1000` (`mCTHth` and `stdCTHth`). [Source](python/All_Sky_AIflux.py)
- **Wavelength lists and normalisers:** the file hard-codes:
  - `visls=[0.47,0.55,0.66,0.86]`
  - `nirls=[0.905,0.935,0.940]`
  - `swirls=[1.24,1.64,2.13,1.375]`
  - `thermalls=[3.75,3.961,3.96,4.05,4.46,4.5,6.7,7.3,8.5,9.7,11.0,12.0,13.33,13.63,13.93,14.23]`
  - normaliser channels `SolarNorm='0.47'`, `MixedNorm='4.05'`, `ThermalNorm='11.0'` (declared but effectively unused: the norms are set to 1.0).

  [Source](python/All_Sky_AIflux.py)
- **Model:**
  - Keras functional API with one Input and one Normalization layer per feature, concatenated.
  - Up to 4 Dense layers with L1 regularisation (0.01 on the first, 0.001 after). By default only Dense(18) is active.
  - Adam with lr 1e-3, batch 5000, 200 epochs, `validation_split` 0.05, no early stopping.
  - Labels are normalised with a Normalization layer.
  - Random shuffle split with `trainprec=0.7` (a row-level split, not by granule).
  - A post-hoc sklearn LinearRegression of predictions against labels.

  [Source](python/All_Sky_AIflux.py)
- **Outputs (written into the data folder itself):**
  - `ANN_model_SW_<nmextension>.keras`
  - `SW_norm_factors_<nm>.npz`
  - `SW_ANNtestarray_<nm>.npz`, with keys `model, Label, RMSE, MAE, ME, R2, MAEperc, Errlow, Errmed, Errhigh, fiteq, fitR2`
  - optionally `Linear_model_*` and `SW_Lintestarray_*`

  `nmextension='3LY_Pac_mrch_tst'`. [Source](python/All_Sky_AIflux.py)
- **Optional import:** `Twilight_module`, inside a try/except (the module is not in the repo). Imported by nothing. [Source](python/All_Sky_AIflux.py)

**python/guys_benchmark_models.py (996 lines)**
- **Purpose:** sweep feature combinations and classical models, pick the best, tune hyperparameters, then train an ANN. [Source](python/guys_benchmark_models.py)
- **CLI flags:**
  - `--data-root` (default `data/paired/Pacific_2014-2015`)
  - `--output-dir` (default `benchmark_outputs`, with a `run_YYYYMMDD_HHMMSS` subfolder)
  - `--random-seed` 42
  - `--test-splits` [0.2]
  - `--cv-folds` 5
  - `--ann-epochs` 150
  - `--ann-batch-size` 64
  - `--ann-trials` 5
  - `--demo`
  - `--skip-classical`
  - `--feature-selection {"",importance}`
  - `--feature-source {full,allsky37}`
  - `--importance-models` (default "Ridge,SVM")

  [Source](python/guys_benchmark_models.py)
- **Library functions** (imported by `guy_model_1.0.py`, `guy_model_2.0.py` and `plot_guy_vs_allsky_comparison.py`):
  - `load_paired_moments_scalars()` builds the 129 features. It concatenates every moments key except `SWwavlngs`/`LWwavlngs`, in NPZ key order, then every scalars key except `Fsw` (the target) and `Flw`. Rows with any non-finite value are dropped.
  - `load_allsky37_subset()`
  - `demo_data()`
  - `get_feature_importances_from_ensemble()`: RF with 50 trees and max_depth 10, fit on standardised data.
  - `generate_importance_based_combinations()`: top 5, 10, 15, 20, 24 and n.
  - `get_classical_models()`: RF, LinReg, Ridge, GB, AdaBoost, XGBoost, CatBoost, SVR, each with a grid.
  - `build_ann_model()`: Normalization, then Dense layers that shrink by a factor of 0.5 (minimum 8), then a linear output.
  - `train_ann()`: EarlyStopping with patience 15 and `restore_best_weights`.
  - `select_best_practical_model()`: hard-coded preference order top_20, top_24, top_15, top_10, using Ridge.

  [Source](python/guys_benchmark_models.py)
- **Methodological flaws found in the code:**
  - Splits are random by row (`train_test_split` with random_state 42).
  - `main()` passes the **test set** as `validation_data` to `train_ann`, so early stopping looks at the test set.
  - The runtime messages and summary hard-code "94.0% accuracy, 83% faster" text taken from the synthetic demo run.

  [Source](python/guys_benchmark_models.py)
- **Outputs:** `benchmark_results.csv`, `performance_heatmap.png`, `feature_importance.png`, `ann_model.keras`, `summary.json`. [Source](python/guys_benchmark_models.py)

**python/guy_model_1.0.py and guy_model_2.0.py (136 and 160 lines)**
- **Purpose:** locked-down recipes. Both import `demo_data`, `get_feature_importances_from_ensemble`, `load_paired_moments_scalars` and `train_ann` from `guys_benchmark_models`; 2.0 also imports `load_allsky37_subset`. The import works via `sys.path.insert(0, <script dir>)`. [Source](python/guy_model_2.0.py)
- **Steps:**
  - RF top-20 features (or all 37 with `--feature-source allsky37`).
  - 80/20 split with seed 42.
  - StandardScaler plus `Ridge(alpha=1.0)`.
  - `train_ann` (again with the test set used as validation).

  [Source](python/guy_model_2.0.py)
- **CLI flags:** `--data-root`, `--output-dir`, `--feature-source` (2.0 only), `--demo`, `--ann-epochs` 150, `--ann-batch-size` 64.
- **Default outputs:**
  - 1.0 writes to `benchmark_outputs/final_model`.
  - 2.0 writes to `benchmark_outputs/final_model` or `benchmark_outputs/final_model_allsky37`.
  - Files: `ridge_model.joblib`, `ridge_scaler.joblib`, `ann_model.keras`, `final_model_summary.json`.
  - Note: the comparison doc refers to `benchmark_outputs/final_model_full/`, which suggests it was run with a custom `--output-dir`.

  [Source](python/guy_model_2.0.py); [Source](Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt)
- **Diff between the two versions:** the only difference is the `--feature-source` option. [Source](python/guy_model_1.0.py)

**python/benchmark_pipeline.py (609 lines)**
- **Purpose:** the generic, older GPU-aware benchmark. It is the Docker CMD. [Source](python/benchmark_pipeline.py); [Source](Dockerfile)
- **CLI flags:** `--data-path`, `--output-dir` (benchmark_outputs), `--demo`, `--random-seed`, `--test-splits` [0.2, 0.3], `--cv-folds` 5, `--ann-epochs` 80, `--ann-batch-size` 64, `--ann-trials` 8, `--require-gpu`. The env var `CTZ_REQUIRE_GPU=1` also forces a GPU. [Source](python/benchmark_pipeline.py)
- **Inputs it accepts:**
  - a CSV or NPZ with a target key from `Flux`, `flux`, `target`, `Target`, `y`, `Y`, `label`, `Label`;
  - a paired directory, including a nested `test/moments` layout.

  [Source](python/benchmark_pipeline.py)
- **Paired loader:** builds SZA, VZA, RAA plus mean and skew for all SW and LW channels. Target priority is Fsw, then Flux, then Flw. [Source](python/benchmark_pipeline.py)
- **Models:**
  - classical: LinReg, Ridge, Lasso, ElasticNet, RF, GB;
  - ANN: a Sequential network with L2 regularisation and dropout; the trial grid ranges from (128,64) to (256,128,64).

  [Source](python/benchmark_pipeline.py)
- **Outputs:** `classical_results.csv`, `benchmark_summary.json`, `best_classical_model.joblib`, `best_ann_model.keras`, `best_ann_scaler.joblib`. [Source](BENCHMARKING.md)
- **Importers:** none.

**python/build_legacy_dataset_from_pairs.py (163 lines)**
- **Purpose:** builds the legacy `.npz`+`.pkl` bundle used by `All_Sky_AIflux` and by `data/legacy_code/new_codes/train_Allsky_moments_narrow_values.py` (that file is not in the repo). [Source](python/build_legacy_dataset_from_pairs.py)
- **Features:** 13 wavelengths (0.47, 0.66, 0.86, 1.375, 2.13, 3.96, 4.05, 4.5, 6.7, 7.3, 9.7, 11.0, 12.0) × 4 moments (m/s/sk/k), plus SZA, VZA, RAA, mCTH, stdCTH = **57 feature columns**, with `Flux` = Fsw as the target. [Source](python/build_legacy_dataset_from_pairs.py)
- **CLI flags:** `--data-root` (Pacific default) and `--output-prefix` (default `projects/F1km/ALLSKY_newDat/mrchsepjunedec_2014-2015_Pacific`).
- **Importers:** none.

**python/consolidate_paired_data.py (120 lines)**
- **Purpose:** symlinks (or, with `--copy`, copies) `moments/`, `scalars/` and `images/` from several roots into `data/paired/Pacific_2014-2015`. [Source](python/consolidate_paired_data.py)
- **Default `--roots`:**
  - `data/test`
  - `data/legacy_code/old_codes/test`
  - `data/paired/North15_March2014/2014`
  - `data/paired/North15_March2014/2015`

  This hints that the "Pacific" folder may partly be a consolidation of other sources. [Source](python/consolidate_paired_data.py)
- **Importers:** none.

**python/sample_paired_data.py (106 lines)**
- **Purpose:** takes a 10% random sample of files and writes `{X, y}` to an NPZ. It duplicates the `load_paired_moments_scalars` logic. [Source](python/sample_paired_data.py)
- **CLI:** positional `argv[1]` = data_root, `argv[2]` = output file.
- **Importers:** none.

**python/Details_of_npz.py (161 lines)**
- **Purpose:** the only code that reads `images/`. It lists every key with its shape, dtype and size. [Source](python/Details_of_npz.py)
  - For images, it `imshow`s `ImSW[idx, ch]` and `ImLW[idx, ch]`.
  - For scalars, it scatters Isw against Fsw, coloured by `glintf`.
  - For moments, it prints the wavelengths.
- **CLI flags:** `--data-root` (data/test), `--file-type {images,scalars,moments}`, `--file-name` (A2014060.0245.npz), `--image-index` 2, `--sw-channels` "1,10", `--lw-channel` 11, `--all-wavelengths`, `--list-files`, `--save-path`. [Source](python/Details_of_npz.py)
- **Importers:** none.

**python/plot_guy_vs_allsky_comparison.py (147 lines)**
- **Purpose:** a 4-panel predicted-vs-true scatter with histograms, in the style of Eytan et al. 2025 Fig. 3. It reports RMSE, MAE, ME, R²(x=y) and the OLS fit with its R². [Source](python/plot_guy_vs_allsky_comparison.py)
- **Imports:** from `guys_benchmark_models`.
- **Hard-coded relative paths:**
  - `DATA_ROOT="data/paired/Pacific_2014-2015"`
  - `benchmark_outputs/final_model_full/...`
  - `benchmark_outputs/final_model_allsky37/...`
  - output `benchmark_outputs/guy_vs_allsky_scatter_comparison.png` and `_stats.json`

  These paths are cwd-relative, so the script must be run from the repo root. [Source](python/plot_guy_vs_allsky_comparison.py)

### Inferences
- **Reusable for the pixel-level/UNet phase** (copy or refactor into a new module rather than import, because the host files are tied to tabular work):
  - the filename-pairing convention (`scalars_dir / moment_file.name`);
  - the nearest-wavelength lookup;
  - `setup_tensorflow_runtime()`;
  - the CTH filter thresholds;
  - `Details_of_npz.py` (the image viewer, directly reusable);
  - `draw_panel`/`compute_stats` from the plot script (the footprint-level evaluation figure);
  - the `CTZ_REQUIRE_GPU` and Docker setup.
- **Legacy-only:**
  - `benchmark_pipeline.py`, `guys_benchmark_models.py` (the sweep logic), `guy_model_1.0/2.0`;
  - `build_legacy_dataset_from_pairs.py`, `sample_paired_data.py`, `consolidate_paired_data.py`;
  - `All_Sky_AIflux.py`, kept as the reference baseline.
- **Bugs not to carry forward:** the test-as-validation early-stopping leak and the random row split both inflate the reported tabular metrics.

### Gaps
- `data/legacy_code/...` and `Twilight_module` are referenced but absent. Their contents are unknown.

---

## Q2. Data formats: moments/, scalars/, images/

### Takeaway
There are three NPZ families per granule, with identical file names (e.g. `A2014060.0245.npz`, which reads as Aqua, year 2014, day 060, time 02:45 UTC), and index-aligned along axis 0 (footprint):

| Family | What it holds | Variables |
|---|---|---|
| `moments/` | Per-footprint spectral moments of MODIS pixels | 4 moments × (11 SW + 16 LW channels) |
| `scalars/` | Geometry, cloud and aerosol scalars plus CERES quantities | 23 keys |
| `images/` | 27×55 pixel cubes per footprint | 3 view-geometry + 11 SW + 16 LW channels, plus `FOVorientation` |

There is no per-pixel lat/lon, no PSF weights and no glint angle (only a `glintf` scalar). Only 6 image granules exist (11,917 footprints), against 2,163 moments/scalars granules (5.26 M raw footprints).

### Cited Findings
- **moments keys:**
  - `SWwavlngs`, `LWwavlngs`;
  - `mSWspec`, `stdSWspec`, `skewSWspec`, `kurtSWspec`;
  - `mLWspec`, `stdLWspec`, `skewLWspec`, `kurtLWspec`.

  108 moment columns = mean/std/skew/kurt × (11 SW + 16 LW). [Source](python/build_legacy_dataset_from_pairs.py); [Source](Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt)
- **scalars keys:** 21 feature scalars plus the targets `Fsw` and `Flw`:
  - CERES radiance: `Isw`, `Ilw`, `mSumIsw`, `stdSumIsw`, `mSumIlw`, `stdSumIlw`
  - `mDFNC`, `stdDFNC`
  - cloud: `CF`, `meanCTH`, `stdCTH`, `LiqPhsf`
  - geometry: `VZA`, `SZA`, `RAA`
  - location: `Lat`, `Lon`
  - `AOD`, `Clrf`, `glintf`, `landf`

  [Source](Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt)
- **Physical meaning (stated in the thesis doc):** CERES flux is computed from CERES radiance, `Fsw = π·Isw/R`. Isw is therefore CERES-derived and acts as a leak. [Source](Thesis_Goal.md)
- **Scale of CF and Clrf:** both are 0–100, not 0–1. Across the 6 image files they correlate with each other at only r ≈ 0.25. About 55% of footprints have Clrf > 95, and about 10.5% have CF < 5. [Source](Documentation/plan_CV.txt)
- **images keys per file:**
  - `Imviews` (N,3,27,55) float32: "3 viewing-geometry channels"
  - `ImSW` (N,11,27,55)
  - `ImLW` (N,16,27,55)
  - `FOVorientation` (N,), a 0/1 flag split roughly 50/50 in every file, meaning unknown

  [Source](Documentation/plan_CV.txt)
- **Image patch size:** 27×55 pixels, described as matching a CERES footprint (~27×55 km) at 1 km. [Source](Documentation/plan_CV.txt)
- **Image granules:** exactly 6 files, under `data/paired/Pacific_2014-2015/images/`, with a byte-identical copy in `data/test/images/`.

  | File | Footprints | Size |
  |---|---|---|
  | A2014060.0245 | 9 | 1 MB |
  | A2014060.0415 | 3,810 | 679 MB |
  | A2014060.0530 | 670 | 67 MB |
  | A2014090.2240 | 4,482 | 799 MB |
  | A2014090.2350 | 502 | 55 MB |
  | A2014090.2355 | 2,444 | 436 MB |
  | **Total** | **11,917** | **~2.04 GB** |

  [Source](Documentation/plan_CV.txt)
- **No code generates `images/`.** No CERES PSF weight map exists anywhere in the repo. [Source](Documentation/plan_CV.txt)
- **Moments/scalars data volume:**
  - 2,163 file pairs; `moments/` is 2.1 GB and `scalars/` 408 MB.
  - The full Pacific folder is 4.4 GB, including images.
  - 5,260,208 raw FOVs.
  - 1,703,803 rows survive the all-finite filter over 129 features (about 32%).
  - 5,260,208 survive with 37 features.
  - 2,909,419 survive the CTH filter.

  [Source](Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt)
- **Data regions:**
  - `data/paired/Pacific_2014-2015` (default; moments, scalars, images)
  - `North15_March2014/{2014,2015}` (moments, scalars)
  - `data/test` (a quick subset)
  - `data/consolidated`

  [Source](CLAUDE.md)
- **Channel order inferred from the code** (11 SW + 16 LW, matching the counts):
  - SW: 0.47, 0.55, 0.66, 0.86, 0.905, 0.935, 0.940, 1.24, 1.64, 2.13, 1.375
  - LW: 3.75, 3.961, 3.96, 4.05, 4.46, 4.5, 6.7, 7.3, 8.5, 9.7, 11.0, 12.0, 13.33, 13.63, 13.93, 14.23

  [Source](python/All_Sky_AIflux.py)
- **The all-wavelength plot of A2014060.0245** shows SW channel 8 and LW channel 15 as horizontal stripes rather than scene structure. [Source](outputs/A2014060.0245_all_wavelengths.png)
- **Image units:** the README calls the SW values "reflectance" and the LW values "brightness temp". The image plot shows SW values roughly 0–275 and LW values roughly 0–6. [Source](Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt); [Source](outputs/A2014060.0245_all_wavelengths.png)

### Inferences
- **Striped channels:** if the channel order above is correct, SW channel 8 = 1.64 µm (Aqua MODIS band 6, known to have dead detectors) and LW channel 15 = 14.23 µm. Exclude or impute both for a CNN.
- **Fill or artefact channels:** LW channel 4 (4.46 µm) and LW channel 6 also look like noise or stripes in the plot.
- **MODIS-only vs CERES-derived variables:**
  - MODIS-only, usable as inputs: images `ImSW`/`ImLW`/`Imviews`; the moments; the geometry scalars SZA, VZA, RAA; MODIS cloud products CF, CTH, LiqPhsf, Clrf; AOD (if MODIS); Lat/Lon.
  - CERES-derived, label or leakage only: Fsw, Flw, Isw, Ilw, and likely mSumIsw/stdSumIsw/mSumIlw/stdSumIlw.
  - Unclear: `mDFNC`/`stdDFNC`, and whether `glintf`/`landf` are flags or fractions.
- **Glint angle:** it is not stored, but it can be computed from the per-pixel `Imviews` (if those are SZA/VZA/RAA) using Bertrand's Eq. 14.

### Gaps
- The exact contents of the 3 `Imviews` channels (presumably per-pixel SZA, VZA, RAA) are unverified.
- The meaning of `FOVorientation` is unverified.
- Units and calibration of `ImSW`/`ImLW` are unverified.
- The meanings of `mSumIsw`, `mDFNC`, `glintf` and `landf` are unverified.
- Pixel geolocation, the mapping between neighbouring footprints and overlap, and scan time are all unverified. These could not be checked because `data/` is absent from this checkout. The key orders and wavelength values above come from docs and code, not from opening the files.

---

## Q3. Results so far (all numbers)

### Takeaway
On Pacific 2014–2015 (Fsw target, W m⁻², random 80/20 split):
- The 129-feature top-20 ANN reached RMSE 5.76, but this relies on CERES `Isw`, which is leakage.
- The MODIS-only 37-feature ANN reached 15.33, against 17.72 for the original All_Sky_AIflux.py.
- Ridge reached 11.03 (129-feature pool) and 22.27 (37-feature pool).

None of these numbers come from a granule-level split.

### Cited Findings
- **guy_model_1.0** (top-20 of 129, 1,703,803 rows):
  - Ridge(α=1): RMSE 11.03, MAE 8.38, R² 0.9796.
  - ANN (Norm, Dense(10), Dense(8), output): RMSE 5.76, MAE 4.06, R² 0.9944.

  [Source](Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt)
- **Top-20 features of the 129-feature run, in rank order:** mSW 0.905, **Isw**, VZA, RAA, mSW 0.47, mLW 3.96 (ch1), SZA, mSW 1.24, mLW 4.05, mLW 11.0, mLW 8.55, mLW 12.0, Clrf, mSW 0.55, mLW 3.75, **Ilw**, mSW 0.66, Lat, mLW 3.96 (ch2), AOD. No std, skew or kurtosis moments made the list. [Source](Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt)
- **All_Sky_AIflux.py** (37 features, CTH filter, 2,909,419 FOVs, 70/30 split):
  - ANN Dense(18): RMSE 17.72.
  - R² ≈ 0.97, derived by the docs rather than printed by the script.
  - Test label mean 182.17, std 106.87.

  [Source](Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt)
- **guy_model_2.0 `--feature-source allsky37`** (5,260,208 rows):
  - Ridge: RMSE 22.27, MAE 16.40, R² 0.9805.
  - ANN (Dense 18, Dense 9, early stop around epoch 63): RMSE 15.33, MAE 10.79, R² 0.9907.

  [Source](Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt)
- **Allsky37 benchmark sweep (Ridge):**

  | Subset | RMSE | MAE | R² |
  |---|---|---|---|
  | top_5 | 28.45 | 20.32 | 0.9682 |
  | top_10 | 23.89 | 17.35 | 0.9775 |
  | top_15 | 22.98 | 16.98 | 0.9792 |
  | top_20 | 22.58 | 16.57 | 0.9799 |
  | top_24 | 22.51 | 16.55 | 0.9801 |
  | top_37 | 22.27 | 16.40 | 0.9805 |

  ANN on top_20: RMSE 15.73, MAE 11.08, R² 0.9903. [Source](Documentation/BENCHMARK_RESULTS_2.0.txt)
- **RF importance within the 37 features:**
  - m0.66 accounts for 0.934 of total importance.
  - Next come m0.47 (0.024), m0.86 (0.020), VZA (0.005), SZA (0.004).
  - Skew features are about zero, except sk9.7 and sk4.5.

  [Source](Documentation/BENCHMARK_RESULTS_2.0.txt)
- **BENCHMARK_RESULTS_1.0 is synthetic demo data only** (2,000 samples, 24 features).
  - Ridge on all features reached R² 0.9492 and RMSE 1.06; ANN reached R² 0.9483.
  - The "top-20 = 94% accuracy, 83% faster" claims come from this synthetic run, not from real data.

  [Source](Documentation/BENCHMARK_RESULTS_1.0.txt)
- **Older run in `projects/`** (`Pac_allMnth_allCld_400Nup_m4_Ly4_Trns1_Cost0`):
  - `SW_ANNtestarray`: N = 287,228 test points; RMSE 20.61; MAE 15.45; ME −2.97; R² 0.956; fit Y = 0.94X + 30.11 (fit R² 0.961); Errlow 15.26, Errmed 15.02, Errhigh 19.08.
  - `SW_norm_factors`: label mean 523.97, std 103.06; `DFmean`/`DFstd` of length 57; `Normlist` ['0.47', '11.0', '4.05'].
  - The ANN scatter shows labels truncated at 400 W m⁻² and above.

  [Source](projects/F1km/ALLSKY_newDat/Pac_allMnth_allCld_400Nup_m4_Ly4_Trns1_Cost0/SW_ANNtestarray_Pac_allMnth_allCld_400Nup_m4_Ly4_Trns1_Cost0.npz); [Source](projects/F1km/ALLSKY_newDat/Pac_allMnth_allCld_400Nup_m4_Ly4_Trns1_Cost0/figs/Pac_allMnth_allCld_400Nup_m4_Ly4_Trns1_Cost0ANN.png)
- **Label distribution plot** (`dataHist.png`): the Fsw label is skewed, with skewness 1.11 and a range of about 70–1000. Skewness after transforms: log1p 0.16, sqrt 0.60, Box-Cox 0.04, Yeo-Johnson 0.04. [Source](projects/F1km/ALLSKY_newDat/Pac_allMnth_allCld_400Nup_m4_Ly4_Trns1_Cost0/Pac_allMnth_allCld_400Nup_m4_Ly4_Trns1_Cost0dataHist.png)
- **Leakage lesson:** the 5.76 result came mostly from `Isw`, because CERES flux is computed directly from it (Fsw = π·Isw/R). The model relearned the CERES ADMs, and it cannot be applied at 1 km. Conclusion: inputs must be MODIS-only, and CERES is used only as the label and constraint. [Source](Thesis_Goal.md); [Source](Documentation/LITERATURE_LESSONS.md)
- **Architecture-only gain:** 15.33 vs 17.72 is about a 14% improvement. It is not controlled: the splits (80/20 vs 70/30) and the row filters differ. [Source](Thesis_Goal.md)

### Inferences
- The `projects/` run uses 57 features (matching `build_legacy_dataset_from_pairs.py`: 13 wavelengths × 4 moments plus 5) and an Fsw ≥ 400 subset ("400Nup"). Its 20.61 is therefore not comparable to the 17.72 or 15.33 results.
- The correct baselines for the new phase are 17.72 and 15.33 (MODIS-only, footprint-level). Both should be re-run with a granule split and proper validation before being used as the comparison bar.

### Gaps
- The exact config of the `projects/` run (layers, epochs, data months) can only be inferred from its name.

---

## Q4. LITERATURE_LESSONS.md and plan_CV.txt: every lesson, recommendation, pitfall and paper

### Takeaway
LITERATURE_LESSONS.md (dated 2026-10-28) gives 13 lessons plus a checklist. The core recommendations are:
- a U-Net per-pixel regressor trained with an MSE loss on the PSF-weighted mean;
- granule (leave-one-granule-out) splits;
- early stopping on held-out granules plus physics checks;
- a synthetic-truth test, including a scene with no CTZ;
- geometry and glint residual checks;
- MODIS-only inputs.

plan_CV.txt (dated 2026-08-02) is an earlier, flat-mean (GlobalAveragePooling) version of the same idea. It adds data facts, a two-branch architecture, homogeneity filters, and staged Phases 0–2.

### Cited Findings: papers cited in LITERATURE_LESSONS.md
All of these are cited in [Source](Documentation/LITERATURE_LESSONS.md):
- **Bertrand, Futyan, Ipe, Gonzalez, Clerbaux 2006**, "Diurnal Asymmetry in the GERB SW Fluxes", IEEE TGRS 44(12): GERB/SEVIRI flux, narrowband-to-broadband conversion, resolution enhancement.
- **Gonzalez, Hermans, Dewitte, Ipe, Sadowski, Clerbaux**, "Resolution Enhancement of GERB Data", RMIB poster (about 2000): constrained optimisation for flux downscaling.
- **Law, Sejdinovic, Cameron, Lucas, Flaxman, Battle, Fukumizu 2018**, "Variational Learning on Aggregate Outputs with Gaussian Processes", NeurIPS (VBAgg).
- **Chau, Bouabid, Sejdinovic 2021**, "Deconditional Downscaling with Gaussian Processes", NeurIPS.
- **Fan, Chen, Wen, Sun, Gomes 2021**, "Resolving Super Fine-Resolution SIF via Coarsely-Supervised U-Net Regression", NeurIPS CCAI workshop.
- **Gao, Kustas, Anderson 2012**, "A Data Mining Approach for Sharpening Thermal Satellite Imagery over Land", Remote Sensing 4(11) (DMS).
- **Meyer, Platnick et al. 2026**, "C7/v2 updates to the MODIS/VIIRS cloud products and impacts of orbital drift", MODIS Science Team slides, March 2026.
- **Not read:** Song, Liang & Zhou, IGARSS 2022 (all-sky TOA albedo over ocean from MODIS; paywalled).
- **Also cited:** Su et al. 2015b (CERES clear-ocean uncertainty about 1.9 W m⁻²).

### Cited Findings: the 13 lessons
All from [Source](Documentation/LITERATURE_LESSONS.md):

1. **Architecture.** A U-Net trained only on aggregate labels works: Fan recovered 30 m SIF from 3 km labels, a 100× gap.
   - Use per-pixel regression with a linear or softplus output.
   - Loss: `MSE(Σ w_i f(pixel_i), F_CERES)`, with w = PSF weights summing to 1.
   - Copy from Fan: a 1×1-conv "pixel encoder"; make the second conv in each down block 1×1; no batch normalisation (it was unstable); AdamW with lr about 2e-4 and weight decay about 1e-4.
   - Compute and report the receptive field. Fan's is about 16 px, i.e. about 16 km at 1 km. Compare it to the CTZ distance scale.
2. **Overfitting.** Coarse supervision produces "extreme maps" that still average correctly: the coarse loss keeps falling while the fine error rises after 20–30 epochs (Fan, App. C, Figs. 4–5).
   - Early stopping options:
     1. Held-out CERES footprints by granule.
     2. Physics checks every few epochs: far-field pure clear-sky (PCS) SW ≈ 80 W m⁻²; CTZ ≈ −9 SW and +1.4 LW; monotonic decay with distance from cloud; no bimodal or extreme per-pixel distribution. Track the 1st and 99th percentiles of predictions; steady widening is the overfitting signature.
     3. Consistency on overlapping footprints.
   - Random-subset regularisation (Fan, App. D): aggregate over a random ~20% of pixels in each step.
   - Weight decay and early stopping are the main defences.
   - A possible extra idea (not from the papers): a penalty on within-footprint spread.
3. **Validation without 1 km truth.** Good aggregate fit does not guarantee pixel accuracy (Law 2018 §4.2: malaria fitted from coordinates alone).
   - Split by granule or by day, never by random footprint, because overlapping footprints share pixels.
   - Use about 70/15/15 by granule. With 6 granules, prefer leave-one-granule-out CV and report the mean and spread.
   - Keep CERES overlap groups and orbit-adjacent granules on the same side of the split.
   - Report by regime: clear, broken, overcast.
   - Reference numbers: thesis baselines 17.72 and 15.33. Meyer 2026 MODIS C7/v2 pixel flux vs CERES SSF (global): SW clear bias +4.4 (2.2%), RMSE 20.6; SW all-sky bias +11.0 (4.2%), RMSE 35.8; ice-only bias +27.3 (6.4%). CERES clear-ocean uncertainty about 1.9 (Su 2015b).
4. **The problem is under-determined.**
   - The Poisson likelihood in Law 2018 is just a count-data likelihood. For flux, use Gaussian, i.e. MSE.
   - Gonzalez's "generalized Lagrange" is in practice a data term plus a smoothness penalty (a quadratic penalty, not true multipliers).
   - Law's Taylor and Jensen approximations are needed only for the GP formulation.
   - What selects the solution in each paper: Gonzalez/Bertrand use smoothness of a multiplicative correction; Law and Chau use a GP prior; Fan uses U-Net architecture plus early stopping plus augmentation; Gao uses a tree trained on homogeneous pixels.
   - In the thesis the CNN makes this choice implicitly. The thesis should state that and test it.
5. **Aggregate in flux, not in inputs:** use mean(f(x)), never f(mean(x)).
   - Law: averaging the covariates performed worst (NLL > 2.4, vs 2.22 for a constant predictor).
   - Gao: T_agg = (mean T⁴)^¼.
   - Bertrand Eqs. 11–13: correct-then-convert differs from convert-then-correct (nonlinear ADM).
   - Fan: averaging baselines get worse as the resolution gap grows.
   - The current tabular models pre-average their inputs, which is why they cannot make a 1 km map.
6. **Regularisation can manufacture an artificial CTZ.** Smoothness leaks corrections across scene edges. A Gonzalez re-implementation shows error concentrated at edges; Bertrand found clear-ocean correction contaminated by clouds and proposed scene-dependent enhancement.
   - Test on a synthetic scene with no CTZ: the flux vs distance-to-cloud curve should be flat.
   - Report the receptive field.
   - Prefer edge-aware or scene-aware regularisation.
7. **Build a known-truth test** (Chau, Gonzalez, Gao).
   - Take a 1 km field from RT simulations, or the current per-pixel model.
   - Aggregate it with the real CERES PSF and footprint geometry.
   - Train exactly as on real data.
   - Compare pixel RMSE, bias by regime, and the distance-to-cloud curve.
   - Caveat (Chau's toy example): within-footprint variability is recoverable only if the inputs carry it.
8. **Residuals must not depend on geometry** (Bertrand: the GERB diurnal asymmetry came from a narrowband-to-broadband conversion with no VZA or RAA dependence; diagnosis by binned flat plots, Fig. 7).
   - Plot residuals against SZA, VZA, RAA and sun-glint angle.
   - Glint angle (Bertrand Eq. 14): `cos γ = μs μv + sqrt(1−μs²) sqrt(1−μv²) cos φ`.
   - Keep Isw out of the inputs.
9. **Features.**
   - Raw bands carry information that derived indices lose (Gao: NDVI-only sharpening gave box-like artefacts and missed thin clouds).
   - Multiplicative band noise (1+ε), ε ~ N(0, 0.2²), applied before normalisation, accounted for about half of Fan's improvement. Cross-normalising against 0.47, 4.05 and 11 µm partly serves the same purpose.
   - MODIS retrieval bands: COT from bands 1, 2, 5 (0.659, 0.865, 1.24 µm); effective radius from bands 6, 7, 20 (1.64, 2.13, 3.75 µm); thin cirrus from band 26 (1.38 µm). Make sure band 26 is included.
   - Sub-km cloudiness matters: the VIIRS 375 m vs 750 m cloud-fraction difference (Meyer).
10. **Training-data diversity.** Within a footprint, the gradient is the same for every pixel, ∝ w_i(mean − F). Pixel types are discriminated only by comparing footprints with different mixtures.
    - Histogram the footprint cloud fraction in the 6 granules.
    - Do not train only on homogeneous footprints. Gao used cv < 0.2, which would drop the CTZ here. A homogeneous-only model is a useful baseline.
11. **Coarse residual as a diagnostic map** (Gao used w ∝ 1/r²).
    - Map the PSF-weighted residual per footprint.
    - Do not add the residual back uniformly: it hides error and creates footprint-shaped artefacts.
12. **External baseline and data versions.**
    - The NASA MODIS C7 / VIIRS v2 pixel-level SW and LW flux (RRTMG from retrievals; it depends on a cloud mask) is the main external baseline.
    - Document versions: moving from C6.1 to C7 changes calibration, the phase algorithm, and GEOS-IT vs GDAS ancillary data.
    - Orbital drift: Terra was about 15 min late by Sept 2022 and Aqua by July 2023, with non-physical cloud-fraction trends. The 2014–2015 data are pre-drift; later years would need retraining on the new angle range.
13. **Optional per-pixel uncertainty.** GP posteriors give per-pixel variance (Law: the f²-link was calibrated; the exp-link was not). For the thesis, use a CNN ensemble or MC dropout.

**Checklist (13 items):**
- U-Net with linear output and PSF-weighted MSE
- receptive field compared with the CTZ scale
- granule split / LOGO CV
- early stopping on held-out granules plus physics checks
- per-epoch spread tracking
- random-subset aggregation
- synthetic-truth experiment, including a no-CTZ scene
- flat residuals vs SZA/VZA/RAA/glint
- band 26 and glint included; Isw excluded
- mixture-diversity check
- homogeneous-footprint baseline
- residual map
- C7 comparison and documented versions

### Cited Findings: plan_CV.txt
All from [Source](Documentation/plan_CV.txt) unless noted:
- **Framing.**
  - All current models are tabular: moments collapse 100% of the spatial information.
  - Paper cited: **Eytan, Gristey & Feingold 2025, AGU Advances**, "The Net Radiative Effect of the Ill-Defined Clear-Sky in the Vicinity of Clouds". It uses the ψ metric (Eq. 16) and the D distance-to-cloud metric, and applies footprint-trained models to pixels only under homogeneity (PCS, ψ < 1).
  - The task is weak or aggregate supervision: one label per about 1,500 pixels.
- **Named techniques:**
  - Learning from Label Proportions (Musicant et al., "Supervised learning by training on aggregate outputs");
  - Multiple-Instance regression with mean pooling (max pooling is wrong for additive flux);
  - attention-MIL (Ilse et al. 2018) to learn the PSF-like weighting;
  - downsampling-consistency loss: `loss = MSE(GAP2D(pred_map), Fsw)`.
- **Small data (11,917 footprints). Mitigations, highest priority first:**
  - (A) Two-branch model: a CNN branch plus a Dense branch on the tabular 129 or 37 features, with the tabular branch pretrained on millions of rows.
  - (B) Flip and rotate augmentation, but first check the meaning of `FOVorientation`.
  - (C) Teacher distillation from guy_model's ANN as a soft auxiliary loss.
  - (D) Ask Eshkol whether more image granules can be generated.
  - (E) 8–16 filters, dropout/L2, k-fold CV, and a split by **granule file**.
- **Phases.**
  - **Phase 0:** a per-pixel MLP where every pixel is a row, with footprint SZA/VZA/RAA and footprint Fsw broadcast to each pixel. Gate: the aggregated RMSE lands near Ridge-level (11–22); otherwise suspect a pairing bug.
  - **Phase 1:** a small CNN plus GAP, which must beat Phase 0.
  - **Phase 2:** a fully convolutional (27,55,1) map, only after Phase 1 passes and validation is in place.
  - Also: a CNN-only ablation.
- **Homogeneity filters:**
  1. `meanCTH < 3000` and `stdCTH < 1000` (already in All_Sky_AIflux).
  2. Clrf > 95 (about 55% of footprints pass).
  3. CF < 5 (about 10.5% pass). Combining 2 and 3 leaves about 1,000–1,300 footprints.
  - Alternative to hard filtering: per-sample loss weighting.
- **Validation:**
  1. Footprint-level RMSE, MAE and R², using the held-out-file split and the scatter style of `plot_guy_vs_allsky_comparison.py`.
  2. Visual overlays of maps on ImSW/ImLW (`Details_of_npz` pattern).
  3. Flux vs distance-to-cloud decay (paper Fig. 1b/c; e-folding about 10 km). Caveat: the paper's result is for Fclr, not all-sky.
  4. Phase 0 vs Phase 1/2 consistency.
- **Proposed naming:** `python/guy_image_model.py`, `load_paired_images(data_root)` (transposed to NHWC), `build_image_ann_model()`; split by granule filename.
- **Open questions:**
  - more granules?
  - meaning of `FOVorientation`?
  - is a CERES PSF map available?
  - does the Fig. 1 decay hold for all-sky Fsw?
- **Ideas for the 37-feature model** (Ideas_For_guy_model_2.0, appended in the same document): [Source](Documentation/Ideas_For_guy_model_2.0.txt)
  - (A) drop Ridge, or rank combos by ANN instead;
  - (B) nonlinear baselines via `--importance-models "Ridge,Gradient Boosting,XGBoost"`;
  - (C) reconstruct a broadband Isw-like feature, `f = (π/R) Σ I_ch α_ch` (paper Eqs. 10–11; α from Loeb & Manalo-Smith 2005);
  - (D) ratios against 0.47, 4.05 and 11.0 µm (paper App. A(e));
  - (E) swap skew for std;
  - (F) cos SZA, cos VZA, sin/cos RAA and interaction terms (the anisotropic factor R ranges from <1 to >3, paper Fig. 5b);
  - (G) a stacked residual model.
  - Suggested order: A → E → B → D/F → C → G.
- **Thesis_Goal.md additions:** [Source](Thesis_Goal.md)
  - Physics features: broadband reconstruction (Eq. 11), cross-normalisation, std instead of skew, angle trigonometry.
  - Success criteria: (1) footprint-level reproduction on held-out granules, beating 17.72/15.33 and approaching about 1.9 W m⁻², broken down by regime; (2) consistency across overlapping footprints, with held-out footprints predicted from neighbours; (3) reproducing the paper: PCS ≈ 80 W m⁻² SW, and CTZ ≈ −9 SW and +1.4 LW (Table B1). Paper CTZ figures: SW −9 W m⁻², LW +1.4 (≈ 90 ppm CO₂); the CTZ covers about 60% of "clear" sky.
  - Risk layers: foundation (per-pixel model plus aggregation; low risk), extension (CNN; medium), full map (high).

### Inferences
- **plan_CV vs LITERATURE_LESSONS:**
  - plan_CV uses a flat GAP and recommends hard homogeneity filtering and flip augmentation. LITERATURE_LESSONS supersedes it with PSF weighting and warns against homogeneous-only training (lesson 10).
  - plan_CV's Phase 0 broadcasts the footprint label to each pixel. That is f(x_i) ≈ F per pixel, not mean(f) = F, so it conflicts with lesson 5 unless Phase 0 is reframed as a per-pixel model trained through the aggregate loss.
- **Pre-work this implies:** a PSF implementation (none exists) and per-pixel geometry (`Imviews` must be verified) are prerequisites.

### Gaps
- No PSF code, PSF parameters or CERES footprint geometry exist in the repo.
- Information about which footprints overlap is not documented for the NPZ files.

---

## Q5. Cross-file dependencies that would break if files moved to legacy/

### Takeaway
Moving `python/*` into `legacy/` would break:
- the sibling imports (via `sys.path` of the script directory, so they keep working if all four files move together);
- the `parents[1]` REPO_ROOT data defaults (they would resolve to `legacy/` rather than the repo root, if moved one level deeper);
- the cwd-relative paths in the plot script;
- the Docker CMD;
- many Markdown links in README, CLAUDE.md, Thesis_Goal and Documentation.

### Cited Findings
- **Imports:**
  - `guy_model_1.0.py`, `guy_model_2.0.py` and `plot_guy_vs_allsky_comparison.py` do `sys.path.insert(0, Path(__file__).resolve().parent)` and then `from guys_benchmark_models import ...`. [Source](python/guy_model_2.0.py); [Source](python/plot_guy_vs_allsky_comparison.py)
  - No other cross-imports exist.
- **`Path(__file__).resolve().parents[1]` used as the repo root.** This assumes the scripts sit exactly one level below the root.
  - `All_Sky_AIflux.py`: `data/test`, `data/March_2014N15_momentsNscalars`
  - `guys_benchmark_models.py`, `guy_model_*.py`: `data/paired/Pacific_2014-2015`
  - `Details_of_npz.py`: `data/test`
  - `build_legacy_dataset_from_pairs.py`: `data/paired/...` and `projects/F1km/ALLSKY_newDat/...`
  - `consolidate_paired_data.py`: `data/test`, `data/legacy_code/old_codes/test`, `data/paired/North15_March2014/{2014,2015}`, `data/paired/Pacific_2014-2015`

  Sources: [Source](python/All_Sky_AIflux.py); [Source](python/build_legacy_dataset_from_pairs.py); [Source](python/consolidate_paired_data.py)
- **cwd-relative paths:**
  - `plot_guy_vs_allsky_comparison.py` uses `data/paired/...` and `benchmark_outputs/final_model_full|allsky37/...`. [Source](python/plot_guy_vs_allsky_comparison.py)
  - `sample_paired_data.py` defaults to `data/paired/...` and `benchmark_outputs/sampled_data.npz`.
  - The default `--output-dir benchmark_outputs` in the benchmark and guy_model scripts is also cwd-relative.
- **Dockerfile:** `CMD ["python3", "python/benchmark_pipeline.py"]`; it also copies `requirements.txt`. [Source](Dockerfile)
- **Doc references (counts of path mentions):**
  - CLAUDE.md: 26
  - README.md: 24
  - GUY_MODEL_VS_ALLSKY_COMPARISON.txt: 15
  - BENCHMARKING.md: 7
  - plan_CV.txt: 7
  - Thesis_Goal.md: 4
  - BENCHMARK_RESULTS_2.0: 3

  CLAUDE.md "Project Rules" names `python/benchmark_pipeline.py` and `python/guys_benchmark_models.py` as the places for ML changes, and documents the guy_model versioning rule (bump to 3.0). [Source](CLAUDE.md)
- **Stale references already present:**
  - BENCHMARKING.md uses root-level paths: `benchmark_pipeline.py`, `All_Sky_AIflux.py`.
  - README uses `data/March_2014N15_momentsNscalars/2014` (the current layout is `data/paired/North15_March2014/2014`).
  - plan_CV refers to `guy_model.py` (renamed to 1.0/2.0).
  - The plot script's docstring also refers to `guy_model.py`.

  [Source](BENCHMARKING.md); [Source](README.md)
- **Artifacts that All_Sky_AIflux writes into the data folder** (`data/paired/Pacific_2014-2015/ANN_model_SW_3LY_Pac_mrch_tst.keras` and `SW_norm_factors_...npz`) are cited in the comparison doc. [Source](Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt)

### Inferences
- **Safe move:** put all of `python/` into `legacy/python/` together, so sibling imports still work, and fix REPO_ROOT. Using `parents[2]`, or a search upward for `.git`, will be needed.
- **Also update:** the Dockerfile CMD, README, CLAUDE.md (rules and run commands) and Thesis_Goal links.
- **Documentation/ reorganisation:** LITERATURE_LESSONS.md and Thesis_Goal.md belong to the "new" phase. BENCHMARK_RESULTS_*, GUY_MODEL_VS_ALLSKY_COMPARISON and Ideas_For_guy_model_2.0 are legacy. plan_CV.txt is a bridge: its data facts are still useful, but its method is superseded.

### Gaps
- None beyond the absent `data/` tree. Paths inside `data/` could not be checked.

---

## Q6. .gitignore and environment

### Takeaway
All data and model artifacts are untracked, and the `projects/` and `outputs/` contents are the only tracked results.

### Cited Findings
- **.gitignore ignores:**
  - `data/`, `Data_toGuy/`, `test/`
  - `benchmark_outputs/`
  - `*.joblib`, `*.keras`, `*.pkl`
  - virtual environments, `__pycache__`, caches, `.ipynb_checkpoints`, OS files

  `*.npz` and `*.png` are **not** ignored, which is why the `projects/` NPZ and PNG files are tracked. [Source](.gitignore)
- **Dockerfile:**
  - base image `nvidia/cuda:12.2.0-runtime-ubuntu22.04`, Python 3.11;
  - environment `TF_FORCE_GPU_ALLOW_GROWTH=true`, `CTZ_REQUIRE_GPU=1`;
  - apt packages: libhdf5, libnetcdf.

  [Source](Dockerfile)
- **requirements.txt:** unpinned numpy, pandas, matplotlib, scipy, scikit-learn, `tensorflow[and-cuda]`, tensorflow-addons (deprecated upstream), netCDF4, h5py, xarray, opencv-python-headless, scikit-image, seaborn, pillow, xgboost, catboost. There is no PyTorch. [Source](requirements.txt)

### Inferences
- A new `legacy/` folder with tracked NPZ or PNG files would stay tracked, which is fine.
- New large outputs should go under `benchmark_outputs/`, or be added to `.gitignore`.
- The requirements are unpinned, so reproducibility of the earlier numbers is not guaranteed.

### Gaps
- None.
