# Methods, studies and outputs (implementation of PLAN.md)

This file explains **what every implemented method is, what it is for, which file holds it, and how to run it**. The plan itself is [PLAN.md](PLAN.md); the package map is [flux1km/README.md](flux1km/README.md).

All commands run from `master/`, after `pip install -e ".[dev,ml]"` (`ml` = PyTorch, needed only for the neural methods).

---

## 1. How to run

### One method alone

Every method module can be run on its own. It goes through the full protocol: frozen leakage-safe folds, a shared tuning budget, scoring through the common aggregation operator, significance tests and the decision rule. The reference baselines B-clim, B-const and M0-ridge are added automatically (`--no-baselines` turns that off).

```bash
python3 -m flux1km.methods.m3_pixel_mlp --demo                       # synthetic data with a known 1 km truth
python3 -m flux1km.methods.m3_pixel_mlp --demo --demo-hard           # the harder synthetic scene (PLAN.md 4.3)
python3 -m flux1km.methods.m3_pixel_mlp --data-root data/paired/Pacific_2014-2015 \
        --manifest benchmark_outputs/e0/split_manifest.json --seeds 0 1 2 3 4
```

All options of `flux1km.cli` work here: `--psf`, `--operator`, `--input-set F1..F5`, `--sampling S0..S7`, `--budget`, `--split`, `--seeds`.

### All methods together

```bash
python3 -m flux1km.cli --demo                                        # every registered method
python3 -m flux1km.cli --data-root DIR --manifest M.json --models M2 M3 M5 M6b
```

Methods that need PyTorch are skipped with a warning when it is not installed.

### Where the results go: one text file per run

Every run creates a new folder `benchmark_outputs/flux1km/run_<UTC time>/` with:

| File | Content |
|---|---|
| **`results.txt`** | **all results in plain text**, readable without code (layout below) |
| `summary.json` | the same, machine-readable |
| `predictions.csv` | one row per (model, fold, seed, test footprint): label, prediction under every operator, regime, angles |

The command line prints the path of `results.txt` at the end. Its sections, in order:

1. **WINNER**: the chosen method, the ranking, the tie set and the full decision log (why each method replaced, or did not replace, the previous one).
2. **Configuration**: models, operators, PSF, budget, seeds, sampling, noise floors, gates.
3. **Folds**: the frozen split, the sizes, and the hyper-parameters chosen for every model in every fold.
4. **Errors**: a method that fails on a fold is skipped and listed here; the run continues.
5. **Scores**: macro RMSE (primary), RMSE, MAE, bias, R², macro RMSE under every sensitivity operator, the interleave score, and pixel RMSE / correlation / SSIM against synthetic truth.
6. **Scores by cloud regime**: clear / broken / overcast.
7. **Residual flatness**: trend effect against SZA, VZA, RAA and glint, and the swath-side asymmetry.
8. **Physics checks of the 1 km maps**: far-field clear-sky flux, fraction above S0·μ0, spectrum peak at the footprint spacing, and the flux-vs-distance-to-cloud curve. On real data, the distance comes from a 0.66 µm brightness proxy.
9. **Pairwise comparisons**: bootstrap CI and p, with Holm correction, Wilcoxon (+Holm) and Diebold–Mariano.
10. **Gate results** and a **glossary** of every number.

---

## 2. The methods (PLAN.md §3)

Common rules for all of them:

- **Inputs are MODIS only.** The inputs are channel radiances (identified by wavelength), SZA/VZA/RAA and the glint angle. A test checks this for every method: it scrambles `Fsw`, `Flw`, `Isw`, `Ilw` and `Clrf` of the test footprints, and the predictions must not change (`tests/test_models.py::TestNoCeresInputs`).
- **Aggregation-loss methods train with the operator they are scored with.** The benchmark passes `operator` and `psf_path` to them.
- **Map networks predict an albedo through a sigmoid** and multiply it by S0·cos(SZA). A map can therefore never exceed the insolation.
- **Some methods read the test footprints' own labels**: B-const, M6e, M7 and M9-local. They are marked `ranked = no`, never ranked on the footprint score, and reported only on synthetic pixel truth and the interleave test.

| ID (registered name) | File | Map? | Reads test labels? | One line |
|---|---|---|---|---|
| B-clim | `flux1km/models.py` | yes | no | training-mean albedo × insolation; the floor |
| B-const | `flux1km/models.py` | yes | **yes** | every pixel = its footprint's CERES value (uniTR) |
| B-bagpixel | `flux1km/models.py` | yes | no | footprint polynomial `f(mean x)` applied per pixel (the naive map) |
| M0-ridge | `flux1km/models.py` | no | no | ridge on footprint mean/std + geometry |
| M0-allsky37 | `methods/m0_legacy.py` | no | no | the 37-feature All_Sky_AIflux ANN |
| M0-advisor | `methods/m0_legacy.py` | no | no | the advisor's moments ANN (log1p, weighted MAE, L1) |
| M1-gbm, M1-ann | `methods/m1_guy.py` | no | no | improved Guy model, GBM / ANN |
| M2 | `flux1km/models.py` | yes | no | per-pixel polynomial trained exactly from the moments |
| M3 | `methods/m3_pixel_mlp.py` | yes | no | per-pixel MLP, aggregation loss |
| M4 | `methods/m4_context_gbm.py` | yes | no | boosted trees + context features, aggregate gradient |
| M5 | `methods/m5_unet.py` | yes | no | UNet, aggregation loss + edge-aware smoothness |
| M6a | `methods/m6a_cnn_tabular.py` | yes | no | CNN + footprint-moment branch |
| M6b | `methods/m6b_mlp_context.py` | yes | no | pixel MLP + residual context encoder |
| M6c | `methods/m6c_unet_from_m2.py` | yes | no | UNet initialised from the M2 map |
| M6d | `methods/m6d_distill_m1.py` | yes | no | pixel MLP distilled from M1 on sub-footprint tiles |
| M6e | `methods/m6e_footprint_norm.py` | yes | **yes** | base map rescaled to each footprint label (GERB-HR style) |
| M6f | `methods/m6f_pseudo_labels.py` | yes | no | pixel MLP + M1-labelled pseudo-bags (mediated pseudo-labels) |
| M7 | `methods/m7_atprk.py` | yes | **yes** | trend map + area-to-point kriging of footprint residuals |
| M8 | `methods/m8_nb2bb.py` | yes | no | narrowband → broadband radiance (label `Isw`) → empirical ADM |
| M9 | `methods/m9_homogeneous.py` | yes | no | footprint model on homogeneous footprints applied per pixel (Gao) |
| M9-local | `methods/m9_homogeneous.py` | yes | **yes** | M9 blended with a per-granule local model (Gao Eq. 2–3) |
| M10 | `methods/m10_gp.py` | yes | no | aggregate GP (VBAgg) with random Fourier features, per-pixel std |
| M10-cmp | `methods/m10_gp.py` | yes | no | mediated / deconditional GP via the footprint moments |

### Null baselines and M0: the reference bar
- **B-clim, B-const, B-bagpixel** (in `models.py`). B-clim is the floor every method must beat. B-const is the reference for synthetic pixel truth and the interleave test. B-bagpixel is the naive way to make a map from a footprint model (Law 2018 found it worse than a constant).
- **M0-ridge**: ridge on footprint mean/std + geometry.
- **M0-allsky37** (`m0_legacy.py`): the All_Sky_AIflux ANN.
  - **Inputs:** SZA, VZA, RAA + mean and skewness of 17 channels (= 37). Means are cross-normalised by 0.47 / 4.05 / 11 µm.
  - **Network:** dense layers N/2, N/4, N/8, 5; MSE loss.
- **M0-advisor** (`m0_legacy.py`): the advisor's `train_Allsky_moments_narrow_values.py`.
  - **Inputs:** SZA, VZA, RAA + mean/std/skew/kurt of 13 channels, cross-normalised the same way.
  - **Training:** log1p label with the weighted MAE (weight log1p|y|+1); L1 0.01 / 0.001; early stop when the validation RMSE changes by less than 0.001 for 3 epochs.
- **What is fair about the re-run:** the group split replaces the random 70/30 row split, early stopping uses the validation fold instead of the test set, and the test set is **not** cut to Fsw > 400.

### M1: improved Guy model (footprint upper bound and teacher)
- **Features** (`guy_features`), from the MODIS moments only:
  - mean and std per channel;
  - channel ratios: SW / 0.47 µm, 3–5 µm / 4.05 µm, thermal / 11 µm;
  - cos SZA, cos VZA, sin/cos RAA, and the glint angle;
  - a broadband reconstruction: the trapezoid integral of the SW channels and its albedo proxy.
- **Learners:** `M1-gbm` (histogram gradient boosting) and `M1-ann` (MLP, Huber loss). Both predict the albedo.
- **Tells us:** the best footprint skill reachable from MODIS. It gives no map. It is the teacher for M6d and M6f.

### M2: pixel-additive polynomial from the moments (already in `models.py`)
E[x]…E[x⁴] follow exactly from mean/std/skew/kurt. So a per-channel polynomial `f` trained on footprint raw moments is exactly a pixel model trained with the aggregation loss. It also trains on moments-only granules (all 5.26M footprints). It is the default on a tie when image data is scarce.

### M3: per-pixel MLP (`m3_pixel_mlp.py`)
- **Network:** 1×1 convolutions, so it sees only the pixel's own channels, geometry and glint. It is trained only through `MSE(Σ w f, F_CERES)`.
- **Regulariser** (`reg`):
  - `none`;
  - `manifold`: feature-space regularisation (Law 2018). Similar inputs get similar outputs, and spatial proximity is not used, so it cannot create an artificial CTZ;
  - `subset`: random-subset aggregation (Fan 2021).
- **Tells us:** pixel-only skill. It is the default on a tie (PLAN.md §6).

### M4: boosted trees with spatial context (`m4_context_gbm.py`)
- **Inputs:** the pixel inputs plus hand-crafted context computed from the radiances only (`methods/_common.context_features`):
  - window mean/std of the 0.66 µm reflectance and the 11 µm radiance (3, 7, 15 px);
  - the local bright (cloud-proxy) fraction;
  - the distance to the nearest bright pixel;
  - a ψ-like cloud-influence index.
- **Training:** functional gradient boosting on the aggregate loss. Each tree fits the footprint residual on subsampled pixels, weighted by `w_i`. The step size along the tree is exact (closed form), and early stopping uses the validation RMSE.
- **Tells us:** the value of hand-crafted context.

### M5: UNet (`m5_unet.py`)
- **Network:** a compact UNet over the 27×55 patch: channels + encoded geometry + glint.
- **Loss:** aggregation loss + edge-aware TV (`|∇f|·exp(−|∇x|/σ)`), which smooths only where the inputs are smooth.
- **Augmentation:** left–right flips of inputs, insolation and weights together, which is the same as toggling FOVorientation.
- **Tells us:** the value of learned spatial context and 3D effects. **It must beat M6b, not only M3** (PLAN.md §6, rule 6).

### M6: hybrids
- **M6a** (`m6a_cnn_tabular.py`): a UNet trunk on the pixels plus an MLP on the footprint moments, broadcast over the patch. Does footprint-level context add to the CNN?
- **M6b** (`m6b_mlp_context.py`): `logit = MLP(pixel) + g(neighbourhood)`. The context branch `g` (3×3 convolutions) is penalised, so it explains only the residual that needs neighbours.
- **M6c** (`m6c_unet_from_m2.py`): fit M2, pretrain the UNet pixel-wise on the M2 map, then fine-tune it with the aggregation loss.
- **M6d** (`m6d_distill_m1.py`): the M1 teacher predicts the flux of 9×11 tiles from the tile moments, and the student (M3) adds `λ·(tile mean of f − teacher)²` to the CERES loss. This is softer, finer-scale supervision. Note that the teacher was trained on 27×55 footprints, so its tile predictions carry a domain shift.
- **M6e** (`m6e_footprint_norm.py`): the base map (M2) is rescaled per footprint to the CERES label. Mode `multiplicative`, or `scene`, where bright pixels absorb more of the correction. **It reads test labels**: it is scored only on the overlap test and on synthetic truth, because uniform redistribution makes footprint-shaped boxes.
- **M6f** (`m6f_pseudo_labels.py`): mediated pseudo-labels (Chau 2021). M1 labels new bags: random windows of the training patches, and windows of **unlabelled MODIS granules** (`unlabeled_root=DIR` with `DIR/images/*.npz`). The student M3 trains on the real footprints plus these bags. This lifts the 6-granule limit.

### M7: ATPRK (`m7_atprk.py`)
- **Method:** a trend map (M2) plus area-to-point simple kriging of the footprint residuals, with an exponential covariance. The covariances are computed with FFT on the granule grid. The result aggregates back to every label (coherent).
- **Overlapping footprints** are kriged jointly when their positions are known (`images/row0`, `col0`: present in synthetic data and in a future in-house footprint index); otherwise each footprint is kriged alone.
- **It reads test labels.** The plan runs it in the focus phase, once the PSF and geolocation are available.

### M8: narrowband → broadband → ADM (`m8_nb2bb.py`)
- **Stage 1:** the Bertrand 2006 Eq. 14 form, `L_BB = c0 + c1 L0.66 + c2 L0.66² + c3 L0.86 + c4 L_swir + c5 θs + c6 γ`.
  - It is fitted per surface type on homogeneous, glint-free (γ > 25°) footprints against **`Isw` used as a label**.
  - E[L0.66²] comes from the moments, so the per-pixel equation aggregates exactly.
  - The default SWIR channel is 2.13 µm, because 1.64 µm is a known bad channel. `swir="1.64"` restores the paper's form.
- **Stage 2:** an empirical ADM, `log(π Isw / Fsw)` as a function of geometry, a glint lobe and the stage-1 scene albedo. So prediction needs only MODIS inputs.
- **Tells us:** whether the physics route is competitive.

### M9: homogeneous-footprint transfer (`m9_homogeneous.py`)
- **Training:** a footprint model (GBM or a single regression tree) trained only on homogeneous footprints (Gao `cv < 0.2` from the moments, weights 1/cv). The same feature columns are then applied per pixel. It is the cheapest legitimate way to turn a footprint model into a map.
- **M9-local** (reads test labels): blends the global model with a local per-granule model, with weights 1/residual² (Gao Eq. 2–3).

### M10: GP aggregate models (`m10_gp.py`)
- **M10 (VBAgg):** random Fourier features turn the GP into a Bayesian linear model. With a Gaussian aggregate likelihood, the posterior is exact. `predict_map_std` gives the **per-pixel uncertainty**.
- **M10-cmp:** mediated / deconditional. The footprint moments z are the mediating variable. A ridge conditional-mean map `z → bag feature` is learned on image bags, and then the posterior is formed from all labelled footprints, including moments-only granules (`fit_mediated(image_batch, moments_batch)`).

---

## 3. Data, E0 and the studies

| What | Command | Output |
|---|---|---|
| **Stage E0** (PLAN.md §2): inventory, value ranges/units, FOVorientation check, moments vs patches, fitted operator H, footprint overlap, bad-channel striping, frozen split manifest with leakage check, advisor bug report, optional M0 re-run | `python3 -m flux1km.e0 --data-root DIR [--zip Data_toGuy.zip] [--psf ceres_psf.csv] [--run-baselines]` | `benchmark_outputs/e0/E0_report.txt`, `split_manifest.json`, `fitted_operator.npz` (usable as `--psf` until `ceres_psf.csv` arrives), `advisor_bug_report.txt` |
| **Hard synthetic data** (§4.3): nonlinear, SZA-dependent NB→BB; anisotropic radiances (glint lobe, limb, forward/backscatter); a CTZ only partly visible in the channels | `--demo --demo-hard [--ctz-visible 0.3] [--no-ctz]` on any command | synthetic granules in the real file layout |
| **Synthetic-truth study** (§4.3): pixel RMSE/MAE/corr/SSIM vs B-const, the flux-vs-distance curve, several footprint counts and aggregation sizes, plus the no-CTZ control (the curve must stay flat) | `python3 -m flux1km.synthetic_study --models M2 M3 M5 M6b` | `synthetic_truth_<time>/synthetic_truth_results.txt` |
| **Sampling study** (§5): S0–S7 × leading methods, Jaccard / KS / MMD, train-on-Si × test-subpopulation matrix, Kendall τ, the τ > 0.8 rule | `python3 -m flux1km.sampling_study --data-root DIR --manifest M.json --models M2 M3 M9` | `sampling_<time>/sampling_results.txt` |
| **Decision rule** (§6) | applied automatically in every run (`flux1km/decision.py`) | the WINNER section of `results.txt` |

### The decision rule as implemented (`flux1km/decision.py`)
1. **Gates.** A method is dropped if:
   - its physics checks fail (clear sky outside 80 ± 10, values above S0·μ0, a peak at the footprint spacing);
   - its residuals trend with SZA/VZA/RAA/glint;
   - its swath-side asymmetry is too large;
   - its ranking flips between the primary and a sensitivity operator. Only map methods are compared here, because footprint-only methods do not depend on the operator.
2. **Primary score:** macro RMSE over the clear / broken / overcast regimes.
3. **Replacement.** Going from simple to complex, a method replaces the incumbent only if:
   - the bootstrap CI excludes 0 in its favour after Holm correction; **and**
   - its gain exceeds the CERES noise floor of the scored regimes (clear 1.9, broken 10, overcast 6 W m⁻²; still to be verified).
4. **Ties**, in order:
   1. synthetic pixel RMSE;
   2. the interleave–LOGO gap;
   3. the default, M3 (or M2 with fewer than 10 image granules);
   4. simplicity.
5. **The UNet rule:** if M5 wins without being significantly better than M6b, M6b wins.
