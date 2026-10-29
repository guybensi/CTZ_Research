# PLAN: 1 km All-Sky TOA Flux Map (new phase)

This plan covers the new phase of the thesis: turning [Thesis_Goal.md](Thesis_Goal.md) into a 1 km flux map. It is deliberately generic. First every candidate direction is benchmarked under one frozen protocol, then work focuses on the method closest to CERES.

The detailed reasoning, citations and numbers are in the research report [reports/תוכנית מחקר מפת שטף 1 קמ.md](reports/תוכנית%20מחקר%20מפת%20שטף%201%20קמ.md) (Hebrew). Its source notes are in [research_notes/](research_notes/). The literature lessons are in [Documentation/LITERATURE_LESSONS.md](Documentation/LITERATURE_LESSONS.md). The previous footprint-level work is frozen under [legacy/](legacy/).

---

## 1. Goal

Learn a function `f(pixel, [neighbourhood], geometry) -> F_SW` (W m⁻²) for every ~1 km MODIS pixel, in all sky conditions and with no cloud mask.

**The only supervision** is the CERES footprint flux:

```
loss = MSE( Σ_i w_i · f(x_i) , F_CERES )     w = CERES PSF (uniform over the 27×55 patch until the PSF is available)
```

This is learning from aggregate observations (LLP / multiple-instance regression), not ordinary regression.

**Hard rules**

1. **Inputs are MODIS only.** That means radiances or reflectances (channel identity comes from the wavelength), sun and view geometry (SZA, VZA, RAA), and the glint angle. Nothing derived from CERES is an input: no `Isw`, `Ilw` or `mSum*`.
2. **Aggregate in flux, not in inputs.** Compute `mean(f(x))`, never `f(mean(x))`.
3. **Footprint-level skill is necessary but not sufficient.** It must be backed by the overlap-consistency test, the synthetic-truth test and the physics checks in §4.
4. **No footprint leakage.** Footprints overlap: a pixel falls inside 3–5 neighbouring footprints. So test footprints never share a granule, orbit or day with training footprints, except in the explicitly labelled interleave test.

SW comes first. LW (`Flw`) is a second head on the same framework once SW works.

## 2. Stage E0: data and infrastructure (prerequisite)

Nothing is benchmarked until these items are done:

- [ ] Unpack `Data_toGuy.zip` (on Drive, shared folder `Guy_Fluxes1KM/Old_codes_data`) into `data/` and inventory it:
  - granules and footprints
  - keys and shapes for `moments/`, `scalars/` and `images/`
- [ ] Confirm what is in each file:
  - the 3 `Imviews` channels: SZA, VZA, RAA?
  - the meaning of `FOVorientation`
  - the units of `ImSW` and `ImLW`
  - the meaning of `glintf`, `landf`, `Clrf` and `CF`
- [ ] Check whether `moments` equal patch statistics of `images` for the same footprints. This tells us whether the moments are PSF-weighted or uniform, and it gates method M2.
- [ ] Compute a per-pixel glint angle from geometry: `cos γ = μs μv + sin θs sin θv cos φ`.
- [ ] Map footprint overlap from footprint-centre Lat/Lon.
- [ ] Build one frozen **split manifest** (§4.1) and an automatic pixel-reuse leakage check.
- [ ] Re-run the legacy baselines inside the new protocol. The old numbers (17.72 and 15.33 W m⁻²) used random row splits, and early stopping on the test set, so they are optimistic.
- [ ] Ask the advisor for more image granules and for the PSF (see §7).

## 3. Candidate methods (exploration phase)

Every method outputs a per-pixel map. Footprint-level methods output one value per footprint and are scored on the same footprints.

| ID | Direction | Inputs | Training signal | What it tells us |
|---|---|---|---|---|
| **M0** | Legacy baselines (All_Sky_AIflux 37-feature ANN, Ridge) | footprint moments + geometry | footprint MSE | the reference bar, re-measured fairly |
| **M1** | **Improved Guy model** using MODIS-only footprint features: mean/std per channel, channel ratios against 0.47/4.05/11 µm, trig of angles, glint, broadband reconstruction | footprint moments | footprint MSE/Huber; GBM + ANN | upper bound on footprint skill, and a teacher for the hybrids. Does not give a map. |
| **M2** | **Pixel-additive model trained from moments.** E[x]…E[x⁴] are recoverable from mean/std/skew/kurt, so a per-channel polynomial `f` can be trained exactly on 5.26M footprints and then applied per pixel. | footprint moments → applied per pixel | exact aggregation loss for polynomial-additive `f` | a pixel model at full data scale; initialises M3 and M5 |
| **M3** | Per-pixel MLP with aggregation loss | per-pixel channels + geometry + glint | `MSE(Σ w f, F)` | pixel-only skill, with no spatial context |
| **M4** | GBM with spatial-context features (window mean/std, distance to bright/cloudy pixel, local cloud fraction, ψ-like index) | pixel + neighbourhood features | custom aggregate gradient `g_i = 2 w_i (Σ w f − F)` | value of hand-crafted context |
| **M5** | **UNet** (CS-SUNet style) on 27×55 patches | 3 geometry + 11 SW + 16 LW channels | aggregation loss + edge-aware smoothness | value of learned spatial context and 3D effects |
| **M6** | **Hybrids**: (a) CNN plus a tabular branch; (b) pixel MLP plus a residual context encoder; (c) UNet initialised from M2; (d) soft distillation from M1; (e) post-hoc per-footprint normalisation, multiplicative and scene-dependent as in GERB-HR. Uniform redistribution creates footprint-shaped boxes (Gao, Bertrand), so (e) is scored only on the overlap test; (f) **mediated pseudo-labels** (Chau 2021): the footprint moments are the mediating variable. M1 is learned on 5.26M labelled footprints, then supplies targets for MODIS image bags that have no CERES label (any MODIS L1B granule). This lifts the 6-granule limit on image data. | mixes | mixes | whether combining helps, and why |
| **M7** | ATPRK / area-to-point kriging of footprint residuals on top of M2/M3 | trend model + PSF | coherence-preserving downscaling | needs PSF and geolocation, so it runs in the focus phase |
| **M8** | Two-stage narrowband→broadband→ADM, with `Isw` used as a **label** in stage 1. Starting form from Bertrand 2006 Eq. 14: `L_BB = c0 + c1 L0.66 + c2 L0.66² + c3 L0.86 + c4 L1.64 + c5 θs + c6 γ`, fitted per surface type, with homogeneous, glint-free (γ > 25°) training footprints | channels → broadband radiance → flux | MSE vs `Isw`, then ADM | the physics-route alternative |
| **M9** | **Homogeneous-footprint transfer** (Gao 2012 DMS). Train a footprint model (regression tree / GBM) only on homogeneous footprints, then apply it to single pixels. On a homogeneous footprint, `f(mean x) ≈ mean f(x)`, so the transfer is valid. Homogeneity is measured by `cv = mean_b(σ_b/μ_b)` < 0.2, computable from the moments; training samples are weighted by 1/cv. Global and local (per-granule) models are blended per footprint with weights ∝ 1/residual² (Gao Eq. 2–3). | footprint moments of homogeneous footprints → applied per pixel | footprint MSE | the cheapest legitimate way to turn the Guy model into a map; uses all 5.26M footprints |
| **M10** | GP aggregate models: VBAgg (Law 2018) and mediated/deconditional downscaling (VarCMP, Chau 2021) | pixel features + footprint covariates | aggregate likelihood / deconditional posterior | per-pixel uncertainty. Low priority: scaling to ~1,500 pixels per bag is hard. |

**Null baselines** (always reported, with no tuning):
- **B-const** (uniTR in Gao, "constant in bag" in Law): every pixel gets its footprint's CERES value, or for held-out footprints the PSF-weighted interpolation of neighbouring footprints. Any method must beat it on the interleave test and on synthetic truth.
- **B-bagpixel** (Law): the legacy footprint model `f(mean x)` applied to single pixels. It is the naïve way to "use the Guy model for a map". Law found it worse than a constant (NLL 2.4 vs 2.22).

**Input ablation (which inputs are really needed).** Each leading method is run on nested input sets:

| Set | Inputs |
|---|---|
| F1 | VIS + geometry |
| F2 | F1 + SWIR (1.24, 1.375, 2.13 µm) |
| F3 | F2 + water-vapour channels |
| F4 | F3 + thermal channels |
| F5 | F4 + glint angle |

Wavelengths are channel identities, not features.

Bad channels are excluded. In the sample image, 1.64 µm (Aqua band 6) and 14.23 µm show only stripes.

MODIS retrieval products (CF, CTH, AOD) and Lat/Lon are **not** inputs. They are used only for stratification and analysis.

**Regularisation choices to compare:**
- Spatial smoothness (TV, Gonzalez/GERB) spreads cloud signal into neighbouring clear pixels, so it can create an artificial CTZ.
- **Feature-space (manifold) regularisation** (Law 2018) penalises different outputs only for pixels with similar inputs. It does not use spatial proximity, so it should not create a CTZ.
- Random-subset aggregation (Fan).

**Fine-tuning:** every method gets the same tuning budget, for example 40 Optuna trials on inner folds. The best configuration is then retrained with 5 seeds.

## 4. Benchmark protocol (one protocol for every method)

### 4.1 Splits (leakage-safe)

- **Arena A (primary, all methods): the 6 image granules in 4 independent groups.**
  - Granules from the same orbit stay together, e.g. 090.2350 with 090.2355, and 060.0245 with 060.0415.
  - Evaluation is leave-one-group-out with nested validation.
- **Arena B (secondary, moments-based methods M0–M2):**
  - group k-fold by day;
  - temporal holdout: train 2014, test 2015;
  - regional holdout: train Pacific, test North15 or Atlantic.
  - The days of the Arena A granules are **always excluded** from M1/M2 training.
- **Spatial blocks** of at least 100 km, with a buffer at least the size of the footprint's 95% PSF ellipse and scaled with VZA. They give more independent units than 4 groups.
- **Interleave test:** train on every other footprint and scan line, and predict those in between from the map. It tests within-footprint distribution. It is leaky by design, so it is reported separately and **never** used to rank methods.
- **Automatic check:** the PSF-weighted fraction of each test footprint's pixels that fall inside any training footprint must be about 0.

### 4.2 Metrics and reporting

- RMSE, MAE, bias and R² at footprint level, using the same aggregation operator H for every method:
  - **uniform patch mean** (primary);
  - **approximate Gaussian PSF** (sensitivity check). The ranking must not flip between the two.
- Breakdown by regime (clear / broken / overcast), SZA, VZA and glint bins, shown next to the CERES ADM noise floor. The floor is about 1.9 W m⁻² for clear ocean and up to about 10 W m⁻² for all-sky (Su et al. 2015). These values are still to be verified.
- Residuals must be flat against SZA, VZA, RAA and glint. Bertrand 2006 found the GERB diurnal asymmetry exactly this way, by plotting the correction factor in bins of each angle.
- **Swath-side asymmetry test.** Aqua passes at a fixed local time, so GERB's morning/afternoon test becomes a comparison of forward-scatter and backscatter sides of the swath (RAA < 90° vs > 90°) at matched VZA. A systematic difference means geometry is missing from the model.
- **Geolocation-jitter sensitivity.** Shift the aggregation weights by ±1 pixel and re-score. MODIS–CERES co-registration errors are largest at cloud edges (Bertrand), which is exactly where the CTZ is.
- When enough independent units exist (Arena B), keep separate early-stop, tuning and test sets, as in Law 2018 (train / early-stop / validation / test). Balance the label distribution across splits.
- Significance on per-unit loss differences:
  - paired bootstrap;
  - Wilcoxon;
  - Diebold–Mariano;
  - Holm correction across pairs.

### 4.3 Evidence beyond the footprint level

- **Synthetic truth:**
  1. Build a known 1 km field (e.g. frozen M2 or NB→BB+ADM applied per pixel).
  2. Aggregate it with H and add CERES-level noise.
  3. Retrain each method and measure pixel RMSE, MAE, correlation and SSIM (Chau 2021), and the flux-vs-distance-to-cloud curve. Compare against B-const.
  4. Repeat for several numbers of footprints and several aggregation sizes, as Law (number and size of bags) and Gao (several coarse resolutions) did. This shows how recovery degrades as the resolution gap grows.
  5. Include a **no-CTZ** scene, where that curve must come out flat. This checks that regularisation does not invent a CTZ.
- **Physics checks**, monitored during training and used for epoch selection:
  - far-field pure clear sky is about 80 W m⁻² SW;
  - flux decays monotonically with distance from cloud;
  - no value exceeds S₀·μ₀;
  - map extremes stay bounded;
  - the power spectrum shows no peak at the footprint spacing.
- **Paper reproduction** (focus phase): the CTZ contribution from the map should be about −9 W m⁻² SW and +1.4 W m⁻² LW.

## 5. Data-selection (sampling) strategies

Sampling changes **only the training set**. The test population is fixed, and broken down into sub-populations.

| ID | Strategy |
|---|---|
| S0 | all valid footprints (reference) |
| S1 | geometry limits (SZA < 70–75°, VZA < 50–55°) |
| S2 | glint exclusion (high `glintf`, or glint angle < 25°, Bertrand's threshold; up to 40°) |
| S3 | homogeneity filter: legacy (`meanCTH<3000`, `stdCTH<1000`) or radiometric (Gao `cv < 0.2` from the moments) |
| S4 | `Fsw ≥ 400` (advisor's newest script). This conflicts with the thesis goal because it removes clear and broken scenes. |
| S5 | stratified by cloud-fraction regime |
| S6 | overlap thinning (every 2nd–3rd footprint). Mostly relevant for evaluation, since overlap *helps* aggregation-loss training. |
| S7 | weighting instead of filtering, with loss weight `1/σ²_ADM(regime)` and a glint weight |

Run the sampling strategies crossed with the 2–3 leading methods only.

**How close the strategies are:**
- Jaccard overlap of their footprint sets;
- KS/MMD distance to S0 on Fsw, CF, SZA, VZA and glint;
- a train-on-Si × test-on-subpopulation matrix;
- Kendall τ between the method rankings under each Si.

If τ > 0.8, sampling does not change the winner: fix S7 (or S5) and move on. Otherwise the sampling strategy is chosen together with the method.

## 6. Decision rule: explore, then focus

1. **Gates.** A method is dropped if:
   - its residuals are not flat against geometry or glint;
   - its pure clear sky falls outside 80 ± 10 W m⁻²;
   - its maps show runaway extremes;
   - its ranking flips between the uniform and PSF operators.
2. **Primary score.** Macro-averaged RMSE over the 3 regimes in Arena A, so broken-cloud scenes are not hidden by clear ones.
3. **When a more complex method replaces a simpler one.** Only if both hold:
   - the paired bootstrap CI excludes 0 after Holm correction;
   - the gain exceeds the CERES noise floor for that regime.
4. **Tie-breakers, in order:**
   1. synthetic-truth pixel RMSE;
   2. a small gap between the interleave and LOGO scores;
   3. simplicity and cost.
5. **Default on a tie:** M3, or M2 if image data stays scarce.
6. **The UNet must beat the hybrid M6b, not just M3.** If it does not, spatial context adds little, and that is itself a result about the scale of the CTZ.

| Stage | Duration (rough) | Content |
|---|---|---|
| E0 | 2–3 weeks | data verification, split manifest, baseline re-runs (§2) |
| E1 | ~2 weeks | null baselines, M0, M1, M2 and M9 in Arenas A and B; input ablation |
| E2 | 3–4 weeks | M3, M4, M5, M6b, M6c and M6f with the same budget; synthetic truth; sampling study |
| E3 | ~1 week | decision, using the rule above |
| F | rest | focus on the winner: more granules, M7 if the PSF arrives, full-granule maps, CTZ analysis, LW head |

## 7. Open questions and risks

**Open questions (for the advisor):**
1. Can more `images/` granules be produced (other seasons, Atlantic, 2015)? Today there are 6 granules from 2 days in March 2014, about 12k footprints, against 2,163 moments/scalars granules (about 5.26M footprints).
2. Is there PSF code or parameters? Are the moments PSF-weighted or uniform over the patch?
3. What are the `Imviews` channels, and what does `FOVorientation` mean? The answer decides whether flip and rotation augmentation is allowed.
4. How does a fixed 27×55 patch represent footprints that grow with VZA?
5. What do `glintf`, `landf`, `Clrf`, `LiqPhsf` and `mDFNC` mean? Why do CF and Clrf correlate so weakly (r ≈ 0.25)?
6. Is the `Fsw ≥ 400` cut in the newest script meant for this thesis?

**Risks:**
- **Too little image data.** This is the main risk. Mitigations: M2 and M1 use all 5.26M footprints, and more granules are requested early.
- **Unverified CERES numbers:** the Su 2015 noise floor, the CTZ magnitude (−9 vs −2.5 W m⁻² in different sources) and the footprint size.
- **Unpinned `requirements.txt`.**
- **The new phase needs PyTorch or TF for UNet work.** Pin versions when the first new-phase code lands.

## 8. Repository layout

```
Thesis_Goal.md        goal and success criteria (unchanged)
PLAN.md               this plan
Documentation/        shared docs: LITERATURE_LESSONS.md
reports/              deep-research reports (Hebrew), the basis for this plan
research_notes/       source notes behind the reports
flux1km/              NEW work: 1 km pipeline, flat modules (see §9 and flux1km/README.md)
tests/                tests for flux1km (tests/test_<module>.py)
scripts/, .github/     CI: lint + tests of changed modules (full suite on main)
legacy/               OLD footprint-level work, frozen for reference and baselines
  python/             All_Sky_AIflux, guys_benchmark_models, guy_model_1.0/2.0, ...
  Documentation/      benchmark results, comparisons, plan_CV, ideas
  projects/, outputs/ old run artefacts
data/                 local data (git-ignored): paired/, test/, consolidated/
```

## 9. Infrastructure (`flux1km/`, built before the data arrives)

Nothing is run on real data until it arrives. Meanwhile the shared machinery is built and tested on synthetic data, so that on day one every method plugs into the same protocol.

### Architecture

```
 io ──► GranuleData(images, moments, scalars, granule_id)
  │
  ├─► schema      channel tables, forbidden CERES keys, input guard
  ├─► geometry    glint angle, angle encoding, S0·cosSZA normalisation
  ├─► moments     raw moments from mean/std/skew/kurt (M2), Gao cv (M9, S3)
  ├─► sampling    S0–S7 masks/weights + similarity (Jaccard, KS, Kendall τ)
  ├─► splits      granule groups, LOGO / day / time / region folds, spatial blocks with buffer,
  │               interleave, pixel-reuse leakage check, JSON manifest
  │
  ▼
 models (interface: fit(train) / predict_map(patches) / predict_footprint)
  │      baselines in numpy now; M1–M10 added as the data arrives (TF/PyTorch in Docker)
  ▼
 aggregation   uniform patch mean, approximate Gaussian PSF, ±1 px jitter  ──►  footprint predictions
  ▼
 metrics       RMSE/MAE/bias/R², regime breakdown, macro RMSE, binned residuals, swath-side asymmetry,
 │             paired bootstrap, Wilcoxon, Diebold–Mariano, Holm
 physics       distance to cloud, flux-vs-distance curve, S0·μ0 bound, percentile spread, spectrum peak
 synthetic     synthetic granules with known 1 km truth (with or without CTZ), used for demo mode and tests
  ▼
 benchmark     runs every method over every fold of a manifest → benchmark_outputs/flux1km/run_*/
```

### Build order

1. `schema`, `geometry`, `moments` (no dependencies)
2. `synthetic` (demo data in the real file layout)
3. `io` (reads real or synthetic NPZ triplets)
4. `aggregation`
5. `splits`, `sampling`
6. `metrics`, `physics`
7. `models` (interface + null baselines + a pixel-linear aggregate model)
8. `benchmark` + `scripts/run_benchmark.py --demo`

The core is NumPy/SciPy only, so the tests run in CI without a GPU. Neural models (M3, M5, M6) live in their own modules, which import TF/PyTorch lazily.

### Decisions taken without the data (revisit in E0)

- The band order follows the advisor's script:
  - SW: 0.47, 0.55, 0.66, 0.86, 0.905, 0.935, 0.940, 1.24, 1.64, 2.13, 1.375
  - LW: 3.75 … 14.23
- `Imviews` channels are assumed to be SZA, VZA, RAA in degrees.
- Moments kurtosis is assumed to be Pearson (not excess); this is a flag in `moments`.
- Granules from the same day and satellite, less than 10 minutes apart, are one group.
