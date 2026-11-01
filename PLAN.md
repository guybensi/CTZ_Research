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

**Data scope.** Data will be plentiful: as much as needed, from any region and time (advisor, Nov 2026). Unlabelled MODIS granules can also be pulled directly from NASA, which makes the mediated pseudo-label hybrid (M6f) feasible. The first phase still starts small, with **Pacific Ocean data only**: 2014–2015, the March, June, September and December files. Ocean-only means a single dark surface, so surface type and land/snow ADM issues are out of the problem at first. Global data (Atlantic, North15, land) comes only in the focus phase F, and only if the Pacific results are good. At that point the regional holdout becomes the generalisation test, and surface type has to enter the model (`landf`, the per-surface regressions of M8).

## 2. Stage E0: data and infrastructure (prerequisite)

Nothing is benchmarked until these items are done:

- [ ] Unpack `Data_toGuy.zip` (on Drive, shared folder `Guy_Fluxes1KM/Old_codes_data`) into `data/` and inventory it:
  - granules and footprints
  - keys and shapes for `moments/`, `scalars/` and `images/`
- [x] Format checked on a sample granule (§2.1). Remaining checks on the full data:
  - the meaning of `FOVorientation`
  - the units of `ImSW` and `ImLW`
  - the meaning and units of `glintf`, `landf`, `Clrf`, `LiqPhsf`, `mDFNC`, `mSumIsw`
- [x] Moments vs image patch statistics, on the sample: means agree to about 2–3%, but the std is about 0.8 of the patch std, so the moments are **not** uniform patch statistics.
- [ ] Fit the aggregation weights H that reproduce the stored moments from the patches. H becomes the primary operator and also gates M2.
- [x] Per-pixel glint angle from `Imviews` (`features.pixel_geometry` + `geometry.glint_angle`). Which RAA convention is specular still needs a glint scene.
- [x] Footprint overlap on the sample: centres about 12.6 km apart, up to 72% shared pixels. Still to repeat on a full overpass.
- [ ] Build one frozen **split manifest** (§4.1) and an automatic pixel-reuse leakage check.
- [ ] Re-run the legacy baselines inside the new protocol. The old numbers (17.72 and 15.33 W m⁻²) used random row splits, and early stopping on the test set, so they are optimistic.
- [ ] Ask the advisor for the PSF / the code that built the triplets (see §7); the raw products are listed in §2.2.
- [ ] Settle the RAA/glint convention with a glint granule and the MYD35 sun-glint bit (§2.2).

### 2.1 What the data looks like

This section is based on one sample granule, `A2014060.0245`, stored in `data/sample/`. It has 9 footprints and is a **format sample only**: it sits at about 50°N, 138°E, so it is not necessarily the training region. Every granule is three NPZ files with the same name, and footprints are in the same order in all three. The facts below are pinned by `tests/test_real_sample.py`.

**`images/<granule>.npz`**: one 27×55 patch per footprint, at about 1 km per pixel.

| Key | Shape | Content (sample range) |
|---|---|---|
| `Imviews` | (N, 3, 27, 55) | per-pixel SZA, VZA, RAA in degrees. **RAA here is 180° − the scalars RAA** (≈110° vs ≈70°). |
| `ImSW` | (N, 11, 27, 55) | SW radiances (0.09–279) |
| `ImLW` | (N, 16, 27, 55) | LW radiances (−0.06–7.5; small negative values exist) |
| `FOVorientation` | (N,) | 0/1 flag, mixed within a granule (meaning unknown) |

**`moments/<granule>.npz`**: footprint statistics of the MODIS pixels.

| Key | Shape | Content |
|---|---|---|
| `mSWspec`, `stdSWspec`, `skewSWspec`, `kurtSWspec` | (N, 11) | mean / std / skewness / **excess** kurtosis (values < 1 occur) |
| `mLWspec`, `stdLWspec`, `skewLWspec`, `kurtLWspec` | (N, 16) | same for LW |
| `SWwavlngs` | (11,) | 0.47, 0.55, 0.66, **0.85**, 0.905, 0.935, 0.94, 1.24, 1.64, 2.13, 1.375 µm (the last channel is out of wavelength order) |
| `LWwavlngs` | (16,) | 3.75, 3.96, 3.96, 4.05, 4.46, 4.5, 6.7, **7.32**, **8.55**, 9.7, 11, 12, 13.33, 13.63, 13.93, 14.23 µm (3.96 appears twice) |

**`scalars/<granule>.npz`**: one value per footprint (23 keys).

| Group | Keys | Notes from the sample |
|---|---|---|
| CERES label / leakage (never inputs) | `Fsw`, `Flw`, `Isw`, `Ilw` | `Fsw/Isw` = π/R ≈ 3.0–3.5 |
| Unclear origin (treated as CERES-derived until verified) | `mSumIsw`, `stdSumIsw`, `mSumIlw`, `stdSumIlw` | same order as the per-pixel sum of `ImSW` channels, so possibly a MODIS-derived broadband reconstruction (Eq. 11), which would make it a legal input |
| Geometry | `SZA`, `VZA`, `RAA` | match the patch means of `Imviews` (RAA as 180° − image value) |
| Location | `Lat`, `Lon` | footprint centres |
| MODIS products (stratification only) | `CF` (**fraction 0–1**), `Clrf` (0–10 here), `meanCTH`, `stdCTH` (m), `LiqPhsf` (≈370–460, not a percentage), `AOD` (all NaN in this cloudy granule), `mDFNC`, `stdDFNC` (≈1.0–1.2) | units and meaning of `Clrf`, `LiqPhsf` and `mDFNC` to be confirmed |
| Flags | `glintf` (0 here), `landf` (1.9–2.6 here: not a 0/1 flag, probably percent land) | |

**What the sample confirms or changes in this plan**

- **Inputs exist as planned**: radiances in 11 SW + 16 LW channels, per-pixel geometry, and per-pixel glint computed from `Imviews` (§3, F1–F5). Wavelengths are channel identities only.
- **Footprint overlap is real and large.** Neighbouring footprint centres are about 12.6 km apart, and two patches share up to 72% of their pixels (1,063 of 1,485 identical 0.66 µm values). Random footprint splits are therefore leaky, and the 20 km leakage radius in `splits` is a sensible start (§4.1).
- **Moments ≠ uniform patch statistics.**
  - Footprint means match the patch means to about 2–3%, but the moments' std is only about 0.8 of the patch std.
  - The moments were probably computed on the CERES field of view (PSF-weighted, or a central ellipse) rather than on the whole 27×55 rectangle. The patch is larger than the footprint.
  - Consequences:
    1. The primary aggregation operator H must be the one that reproduces the stored moments. Fitting it is an E0 task, and the uniform patch mean is only a fallback.
    2. The M2 bridge (train on moments, apply to pixels) is exact only under that same operator. Until H is known, M2 can also be trained on moments recomputed from the images with H.
- **Storage conventions now encoded in `flux1km.schema`**:
  - `MOMENTS_EXCESS_KURTOSIS`;
  - `CF_TO_PERCENT` (CF is converted to percent in the footprint table, so the regime thresholds 5/95 hold);
  - `IMVIEWS_RAA_IS_SUPPLEMENT` (pixel RAA is converted to the scalars convention, so pixel and footprint geometry agree).
- **Which RAA convention puts the specular point at 0°** still needs a glint scene to settle; `glintf` is 0 in this sample.
- **Ocean-only phase:** `landf` is non-zero even in an ocean granule, so phase 1 needs an explicit `landf` threshold once its unit is confirmed.
- **Not inputs:** `AOD` is NaN under cloud and `CF` / `CTH` are retrieval products. They are used only for regimes and filters, as already planned.

### 2.2 Data sources (raw products)

The granule triplets of §2.1 are built from public NASA products. Where to get them (advisor's email):

- **MODIS (Aqua)** from LAADS DAAC: <https://ladsweb.modaps.eosdis.nasa.gov/search/>
- **CERES** from the CERES data portal: <https://ceres.larc.nasa.gov/data/>

| Product | What it holds | Role in this thesis |
|---|---|---|
| **MYD021KM** (MODIS/Aqua L1B 1 km calibrated radiances, 5-min swath) | radiances/reflectances of 36 bands | the image inputs: `ImSW`/`ImLW` and the moments are 27 of these bands |
| **MYD03** (geolocation) | per-pixel lat/lon, solar and sensor zenith and azimuth, land/sea mask, height | the per-pixel geometry (`Imviews`) and, presumably, `landf` |
| **MYD35_L2** (cloud mask) | per-pixel cloud-mask confidence, day/night, **sun-glint flag**, snow/ice and land/water bits | cloud regimes, distance to cloud, `Clrf`/`glintf` (presumably) |
| **MYD06_L2** (cloud properties) | cloud-top height/pressure/temperature, phase, optical thickness, effective radius | stratification only (`meanCTH`, `stdCTH`, `LiqPhsf`, presumably); never model inputs |
| **CERES SSF Level 2, Aqua** (Single Scanner Footprint, Edition 4A, FM3/FM4) | per footprint: time, location, viewing geometry, unfiltered radiances, TOA SW/LW flux from the ADMs, scene information | the label (`Fsw`, `Flw`) and the CERES-side scalars (`Isw`, `Ilw`, `SZA`, `VZA`, `RAA`). Only a subset of the SSF variables is needed; to be agreed with the advisor. |

**Channel ↔ MODIS band mapping** (`schema.SW_MODIS_BANDS`, `schema.LW_MODIS_BANDS`):

| Group | Channels and bands |
|---|---|
| SW | 0.47 = B3, 0.55 = B4, 0.66 = B1, 0.85 = B2, 0.905 = B17, 0.935 = B19, 0.94 = B18, 1.24 = B5, 1.64 = B6, 2.13 = B7, 1.375 = B26 |
| LW | 3.75 = B20, 3.96 = B21 (low-gain fire channel) and B22, 4.05 = B23, 4.46 = B24, 4.5 = B25, 6.7 = B27, 7.32 = B28, 8.55 = B29, 9.7 = B30, 11 = B31, 12 = B32, 13.33–14.23 = B33–B36 |

The ocean-colour bands 8–16 are not used. Aqua band 6 (1.64 µm) has known dead detectors, which explains its stripes.

**What this explains, and what it changes in the plan**

- **The two RAA conventions** most likely come from two sources. The scalars `RAA` is the CERES SSF relative azimuth; `Imviews` is computed from the MYD03 solar and sensor azimuths, where the specular direction sits at the opposite end. **E0 can settle the glint convention without waiting for the advisor**: in a glint granule, the convention whose glint angle is small exactly where the MYD35 sun-glint bit is set is the right one.
- **The CERES-side and MODIS-side variables can now be separated by origin.** The leakage guard (`schema.CERES_DERIVED_KEYS`) excludes everything that comes from SSF. Products from MYD35/MYD06 are retrieval products, so they serve stratification only, as already planned.
- **We are not limited to the advisor's pre-built triplets.** With MYD021KM + MYD03 + SSF footprint centres and the CERES PSF, footprints can be built in-house, with a **known** aggregation operator H, which removes the moments-vs-patch ambiguity of §2.1. This is a candidate E0/F task if the advisor's construction code is not available.
- **Unlabelled MODIS granules** (MYD021KM + MYD03 without SSF) are freely available, which makes the mediated pseudo-label hybrid M6f practical.

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

Channels that are **systematically** bad are excluded (agreed with the advisor). In the sample image, 1.64 µm (Aqua band 6) and 14.23 µm show only stripes; E0 confirms on the full data which channels are systematically bad.

**Model inputs, as agreed with the advisor:** MODIS channel radiances (identified by wavelength) and the geometric angles. These are the same input types as `All_Sky_AIflux.py`. The glint angle is a function of SZA, VZA and RAA, so it is a derived geometric feature, not a new data source. MODIS retrieval products (CF, CTH, AOD) and Lat/Lon are **not** inputs. They are used only for stratification and analysis.

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
  - seasonal holdout: leave one month out (March / June / September / December), which tests generalisation across solar geometry and cloud regimes inside the Pacific;
  - regional holdout (train Pacific, test Atlantic / North15) is deferred to phase F, when global data is added.
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
- **Paper reproduction** (focus phase): the CTZ contribution from the map should be about −9 W m⁻² SW and +1.4 W m⁻² LW. The reference numbers are those of Eshkol Eytan's CTZ paper (Table B1), as agreed with the advisor; other published values (e.g. −2.5 W m⁻²) are not used as targets.

## 5. Data-selection (sampling) strategies

Sampling changes **only the training set**. The test population is fixed, and broken down into sub-populations.

| ID | Strategy |
|---|---|
| S0 | all valid footprints (reference) |
| S1 | geometry limits (SZA < 70–75°, VZA < 50–55°) |
| S2 | glint exclusion (high `glintf`, or glint angle < 25°, Bertrand's threshold; up to 40°) |
| S3 | homogeneity filter: legacy (`meanCTH<3000`, `stdCTH<1000`) or radiometric (Gao `cv < 0.2` from the moments) |
| S5 | stratified by cloud-fraction regime |
| S6 | overlap thinning (every 2nd–3rd footprint). Mostly relevant for evaluation, since overlap *helps* aggregation-loss training. |
| S7 | weighting instead of filtering, with loss weight `1/σ²_ADM(regime)` and a glint weight |

S4 (`Fsw ≥ 400`, from the advisor's newest script) was **dropped**: it removes the clear and broken scenes where the CTZ lives, so it works against the thesis goal. The IDs of the other strategies are kept unchanged.

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
| F | rest | focus on the winner: more Pacific granules, M7 if the PSF arrives, full-granule maps, CTZ analysis, LW head; if the Pacific results are good, global data and the regional holdout |

## 7. Open questions and risks

**Open questions (for the advisor):**
1. ~~Can more `images/` granules be produced?~~ **Answered:** as much data as needed, any region and time; we start small (Pacific, 2014–2015, edge months). Unlabelled MODIS granules can be pulled from NASA.
2. Is there PSF code or parameters? How exactly were the moments computed from the pixels? The sample shows they are not uniform patch statistics (the std is about 0.8 of the patch std).
3. What does `FOVorientation` (0/1) mean? The answer decides whether flip and rotation augmentation is allowed. (`Imviews` is now known: SZA, VZA, 180° − RAA.)
4. How does a fixed 27×55 patch represent footprints that grow with VZA?
5. What are the units and meaning of `glintf`, `landf` (≈2 in an ocean granule), `Clrf` (0–10), `LiqPhsf` (≈400) and `mDFNC` (≈1.0–1.2)? Why do CF and Clrf correlate so weakly (r ≈ 0.25)?
6. Is `mSumIsw` a MODIS-derived broadband reconstruction (then it is a legal input) or CERES-derived (then it stays excluded)?

**Risks:**
- **Too little image data.** Largely resolved: the advisor will supply as much data as needed. The remaining risk is timing, so the infrastructure stays testable on synthetic data and on `data/sample/`.
- **Unverified CERES numbers:** the Su 2015 noise floor and the footprint size. The CTZ magnitude is fixed to the values of Eytan's CTZ paper.
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
  ├─► splits      granule groups, LOGO / day / month / year folds, spatial blocks with buffer,
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

### Decisions taken without the data, and their status after the sample

| Decision | Status |
|---|---|
| Band order and wavelengths | **confirmed** from `SWwavlngs`/`LWwavlngs` (labels corrected to 0.85, 7.32, 8.55) |
| `Imviews` = SZA, VZA, RAA in degrees | **confirmed**, with RAA stored as 180° − scalars RAA |
| Moments kurtosis type | **excess** (was assumed Pearson; fixed) |
| CF scale | **fraction 0–1** (converted to percent in the table) |
| Primary aggregation operator = uniform patch mean | **doubtful**: the moments' std is about 0.8 of the patch std; fit H to the stored moments in E0 |
| RAA convention for glint (specular at 0° vs 180°) | open: needs a glint scene |
| Footprint radius for the leakage check (20 km) | plausible: centres are about 12.6 km apart and neighbouring patches share up to 72% of pixels |
| Granules of one satellite less than 10 minutes apart are one group | unchanged |
