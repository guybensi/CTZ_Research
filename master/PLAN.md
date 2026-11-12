# PLAN: 1 km All-Sky TOA Flux Map (new phase)

This plan covers the new phase of the thesis: turning [Thesis_Goal.md](../Thesis_Goal.md) into a 1 km flux map. It is deliberately generic. First every candidate direction is benchmarked under one frozen protocol, then work focuses on the method closest to CERES.

The detailed reasoning, citations and numbers are in the research report [reports/תוכנית מחקר מפת שטף 1 קמ.md](research/reports/%D7%AA%D7%95%D7%9B%D7%A0%D7%99%D7%AA%20%D7%9E%D7%97%D7%A7%D7%A8%20%D7%9E%D7%A4%D7%AA%20%D7%A9%D7%98%D7%A3%201%20%D7%A7%D7%9E.md) (Hebrew). Its source notes are in [research_notes/](research/research_notes/). The literature lessons are in [Documentation/LITERATURE_LESSONS.md](../Documentation/LITERATURE_LESSONS.md). The previous footprint-level work is frozen under [pre_master/](../pre_master/).

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

- [ ] Unpack `Data_toGuy.zip` (on Drive, shared folder `Guy_Fluxes1KM/Old_codes_data`) into `data/` and inventory it. *Tool ready:* `python3 -m flux1km.e0 --zip Data_toGuy.zip --data-root data/paired` (`e0.unpack_zip`, `e0.inventory`); waiting for the data:
  - granules and footprints
  - keys and shapes for `moments/`, `scalars/` and `images/`
- [x] Format checked on a sample granule (§2.1). Remaining checks on the full data (*tools ready:* `e0.value_ranges` flags unit/range anomalies, `e0.check_fov_orientation` compares the stored flag with its inverse):
  - the meaning of `FOVorientation`
  - the units of `ImSW` and `ImLW`
  - the meaning and units of `glintf`, `landf`, `Clrf`, `LiqPhsf`, `mDFNC`, `mSumIsw`
- [x] Moments vs image patch statistics, on the sample: means agree to about 2–3%, but the std is about 0.8 of the patch std, so the moments are **not** uniform patch statistics.
- [ ] Fit the aggregation weights H that reproduce the stored moments from the patches. H becomes the primary operator and also gates M2. *Tool ready:* `e0.fit_aggregation_operator` fits the ellipse and, with ≥ 60 footprints, recovers the full PSF; it saves `fitted_operator.npz`, which `--psf` accepts.
- [x] Per-pixel glint angle from `Imviews` (`features.pixel_geometry` + `geometry.glint_angle`). The RAA convention is settled by the advisor's code (§2.3).
- [x] Footprint overlap on the sample: centres about 12.6 km apart, up to 72% shared pixels. Still to repeat on a full overpass (*tool ready:* `e0.footprint_overlap`).
- [x] Build one frozen **split manifest** (§4.1) and an automatic pixel-reuse leakage check: `e0.build_frozen_manifest` (Arena A LOGO + interleave, Arena B day k-fold / year / leave-one-month-out with the Arena A days excluded from training, spatial blocks), each fold checked with `splits.pixel_reuse_fraction`. To freeze on the full data.
- [ ] Re-run the legacy baselines inside the new protocol. The old numbers (17.72 and 15.33 W m⁻²) used random row splits, and early stopping on the test set, so they are optimistic. *Code ready:* `M0-allsky37`, `M0-advisor`, `M0-ridge` (`methods/m0_legacy.py`); run with `python3 -m flux1km.e0 --data-root DIR --run-baselines` or `python3 -m flux1km.methods.m0_legacy --data-root DIR --manifest M.json`.
- [x] The advisor's data-building code received (§2.3, `pre_master/advisor_code/`).
- [x] `ceres_psf.csv` received from the advisor (`master/resources/ceres_psf.csv`); switch the primary operator to `ceres_psf` (`--psf master/resources/ceres_psf.csv`). Fallback: `aggregation.recover_psf_from_moments` on ≥ 60 footprints.
- [ ] Report the `LiqPhsf` bug (and the minor ones) to the advisor (§2.3). *Text ready:* `e0.advisor_bug_report()`, written by every E0 run to `advisor_bug_report.txt`.

### 2.1 What the data looks like

This section is based on one sample granule, `A2014060.0245`, stored in `data/sample/`. It has 9 footprints and is a **format sample only**: it sits at about 50°N, 138°E, so it is not necessarily the training region. Every granule is three NPZ files with the same name, and footprints are in the same order in all three. The facts below are pinned by `tests/test_real_sample.py`.

**`images/<granule>.npz`**: one 27×55 patch per footprint, at about 1 km per pixel.

| Key | Shape | Content (sample range) |
|---|---|---|
| `Imviews` | (N, 3, 27, 55) | per-pixel SZA, VZA, RAA in degrees. **RAA here is 180° − the scalars RAA** (≈110° vs ≈70°). |
| `ImSW` | (N, 11, 27, 55) | SW radiances (0.09–279) |
| `ImLW` | (N, 16, 27, 55) | LW radiances (−0.06–7.5; small negative values exist) |
| `FOVorientation` | (N,) | 1 = the next CERES footprint lies at a larger MODIS column (PSF used as stored); 0 = at a smaller column (PSF mirrored left-right). It encodes the scan direction. |

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
| MODIS broadband sanity check | `mSumIsw`, `stdSumIsw`, `mSumIlw`, `stdSumIlw` | sum of the MODIS narrowband radiances, meant to mimic the integral CERES measures; a sanity check (advisor). MODIS-derived, so not leakage, but it adds nothing beyond the channels the models already see |
| Geometry | `SZA`, `VZA`, `RAA` | match the patch means of `Imviews` (RAA as 180° − image value) |
| Location | `Lat`, `Lon` | footprint centres |
| CERES product (stratification only, never an input) | `Clrf` (0–10 here) | SSF `Cloud_mask_clear_strong_percent_coverage`: percent of the footprint that CERES's own cloud mask calls strongly clear |
| MODIS-derived products (stratification only) | `CF` (**fraction 0–1**), `meanCTH`, `stdCTH` (m), `LiqPhsf`, `AOD` (all NaN in this cloudy granule), `mDFNC`, `stdDFNC` | exact definitions from the code in §2.3. In short: `CF` is the unweighted cloudy fraction of the patch (MYD35, "relaxed" mask); `meanCTH`/`stdCTH` and `mDFNC`/`stdDFNC` are PSF-weighted over cloudy and clear pixels respectively; `mDFNC` is in pixels. **`LiqPhsf` is unusable in the current files** (a bug, §2.3). `AOD` is the SSF PSF-weighted MOD04 ocean AOD. |
| Flags | `glintf` (0 here), `landf` (1.9–2.6 here) | `glintf`: unweighted fraction of patch pixels with the MYD35 sun-glint flag. `landf`: patch mean of the MYD35 2-bit land/water code, **not a percentage**; `landf == 0` means all pixels are water (§2.3) |

**What the sample confirms or changes in this plan**

- **Inputs exist as planned**: radiances in 11 SW + 16 LW channels, per-pixel geometry, and per-pixel glint computed from `Imviews` (§3, F1–F5). Wavelengths are channel identities only.
- **Footprint overlap is real and large.** Neighbouring footprint centres are about 12.6 km apart, and two patches share up to 72% of their pixels (1,063 of 1,485 identical 0.66 µm values). Random footprint splits are therefore leaky, and the 20 km leakage radius in `splits` is a sensible start (§4.1).
- **The moments are PSF-weighted**, not uniform patch statistics. They are `Σ w·x` with the CERES PSF `w` resampled to 27×55 and mirrored for `FOVorientation == 0` (§2.3).
  - On the sample, uniform patch statistics miss the stored moments by 23% RMS; a fitted tilted elliptical Gaussian (`fov_ellipse`) gets to 8.6%. The rest is the real PSF's along-scan skew.
  - The exact operator is `ceres_psf`, available: the advisor's `ceres_psf.csv` is in `master/resources/`. `fov_ellipse`, `uniform` and `gaussian_psf` are sensitivity checks.
  - The M2 bridge (train on moments, apply to pixels) is exact only under this same operator. The synthetic generator therefore builds its labels and moments with it too.
- **Storage conventions now encoded in `flux1km.schema`**:
  - `MOMENTS_EXCESS_KURTOSIS`;
  - `CF_TO_PERCENT` (CF is converted to percent in the footprint table, so the regime thresholds 5/95 hold);
  - `IMVIEWS_RAA_IS_SUPPLEMENT` (pixel RAA is converted to the scalars convention, so pixel and footprint geometry agree).
- **RAA convention settled** (§2.3): `Imviews` RAA is the folded difference of the MYD03 solar and sensor azimuths, so specular = 180°; the SSF RAA has specular = 0°. `features.pixel_geometry` converts to the SSF convention, and the glint angle uses specular = 0°.
- **Ocean-only phase:** `landf` is a mean land/water *code*, and it is about 2 here, so this footprint is mostly coastal pixels. Phase 1 keeps only `landf == 0` (all-water) footprints.
- **`FOVorientation` and augmentation:** the flag is the PSF's left-right mirror. A left-right flip of a patch must toggle the flag; a top-bottom flip leaves it unchanged, because the PSF is centred in rows. Flip augmentation is allowed only with that update.
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
| **CERES SSF Level 2, Aqua** (Single Scanner Footprint, Edition 4A, FM3/FM4) | per footprint: time, location, viewing geometry, unfiltered radiances, TOA SW/LW flux from the ADMs, scene information | the label (`Fsw`, `Flw`) and the CERES-side scalars. The advisor's code reads exactly 12 SSF variables (§2.3). |

**Channel ↔ MODIS band mapping** (`schema.SW_MODIS_BANDS`, `schema.LW_MODIS_BANDS`):

| Group | Channels and bands |
|---|---|
| SW | 0.47 = B3, 0.55 = B4, 0.66 = B1, 0.85 = B2, 0.905 = B17, 0.935 = B18 (0.936 µm), 0.94 = B19, 1.24 = B5, 1.64 = B6, 2.13 = B7, 1.375 = B26 |
| LW | 3.75 = B20, 3.96 = B21 (low-gain fire channel) and B22, 4.05 = B23, 4.46 = B24, 4.5 = B25, 6.7 = B27, 7.32 = B28, 8.55 = B29, 9.7 = B30, 11 = B31, 12 = B32, 13.33–14.23 = B33–B36 |

The ocean-colour bands 8–16 are not used. Aqua band 6 (1.64 µm) has known dead detectors, which explains its stripes.

**What this explains, and what it changes in the plan**

- **The two RAA conventions** come from two sources, as the advisor's code confirms (§2.3). The scalars `RAA` is the CERES SSF relative azimuth; `Imviews` is computed from the MYD03 solar and sensor azimuths.
- **The CERES-side and MODIS-side variables can now be separated by origin.** The leakage guard (`schema.CERES_DERIVED_KEYS`) excludes everything that comes from SSF. Products from MYD35/MYD06 are retrieval products, so they serve stratification only, as already planned.
- **We are not limited to the advisor's pre-built triplets.** With MYD021KM + MYD03 + SSF footprint centres and the CERES PSF, footprints can be built in-house, with a **known** aggregation operator H, which removes the moments-vs-patch ambiguity of §2.1. This is a candidate E0/F task if the advisor's construction code is not available.
- **Unlabelled MODIS granules** (MYD021KM + MYD03 without SSF) are freely available, which makes the mediated pseudo-label hybrid M6f practical.

### 2.3 How the triplets are built (advisor's `MOD_CERES_allsky`)

Source: `pre_master/advisor_code/` (driver `Generate_AllSky_2.py`; the function is in `Twilight_module.py`). For every MODIS granule and every CERES footprint it does the following.

1. **Inputs read.**
   - **MYD021KM radiances**, with `radiance_scales`/`offsets` applied, in W m⁻² sr⁻¹ µm⁻¹:
     - SW = bands 3, 4, 1, 2, 17, 18, 19, 5, 6, 7, 26;
     - LW = bands 20–25 and 27–36.
   - **MYD03:** lat/lon, solar and sensor zenith, and azimuths.
   - **MYD35:** cloud mask, sun-glint and land/water bits.
   - **MYD06:** `cloud_top_height_1km`, `Cloud_Phase_Infrared_1km`.
   - **SSF:** 12 variables:
     - `CERES_SW/LW_TOA_flux___upwards` → `Fsw`, `Flw`;
     - `CERES_SW/LW_radiance___upwards` → `Isw`, `Ilw`;
     - `CERES_solar_zenith/viewing_zenith/relative_azimuth_at_surface`;
     - `Colatitude/Longitude_of_CERES_FOV_at_surface`;
     - `time`;
     - `Cloud_mask_clear_strong_percent_coverage` → `Clrf`;
     - `PSF_wtd_MOD04_effective_optical_depth_average_ocean__0_550_` → `AOD`.
2. **The PSF.**
   - The source is `ceres_psf.csv`, the NOAA-20 CERES PSF on a 250 m grid. It is in the repo: `master/resources/ceres_psf.csv`. Its column peak is at column 18 of 55 (not the patch centre): the advisor aligns that peak with the footprint pixel.
   - It is Fourier-resampled to 27×55 at about 1 km, which rescales it to Aqua's altitude, and then normalised.
   - The 27×55 patch is exactly the PSF's support. That is why the patch is a fixed rectangle, and the footprint ellipse sits inside it.
   - This answers open question 4 (§7) for the stored data: the PSF is not rescaled with VZA, so off-nadir footprints are represented by the same 27×55 weights.
3. **Matching.**
   - Footprints are matched to the granule by date + hour and the granule's lat/lon box.
   - The footprint centre is the MODIS pixel nearest to the SSF lat/lon. The patch has rows centre ± 13, and its columns align the PSF's peak column with the centre.
   - `FOVorientation` is 1 if the next SSF footprint is at a larger column (PSF as stored), 0 otherwise (PSF mirrored with `np.fliplr`).
   - Footprints are skipped:
     - at night (`Fsw` < 1);
     - when the patch leaves the granule;
     - when invalid pixels carry more than 0.1% of the PSF weight.
4. **What is stored.**
   - **Moments:** for each channel, the PSF-weighted mean, std, skew and **excess** kurtosis (`calc4moms` logic, population std).
   - **`mSumIsw`/`mSumIlw`:** the PSF-weighted mean and std of the sum of the SW (LW) channel radiances.
   - **`meanCTH`/`stdCTH`:** PSF-weighted over pixels with a cloud top.
   - **`mDFNC`/`stdDFNC`:** PSF-weighted over clear pixels, of the distance in pixels to the nearest cloudy pixel. It is NaN beyond the granule-edge distance.
   - **`CF`, `glintf`, `landf`:** unweighted patch means. The cloud mask is MYD35 "relaxed" (cloudy = the most confident cloudy class).
   - **`Imviews`:** per-pixel SZA, VZA and RAA, where RAA is the folded |SolarAzimuth − SensorAzimuth| ∈ [0°, 180°], specular = 180°. The scalars carry the SSF angles instead, which have specular = 0°.
   - **`images/`:** written only when `GenIm=True`, which is off by default. That is why only 6 image granules exist.

**Problems found in the code** (to report to the advisor; none affects the radiances, moments or labels):

| Problem | Effect | What we do |
|---|---|---|
| `LiqPhsf = nansum(CldPhs == 1) / nansum(cldmskfov)` uses the **whole granule's** phase array (`CldPhs`) instead of the footprint's (`cphsfov`) | the stored values (~400) are a granule-wide count over a footprint count: meaningless | `LiqPhsf` is not used until fixed |
| `land` is parsed from the reversed bit string, so the 2-bit code is read with swapped bits (coastal ↔ desert) | `landf` is a mean *code*, not a fraction | use only `landf == 0` (all water) |
| The same bit-order swap affects the 2-bit cloudiness code (uncertain ↔ probably clear) | none for the stored files: the default "relaxed" mask uses only the unambiguous "cloudy" class; the "strict" option would be wrong | keep "relaxed" if data is rebuilt |
| `binary2 = binary[::-1]` re-reads byte 1 instead of byte 2 | the thin-cirrus and non-cloud-obstruction bits are wrong (not stored, so no effect on our files) | none |
| `cldmskfov = BWmsk[:subsubRi:, ...]` in the bottom-edge branch (stray colon) | wrong `CF` for footprints touching the granule's last row (rare) | edge footprints can be dropped in E0 |
| The RAA fold can exceed 180° when SolarAzimuth − SensorAzimuth < −180° | rare wrong pixel RAA | `features.pixel_geometry` folds it back into [0, 180] |
| `CERESraa[fov+1]` is read for orientation without a bounds check | crash on the last footprint of a file (no wrong data) | none |

**Consequences for the plan**

- **Storage.** Patches overlap by up to 72% (§2.1), so storing 27×55×30 float32 patches for 5M footprints would be about 0.9 TB, with most pixels duplicated 3–5 times. The scalable format keeps the granule-level MODIS arrays once, plus a footprint index (granule, centre row/column, orientation, PSF). Patches are then cut on the fly, as `MOD_CERES_allsky` itself does. This is the recommended format for phase F; `GenIm=True` triplets are fine for the first Pacific subset.
- **In-house data building is now possible**, with the same geometry and weights as the advisor's files, using the code above as the reference implementation.
- **M0 now includes the advisor's current model** (`train_Allsky_moments_narrow_values.py`), re-run under our protocol. As written, it uses a random 70/30 split and filters **both** training and test to `Fsw > 400`, so its reported RMSE is not comparable to ours.
  - Inputs: SZA, VZA, RAA, plus mean/std/skew/kurt of 13 channels (0.47, 0.66, 0.86, 1.375, 2.13, 3.96, 4.05, 4.5, 6.7, 7.3, 9.7, 11, 12 µm).
  - Normalisation: channels cross-normalised by 0.47, 4.05 and 11 µm.
  - Loss: log1p label transform with a weighted MAE.
  - Network: dense layers of N/2, N/4, N/8 and 5 units with L1 regularisation, and early stopping on the change in RMSE.

## 3. Candidate methods (exploration phase)

Every method outputs a per-pixel map. Footprint-level methods output one value per footprint and are scored on the same footprints.

| ID | Direction | Inputs | Training signal | What it tells us |
|---|---|---|---|---|
| **M0** | Legacy baselines: All_Sky_AIflux 37-feature ANN, the advisor's current moments ANN (`train_Allsky_moments_narrow_values.py`, §2.3), and Ridge | footprint moments + geometry | footprint MSE / the advisor's log1p-weighted MAE | the reference bar, re-measured fairly (group splits, no `Fsw > 400` cut on the test set) |
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
2. ~~How were the moments computed?~~ **Answered by the code** (§2.3): CERES-PSF-weighted. **Still needed: the file `ceres_psf.csv`**, to use the exact operator.
3. ~~What does `FOVorientation` mean?~~ **Answered by the code:** the scan direction; 0 = PSF mirrored left-right.
4. ~~How does a fixed 27×55 patch represent footprints that grow with VZA?~~ **Answered by the code:** it does not. The same 27×55 PSF is used at every VZA. Whether that matters off-nadir is a question for phase F (the CERES PSF in km grows with VZA).
5. ~~Meaning of `glintf`, `landf`, `Clrf`, `LiqPhsf`, `mDFNC`?~~ **Answered by the code** (§2.1, §2.3). `LiqPhsf` has a bug and `landf` is a code, not a percentage. `CF` (MODIS relaxed mask, unweighted) and `Clrf` (CERES strong-clear %) come from different masks and definitions, which explains their weak correlation.
6. ~~Origin of `mSumIsw`?~~ **Answered:** a MODIS sum built as a sanity check of the CERES integral.

**Risks:**
- **Too little image data.** Largely resolved: the advisor will supply as much data as needed. The remaining risk is timing, so the infrastructure stays testable on synthetic data and on `data/sample/`.
- **Unverified CERES numbers:** the Su 2015 noise floor and the footprint size. The CTZ magnitude is fixed to the values of Eytan's CTZ paper.
- **Unpinned `requirements.txt`.**
- **The new phase needs PyTorch or TF for UNet work.** Done: the neural methods use PyTorch (`torch>=2.2`, the `ml` extra), imported only inside `fit`; the GPU Docker image installs it.

## 8. Repository layout

```
Thesis_Goal.md          goal and success criteria (unchanged)
Documentation/          shared docs: LITERATURE_LESSONS.md
data/                   local data (git-ignored) + data/sample/ (one real granule, tracked)
master/                 THE M.Sc. WORK: this plan and its implementation
  PLAN.md               this plan
  research/             deep-research report (Hebrew) and source notes behind this plan
  flux1km/              the 1 km pipeline, flat modules (see §9 and flux1km/README.md)
  tests/                tests for flux1km (tests/test_<module>.py)
  scripts/              CI helper: run only the tests of changed modules
  pyproject.toml        package + lint/test config (install from master/)
pre_master/             OLD work from before the M.Sc. plan, frozen for reference and baselines
  python/               All_Sky_AIflux, guys_benchmark_models, guy_model_1.0/2.0, ...
  Documentation/        benchmark results, comparisons, plan_CV, ideas
  advisor_code/         the advisor's data-building and training code (§2.3)
  projects/, outputs/   old run artefacts
.github/workflows/      CI (lint + tests, run from master/)
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
  │      baselines in models.py; M0-M10 and the hybrids in methods/ (PyTorch imported lazily)
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

All paths in this section are relative to `master/`.

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
| Primary aggregation operator | **CERES PSF** (`ceres_psf`, from the advisor's code). It needs `ceres_psf.csv`; until then `fov_ellipse` (fitted to 8.6% RMS) |
| RAA convention for glint (specular at 0° vs 180°) | **settled** by the code: SSF specular at 0°, `Imviews` at 180° |
| Footprint radius for the leakage check (20 km) | plausible: centres are about 12.6 km apart and neighbouring patches share up to 72% of pixels |
| Granules of one satellite less than 10 minutes apart are one group | unchanged |

## 10. Implementation status

Everything in this plan that can be built before the full data arrives is implemented in `flux1km/` and tested on synthetic data and on `data/sample/`. Nothing has been run on the full data yet. What each method is and how to run it: [MODELS.md](MODELS.md).

| Plan item | Where | Status |
|---|---|---|
| §2 E0 checks, manifest, operator fit, advisor report | `flux1km/e0.py` (`python3 -m flux1km.e0`) | tools done; the data-dependent items above wait for `Data_toGuy.zip` |
| §3 null baselines, M0 (ridge, All_Sky 37-feature ANN, advisor's moments ANN) | `models.py`, `methods/m0_legacy.py` | done |
| §3 M1 (GBM + ANN), M2 | `methods/m1_guy.py`, `models.py` | done |
| §3 M3, M4, M5 | `methods/m3_pixel_mlp.py`, `m4_context_gbm.py`, `m5_unet.py` | done (manifold / random-subset / edge-aware regularisers) |
| §3 M6a–M6f hybrids | `methods/m6a_*.py` … `m6f_*.py` | done; M6e reads test labels (scored on overlap test and synthetic truth only) |
| §3 M7 ATPRK | `methods/m7_atprk.py` | done; joint kriging needs footprint positions (`row0`/`col0`), available in synthetic data and in a future in-house footprint index |
| §3 M8 NB→BB→ADM | `methods/m8_nb2bb.py` | done (default SWIR 2.13 µm because 1.64 µm is a bad channel) |
| §3 M9 (+ M9-local), M10 (VBAgg, mediated) | `methods/m9_homogeneous.py`, `methods/m10_gp.py` | done (RFF approximation for M10) |
| §3 input ablation F1–F5 | `--input-set` on every command | done |
| §4.1 splits, leakage check | `splits.py`, `e0.build_frozen_manifest` | done |
| §4.2 metrics, sensitivity operators, significance | `metrics.py`, `benchmark.py` (bootstrap + Holm, Wilcoxon, Diebold–Mariano) | done |
| §4.3 harder synthetic truth | `synthetic.py` (`hard=True`, `--demo-hard`): nonlinear angle-dependent NB→BB, anisotropic radiances, CTZ only partly visible | done |
| §4.3 synthetic-truth experiment, footprint counts × aggregation sizes, no-CTZ control | `flux1km/synthetic_study.py` | done |
| §4.3 physics checks | `physics.py`; on real data the distance to cloud comes from a 0.66 µm brightness proxy | done |
| §5 sampling S0–S7 and their comparison | `sampling.py`, `flux1km/sampling_study.py` | done |
| §6 decision rule | `flux1km/decision.py` | done |
| Results as text | `report.py`: every run writes `run_<time>/results.txt` | done |
