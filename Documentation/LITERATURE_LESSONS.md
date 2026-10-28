# Literature Lessons for the 1 km Flux Model

**Date:** 2026-10-28
**Purpose:** what the papers reviewed in October 2026 teach about how the thesis model (a 1 km all-sky TOA flux map from MODIS, trained against CERES footprints) should be built, trained and validated, and what to watch out for.
**Status:** guidance notes, not an implementation. Complements [plan_CV.txt](plan_CV.txt) and [../Thesis_Goal.md](../Thesis_Goal.md).

## Sources

| Short name | Reference | Main topic |
|---|---|---|
| Bertrand 2006 | Bertrand, Futyan, Ipe, Gonzalez, Clerbaux, *Diurnal Asymmetry in the GERB SW Fluxes*, IEEE TGRS 44(12), 2006 | GERB/SEVIRI flux, NB-to-BB conversion, resolution enhancement |
| Gonzalez | Gonzalez, Hermans, Dewitte, Ipe, Sadowski, Clerbaux, *Resolution Enhancement of GERB Data*, RMIB poster (undated, ~2000) | Constrained optimization for flux downscaling |
| Law 2018 | Law, Sejdinovic, Cameron, Lucas, Flaxman, Battle, Fukumizu, *Variational Learning on Aggregate Outputs with Gaussian Processes*, NeurIPS 2018 | Learning from aggregate labels (VBAgg) |
| Chau 2021 | Chau, Bouabid, Sejdinovic, *Deconditional Downscaling with Gaussian Processes*, NeurIPS 2021 | Downscaling with unmatched data, CMP posterior |
| Fan 2021 | Fan, Chen, Wen, Sun, Gomes, *Resolving Super Fine-Resolution SIF via Coarsely-Supervised U-Net Regression*, NeurIPS 2021 CCAI workshop | U-Net trained only on coarse labels |
| Gao 2012 | Gao, Kustas, Anderson, *A Data Mining Approach for Sharpening Thermal Satellite Imagery over Land*, Remote Sensing 4(11), 2012 | Thermal sharpening with regression trees (DMS) |
| Meyer 2026 | Meyer, Platnick et al., *C7/v2 updates to the MODIS/VIIRS cloud products and impacts of orbital drift*, MODIS Science Team meeting, March 2026 (slides) | MODIS C7 pixel-level fluxes, orbital drift |

Not covered: Song, Liang & Zhou, *All-Sky TOA albedo estimation over ocean based on MODIS*, IGARSS 2022 (full text behind IEEE paywall, not read).

---

## 1. Architecture: a U-Net for per-pixel regression

**Lesson.** A U-Net that predicts a value for every pixel, trained only so that the *aggregate* of its predictions matches the coarse label, works (Fan 2021: 30 m SIF from 3 km labels, a 100× resolution gap).

**What this means for the thesis.**

- It is **per-pixel regression**, not segmentation: a linear (or softplus, if flux must stay positive) output layer producing continuous flux in W m⁻², not class probabilities.
- The aggregation must be the **CERES PSF-weighted mean**, not the plain mean Fan used:

  ```
  loss = MSE( Σ_i w_i · f(pixel_i) ,  F_CERES ),   w_i = PSF weight of pixel i, Σ w_i = 1
  ```

- Implementation details worth copying from Fan (Appendix F):
  - start with a 1×1 conv as a "pixel encoder";
  - in each down block, make the second conv 1×1 (not 3×3) to keep the receptive field small and the output local;
  - no batch normalization (it made training unstable);
  - AdamW, learning rate ~2e-4, weight decay ~1e-4.
- **Compute and report the receptive field.** Fan's down path sees ~16 pixels; at 1 km that is ~16 km. The receptive field sets how far from a cloud the model can "see", so it should be compared with the distance range over which the CTZ acts. Too small: the model cannot learn cloud-proximity effects. Too large: see the artificial-CTZ risk in section 6.

## 2. Overfitting and early stopping

**Lesson.** Coarse supervision has its own overfitting mode (Fan 2021, Appendix C, Figs. 4–5): the model outputs **extreme maps** (too high where the truth is high, too low where it is low) that **still average to the correct label**. The coarse loss keeps falling while the fine-resolution error rises after ~20–30 epochs. Nothing in the training loss signals it.

**What this means for the thesis.**

- Fan stopped training using a small **fine-resolution validation set**. The thesis has no 1 km truth, so early stopping needs a different criterion. Options:
  1. **Held-out CERES footprints** (by granule, see section 3): stop when the PSF-weighted error on unseen granules stops improving.
  2. **Physics checks computed every few epochs** and used for model selection, not only at the end:
     - far-field pure clear-sky SW flux ≈ 80 W m⁻² (ocean, near noon);
     - mean CTZ effect near −9 W m⁻² SW / +1.4 W m⁻² LW;
     - flux decaying monotonically with distance from cloud;
     - the per-pixel flux distribution not becoming bimodal or extreme (track e.g. the 1st/99th percentiles of predicted flux per epoch; a steady widening is the overfitting signature).
  3. **Consistency on overlapping footprints**: predict a held-out footprint from the map built on its neighbours.
- **Random subset regularization** (Fan, Appendix D): compute the aggregate over a random ~20% of the pixels in each step. This demands that every large subset averages correctly, a stronger constraint. Small but positive effect; cheap to add.
- Weight decay and early stopping are the main defences; also consider a mild penalty on the spread of per-pixel predictions within a footprint (not from the papers; an idea to test).

## 3. Validation without 1 km truth

**Lesson.** Every method here must be evaluated against the coarse measurement, but **good aggregate performance does not guarantee good pixel-level performance** (Law 2018, Section 4.2: a model can fit district-level malaria counts using only coordinates while being wrong at pixel level).

**What this means for the thesis.**

- **How to split matters more than the percentage.**
  - Split **by granule (or by day)**, never randomly by footprint. CERES footprints overlap, so a random split puts the same MODIS pixels in both train and test: leakage.
  - A standard split is ~70 / 15 / 15 (train / validation / test) by granule. Validation is for early stopping and tuning; test is touched once, at the end.
  - With only six granules of pixel-level data, a fixed split is fragile. Prefer **leave-one-granule-out cross-validation** for the CNN stage, and report the mean and spread across folds.
  - Keep CERES overlap groups and orbit-adjacent granules on the same side of the split.
- **Report results by regime** (clear / broken / overcast), not only overall, since a good overall score can hide a poor CTZ.
- **Aggregate fit is necessary, not sufficient.** Always pair it with the physics checks (section 2) and the synthetic-truth test (section 7).
- **Reference numbers for comparison tables:**
  - Current thesis baselines: RMSE 17.72 W m⁻² (existing all-sky model) and 15.33 W m⁻² (deeper network), Pacific 2014–2015.
  - MODIS C7/v2 pixel-level flux vs CERES SSF (global, Meyer 2026): SW clear-sky bias +4.4 W m⁻² (2.2%), RMSE 20.6; SW all-sky bias +11.0 (4.2%), RMSE 35.8; SW ice-only bias +27.3 (6.4%). Not directly comparable (global vs ocean, different matching), but useful context.
  - CERES clear-ocean uncertainty ~1.9 W m⁻² (Su et al. 2015b).

## 4. The under-determined problem: what actually resolves it

**Correction to a common misreading.** Poisson likelihoods and Lagrange-style methods in these papers are **not** general tools for "too many unknowns":

- **Poisson** (Law 2018) is just the right likelihood for **count** data (malaria cases). Flux is continuous; a Gaussian likelihood, i.e. MSE, is the right choice and is what the thesis already uses.
- **The "generalized Lagrange function"** in Gonzalez is in practice a **regularized loss**: a data term (fit to the coarse GERB flux) plus a smoothness penalty. It is a quadratic-penalty method, not true Lagrange multipliers.
- The Taylor and Jensen approximations in Law 2018 were needed only because of the Bayesian GP formulation (an expectation of a log of a sum). With a network and MSE they are not needed.

**The real lesson.** With ~1,500 MODIS pixels per CERES label there are **infinitely many** pixel maps that match CERES exactly. Something must choose among them:

| Paper | What chooses the solution |
|---|---|
| Gonzalez / Bertrand | explicit smoothness of a multiplicative correction field |
| Law 2018, Chau 2021 | a Gaussian-process prior (kernel = assumed smoothness/similarity) |
| Fan 2021 | the U-Net architecture itself, plus early stopping and augmentation |
| Gao 2012 | a regression tree learned only on homogeneous pixels |

In the thesis this choice is made **implicitly** by the CNN (its architecture, receptive field, regularization and stopping point). The thesis should state this explicitly and test its consequences (sections 6–7).

## 5. Aggregate in flux, not in inputs

**Lesson (from almost every paper).** The aggregation must be applied to the **predicted flux**, `mean(f(x))`, never to the inputs, `f(mean(x))`, because flux is nonlinear in the inputs.

- **Law 2018:** the "bag-pixel" baseline (average the covariates, then predict) performs worst of all (NLL above 2.4 vs 2.22 for a constant-in-bag predictor).
- **Gao 2012:** temperature is aggregated through radiance, `T_agg = (mean(T⁴))^(1/4)`, not as a mean temperature.
- **Bertrand 2006, Eqs. 11–13:** correcting radiance and then converting to flux gives a different answer from converting and then correcting, because the radiance-to-flux conversion (with the ADM aerosol LUT) is nonlinear.
- **Fan 2021:** the averaging-based SIF baselines lose more accuracy the larger the resolution gap (equal at 600 m, clearly worse at 30 m).

**Implication.** Keep the PSF weighting on the per-pixel flux output inside the loss; avoid any shortcut that pre-averages MODIS features per footprint for the CNN stage (that is exactly what the current tabular models do, and why they cannot produce a 1 km map).

## 6. Regularization can manufacture an artificial CTZ

**Lesson.** Smoothness spreads the correction across scene boundaries. In a synthetic re-implementation of Gonzalez's method, the residual error concentrated **at object edges**, because the correction from one scene type leaked into its neighbour. Bertrand 2006 found the real-world version: the high-resolution correction factor over **clear ocean was contaminated by clouds** in the surrounding GERB pixel, and they proposed scene-dependent resolution enhancement to fix it.

**Why it matters here.** The CTZ is, by definition, a change in clear-sky flux near clouds. A model that smears cloud influence into nearby clear pixels (through explicit smoothness, a large receptive field, or heavy regularization) will **produce a CTZ-like signal even if none exists**.

**What to do.**

- Run the model on a **synthetic scene with no CTZ** (clear-sky flux independent of distance to cloud) and confirm the predicted flux vs distance-from-cloud curve is flat. Any decay seen there is an artefact and sets a floor on what the real result can claim.
- Report the receptive field (section 1) alongside the CTZ distance scale.
- Prefer edge-aware / scene-aware choices over global smoothing.

## 7. Build a test problem with known truth

**Lesson (Chau 2021, also Gonzalez and Gao).** All of these methods were validated on **simulated** problems: take a fine field with known truth, aggregate it, try to recover it.

**What to do.**

1. Take a 1 km flux field with known values, e.g. from radiative-transfer simulations, or from the current per-pixel model used as a synthetic "truth".
2. Aggregate it with the real CERES PSF and footprint geometry to produce synthetic CERES labels.
3. Train the CNN on those labels exactly as on real data.
4. Compare the recovered 1 km map with the known truth: pixel RMSE, bias by regime, flux-vs-distance-to-cloud curve.

This is the only way to measure pixel-level accuracy directly, and it doubles as the artificial-CTZ check in section 6.

**Caveat from Chau's toy example.** What aggregation erases (variability inside a footprint) can be recovered **only if the inputs carry that information**. CTZ flux is recoverable only to the extent MODIS "sees" what drives it (humidity, humidified aerosol, small or thin clouds). This supports physically motivated features (section 9).

## 8. Check that residuals do not depend on geometry

**Lesson (Bertrand 2006).** A systematic diurnal asymmetry in GERB SW flux was traced to a narrowband-to-broadband conversion that had **no dependence on viewing zenith (VZA) or relative azimuth (RAA)** (it was built from Lambertian simulations). The diagnostic was simple: plot the correction factor (or residual) binned by SZA, VZA and RAA; it should be flat (their Fig. 7).

**What to do.**

- Plot model residuals vs CERES binned by SZA, VZA, RAA and sun-glint angle. Any trend means missing geometric information.
- Consider the **sun-glint angle** as a feature over ocean (Bertrand Eq. 14):

  ```
  cos γ = μs·μv + sqrt(1 − μs²)·sqrt(1 − μv²)·cos φ
  ```

  where μs, μv are the cosines of the solar and viewing zenith angles and φ is the relative azimuth.
- Keep CERES radiance (Isw) out of the features, as already planned, so the model does not relearn the CERES ADMs.

## 9. Features

- **Raw bands carry information derived indices lose.** Gao 2012: thin clouds and terrain shadows were visible in the shortwave bands but not in NDVI, and NDVI-only sharpening produced box-like artefacts there. Use raw (cross-normalized) MODIS bands, not only derived products such as a cloud mask.
- **Band ratios / multiplicative noise** (Fan 2021): multiplying all reflectance bands by a random factor (1 + ε), ε ~ N(0, 0.2²), forces the model to rely on ratios between bands. It gave about half of Fan's improvement. If used, apply it to reflectance/radiance **before** normalization; after standardization the ratios are no longer preserved. Note the thesis's cross-normalization against 0.47, 4.05 and 11 µm already partly serves this purpose.
- **Bands used by the MODIS cloud retrievals** (Meyer 2026): cloud optical thickness from bands 1, 2, 5 (0.659, 0.865, 1.24 µm); effective radius from bands 6, 7, 20 (1.64, 2.13, 3.75 µm); thin cirrus from band 26 (1.38 µm). Make sure band 26 is among the features, since thin cirrus is relevant to the CTZ.
- **Sub-kilometre cloudiness matters** (Meyer 2026): the 375 m VIIRS cloud mask finds a real cloud-fraction difference relative to 750 m. Many small clouds are invisible at 1 km, which is part of what the CTZ is.

## 10. Training data diversity

**Lesson.** Within one footprint, the gradient of the aggregate loss is identical for every pixel (`∂loss/∂f_i ∝ w_i · (mean − F_CERES)`). A single footprint can only say "move all pixels up or down together". Discrimination between cloud, CTZ and clear pixels is learned **only by comparing footprints with different mixtures** (Fan 2021).

**What to do.**

- Count how many distinct cloud/clear mixtures the six pixel-level granules contain (e.g. a histogram of footprint cloud fraction). Too few distinct mixtures limits what the CNN can learn.
- **Do not train only on homogeneous footprints.** Gao 2012 learned only from homogeneous coarse pixels (cv < 0.2), which works for land-surface temperature but here would systematically drop the CTZ and cloud edges, the cases that matter most. A homogeneous-only per-pixel model is, however, a good **baseline** to compare against.

## 11. Use the coarse residual as a diagnostic map

**Lesson (Gao 2012, Bertrand 2006).** After aggregating the fine prediction back to the coarse grid, the residual against the coarse measurement is a model-quality map. Gao used it to weight local vs global models per pixel (`w ∝ 1/r²`); spreading the residual uniformly gives perfect aggregate agreement but "no new sub-pixel information" and can create box-like artefacts.

**What to do.**

- Map the PSF-weighted residual vs CERES per footprint. Residuals concentrated near clouds mean the model misses the CTZ.
- Do **not** add the residual back uniformly to force exact agreement; it hides model error and creates footprint-shaped artefacts.

## 12. External baseline and data versions

- **NASA's own pixel-level flux** (Meyer 2026): MODIS C7 / VIIRS v2 will provide pixel-level TOA SW and LW flux computed with RRTMG from retrieved cloud properties. This is the main external baseline. The thesis contribution should be stated against it: the NASA product depends on a cloud mask and retrievals, so "clear" CTZ pixels receive clear-sky flux, whereas the thesis aims for continuous flux without a mask.
- **Document data versions.** If the thesis uses C6.1 MODIS products, moving to C7 changes calibration, the phase algorithm and ancillary data (GEOS-IT replaces GDAS). Record versions precisely and test sensitivity if switching.
- **Orbital drift.** Terra drifted ~15 min by September 2022 and Aqua by July 2023; all MODIS products show non-physical cloud-fraction trends in the drift period, and solar and viewing zenith sampling changed. The thesis data (2014–2015) are pre-drift. Extending to later years would require training on the new angle range, otherwise the model extrapolates in its angular features.

## 13. Optional: per-pixel uncertainty

**Lesson (Chau 2021, Law 2018).** Gaussian-process posteriors give a per-pixel variance that measures how much the aggregate data constrain each pixel. Law found the f²-link model well calibrated and the exp-link model (which used an extra bound) poorly calibrated.

**Idea.** An uncertainty map (e.g. from an ensemble of CNNs or MC dropout) would show where CERES is not enough to pin down the 1 km flux. Useful for honest reporting; not required for the core result.

---

## Checklist

- [ ] U-Net with linear output; PSF-weighted mean inside an MSE loss
- [ ] Receptive field computed and compared with the CTZ distance scale
- [ ] Granule-level split (or leave-one-granule-out CV); no random footprint split
- [ ] Early stopping on held-out granules plus physics checks (≈80 W m⁻² far clear sky, CTZ ≈ −9 SW / +1.4 LW, decay with distance)
- [ ] Track per-pixel prediction spread per epoch (extreme-map overfitting)
- [ ] Random-subset aggregation as extra regularization
- [ ] Synthetic-truth experiment, including a no-CTZ scene
- [ ] Residuals vs SZA / VZA / RAA / glint angle are flat
- [ ] Band 26 and glint angle considered as features; CERES radiance excluded
- [ ] Footprint-mixture diversity checked in the six granules
- [ ] Homogeneous-footprint per-pixel model as a baseline
- [ ] Residual map vs CERES per footprint
- [ ] Comparison with MODIS C7 pixel-level flux; data versions documented
