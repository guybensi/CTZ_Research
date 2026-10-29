# Thesis Overview: All-Sky Radiative Flux at 1 km Resolution

## Thesis goal

The goal is to develop a model that produces maps of outgoing top-of-atmosphere (TOA) radiative flux, for all sky conditions, at 1 km resolution. No such product exists today:

- **CERES** measures flux reliably, but only in footprints of roughly 27×55 km. Each footprint mixes clouds, clear sky and transition regions.
- **MODIS** provides sharp multispectral imagery at 1 km resolution, but does not measure flux.

The thesis bridges the two: MODIS sets the spatial structure, and CERES anchors the flux level.

## The paper's scientific contribution

The paper challenges an assumption underlying most energy-budget work: that the sky can be split into cloudy and non-cloudy. It shows that the sky a cloud mask labels as clear is not uniform. Near clouds there is a transition region (the Cloud Transition Zone, CTZ) containing:

- higher humidity and swollen (humidified) aerosols;
- small or thin clouds that go undetected;
- 3D scattering of light from neighboring clouds.

This region covers about 60% of the "clear" sky.

To measure it, the authors:

1. Defined, for the first time, a truly clean background: **pure clear-sky** (PCS). It is identified using the spectral metric ψ (Equation 16) and distance from clouds.
2. Trained a neural network that predicts the background flux (Fclr) from MODIS channels, anchored to clean CERES footprints.
3. Recast the TOA radiative budget equation so that the CTZ contribution can be extracted from ratios of MODIS radiances (Equations 13–14).

**Result:** over the ocean, near noon, the CTZ contribution is:

- in the shortwave (SW): about **−9 W/m²**, larger than the aerosol direct effect over the same regions;
- in the longwave (LW): about **+1.4 W/m²**, equivalent to roughly 90 ppm of CO₂.

This leads to the paper's conceptual proposal: move from two sky categories to three (cloud, CTZ, far-field clear), and eventually to a continuous treatment of the sky. The paper leaves that last direction as future work (Equation 15).

## The thesis contribution: from a scientific finding to an engineering tool

The paper provides **a single averaged number** for the CTZ contribution, and computes flux **only for clear sky**. The thesis aims to build the tool the paper points toward: a model that produces continuous flux for every pixel, with no dependence on a cloud mask. Such a tool would show **where and how strongly** clouds and transition regions change the flux, not just the average.

The question is no longer whether the CTZ matters, which the paper already showed, but how to recover it in a map.

**The core challenge:** there is no 1 km ground truth. CERES provides one label per roughly 1,500 pixels, so this is a learning-from-aggregate-supervision problem (Learning from Label Proportions / Multiple-Instance Regression), not standard regression.

## Work done so far

1. **Benchmarking infrastructure.** A framework that systematically compares classical models (Ridge, tree ensembles, SVM) against neural networks, with importance-based feature selection. Everything runs on Pacific Ocean data for 2014–2015, about 5 million footprints. See [`legacy/python/guys_benchmark_models.py`](legacy/python/guys_benchmark_models.py) and [`legacy/python/guy_model_2.0.py`](legacy/python/guy_model_2.0.py).

2. **Initial improvement over the existing all-sky model.** On the same 37-feature set used by [`legacy/python/All_Sky_AIflux.py`](legacy/python/All_Sky_AIflux.py), a deeper architecture with early stopping reduced the error by about **14%**: RMSE of 15.33 vs. 17.72 W/m².

   > This is not yet a fully controlled comparison: row filtering and the train/test split ratio differ between the scripts. Details in [`legacy/Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt`](legacy/Documentation/GUY_MODEL_VS_ALLSKY_COMPARISON.txt).

3. **Methodological lesson: no CERES-resolution features.** A 129-feature version reached an RMSE of 5.76, but most of that gain came from the CERES radiance itself (`Isw`). CERES flux is computed directly from that radiance (Fsw = π·Isw/R), so the model mostly relearned the CERES conversion tables (ADMs). It also cannot be applied at 1 km resolution.

   **Conclusion:** model inputs must come from MODIS alone, and CERES serves only as the label and as a constraint.

4. **A work plan for a CNN** on per-footprint image cubes (27×55 pixels), including an analysis of the existing data. See [`legacy/Documentation/plan_CV.txt`](legacy/Documentation/plan_CV.txt).

## Main next directions

### 1. From footprint-level to pixel-level prediction

The current models take footprint-averaged inputs, so they cannot be applied to a single pixel. The plan is to run the model on every pixel, average the predictions using the CERES point spread function (PSF), and train against the measured flux:

```
loss = MSE( PSF_weighted_mean( f(pixel_i) ) , F_CERES )
```

In cloudy footprints, `mean(f(x))` differs substantially from `f(mean(x))`, because flux is nonlinear in the inputs.

### 2. A convolutional network for spatial context

The effects the paper describes (cloud proximity, 3D scattering and humidity) depend on a pixel's surroundings, not just its spectrum. A CNN can learn them directly, rather than relying on hand-built metrics such as D and ψ. The plan is staged:

- **Phase 0:** a per-pixel model with no spatial context, as a baseline.
- **Phase 1:** a CNN with an aggregation constraint, evaluated at footprint level.
- **Phase 2:** a full 1 km flux map.

Each phase proceeds only if it improves on the previous one.

### 3. Physics-based features

- Reconstruct broadband radiance from the narrow-band channels (Equation 11 in the paper).
- Cross-normalize channels against baseline channels: 0.47, 4.05 and 11 μm (Appendix A of the paper).
- Use standard deviations instead of moments that contribute little (skew).
- Add trigonometric functions of the viewing and solar angles, to approximate the anisotropic factor R.

### 4. An evaluation framework without 1 km ground truth

- Split train/test **by granule**, to avoid spatial leakage.
- Check consistency across **overlapping CERES footprints**.
- Check agreement with the paper's background flux (Fclr) in pure clear-sky pixels.
- Check that flux **decays with distance from cloud**, similar to Figure 1 of the paper.

## Assumptions

We assume that the findings of the paper hold within a given margin of error, and that sufficient data will be available.

## Success criteria

There is no 1 km ground truth, so no single test can prove the map is correct. Instead, three tests build a chain of evidence, each closing a gap the previous one leaves open.

### 1. Reproducing CERES at footprint level (necessary condition)

Average the map with the PSF and compare it to the measured flux, on granules held out from training.

- **The bar:** beat the existing baselines (17.72 and 15.33 W/m²), and approach the CERES uncertainty itself (about 1.9 W/m² for clear sky over ocean, per Su et al. 2015b; higher for cloudy scenes).
- **Break down the error by regime:** clear, broken cloud and overcast. A good overall error can hide failure in broken-cloud scenes, which is exactly where the CTZ lives.

**Limitation:** infinitely many different maps produce the same average. This test shows the level is right, not that the distribution within the footprint is right.

### 2. Consistency across overlapping footprints (tests the spatial distribution)

CERES footprints overlap heavily: two neighboring footprints share some pixels and differ in others. If the map puts too much flux on one side of a footprint and too little on the other, that footprint's own average can still come out right. But the average of a neighboring footprint that mostly covers one side will come out wrong.

**The method:** hold some footprints out of training, and evaluate them from the map built from their neighbors. This directly measures the spatial distribution of flux, not just its level.

### 3. Reproducing the paper's result (tests the physics)

A good map makes it possible to compute RE_ctz independently, from the map itself:

- In pure clear-sky pixels far from clouds, the map should give about 80 W/m² in the SW, consistent with the paper's Fclr model.
- The average over the whole CTZ should come out near −9 W/m² in the SW and +1.4 W/m² in the LW (Table B1 of the paper).

The model is not trained on these numbers. If it still reproduces them, that is strong evidence it captures the physics of the CTZ, rather than just fitting averages.

## Scope and risks

A validated all-sky flux map at 1 km resolution is an ambitious goal for a two-year thesis. It has no direct ground truth, the aggregation constraint does not determine a unique distribution within each footprint, and pixel-level image data is currently limited to six granules. Even so, with the right structure the work can reach a satisfactory result even if the full goal is not completely achieved. It is built in layers, each of which stands on its own:

1. **Foundation (low risk):** a per-pixel model with an aggregation constraint, using MODIS-only inputs, together with the evaluation framework and a reproduction of the paper's RE_ctz. This is already a complete contribution: a method, an evaluation framework, and a result that connects to a published paper. Much of the infrastructure already exists.
2. **Extension (medium risk):** a convolutional network with spatial context, in a controlled comparison against the foundation. Even a negative result here is valuable, because it says something about the nature of the CTZ.
3. **Full map (high risk):** complete granule maps and a spatial analysis of the CTZ. This is the ambitious target, not a condition for the thesis to succeed.

The main risk, availability of image data, will be resolved early, before investing in layers 2 and 3.
