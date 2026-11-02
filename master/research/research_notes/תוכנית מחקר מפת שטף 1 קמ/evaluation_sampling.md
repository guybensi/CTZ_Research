# Evaluation and Data-Sampling Methodology for a 1 km TOA Flux Model Validated against PSF-Averaged CERES Footprints

Scope: how to split and sample MODIS/CERES data without leakage, and how to evaluate a 1 km TOA-flux model fairly when the only reference is the CERES footprint flux. The benchmark compares a per-pixel MLP with an aggregation loss, a UNet, and hybrids after fine-tuning.

Access note: ceres.larc.nasa.gov, ntrs.nasa.gov, amt.copernicus.org and webarchive.library.unt.edu were blocked in this session, so I could not open the CERES ATBDs and AMT PDFs in full. Numbers below come from search-result excerpts of those documents. Items marked "verify" should be checked against the PDFs once those domains are allowed.

---

## 1. CERES footprint geometry, PSF, and overlap: implications for leakage

### Takeaway
A CERES Terra/Aqua footprint is nominally "20 km" at nadir. The PSF's 95%-energy oval is about 38 x 31 km at nadir and grows to about 253 x 70 km at VZA 70°. SSF footprints are spaced about 8 km apart along-scan at nadir, so neighbouring footprints share most of their MODIS pixels. A random footprint-level split therefore leaks heavily, because test footprints are largely built from the same 1 km pixels the model was trained on.

### Cited Findings
- The CERES FAQ gives the Terra/Aqua (705 km orbit) optical FOV at nadir as **16 x 32 km**, called "20 km resolution". The footprint size it quotes refers to an oval holding **~95% of the PSF response**: **38 x 31 km at nadir**, rising to **253 x 70 km at 70° VZA**. The 16 x 32 km optical figure and the 38 x 31 km PSF oval measure different things. — [CERES FAQ](https://ceres.larc.nasa.gov/resources/faqs) (search excerpt; page blocked here)
- The PSF is described fully in Smith (1994), and the CERES convolution ATBD (subsystem 4.4, Release 2.2, 1997) draws the footprint as the PSF ellipse. — [CERES ATBD 4.4 (archived)](https://webarchive.library.unt.edu/eot2008/20090825013213/http:/science.larc.nasa.gov/ceres/ATBD/pdf/ceres-atbd2.2-s4.4.pdf)
- Smith, G. L. (1994), "Effects of time response on the point spread function of a scanning radiometer," *Applied Optics* 33(30), 7031–7037, doi:[10.1364/AO.33.007031](https://doi.org/10.1364/AO.33.007031). The detector and electronics time response widens and skews the PSF in the scan direction. The PSF can be derived from the step response, and the time response limits resolution along the scan. The paper uses the CERES scanner as its example. — [DOI landing](https://0-doi-org.brum.beds.ac.uk/10.1364/AO.33.007031)
- Later work on the measured point response function: "Point-response function of the Clouds and Earth Radiant Energy System scanning radiometers," *J. Appl. Remote Sens.* 8, 084991 (2014), doi:10.1117/1.JRS.8.084991. — [SPIE](https://www.spiedigitallibrary.org/journals/JARS/volume-8/issue-01/084991/Point-response-function-of-the-Clouds-and-Earth-Radiant-Energy/10.1117/1.JRS.8.084991.full)
- In SSF, MODIS/VIIRS spectral radiances and cloud retrievals are averaged over each CERES footprint with CERES-PSF weights (Smith 1994). Nadir footprint size is about 20 km for Terra/Aqua and about 24 km for S-NPP. — [Su et al. 2017, AMT 10, 4001](https://amt.copernicus.org/articles/10/4001/2017/) (search excerpt)
- Sampling: detectors are sampled every **10 ms** in a **6.6 s** scan cycle, giving **660 samples per scan**. — [CERES ATBD 1.0](https://ceres.larc.nasa.gov/documents/ATBD/pdf/r2_2/ceres-atbd2.2-s1.0.pdf); [BDS DPC](https://asdc.larc.nasa.gov/documents/ceres/readme/DPC_BDS_R6V1.pdf)
- Footprints overlap enough in the scan direction that "using every other footprint" still leaves no gaps in spatial coverage. — [CERES SSF NOAA-20 Ed1B/1C Data Quality Summary](https://ceres.larc.nasa.gov/documents/DQ_summaries/CER_SSF_NOAA20_Edition1B.pdf)
- SSF footprints are about **8 km apart at nadir**, growing to about **35 km at 65° VZA**, with about **86 SSF footprints per scan**. This comes from an Earthdata forum answer, which is less authoritative than the product documents. — [Earthdata Forum](https://forum.earthdata.nasa.gov/viewtopic.php?p=18782)
- Su et al. (2017) convolved MODIS pixel data with the Aqua and NPP CERES PSFs. Footprint size alone causes **<4 W m⁻²** instantaneous SW difference, rising to **~20 W m⁻²** in the Arctic when cloud-property differences are added. — [Su et al. 2017, AMT 10, 4001–4017](https://amt.copernicus.org/articles/10/4001/2017/)
- Relevant product documents: SSF Collection Guide ([SSF_CG_R2V1](https://ceres.larc.nasa.gov/documents/collect_guide/pdf/SSF_CG_R2V1.pdf)) and the SSF Ed4 Data Products Catalog ([DPC_SSF-Ed4_R5V1](https://ceres.larc.nasa.gov/documents/DPC/DPC_current/pdfs/DPC_SSF-Ed4_R5V1.pdf)).

### Inferences
- Along-scan overlap: footprint spacing (~8 km) is much smaller than the effective footprint (~20 km, and 31–38 km for the 95% oval). Each 1 km MODIS pixel near nadir therefore sits in the PSF of roughly 3–5 neighbouring footprints along-scan, and of neighbouring scan lines too. In a random footprint split, test targets are mostly weighted sums of training pixels. That is close to direct target leakage for a model trained with an aggregation loss.
- Along-track spacing is my estimate, not sourced: ground speed ≈ 6.8 km/s and a 6.6 s cycle give ~45 km per full cycle, or ~22 km per Earth-view sweep if the scan sweeps both ways in each cycle. This is comparable to the PSF size, so along-track overlap is likely partial. Verify in ATBD 1.0 or 4.4.
- The PSF is asymmetric (skewed in the scan direction) and grows quickly with VZA. A leakage buffer sized in kilometres must therefore depend on VZA. Splitting by granule, orbit, or day avoids having to compute footprint-to-footprint geometry at all.
- Practical leakage test: for every test footprint, compute the PSF-weighted fraction of its MODIS pixels that fall inside any training footprint's 95% oval. Report this "pixel-reuse fraction" for each split strategy. It should be ≈0 for the leakage-safe splits.

### Gaps
- I could not retrieve the analytic PSF (the ATBD 4.4 equations, including the optical FOV and the time-response terms). Get it from ATBD 4.4 or Smith (1994) directly.
- I found no authoritative statement of along-track footprint spacing or overlap.

---

## 2. Leakage-safe splitting

### Takeaway
Use grouped or blocked splits whose units (granule, day, or orbit, plus spatial blocks) are larger than the footprint and the cloud-field autocorrelation range. Add a temporal holdout (2014 vs 2015) and a regional holdout (Pacific vs North15) to test generalisation. Random splits overstate skill. One caveat: blocking can also overstate error, because it forces extrapolation (Roberts et al.) and is not a design-based estimate of map accuracy (Wadoux et al.).

### Cited Findings
- Roberts et al. (2017), *Ecography* 40(8), 913–929, doi:[10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881). Random CV on dependent data "can lead to serious underestimation of predictive error." Block CV is "nearly universally more appropriate than random cross-validation if the goal is predicting to new data or predictor space." Blocking "may also unwittingly induce extrapolations … thus overestimating interpolation errors." Recommendation: use block CV wherever dependence exists, even if residuals look uncorrelated. — [Wright State record](https://research.wright.edu/en/publications/cross-validation-strategies-for-data-with-temporal-spatial-hierar/); [Dryad data](https://datadryad.org/dataset/doi:10.5061/dryad.737gk)
- Roberts et al., as quoted in the blockCV source: blocks should be substantially larger than the residual autocorrelation range, and a buffer about that size gives a good error estimate. blockCV (Valavi et al. 2019, *Methods Ecol. Evol.* 10(2), 225–232) estimates autocorrelation ranges, offers block-size selection (`cv_block_size`) and buffered LOO (`cv_buffer`, which excludes points within the autocorrelation range). Its docs warn that ranges estimated from the raw response "may mis-size blocks for residual autocorrelation." — [blockCV manual](https://rvalavi.r-universe.dev/blockCV/doc/manual.html); [cv_spatial.R](https://rdrr.io/cran/blockCV/src/R/cv_spatial.R)
- Ploton et al. (2020), *Nat. Commun.* 11, 4540, doi:[10.1038/s41467-020-18321-y](https://doi.org/10.1038/s41467-020-18321-y). A random forest biomass model explained more than half of the variance under non-spatial CV but showed almost no predictive power under spatial CV, because nonspatial validation ignores autocorrelation. — [IRD record](https://www.documentation.ird.fr/hor/fdi:010080646/mods); [HAL](https://hal.inrae.fr/hal-02936567v1/document)
- Meyer & Pebesma (2021), "Predicting into unknown space? Estimating the area of applicability of spatial prediction models," *Methods Ecol. Evol.* They propose a dissimilarity index (importance-weighted distance to training data in predictor space) and an AOA threshold derived via CV, and suggest reporting the AOA with predictions. — [arXiv 2005.07939](https://ar5iv.arxiv.org/html/2005.07939)
- Wadoux et al. (2021), "Spatial cross-validation is not the right way to evaluate map accuracy," *Ecol. Model.* 457, 109692. Unbiased map-accuracy estimates come from probability sampling with design-based inference. In their experiment, standard CV had smaller bias than spatial CV. Meyer & Pebesma (2022, *Nat. Commun.* comment) reply that random CV is valid only for iid data and that spatial CV is the way to control dependence when a probability sample is unavailable. — [IDEAS](https://ideas.repec.org/a/eee/ecomod/v457y2021ics0304380021002489.html); [PMC9033849](https://pmc.ncbi.nlm.nih.gov/articles/PMC9033849)
- Cloud-field autocorrelation scales: Bley, Deneke & Senf (2016, *JAMC*) give a typical spatial decorrelation length of **~7 km** and a Lagrangian decorrelation time of **~30 min** for warm convective cloud fields (SEVIRI, central Europe), for 0.6 µm reflectance, COD and LWP. — [TROPOS PDF](https://www.tropos.de/fileadmin/user_upload/Institut/Abteilungen/Fernerkundung/Daten_PDF/2016_Bley-Deneke-Senf_JAMC.pdf)

### Inferences
Recommended split hierarchy, from strictest to least strict:
1. **Region holdout:** train on Pacific 2014–2015, test on North15. This tests transfer across regime and SZA.
2. **Temporal holdout:** train 2014, test 2015, or hold out whole months. Within each test period, keep complete days.
3. **Day- or granule-grouped CV within the training set:** use GroupKFold on day ID for tuning. A single MODIS granule (5 min, ~2330 x 2030 km) covers thousands of footprints, and Terra and Aqua overpasses on the same day share synoptic systems. Group by day rather than by granule if both satellites are used.
4. **Spatial block CV inside granules:** only if test data must come from the same scenes. Use blocks of at least about 100 km. The footprint's 95% oval is 31–38 km, overlapping footprints add more, and the cloud decorrelation length is about 7 km, though mesoscale organisation of stratocumulus can be tens to hundreds of km. Leave a buffer of at least one footprint 95%-oval diameter, scaled with VZA, between train and test blocks.

Further points:
- Size buffers from the variogram of the **residuals** (CERES minus model prediction), not of raw flux, as the blockCV docs warn. Compute this variogram on footprint centres as part of the benchmark.
- Report both a grouped-CV estimate (for model selection) and a fixed, untouched, temporally and regionally separated test set (for the final number). This answers the Roberts concern (dependence) and partly the Wadoux concern: test days drawn at random from 2015 approximate a probability sample of the target period.
- Use the AOA or dissimilarity index to flag North15 scenes that fall outside the Pacific training distribution (for example, high SZA). Report metrics separately inside and outside the AOA.

### Gaps
- I found no published CERES/MODIS ML-flux paper that documents its split protocol in detail. CERES ADM validation (Su et al. 2015) uses consistency between different viewing geometries rather than ML-style splits.
- I found no source for a horizontal decorrelation length of TOA SW flux over marine stratocumulus. The ~7 km figure is for continental cumulus.

---

## 3. Evaluation without 1 km ground truth

### Takeaway
The reference is the CERES footprint flux. Its instantaneous ADM uncertainty is about 1.9 W m⁻² (2.3%) for clear ocean SW and up to about 9.9 W m⁻² (3.7%) for all-sky SW, and largest for thin broken clouds. A footprint RMSE below these levels cannot be told apart from reference error. Evaluate:
- at footprint level, by cloud regime;
- against held-out overlapping footprints;
- against domain-mean bias;
- against physical signatures such as CTZ flux decay;
- by recovering a synthetic fine-scale truth.

### Cited Findings
- Su et al. (2015), "Next-generation angular distribution models for top-of-atmosphere radiative flux calculation from CERES instruments: validation," *AMT* 8, 3297–3313, doi:[10.5194/amt-8-3297-2015](https://doi.org/10.5194/amt-8-3297-2015). Edition 4 instantaneous SW uncertainties: **2.3% (1.9 W m⁻²) clear ocean**, **1.6% (4.5 W m⁻²) clear land**, **2.0% (6.0 W m⁻²) clear snow/ice**, and up to **3.7% (9.9 W m⁻²) all-sky**. Uncertainties are larger for thin broken clouds than for moderate or thick overcast. Daytime LW is about 0.5–1% clear and 1–1.5% all-sky. Scene-identification errors bias SW by about −0.6 W m⁻². A reviewer flagged an abstract/Table 5 inconsistency in the discussion version; check the final tables. — [AMT article](https://amt.copernicus.org/articles/8/3297/2015/); [PDF](https://amt.copernicus.org/articles/8/3297/2015/amt-8-3297-2015.pdf)
- Loeb, Kato, Loukachine & Manalo-Smith (2005), "Angular distribution models for TOA radiative flux estimation from CERES on Terra. Part I: Methodology," *J. Atmos. Oceanic Technol.* 22(4), 338–351, doi:[10.1175/JTECH1712.1](https://doi.org/10.1175/JTECH1712.1). ADMs built from 24 months of CERES radiances with coincident MODIS cloud and aerosol retrievals. — [MODIS pubs](https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=02986)
- The FluxByCldTyp product partitions each footprint into clear, lower-cloud and upper-cloud sub-footprints using 2 km MODIS cloud properties. Sub-footprint fluxes are rescaled to sum to the observed footprint flux. MODIS-derived fluxes are within **1% (LW) and 2.5% (SW)** of CERES footprint fluxes. This is a direct precedent for the "consistency with footprint" constraint. — [NTRS 20230000125](https://ntrs.nasa.gov/citations/20230000125); [AMS 2017 abstract](https://ams.confex.com/ams/97Annual/webprogram/Paper316068.html); [CERES STM 2019](https://ceres.larc.nasa.gov/documents/STM/2019-05/09_CERES.Langley.FBCT.201905STM_Moguo.pdf)
- CTZ physical signature: an Eytan CERES STM (May 2024) presentation reports SW twilight-zone radiative effect **≈ −10 W m⁻²**, and **≈ −6/−10 (confident mask)** and **−4.4/−7.4 W m⁻² (relaxed mask)**, obtained by co-locating CERES FOVs with MODIS and regressing clear-sky flux. This is a presentation, not peer-reviewed. Eytan, Gristey & Feingold (2025, *AGU Advances*, doi:10.1029/2024AV001407) report near-cloud regions reducing solar radiation by **~2.5 W m⁻²** relative to the clear-sky direct aerosol effect. The LW CTZ effect is **~0.75 W m⁻²** (Eytan et al. 2020, *Nat. Geosci.*). — [Eytan STM 2024](https://ceres.larc.nasa.gov/documents/STM/2024-05/18_Eshkol_Eytan.pdf); [Eos highlight](https://eos.org/editor-highlights/characterizing-the-space-between-clouds-and-clear-sky)
- Prior ML-to-CERES skill levels, for context:
  - Gradient boosting from AVHRR radiances to instantaneous TOA SW over land, trained on CERES: RMSE **25.16 W m⁻²**. — [ResearchGate](https://www.researchgate.net/publication/316173314_Estimating_Top-of-Atmosphere_Daily_Reflected_Shortwave_Radiation_Flux_Over_Land_From_MODIS_Data)
  - OMI-based neural network trained on Aqua CERES SSF footprints: global daily mean within ±1% of CERES. — [ResearchGate](https://www.researchgate.net/publication/305035683_Top-of-the-atmosphere_shortwave_flux_estimation_from_satellite_observations_An_empirical_neural_network_approach_applied_with_data_from_the_A-train_constellation)
  - CERES ML scene classification. — [Part I, JTECH 2017](https://journals.ametsoc.org/view/journals/atot/34/10/jtech-d-16-0183.1.xml); [CERES STM 2020 ML](https://ceres.larc.nasa.gov/documents/STM/2020-04/46_Bijoy_CERES_STM_May2020_ver4.pdf)

### Inferences
Evaluation suite:
1. **Footprint metrics:** RMSE, MAE, bias and R² of the PSF-convolved prediction against CERES SW, stratified by:
   - regime: clear (CF < 5%), broken (5–95%), overcast (> 95%), using SSF/MODIS cloud fraction;
   - SZA and VZA bins;
   - sun-glint angle.

   Put Su et al. (2015) ADM uncertainty next to each bin as a noise floor. Skill differences between methods that are smaller than about 2 W m⁻² (clear) or 10 W m⁻² (all-sky) need paired tests (section 5), not raw comparison.
2. **Overlapping-footprint consistency:** train with every other footprint (the DQS says these alone cover the swath without gaps), then predict the interleaved held-out footprints from the 1 km map. Report this in-scene interpolation score separately, because it is leaky by construction. A much larger gap between it and the day-held-out score points to overfitting the training footprints rather than learning sub-footprint structure.
3. **Domain-mean bias:** mean prediction vs mean CERES per granule and per day. This tests energy conservation, which matters for CTZ budget estimates.
4. **Physical sanity checks:**
   - mean 1 km SW as a function of distance to the nearest MODIS cloudy pixel should decay monotonically toward the clear-sky value over a few km;
   - the implied CTZ SW effect should be on the order of −2.5 to −10 W m⁻², as in the Eytan work;
   - no flux below clear ocean or above incoming SW (SW ≤ S₀·μ₀);
   - no blocky or checkerboard artifacts at the footprint spacing. Check the power spectrum of the 1 km map against the MODIS reflectance spectrum at scales finer than 20 km.
5. **Area-to-point coherence:**
   - re-convolve the 1 km map with the PSF at other VZAs and footprint shapes (e.g., S-NPP 24 km) and check against those observations;
   - check the scaling of fine-scale variance against MODIS 0.65 µm reflectance variance within each footprint.
6. **Synthetic-truth experiment:** build a pseudo-truth 1 km flux, for example by applying a fixed narrowband-to-broadband plus ADM-like transform to MODIS reflectances, or by using 3D-RT/LES scenes. Convolve it with the PSF to get pseudo-CERES targets, add noise at the Su 2015 levels, then train each method and score it at 1 km against the known truth. This is the only way to measure 1 km skill directly. It also shows which method best recovers sub-footprint variance, as opposed to footprint means.

### Gaps
- I found no published paper that validates 1 km downscaling of CERES flux against an LES or 3D-RT synthetic truth. This looks like an open niche and could be a contribution.
- I could not verify the peer-reviewed SW CTZ magnitude: the −9 or −10 W m⁻² figure appears only in the presentation.
- Loeb et al. (2007) ADM Part II validation (Terra) is a likely source of further instantaneous uncertainty numbers. I could not retrieve it.

---

## 4. Sampling strategies for training data

### Takeaway
Overlapping footprints, regime imbalance (clear ocean dominates the Pacific scenes) and angular extremes all reduce the effective sample size and bias training. The literature supports:
- quality and angle filters (SZA, VZA, glint);
- stratifying by cloud fraction;
- thinning overlapping footprints to a near-independent lattice for evaluation, while keeping all footprints for aggregation-loss training only inside training groups.

### Cited Findings
- Using every other footprint along the scan still leaves no coverage gaps, which shows heavy redundancy between adjacent footprints. — [NOAA-20 SSF DQS](https://ceres.larc.nasa.gov/documents/DQ_summaries/CER_SSF_NOAA20_Edition1B.pdf)
- Footprint area grows from 38 x 31 km at nadir to 253 x 70 km at 70° VZA. Off-nadir footprints therefore average far more pixels and carry little sub-footprint information. — [CERES FAQ](https://ceres.larc.nasa.gov/resources/faqs)
- ADM uncertainty is regime-dependent (clear ocean 1.9 W m⁻² vs all-sky up to 9.9 W m⁻², highest for thin broken clouds). Target noise is therefore heteroscedastic across regimes. — [Su et al. 2015](https://amt.copernicus.org/articles/8/3297/2015/)
- Footprint-size and cloud-retrieval differences alone cause up to ~20 W m⁻² SW differences in the Arctic, and less than 4 W m⁻² elsewhere. — [Su et al. 2017](https://amt.copernicus.org/articles/10/4001/2017/)
- Blocking or stratifying can restrict predictor coverage and induce extrapolation. — [Roberts et al. 2017](https://doi.org/10.1111/ecog.02881)

### Inferences
- **Filters:**
  - SZA < 70–75°, to avoid plane-parallel failure and large ADM error at low sun;
  - VZA < ~50–55°, where the footprint is still smaller than about 2x its nadir size;
  - exclude glint angle < ~25–40° over ocean;
  - require SSF imager coverage ≥ 99% of the footprint, and drop footprints with missing MODIS pixels.

  These thresholds are common practice, not taken from a source here. Verify them against the SSF DPC quality flags.
- **Regime balance:** stratify sampling and loss weighting by cloud-fraction bins (for example 0, 0–20, 20–50, 50–80, 80–100%), so broken scenes (the CTZ-relevant ones) are not swamped by clear and overcast scenes. An alternative is to weight the loss by 1/σ²_ADM(regime) to account for heteroscedastic label noise.
- **Overlap and effective sample size:** for any metric that assumes independence (confidence intervals, significance tests), use a non-overlapping lattice: every 2nd–3rd footprint along-scan and every other scan line, or blocks of at least 40 km. Or treat each granule or day as the independent unit. A rough effective-N estimate is N_eff ≈ N / (1 + 2Σρ_k), using the footprint-residual autocorrelation.
- For an aggregation-loss model, overlapping training footprints are not harmful: they give extra constraints on shared pixels. They must, however, all lie inside the training groups.
- Ocean-only (Pacific) training simplifies surface effects. Land or high-SZA North15 scenes should be reported as out-of-AOA transfer.

### Gaps
- I found no paper that quantifies the effective sample size of CERES SSF footprints, or recommends a specific thinning lattice for ML.

---

## 5. Fair benchmark design

### Takeaway
Hold everything fixed across methods:
- splits (same day/granule groups and the same final temporal and regional test sets);
- tuning budget (same number of HP trials on the same validation groups, or nested CV);
- fine-tuning protocol;
- repeated seeds (≥ 3–5).

Compare methods with paired tests on dependent-unit-level losses (granule or day), for example a paired block bootstrap, a Diebold–Mariano-style test with HAC variance, or Wilcoxon signed-rank across days.

### Cited Findings
- Diebold & Mariano (1995), "Comparing Predictive Accuracy," *J. Bus. Econ. Stat.* 13(3), 253–263. They test the null of equal accuracy of two forecasts using a t-statistic on the mean loss differential with a HAC variance. Diebold (2012, NBER WP 18391) cautions that DM was designed for comparing forecasts, not models. — [RePEc](https://econpapers.repec.org/RePEc:bes:jnlbes:v:13:y:1995:i:3:p:253-63); [NBER w18391](https://www.nber.org/papers/w18391.pdf); [statsmodels implementation](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.diebold_mariano_test.html)
- Random CV under dependence underestimates error, so variance estimates for model comparison must also respect blocking. — [Roberts et al. 2017](https://doi.org/10.1111/ecog.02881)

### Inferences
Protocol:
1. Freeze the split manifest (day or granule IDs for train, val and test, and the North15 holdout) in a versioned file. Every method reads the same manifest.
2. Tune with an equal budget, for example N Optuna trials per method on the same day-grouped validation folds. Use nested CV only if data are small, otherwise one fixed validation block. Never tune on the test days.
3. Train the final models with 5 seeds each. Report mean ± sd across seeds.
4. Compute a per-day (or per-granule) loss differential between methods, d_g = MSE_A,g − MSE_B,g. Then test it with:
   - a paired bootstrap over days or granules (resample units, 10k reps, 95% CI on mean d);
   - a Wilcoxon signed-rank test;
   - a DM test with HAC lag spanning the within-day dependence.

   Adjust for multiple comparisons (Holm) when testing several method pairs.
5. Report by regime (clear/broken/overcast) and by test type (interleaved footprints, day-holdout, 2015 holdout, North15 holdout), and the synthetic-truth 1 km score. Add the parameter count and compute cost, and the ADM noise floor for context.
6. Two further notes:
   - **Dietterich (1998)**, *Neural Computation* 10(7), 1895–1923: proposes the 5x2cv paired t-test for comparing classifiers. Cited from memory, not verified in this session.
   - **Demšar (2006)**, *JMLR* 7, 1–30: recommends Wilcoxon and Friedman/Nemenyi tests for comparisons across multiple datasets. Also from memory, not verified.

   The multi-dataset framing fits here, with each region or year treated as a "dataset".

### Gaps
- Dietterich (1998) and Demšar (2006) details are from memory and unverified in this session.
- I found no CERES/MODIS ML paper that reports significance tests between methods. This looks like common practice that the field has not adopted.
