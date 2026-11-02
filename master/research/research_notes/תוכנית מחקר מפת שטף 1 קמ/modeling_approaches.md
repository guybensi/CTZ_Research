# Modeling Approaches for ~1 km TOA SW/LW Flux Maps from MODIS with Coarse (CERES-footprint) Supervision

Scope: candidate methods the thesis can benchmark. Inputs restricted to MODIS radiances/reflectances, SZA/VZA/RAA and sun-glint angle; labels are CERES footprint fluxes (~20 km at nadir, PSF-weighted). Note: ceres.larc.nasa.gov was not reachable from this research session (network block), so CERES STM slides were seen only via search snippets; NASA NTRS mirrors were used where possible.

## 1. Pixel-level regressors trained with aggregate supervision (LLP / MIR / aggregate-output losses)

### Takeaway
There is a mature ML theory for training a per-instance regressor from set-level mean labels ("learning from aggregate observations"), and remote-sensing downscaling work (CS-SUNet, Label Super-Resolution) shows it works at 100x resolution gaps if smoothness/prior regularizers are added. This is the most direct formalization of the thesis problem: loss = MSE(Σ_i w_PSF,i · f(x_i), F_CERES).

### Cited Findings
- **Zhang, Charoenphakdee, Wu & Sugiyama (2020), "Learning from Aggregate Observations", NeurIPS 2020 (arXiv:2004.06316).** Generalizes multiple-instance learning to multiclass classification and regression where supervision is given to sets of instances while the goal is predicting individual labels; covers "mean / difference / rank observation" for regression; maximum-likelihood formulation applicable to DNNs and gradient boosting machines; proves consistency "up to an equivalence relation". — [NeurIPS abstract](https://papers.nips.cc/paper/2020/hash/5b0fa0e4c041548bb6289e15d865a696-Abstract.html); [arXiv](https://arxiv.org/abs/2004.06316v3)
- A secondary review summarizes that learning from **mean observations** is consistent (MSE comparable to direct supervision), while **rank observations** are consistent only up to an additive constant. — [liner.com review (secondary)](https://liner.com/review/learning-from-aggregate-observations); full PDF not read, verify in paper.
- **Fan, Chen, Wen, Sun & Gomes (2022), "Monitoring Vegetation From Space at Extremely Fine Resolutions via Coarsely-Supervised Smooth U-Net" (CS-SUNet), IJCAI 2022, pp. 5066–5072, arXiv:2207.08022.** Labels (SIF) only at ~3 km, predictions at ~30 m (100x); the average of U-Net pixel predictions over the coarse cell is trained to match the coarse label; smoothness loss "crucial for preventing overfitting"; ~10% RMSE reduction vs all baselines. — [IJCAI](https://www.ijcai.org/proceedings/2022/703); [arXiv PDF](https://web3.arxiv.org/pdf/2207.08022)
- **Malkin et al. (2019), "Label Super-Resolution Networks", ICLR 2019.** Loss compares the distribution implied by a set of model outputs with the distribution implied by low-resolution labels; demonstrated 30 m → 1 m land cover; adding any available fine labels helps. — [mlanthology](https://mlanthology.org/iclr/2019/malkin2019iclr-label)
- **Progressive self-training LST downscaling (arXiv 2603.29478, March 2026):** pretrain at coarse scale, then pass pseudo-labels down through 240→120→60→30 m stages with fine auxiliary inputs; beats linear and standard DL baselines on simulated and real data. — [arXiv](https://arxiv.org/pdf/2603.29478)
- **SRDA-Net** (MODIS LST to 500/250 m) uses thermodynamic and spatial-structure constraint losses (physics + reconstruction). — [MDPI Remote Sensing 17, 3510](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-17-03510/article_deploy/remotesensing-17-03510.pdf)

### Inferences
- The thesis loss maps directly onto Zhang et al.'s "mean observation" case with known non-uniform weights (the CERES PSF), so consistency results should carry over if the PSF weights are correct and the per-pixel function is identifiable from the variety of footprint compositions (many footprints with different cloud fractions/mixtures).
- Benchmark set: (a) per-pixel MLP with PSF-weighted aggregate loss; (b) LightGBM/XGBoost with custom aggregate gradient (Zhang et al. explicitly say GBMs work); (c) CS-SUNet-style U-Net with aggregate loss + total-variation/smoothness prior; (d) two-stage "pseudo-label" baseline: train footprint-level model on PSF-averaged features, apply to pixels, then refine with aggregate loss (similar to progressive self-training).
- Fine-tuning: the key hyperparameters are smoothness-loss weight, footprint batch composition (sample many footprints with heterogeneous cloud cover), and whether to add a weak per-pixel prior (e.g., clear-sky pixels near a known clear-sky flux). Validate by (i) holding out footprints, (ii) re-aggregating predictions to overlapping footprints from the other satellite / other scans, and (iii) synthetic tests with known 1-km truth (e.g., RT-simulated fields) because footprint-level RMSE does not prove sub-footprint correctness.

### Gaps
- I did not locate a specific "Hsu et al." aggregate-label paper or multiple-instance regression (MIR) papers for remote sensing in this session; PM2.5 downscaling-with-coarse-labels papers were not retrieved.
- No paper found applying aggregate-observation learning specifically to TOA radiative flux.

## 2. Existing high-resolution flux / NB-to-BB / ADM work (CERES, GERB, GEO, EPIC, cloud twilight zone)

### Takeaway
The operational lineage is MODIS narrowband → broadband radiance → ADM → flux. CERES itself is moving (FBCT/Ed5) to DNN-based MODIS→broadband conversion at sub-footprint level, normalized to CERES — essentially the thesis problem — with footprint-level SW RMS ~3.2 W/m² reported. GERB-HR shows the classic "SEVIRI-derived high-res flux made PSF-consistent with the coarse broadband" approach.

### Cited Findings
- **CERES ADMs (foundational):** Loeb, Kato, Loukachine & Manalo-Smith (2005), "Angular Distribution Models for TOA Radiative Flux Estimation from CERES on Terra. Part I: Methodology", J. Atmos. Oceanic Technol. 22(4), 338–351, doi:10.1175/JTECH1712.1. Built from 24 months of CERES radiances with coincident MODIS cloud/aerosol retrievals and GMAO meteorology; scene types over ocean, land, desert, snow, clear and cloudy. — [sciencegate record](https://www.sciencegate.app/document/10.1175/jtech1712.1); [MODIS pub abstract](https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=02986)
- Earlier TRMM ADM paper: Loeb et al. (2003), J. Appl. Meteor. 42, 1748, doi:10.1175/1520-0450(2003)042<1748:ADMFTR>2.0.CO;2. — [PDF](https://satcorps.larc.nasa.gov/ceres/pub/journals/Loeb.etal.JAM.03.pdf)
- **Su, Corbett, Eitzen & Liang (2015a), "Next-generation angular distribution models for TOA radiative flux calculation from CERES instruments: methodology", AMT 8, 611–632, doi:10.5194/amt-8-611-2015** (has a corrigendum). Uses Terra/Aqua RAPS data, MODIS cloud/aerosol + GEOS-5.4.1; largest changes over clear and polar scenes; regional monthly instantaneous flux changes up to 5 W/m² at 1°, global < 0.5 W/m². Companion validation paper: Su et al. (2015b), AMT 8, 3297–3313, doi:10.5194/amt-8-3297-2015. — [AMT 8, 611](https://amt.copernicus.org/articles/8/611/2015/); [AMT 8, 3297 metrics page](https://amt.copernicus.org/articles/8/3297/2015/amt-8-3297-2015-metrics.html)
- **CERES FBCT DNN (AGU Dec 2023 poster, NTRS 20230017391):** two variants — DNN converts MODIS NB radiances → BB radiance then ADM → flux ("DNN+ADM"), or DNN direct to flux without ADM ("DNNOnly"); trained nets applied to mixed-scene sub-footprints, sub-footprint fluxes summed to footprint flux. Reported global SW RMS: Ed4 baseline 3.99, DNN+ADM 3.21, DNN-only 3.15 W/m² (bias 0.42/0.85/0.60) — column mapping ambiguous in extracted text, treat as tentative. Authors conclude a future edition can bypass the ADM. — [NTRS poster PDF](https://ntrs.nasa.gov/api/citations/20230017391/downloads/AGU_202312_poster.pdf); [NTRS record](https://ntrs.nasa.gov/citations/20230017391)
- CERES 2022 STM (Doelling): Ed5 plan "utilize Machine learning to derive the subfootprint BB radiances from MODIS channels", add more MODIS channels, and "normalize" sub-footprint fluxes with CERES observations; a 2021 SPIE abstract analyzed 19 MODIS channels for next-gen FBCT. — [Doelling STM 2022 slides](https://ceres.larc.nasa.gov/documents/STM/2022-04/07_Doelling_CERES_STM_apr2022_final.pdf) (seen via search snippet only); [SPIE 2021, doi:10.1117/12.2600060](https://www.spiedigitallibrary.org/conference-presentations/11859/118590F/Optimized-MODIS-channel-selection-to-improve-the-NASA-CERES-FluxByCloudTyp/10.1117/12.2600060)
- **GERB/SEVIRI:** Clerbaux et al. (2008) "Unfiltering of GERB data" Parts I (SW) and II (LW), JTECH; GERB filtered BB radiance scaled by SEVIRI-derived unfiltering factor. SEVIRI narrowband radiances are used to produce ~9 km (3×3 SEVIRI pixel) GERB-like fluxes made consistent with GERB native ~45 km via the PSF; ADM selected using SEVIRI scene ID from CERES TRMM ADMs; Dewitte et al. (2008) describe Edition 1 processing. — [Dewitte, "Generation of GERB unfiltered radiances and fluxes" (PDF)](https://gerb.oma.be/Documents/dewitte.pdf); [RMIB poster](https://gerb.oma.be/PapPrePos/2014EGU/poster_aipe.pdf); [orfeo record](https://orfeo.belnet.be/handle/internal/8874)
- **Himawari-8/AHI:** Lee et al. (2018), Remote Sensing 10(2), 213, doi:10.3390/rs10020213 — non-ML reflected SW at TOA vs Terra/Aqua/S-NPP CERES (Jul 2015–Feb 2017): RMSE 52.12 W/m², R² 0.903, bias −2.34 W/m² (instantaneous). Zhan et al. (2022), Remote Sensing 14(22), 5696 — GBRT for TOA OLR from AHI, ~662,000 collocated samples, daily-mean RMSE ~6 W/m² (~3%). — [DOAJ Lee 2018](https://doaj.org/article/9fd7d6d817824d83abc0b82652a18fdd); [MDPI Zhan 2022](https://www.mdpi.com/2072-4292/14/22/5696)
- **DSCOVR/EPIC (2025):** IEEE TGRS (Oct 2025) deep-learning framework: 18 CNNs (9 scene types × clear/cloudy) trained on matched EPIC–CERES SSF, plus MLP linking to CERES SYN; motivated by absence of EPIC ADMs; companion J. Remote Sensing paper reports 97% of daily global TOA SW variance explained. No RMSE retrieved. — [SZU news](https://ias.szu.edu.cn/en/info/1027/4005.htm); [EurekAlert](https://www.eurekalert.org/news-releases/1092795); [JRS record](https://library.kab.ac.ug/Record/doaj-art-578f0f2c5f3340d092ed3b505a06b6ad/Description)
- **OMI neural network**: OMI data + ancillary as inputs, CERES TOA SW flux as training target, reproduces independent CERES with high fidelity (AMT, 2016, vol. 9, 2813). — [AMT 9, 2813 XML](https://amt.copernicus.org/articles/9/2813/2016/amt-9-2813-2016.xml)
- **Cloud twilight zone:** Eytan, Koren, Altaratz, Kostinski & Ronen (2020), "Longwave radiative effect of the cloud twilight zone", Nature Geoscience, doi:10.1038/s41561-020-0636-8: LW twilight-zone effect ~0.75 W/m² over oceans (≈ forcing of +75 ppm CO₂), twilight zone occupies >60% of apparent clear sky in low-cloud fields; method co-locates CERES footprints with MODIS imagery and uses regression to separate clear-sky flux. Eytan CERES STM 2024 slides report SW twilight effect of order −6 to −10 W/m² depending on cloud-mask confidence (snippet only). Jahani et al. (ACP-2021-421) found ~0.8 W/m² LW over SE Atlantic (Aug 2010, Aqua). — [Nature Geoscience](https://www.nature.com/articles/s41561-020-0636-8); [Eytan STM 2024 slides](https://ceres.larc.nasa.gov/documents/STM/2024-05/18_Eshkol_Eytan.pdf); [Jahani ACP supplement](https://acp.copernicus.org/preprints/acp-2021-421/acp-2021-421-AC1-supplement.pdf)

### Inferences
- The CERES FBCT DNN is the closest published analogue and the right external baseline: footprint-level SW RMS ~3–4 W/m² (monthly/regional-scale statistic, likely not instantaneous) sets a reference; instantaneous footprint RMSE for direct ML retrievals (AHI SW non-ML ~52 W/m²) is much larger, so the thesis should report both instantaneous footprint RMSE and aggregated statistics, clearly labeled.
- The "DNN-only bypasses ADM" result supports the thesis's no-ADM end-to-end regression; GERB-HR's PSF-consistency step suggests a post-hoc "normalize/redistribute to match CERES" option (multiplicative correction per footprint) as a cheap baseline.
- The twilight-zone ψ/regression approach (Eytan) is a natural evaluation target: does the 1-km map reproduce the SW/LW twilight-zone signal near cloud edges?

### Gaps
- Kato, Doelling, Painemal individual ML flux papers and an "All-sky AI flux" publication were not found in this session; full RMSE tables for FBCT DNN and EPIC CNN are unverified.
- ceres.larc.nasa.gov blocked; Ed5 ATBD not found (likely not yet released).

## 3. CNN / U-Net approaches: super-resolution with coarse supervision and 3D cloud radiative effects

### Takeaway
Multi-pixel CNNs demonstrably capture 3D radiative effects (neighbor illumination/shadowing) that per-pixel (IPA) methods miss, which matters at 1 km near cloud edges. Combined with aggregate losses (Section 1), a U-Net is the main deep candidate.

### Cited Findings
- **Okamura, Iwabuchi & Schmidt (2017), AMT 10, 4747–4759, doi:10.5194/amt-10-4747-2017:** DNNs retrieve COT and effective radius over multiple pixels from 4-wavelength reflectances; trained on LES + 3D RT; convolutional variant more accurate than correcting bispectral retrievals and reduces 3D effects, robust for thick clouds (feasibility study). — [AMT](https://amt.copernicus.org/articles/10/4747/2017/)
- **Nataraja et al. (2022), AMT 15, 5181–5205, doi:10.5194/amt-15-5181-2022:** segmentation-style CNN, single 600 nm channel → COT fields; trained on 3D RT of LES (Sulu Sea), tested on Atlantic; reduces IPA bias, overestimates over Atlantic; code (Cloud-Retrieval-CNN v2.0.0) and weights released on Zenodo. Follow-up: Chen et al. (2023) EaR³T toolbox, AMT 16, 1971–2000, applied the CNN to CAMP2Ex imagery with better 3D radiance consistency. — [DOAJ record](https://doaj.org/article/851777fe33f74341baa053dab3ec14e0); [AMT assets](https://amt.copernicus.org/articles/15/5181/2022/amt-15-5181-2022-assets.html)
- CS-SUNet (above) is a U-Net with aggregate (mean-of-pixels) loss + smoothness prior. — [IJCAI](https://www.ijcai.org/proceedings/2022/703)
- Pan-Arctic LST: deep anisotropic diffusion model (DADA) downscales AVHRR LST to 1 km trained with MODIS (guided super-resolution style). — [ISPRS Archives XLVIII-4-W18-2025, 35](https://isprs-archives.copernicus.org/articles/XLVIII-4-W18-2025/35/2026/)

### Inferences
- A U-Net receiving MODIS 1-km bands + geometry and outputting a 1-km flux map, trained with PSF-weighted aggregate loss, is the "context" counterpart to a per-pixel MLP; comparing the two isolates the value of 3D/neighborhood information.
- LES+3D RT synthetic scenes (as in Okamura/Nataraja; EaR³T) could provide a ground-truth 1-km flux testbed to validate sub-footprint correctness, which CERES alone cannot.
- Fine-tuning: receptive field ≥ a few km (cloud shadows/side illumination scale with SZA); random crops aligned to footprint centers; flips/rotations only if geometry inputs (RAA) are transformed consistently — otherwise avoid rotation augmentation; mask/flag glint pixels or provide glint angle as input.

### Gaps
- Masuda (3D cloud ML) and deep image prior / weakly supervised regression papers in atmospheric remote sensing were not retrieved.

## 4. Hybrid approaches (per-pixel MLP + CNN context, physics constraints, GBM with spatial features, ensembles)

### Takeaway
Evidence supports GBMs for flux regression (AHI OLR GBRT, ~6 W/m² daily RMSE) and supports CERES's own hybrid (scene-dependent NN + normalization to CERES). Hybrid designs for the thesis are largely inferred rather than directly published.

### Cited Findings
- GBRT on ~662k AHI–CERES collocations for OLR, daily RMSE ~6 W/m². — [Zhan et al. 2022](https://www.mdpi.com/2072-4292/14/22/5696)
- CERES FBCT: scene-type-specific DNNs applied per sub-footprint and normalized to CERES footprint observation. — [NTRS poster](https://ntrs.nasa.gov/api/citations/20230017391/downloads/AGU_202312_poster.pdf)
- EPIC: one CNN per scene type (18 models) — scene-stratified ensembles. — [SZU news](https://ias.szu.edu.cn/en/info/1027/4005.htm)
- Zhang et al. (2020) aggregate-observation ML framework applies to GBMs as well as DNNs. — [NeurIPS](https://papers.nips.cc/paper/2020/hash/5b0fa0e4c041548bb6289e15d865a696-Abstract.html)
- SRDA-Net combines physical-constraint loss with reconstruction loss. — [MDPI](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-17-03510/article_deploy/remotesensing-17-03510.pdf)

### Inferences
- Candidate hybrids: (1) GBM with spatial-context features (3×3/5×5/15×15 neighborhood mean/std of key bands, distance-to-cloud, local cloud fraction, ψ-like twilight metric) trained with aggregate gradient; (2) per-pixel MLP + small CNN context encoder; (3) physics-informed regularizer: penalize deviation from an ADM-style anisotropy prior, enforce 0 ≤ SW flux ≤ S0·cos(SZA), monotonicity of SW in visible reflectance; (4) post-hoc per-footprint multiplicative normalization to CERES (GERB-HR/CERES Ed5 style); (5) deep ensembles/stacking for uncertainty maps.
- Scene stratification (clear/cloudy × surface) is a cheap, published way to improve accuracy (CERES, EPIC).

### Gaps
- No published GBM-with-distance-to-cloud features for TOA flux found; ψ metric definitions not retrieved here.

## 5. Other directions: transformers / foundation models, GPs, ATPK kriging, two-stage NB→BB→ADM

### Takeaway
ATPRK is a well-established, PSF-aware geostatistical downscaling baseline that explicitly enforces coherence with coarse data; foundation models exist (Prithvi-EO 2.0, SatMAE, Prithvi WxC) but none are pretrained on MODIS-like 36-band radiances for flux, so their benefit is uncertain.

### Cited Findings
- **ATPRK:** Wang, Shi, Atkinson & Zhao (2015), "Downscaling MODIS images with area-to-point regression kriging", RSE 166, 191–204 (MODIS bands 3–7 to 250 m; best of 5 methods). Wang, Shi, Atkinson & Pardo-Igúzquiza (2016), "A new geostatistical solution to remote sensing image downscaling", IEEE TGRS 54(1), 386–396 — accounts for support size, spatial correlation and **sensor PSF**. Regression step for trend + ATPK on coarse residuals; extended with GWR (information-loss-guided fusion). — [Lancaster eprint](https://eprints.lancs.ac.uk/id/eprint/79122); [PolyU record](https://research.polyu.edu.hk/en/publications/a-new-geostatistical-solution-to-remote-sensing-image-downscaling/); [MODIS pub abstract](https://modis.gsfc.nasa.gov/sci_team/pubs/abstract.php?id=19107); [ISPRS 2017 LST comparison](https://isprs-archives.copernicus.org/articles/XLII-2-W7/913/2017/isprs-archives-XLII-2-W7-913-2017.pdf)
- **Prithvi-EO-2.0** pretrained on 4.2M HLS time-series samples at 30 m; original Prithvi on >1 TB HLS; SatMAE adapts MAE to satellite imagery (spectral and temporal variants, not unified). **Prithvi WxC** is a 2.3B-parameter MERRA-2 weather/climate model with a downscaling fine-tune (frozen encoder + upscaling modules), CORDEX mean spatial RMSE 0.44 K. — [Prithvi-EO-2.0 arXiv 2412.02732](https://arxiv.org/pdf/2412.02732v3); [Prithvi WxC arXiv 2409.13598 review](https://liner.com/review/prithvi-wxc-foundation-model-for-weather-and-climate); [Prithvi (Jakubik et al.) arXiv 2310.18660](https://arxiv.org/abs/2310.18660v2)
- Two-stage NB→BB→ADM is the CERES "DNN+ADM" path; with similar accuracy to DNN-only. — [NTRS poster](https://ntrs.nasa.gov/api/citations/20230017391/downloads/AGU_202312_poster.pdf)

### Inferences
- ATPRK (regression = the thesis's pixel model, kriging = residual redistribution honoring the CERES PSF) is a principled, cheap baseline/post-processor that guarantees re-aggregation to CERES.
- Gaussian processes with aggregated (integral) observations are the probabilistic analogue of ATPK and give uncertainty; scaling requires sparse/inducing-point methods.
- Foundation models: Prithvi-EO bands (HLS: 6 reflective bands) don't cover MODIS thermal/IR, so fine-tuning would need input-adapter layers; likely low priority relative to aggregate-loss U-Net. A ViT/Swin encoder trained from scratch on MODIS granules with masked pretraining (SatMAE-style) on unlabeled MODIS is a plausible later experiment.

### Gaps
- No source found applying foundation models or transformers to TOA flux; no GP-aggregate-observation flux paper retrieved.

## 6. Fine-tuning practices (cross-cutting)

### Takeaway
Published practices that transfer: smoothness priors for coarse supervision, scene-stratified models, PSF-consistent normalization, and synthetic 3D-RT testbeds. Specific hyperparameter recipes for flux are not published; the items below marked as inference are recommendations.

### Cited Findings
- Smoothness loss is crucial against overfitting under coarse labels. — [CS-SUNet](https://www.ijcai.org/proceedings/2022/703)
- Adding available fine labels improves label super-resolution. — [Malkin et al. 2019](https://mlanthology.org/iclr/2019/malkin2019iclr-label)
- Scene-type-specific models (CERES ADMs, FBCT DNN, EPIC 18 CNNs). — [Su et al. 2015](https://amt.copernicus.org/articles/8/611/2015/); [SZU news](https://ias.szu.edu.cn/en/info/1027/4005.htm)
- CERES analyzed 19 MODIS channels for optimal channel selection in next-gen FBCT. — [SPIE 2021](https://www.spiedigitallibrary.org/conference-presentations/11859/118590F/Optimized-MODIS-channel-selection-to-improve-the-NASA-CERES-FluxByCloudTyp/10.1117/12.2600060)
- ADM-related changes are largest in clear-sky and polar scenes (relevant for where anisotropy/glint matters). — [Su et al. 2015](https://amt.copernicus.org/articles/8/611/2015/)

### Inferences
- Loss: PSF-weighted MSE on footprint flux; consider Huber for glint/outlier footprints; optionally normalize target by S0·cos(SZA) (albedo-like) for SW to reduce dynamic range.
- Split by granule/date (not random footprints) to avoid spatial leakage; check generalization across regions (Pacific vs North15 data already in the repo).
- Sun glint: include glint angle as input, stratify or weight footprints by glint angle, and report errors in glint bins.
- Separate SW and LW models (or heads); LW needs thermal bands, is less angle dependent.
- Hyperparameters to tune: learning rate/schedule, depth/width, smoothness weight, receptive field (U-Net levels), GBM leaves/learning rate/regularization; use Optuna-style search with footprint-level CV.

### Gaps
- No published flux-specific hyperparameter studies found.
