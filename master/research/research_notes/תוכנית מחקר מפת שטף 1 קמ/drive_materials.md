# Google Drive materials for the M.Sc. thesis (all-sky TOA flux at 1 km, CTZ)

Scope: a read-only search of guy1998@gmail.com's Google Drive, including files shared with him, on 2026-10-29. Nothing was modified.

Summary: the advisor (Eytan, `eshkoleytan@gmail.com`) shared a folder tree called **Guy_Fluxes1KM**. It contains "Data_toGuy.zip", the advisor's training code, and paired MODIS/CERES NPZ data. The data are per-footprint *moments* and *scalars* for 2014 and 2015, Atlantic and Pacific, four months each. **No unpacked `images/` NPZ files and no PNG imagery were found on Drive.** The only per-pixel image cubes are probably inside the 1.2 GB `Data_toGuy.zip`, which could not be opened through the connector. **No paper-summary or literature-review documents for the thesis exist on Drive.**

## 1. What is in "DATA TO GUY" (Data_toGuy.zip) and the Guy_Fluxes1KM folder?

### Takeaway
"DATA TO GUY" is the zip `Data_toGuy.zip` (1,230,081,230 bytes), owned by the advisor and shared on 2026-07-15. It sits in `Guy_Fluxes1KM/Old_codes_data/` next to the advisor's code and paired moments/scalars NPZ files. The zip's internal listing could not be read through the connector because the file is too large to download as base64. Its contents are inferred from the sibling files, the advisor's `Details_of_npz.py`, and the local repo, which appears to be the extracted copy.

### Cited Findings
**Folder tree.** Everything is owned by `eshkoleytan@gmail.com`.
- Root folder `Guy_Fluxes1KM`, id `1JksxgR_nNQLAoZoDaxudhvL5I_qp8yvH`, shared 2026-07-15 — [folder](https://drive.google.com/drive/folders/1JksxgR_nNQLAoZoDaxudhvL5I_qp8yvH)
  - `Old_codes_data/`, id `1jrdR6sV0JfrWKKARwU7r-yTz3LVNqNtO` — [folder](https://drive.google.com/drive/folders/1jrdR6sV0JfrWKKARwU7r-yTz3LVNqNtO)
    - **`Data_toGuy.zip`**, id `1Xf4lZti-pAJI5DwhfBSndU7J_yFI1ah9`, 1.23 GB, modified 2026-07-15 — [file](https://drive.google.com/file/d/1Xf4lZti-pAJI5DwhfBSndU7J_yFI1ah9/view)
    - `March_2014N15_momentsNscalars.zip`, id `1Ra6x1JW-LmuudLpbzoZD23GT09zzfRjb`, 2.44 GB, 2026-07-22 — [file](https://drive.google.com/file/d/1Ra6x1JW-LmuudLpbzoZD23GT09zzfRjb/view). This is the "North15 March 2014" validation region.
    - `Details_of_npz.py`, id `1LHz36f4xi2c25RcN7OF2pmer4o4pbWk3` — [file](https://drive.google.com/file/d/1LHz36f4xi2c25RcN7OF2pmer4o4pbWk3/view)
    - `All_Sky_AIflux.py`, id `1lIP2EYqy46eUMJSBI7eB5n4yYYG5HxA4`. It imports `Twilight_module as tw`; the header says "Created on Wed Dec 4 2024, @author: eeytan". — [file](https://drive.google.com/file/d/1lIP2EYqy46eUMJSBI7eB5n4yYYG5HxA4/view)
    - Folder `2014/` (`1SHTY5d8AarjSOCXhk_RrU_d2Kw2q4WB1`) and folder `2015/` (`1ElLanDYFnRn7IOlhiOoHX5u6b6f2zVWV`). Each holds `Atlantic/` and `Pacific/`, each of those holds `mrch/ june/ sep/ dec/`, and each month holds **`moments/` and `scalars/` only**. For example, 2015/Atlantic/sep has moments `1G8hC9_SlcHc82nUrIgAR2w-EwjJtLlEG` and scalars `1ZQhd_SHy_7sHlTZZvmUA8TkqPr6Ue40u`. — [2014](https://drive.google.com/drive/folders/1SHTY5d8AarjSOCXhk_RrU_d2Kw2q4WB1), [2015](https://drive.google.com/drive/folders/1ElLanDYFnRn7IOlhiOoHX5u6b6f2zVWV)
  - `new_codes_data/`, id `17qxYkLOgUFb-MOPBvwSZbZiT9-_kBokL` — [folder](https://drive.google.com/drive/folders/17qxYkLOgUFb-MOPBvwSZbZiT9-_kBokL)
    - `train_Allsky_moments_narrow_values.py`, id `1hXHhQo_iO-2ii7ufcn4CIVNnWIP26XYG`, 2026-07-22. This is the advisor's newest training script. — [file](https://drive.google.com/file/d/1hXHhQo_iO-2ii7ufcn4CIVNnWIP26XYG/view)
    - `mrchsepjunedec_2014-2015_Pacific.zip`, id `1Y5UThDFuiYPXdqLPfrYWE7hmsOMd2beE`, 2.38 GB. This is the consolidated Pacific training set; the script loads `mrchsepjunedec_2014-2015_Pacific.npz`. — [file](https://drive.google.com/file/d/1Y5UThDFuiYPXdqLPfrYWE7hmsOMd2beE/view)

**File naming.** Granules are named in MODIS Aqua style, `AYYYYDDD.HHMM.npz`, for example `A2015253.1225.npz` (day of year 253, 12:25 UTC). The same name is used across `moments/`, `scalars/` and (in the zip/repo) `images/`. — [Drive search results, e.g. A2015253.1225.npz](https://drive.google.com/file/d/1gdgiYeQzrzSGLbSBDTnINcn9tC2mmbHE/view)

**Moments NPZ.** These are footprint-level statistics of MODIS radiances inside each CERES footprint. I downloaded and inspected `A2015170.1855.npz` (id `1nn128jFQg58WtOeYlvmhoN3khbzyvBME`, 2015 Atlantic June moments), which holds 118 footprints:
- `mSWspec`, `stdSWspec`, `skewSWspec`, `kurtSWspec`: shape (118, 11), float32. These are the mean, standard deviation, skewness and kurtosis of the 11 SW bands. mSWspec ranges from 0.011 to 536.
- `mLWspec`, `stdLWspec`, `skewLWspec`, `kurtLWspec`: shape (118, 16), float32, for the 16 LW bands.
- `SWwavlngs`: (11,), from 0.47 to 2.13 µm.
- `LWwavlngs`: (16,), from 3.75 to 14.23 µm.

— [A2015170.1855.npz](https://drive.google.com/file/d/1nn128jFQg58WtOeYlvmhoN3khbzyvBME/view)

**Band list.** The advisor's script lists the column names as `['SZA','VZA','RAA','0.47','0.55','0.66','0.86','0.905','0.935','0.940','1.24','1.64','2.13','1.375','3.75','3.96a','3.96','4.05','4.46','4.5','6.7','7.3','8.5','9.7','11.0','12.0','13.33','13.63','13.93','14.23']`. That is 3 geometry columns, 11 SW bands and 16 LW bands, and each band gets m/std/skew/kurt features. — [train_Allsky_moments_narrow_values.py](https://drive.google.com/file/d/1hXHhQo_iO-2ii7ufcn4CIVNnWIP26XYG/view)

**Scalars NPZ.** These are per-footprint CERES and auxiliary values. I decoded the archive directory of `A2015253.1225.npz` (id `1gdgiYeQzrzSGLbSBDTnINcn9tC2mmbHE`, 2015 Atlantic Sep scalars, 5 KB). It contains exactly 23 arrays:
- CERES flux and radiance: `Fsw, Isw, Flw, Ilw`
- `mSumIsw, stdSumIsw, mSumIlw, stdSumIlw`
- `mDFNC, stdDFNC`
- Cloud: `CF, meanCTH, stdCTH, LiqPhsf`
- Geometry: `VZA, SZA, RAA`
- Location and aerosol: `Lat, Lon, AOD`
- Flags: `Clrf, glintf, landf`

So **glint (`glintf`), land fraction (`landf`), solar/view geometry (SZA, VZA, RAA), AOD and lat/lon are all present per footprint.** — [A2015253.1225.npz](https://drive.google.com/file/d/1gdgiYeQzrzSGLbSBDTnINcn9tC2mmbHE/view)

**What the advisor's `Details_of_npz.py` shows about images.**
- The file types are `images`, `scalars` and `moments`.
- For images, it reads `ImSW` and `ImLW`, indexed `[footprint, channel, :, :]`. Channel 1 of SW is "green light (0.55 microns)", channel 10 of SW is the "Cirrus cloud detection channel (1.375 microns)", and channel 11 of LW is the "atmospheric window channel (11 microns)".
- For scalars, it plots `Isw` vs `Fsw` coloured by `glintf` and counts samples as non-NaN `Fsw`.
- The original path is `/userdata/eeytan/testdata/2014/Pacific/mrch/test/images/A2014060.0245.npz`.

— [Details_of_npz.py](https://drive.google.com/file/d/1LHz36f4xi2c25RcN7OF2pmer4o4pbWk3/view)

**Advisor's newest training recipe.** This is the closest thing to advisor guidance on Drive. `train_Allsky_moments_narrow_values.py`:
- loads `projects/F1km/ALLSKY_newDat/mrchsepjunedec_2014-2015_Pacific.npz`;
- trains only on footprints with all-sky flux ≥ 400 (`train_label_low = 400 # limit the training to scenes with a minimal all-sky flux`), with an upper outlier cut;
- uses base features SZA, VZA, RAA and the MODIS bands 0.47, 0.66, 0.86, 1.375, 1.24, 2.13, 3.75, …;
- uses `trainprec=0.7`, `validation_split=0.1` and `Nepochs=1000`, with custom early stopping on delta-RMSE in original units (threshold 5e-5, patience 3);
- offers label transforms 0 = none, 1 = log1p + weighted MSE, 2 = Yeo-Johnson;
- builds a Dense MLP with up to 4 hidden layers and also fits a linear-regression baseline.

— [train_Allsky_moments_narrow_values.py](https://drive.google.com/file/d/1hXHhQo_iO-2ii7ufcn4CIVNnWIP26XYG/view)

### Inferences
- `Data_toGuy.zip` most likely holds the `test/` subset (images, moments, scalars) for Pacific March 2014. The local repo's `data/paired/Pacific_2014-2015/images/` and `data/test/` sets (6 image granules from 2014-03-01 and 2014-03-31) match the path in `Details_of_npz.py`, and the local `Documentation/plan_CV.txt` describes them in detail (see section 2). Both the timestamps (zip uploaded 2026-07-15, the same day `Details_of_npz.py` was uploaded) and the paths are consistent with this, but I could not confirm the zip's contents directly.
- **Pixel-to-footprint mapping:** each footprint is a fixed 27×55 MODIS pixel patch (about 1 km per pixel), and footprint index i is aligned across images, moments and scalars. **No PSF weights and no explicit footprint/pixel index arrays** were found on Drive.
- Scalars contain `Isw`/`Ilw` (CERES radiances). Using them as model inputs would leak the label, because Fsw is computed from Isw. The repo already recorded this lesson in `Thesis_Goal.md`.

### Gaps
- I could not list or read inside `Data_toGuy.zip` (1.2 GB), `March_2014N15_momentsNscalars.zip` or `mrchsepjunedec_2014-2015_Pacific.zip`, because the connector only returns base64 content and these files are far too large. To confirm, download the zip manually and run `unzip -l`.
- No README, readme doc or written advisor instructions accompany the data on Drive. The meaning of `glintf`, `landf`, `Clrf` and `LiqPhsf` (fraction? flag?), the units, and the meaning of `FOVorientation` are not documented there.
- I did not count the exact number of NPZ files per month folder (the search was paginated). Sizes range from about 5 KB to about 1 MB each.

## 2. Example MODIS/CERES image files usable by a UNet

### Takeaway
On Drive, the only directly downloadable NPZ files are per-footprint **moments** (N×11 / N×16 statistics) and **scalars** (N per variable). They are not images. No `images/` folder, PNG or JPG MODIS imagery exists unpacked on Drive. The image cubes (`ImSW`, `ImLW`, `Imviews`) live only in the zip and in the local repo copy.

### Cited Findings
- Every month folder under `Old_codes_data/2014|2015/Atlantic|Pacific/<month>/` contains only `moments` and `scalars` subfolders. — [2015/Atlantic/sep folder](https://drive.google.com/drive/folders/1EmLxs9WIIdVeAUQJ5T64IeOvGyEsFUBa); [2014/Pacific/mrch folder](https://drive.google.com/drive/folders/1SLnZ8vM7lWhb7jJrEDs2QaSdmAwmnMF7)
- Example granule files on Drive:

  | Region and month | Granules | Folder id |
  |---|---|---|
  | 2015 June | A2015166.0240, A2015169.1810, A2015170.1855, A2015171.0300, A2015170.2020, A2015176.1940, A2015178.0450 | `1EPglu1MVtIzndOwVAT4H43C_JEIPsyNg` |
  | 2015 Atlantic Sep scalars | A2015246.1225 … A2015272.1440 | `1ZQhd_SHy_7sHlTZZvmUA8TkqPr6Ue40u` |
  | 2015 Atlantic Sep moments | A2015249.1600, A2015257.1720, A2015261.1655 | `1G8hC9_SlcHc82nUrIgAR2w-EwjJtLlEG` |
  | 2015 March | A2015061 … A2015090 | `1m-5fY7wj7SX8vIzJRwHge1zli1FNcsdV`, `1CTUAoNW5CH9n4KC_tjy9FlETmUjLq7x5` |
  | 2015 December | A2015336 … A2015362 | `12XFGCK-5UoQukLiw7OP36Bd-Ni_-FYZN` |

  — [Drive search](https://drive.google.com/file/d/1Ar2lUR0vj2PiEIgFDMFbVmSJ1nlNsswu/view)
- `Details_of_npz.py` confirms that image NPZs exist with the arrays `ImSW` and `ImLW`, laid out as [footprint, channel, y, x]. — [Details_of_npz.py](https://drive.google.com/file/d/1LHz36f4xi2c25RcN7OF2pmer4o4pbWk3/view)
- Local repo, not Drive (`Documentation/plan_CV.txt`, verified there): **exactly 6 image granules** exist, holding 11,917 footprints and about 2.04 GB in total.

  | Granule | Footprints |
  |---|---|
  | A2014060.0245 | 9 |
  | A2014060.0415 | 3,810 |
  | A2014060.0530 | 670 |
  | A2014090.2240 | 4,482 |
  | A2014090.2350 | 502 |
  | A2014090.2355 | 2,444 |

  Arrays in each file:
  - `Imviews` (N,3,27,55)
  - `ImSW` (N,11,27,55)
  - `ImLW` (N,16,27,55), all float32
  - `FOVorientation` (N,), a 0/1 flag split about 50/50

  Each 27×55 patch equals one CERES footprint at about 1 km. Files are index-aligned with moments and scalars, so the per-footprint labels `Fsw`/`Flw` are aligned with the image patches. `CF` and `Clrf` are on a 0–100 scale. No PSF map exists. — local file `/home/user/CTZ_Research/Documentation/plan_CV.txt` (lines 53–120)
- Local `data/` is **not present** in this checkout. `ls data` fails, so the images cannot be inspected here. — local check of `/home/user/CTZ_Research`

### Inferences
- **For a UNet:** the input is 30 channels on a 27×55 grid (3 view + 11 SW + 16 LW). The label is one scalar per patch (CERES Fsw or Flw), which naturally gives an aggregation loss: mean (or PSF-weighted mean) of the predicted pixel fluxes compared with the footprint flux.
- The patches are per footprint, not contiguous swaths. Neighbouring CERES footprints overlap, so patches from the same granule share pixels. Train/test splits must therefore be made by granule, date or region, never by random footprint.
- About 12k image footprints from only 2 dates is tiny compared with the roughly 5M tabular footprints. More image granules would have to be requested from the advisor.

### Gaps
- No PNG or JPG quick-looks of MODIS/CERES scenes were found on Drive.
- I could not confirm whether `Data_toGuy.zip` contains more than the 6 image granules.
- `Imviews` channel semantics (probably SZA/VZA/RAA per pixel) and `FOVorientation` are undocumented on Drive.

## 3. Paper summaries and literature review for the thesis on Drive

### Takeaway
**None found.** Searches in Hebrew and English returned only unrelated personal or course material: wedding documents, BGU course summaries, school lab reports and similar. The chats "סיכומי מאמרים התחלתי" and "סקירת ספרות לתזה" evidently did not save documents to Drive. The only thesis-adjacent non-code items are an audio file and an almost empty Colab notebook.

### Cited Findings
- `Invisible_Cloud_Halos_Rival_CO2.m4a`, id `1j7yqd_Gw4RyGP4oAJmRqkkcyCtv7eOYG`, 35.6 MB, owned by Guy, 2026-07-05. Judging by its title, this is probably an audio overview of the CTZ / "twilight zone" paper (the halo LW effect compared with CO₂). The content was not transcribed. — [file](https://drive.google.com/file/d/1j7yqd_Gw4RyGP4oAJmRqkkcyCtv7eOYG/view)
- `CTZ_Research.ipynb` (Colab), id `1OwlZZ3FY2fv_LTf7eunenjngRjK2rSC9`, 1.6 KB, 2026-07-16. Its size suggests it is essentially empty; the connector could not read the mime type. — [file](https://colab.research.google.com/drive/1OwlZZ3FY2fv_LTf7eunenjngRjK2rSC9)
- Searches with no thesis-relevant hits:
  - `fullText contains` 'CERES', 'MODIS', 'סקירת ספרות', 'twilight zone', 'סיכום', 'מאמר', 'Koren', 'Altaratz', 'Loeb', 'cloud transition', 'radiative flux', 'footprint', 'UNet', 'aerosol', 'shortwave'
  - `title contains` 'literature', 'Thesis', 'תזה', 'paper', 'summary', 'review', 'CTZ', 'Eytan'

  — Drive search, 2026-10-29

### Inferences
- The literature lessons that do exist live **in the local repo**: `/home/user/CTZ_Research/Documentation/LITERATURE_LESSONS.md` (204 lines; sections on U-Net for per-pixel regression, validation without 1 km truth, aggregating in flux rather than in inputs, regularization creating an artificial CTZ, synthetic known-truth tests, geometry-dependence of residuals, data diversity, coarse-residual diagnostics) and `/home/user/CTZ_Research/Thesis_Goal.md`. The planner should use those, not Drive, for paper lessons.
- Per-paper summaries (method, data, result, lessons) cannot be produced from Drive because no such documents exist there.

### Gaps
- Paper PDFs, for example Eytan et al. on the CTZ, were not found on Drive.
- No explicit thesis requirements document (proposal, committee notes or advisor email-as-doc) was found on Drive. The only implicit guidance is the advisor's code choices (section 1). The repo's `Thesis_Goal.md` states the goal: a 1 km all-sky TOA flux map, with MODIS-only inputs and CERES used as label and constraint.
