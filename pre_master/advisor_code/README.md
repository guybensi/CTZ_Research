# Advisor's data-building and training code (Eshkol Eytan, received Nov 2026)

Kept unmodified, for reference. It is not run from this repository: it reads raw NASA files from
paths on the advisor's server.

| File | What it does |
|---|---|
| `Generate_AllSky_2.py` | Driver. Loads the CERES PSF (`ceres_psf.csv`, not included) and resamples it to the 27x55 patch. Reads the CERES SSF variables per month and region, then runs `MOD_CERES_allsky` on every MODIS granule in parallel. |
| `Twilight_module.py` | Helper module. `MOD_CERES_allsky` (line ~2686) builds the `scalars/`, `moments/` and `images/` NPZ triplets from MYD021KM + MYD03 + MYD35 + MYD06 and the SSF footprints. `calc4moms` and `readMODIS` are the helpers it uses. |
| `train_Allsky_moments_narrow_values.py` | The advisor's current footprint-level ANN (moments of 13 channels plus geometry). It uses a random 70/30 split and an optional `Fsw >= 400` training/test cut. |
| `Plotting_AllSky_ML.py` | Comparison figure of two trained models (needs `esh_fun_module`, not included). |

What this code tells us about the data, and the problems found in it, are documented in
[PLAN.md](../../master/PLAN.md) section 2.3.
