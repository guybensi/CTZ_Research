"""Candidate methods of PLAN.md section 3, one module per method family.

Every module registers its models in flux1km.models.MODELS on import and can be run on its own:

    python3 -m flux1km.methods.<module> --demo            # synthetic data with known 1 km truth
    python3 -m flux1km.methods.<module> --data-root DIR   # real granule triplets

| Module             | Registered models          | Direction |
|--------------------|----------------------------|-----------|
| m0_legacy          | M0-allsky37, M0-advisor    | legacy footprint ANNs (M0-ridge lives in models.py) |
| m1_guy             | M1-gbm, M1-ann             | improved Guy model, footprint-level |
| (models.py)        | M2, B-clim, B-const, B-bagpixel | pixel polynomial from moments, null baselines |
| m3_pixel_mlp       | M3                         | per-pixel MLP, aggregation loss |
| m4_context_gbm     | M4                         | boosted trees + context features, aggregate gradient |
| m5_unet            | M5                         | UNet, aggregation loss + edge-aware smoothness |
| m6a_cnn_tabular    | M6a                        | CNN + tabular branch |
| m6b_mlp_context    | M6b                        | pixel MLP + residual context encoder |
| m6c_unet_from_m2   | M6c                        | UNet initialised from M2 |
| m6d_distill_m1     | M6d                        | soft distillation from M1 |
| m6e_footprint_norm | M6e                        | post-hoc per-footprint normalisation (reads test labels) |
| m6f_pseudo_labels  | M6f                        | mediated pseudo-labels from M1 |
| m7_atprk           | M7                         | area-to-point regression kriging (reads test labels) |
| m8_nb2bb           | M8                         | narrowband -> broadband -> ADM |
| m9_homogeneous     | M9, M9-local               | homogeneous-footprint transfer (Gao) |
| m10_gp             | M10, M10-cmp               | GP aggregate models with RFF (VBAgg, mediated) |

Neural modules import PyTorch only inside fit (methods._torch); tree models import scikit-learn
only inside fit.
"""
from . import (  # noqa: F401
    m0_legacy,
    m1_guy,
    m3_pixel_mlp,
    m4_context_gbm,
    m5_unet,
    m6a_cnn_tabular,
    m6b_mlp_context,
    m6c_unet_from_m2,
    m6d_distill_m1,
    m6e_footprint_norm,
    m6f_pseudo_labels,
    m7_atprk,
    m8_nb2bb,
    m9_homogeneous,
    m10_gp,
)
