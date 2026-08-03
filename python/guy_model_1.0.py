#!/usr/bin/env python3
"""Guy Model v1.0 - predicts Fsw (shortwave flux, W/m^2) at each satellite
field of view from per-scene moments (mean/std/skew/kurtosis of the SW and LW
spectra) and scalar geometry/cloud/aerosol variables (VZA/SZA/RAA, Isw/Ilw,
CF, CTH, AOD, Lat/Lon, etc.), trained on paired moments/scalars NPZ data.
It is the recipe locked in by guys_benchmark_models.py's model/feature-combo
sweep: rank the 129 candidate features by Random Forest importance, keep the
top 20, then fit both a Ridge baseline and an ANN on that fixed subset so the
two model types are compared head-to-head. This version has no
--feature-source option; see guy_model_2.0.py for the variant that can
instead run on the restricted 37-feature All_Sky_AIflux-style pool.

python3 python/guys_benchmark_models.py --feature-selection importance identified:
- Feature selection: top 20 features by Random Forest importance (best speed/accuracy tradeoff)
- Classical baseline: Ridge (alpha=1.0)
- ANN on the same 20 features outperforms the classical baseline

This script re-derives the same top-20 feature subset (Random Forest importance
ranking is deterministic given the fixed random seed) and trains both the Ridge
baseline and the ANN on it, without re-running the full model/combo sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guys_benchmark_models import (
    demo_data,
    get_feature_importances_from_ensemble,
    load_paired_moments_scalars,
    train_ann,
)

RANDOM_SEED = 42
TOP_N_FEATURES = 20
RIDGE_ALPHA = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the final Ridge + ANN model on top-20 importance-selected features")
    parser.add_argument("--data-root", type=str, default="", help="Override data directory (defaults to data/paired/Pacific_2014-2015)")
    parser.add_argument("--output-dir", type=str, default="benchmark_outputs/final_model", help="Output directory for the trained model and metrics")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data instead of real paired data")
    parser.add_argument("--ann-epochs", type=int, default=150, help="Maximum ANN epochs")
    parser.add_argument("--ann-batch-size", type=int, default=64, help="ANN batch size")
    return parser.parse_args()


def select_top_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    importances = get_feature_importances_from_ensemble(X, y, seed=RANDOM_SEED)
    return np.argsort(-importances)[:TOP_N_FEATURES]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        print("🔬 Using synthetic demo data...")
        X, y = demo_data(RANDOM_SEED)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        data_root = args.data_root if args.data_root else str(repo_root / "data" / "paired" / "Pacific_2014-2015")
        print(f"Loading from {data_root}...")
        X, y, _ = load_paired_moments_scalars(data_root)

    print("📊 Selecting top features by Random Forest importance...")
    top_indices = select_top_features(X, y)
    X_top = X[:, top_indices]
    print(f"✓ Using {X_top.shape[1]} of {X.shape[1]} features: {top_indices.tolist()}")

    X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=RANDOM_SEED)

    # Classical baseline: Ridge
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    ridge_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, ridge_pred))),
        "mae": float(mean_absolute_error(y_test, ridge_pred)),
        "r2": float(r2_score(y_test, ridge_pred)),
    }
    print(f"Ridge  -> RMSE={ridge_metrics['rmse']:.4f} MAE={ridge_metrics['mae']:.4f} R²={ridge_metrics['r2']:.4f}")

    joblib.dump(ridge, output_dir / "ridge_model.joblib")
    joblib.dump(scaler, output_dir / "ridge_scaler.joblib")

    # ANN (best performer)
    print("Training ANN...")
    ann_model, _ = train_ann(
        X_train, y_train, X_test, y_test,
        input_shape=X_train.shape[1],
        epochs=args.ann_epochs,
        batch_size=args.ann_batch_size,
        seed=RANDOM_SEED,
    )
    ann_pred = ann_model.predict(X_test, verbose=0).flatten()
    ann_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, ann_pred))),
        "mae": float(mean_absolute_error(y_test, ann_pred)),
        "r2": float(r2_score(y_test, ann_pred)),
    }
    print(f"ANN    -> RMSE={ann_metrics['rmse']:.4f} MAE={ann_metrics['mae']:.4f} R²={ann_metrics['r2']:.4f}")

    ann_model.save(output_dir / "ann_model.keras")

    summary = {
        "feature_selection": "top_20_random_forest_importance",
        "top_feature_indices": top_indices.tolist(),
        "ridge": {"alpha": RIDGE_ALPHA, **ridge_metrics},
        "ann": ann_metrics,
        "data_shape": list(X.shape),
    }
    with open(output_dir / "final_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved model artifacts and summary to {output_dir}")


if __name__ == "__main__":
    main()
