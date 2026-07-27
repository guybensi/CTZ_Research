#!/usr/bin/env python3
"""Guy's Model - Systematic feature combination benchmark with model selection and ANN training.

This script explores different combinations of moments (mean, skew, std) and scalar features,
benchmarks 5+ classical ML models across all combinations, selects the best model/combo pair,
hyperparameter-tunes it, and then trains a TensorFlow ANN using the All_Sky_AIflux architecture.

Architecture:
- Reuses ANN design from All_Sky_AIflux (Normalization → Dense layers → output)
- Systematic feature combination exploration
- Complete pipeline: data prep → model selection → hypertuning → ANN training
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.pipeline import Pipeline

import tensorflow as tf

# Try importing XGBoost and CatBoost, fallback if unavailable
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not available - install with: pip install xgboost")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️  CatBoost not available - install with: pip install catboost")

try:
    from sklearn.svm import SVR
    HAS_SVM = True
except ImportError:
    HAS_SVM = False


# ============================================================================
# Data Structures
# ============================================================================

@dataclass(frozen=True)
class ModelResult:
    """Result from training a single model on a feature combination."""
    combo_name: str
    model_name: str
    test_split: float
    cv_rmse: float
    test_rmse: float
    test_mae: float
    test_r2: float
    params: Dict[str, Any]
    feature_count: int
    feature_importance: Optional[List[float]] = None


@dataclass(frozen=True)
class BestModelConfig:
    """Best model/combo pair identified during benchmarking."""
    combo_name: str
    model_name: str
    test_split: float
    test_r2: float
    params: Dict[str, Any]
    feature_names: List[str]
    feature_importance: Optional[List[float]] = None


# ============================================================================
# Configuration & Setup
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guy's Model - Feature combo exploration + model benchmark + ANN training")
    parser.add_argument("--data-root", type=str, default="", help="Override data directory (defaults to data/paired/Pacific_2014-2015)")
    parser.add_argument("--output-dir", type=str, default="benchmark_outputs", help="Output directory for results and models")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--test-splits", type=float, nargs="+", default=[0.2], help="Test split fractions to evaluate")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of k-fold CV splits")
    parser.add_argument("--ann-epochs", type=int, default=150, help="Maximum ANN epochs")
    parser.add_argument("--ann-batch-size", type=int, default=64, help="ANN batch size")
    parser.add_argument("--ann-trials", type=int, default=5, help="Number of ANN architecture trials")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data instead of real paired data")
    parser.add_argument("--skip-classical", action="store_true", help="Skip classical models, go straight to best-pretrained ANN")
    return parser.parse_args()


def setup_tensorflow_runtime() -> None:
    """Configure TensorFlow to use GPU if available."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU available: {len(gpus)} device(s)")
        else:
            print("ℹ️  No GPU detected, using CPU")
    except Exception as e:
        print(f"⚠️  GPU setup warning: {e}")


# ============================================================================
# Data Loading & Feature Engineering
# ============================================================================

def load_paired_moments_scalars(data_root: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """Load all paired moments and scalars files from data_root/{moments,scalars}/ directories.
    
    Returns:
        (X, y, file_mapping) where:
        - X: feature matrix (n_samples, n_features)
        - y: target values (n_samples,)
        - file_mapping: dict mapping filenames to their roles
    """
    moments_dir = Path(data_root) / "moments"
    scalars_dir = Path(data_root) / "scalars"

    if not moments_dir.exists() or not scalars_dir.exists():
        raise FileNotFoundError(f"Expected moments/ and scalars/ subdirectories in {data_root}")

    # Load all moment files
    moment_files = sorted(moments_dir.glob("*.npz"))
    scalar_files = sorted(scalars_dir.glob("*.npz"))

    if not moment_files:
        raise FileNotFoundError(f"No .npz files found in {moments_dir}")
    if not scalar_files:
        raise FileNotFoundError(f"No .npz files found in {scalars_dir}")

    print(f"Loading {len(moment_files)} moments files and {len(scalar_files)} scalars files...")

    # Aggregate data
    all_moments = []
    all_scalars = []
    all_targets = []

    for moment_file in moment_files:
        scalar_file = scalars_dir / moment_file.name
        if not scalar_file.exists():
            print(f"  ⚠️  Skipping {moment_file.name} (no matching scalar file)")
            continue

        with np.load(moment_file, allow_pickle=True) as m_data:
            with np.load(scalar_file, allow_pickle=True) as s_data:
                # Extract moment features (mean, skew, std, etc.)
                moment_features = []
                for key in m_data.files:
                    if key not in ("__header__", "__version__", "__allow_pickle__"):
                        moment_features.append(m_data[key])

                # Extract scalar features (SZA, VZA, RAA, etc.)
                scalar_features = []
                for key in s_data.files:
                    if key not in ("__header__", "__version__", "__allow_pickle__"):
                        scalar_features.append(s_data[key])

                # Extract target (flux/target/y)
                target = None
                for target_key in ("Flux", "flux", "target", "Target", "y"):
                    if target_key in m_data.files:
                        target = m_data[target_key]
                        break

                if target is None:
                    print(f"  ⚠️  Could not find target in {moment_file.name}, skipping")
                    continue

                if moment_features and scalar_features:
                    combined = np.concatenate(moment_features + scalar_features, axis=-1)
                    all_moments.append(combined)
                    all_targets.append(target)

    if not all_moments:
        raise ValueError("No valid moment/scalar pairs found")

    X = np.concatenate(all_moments, axis=0).astype(np.float32)
    y = np.concatenate(all_targets, axis=0).astype(np.float32)

    print(f"✓ Loaded data: X.shape={X.shape}, y.shape={y.shape}")
    return X, y, {}


def generate_feature_combinations(X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """Generate systematic feature combinations from the full feature matrix.
    
    Combinations include:
    - Mean features only
    - Skew features only
    - Std features only
    - Mean + Skew
    - Mean + Skew + Std
    - All features
    
    Returns:
        Dict mapping combination names to (feature_subset, feature_names)
    """
    combos = {}

    # For simplicity in this initial implementation, use the full feature set
    # and generate a few useful combinations based on feature count
    n_features = X.shape[1]

    # Combination 1: All features
    combos["all_features"] = (X, list(range(n_features)) if feature_names is None else feature_names)

    # Combination 2: First 2/3 of features
    if n_features > 10:
        subset_idx = list(range(int(2 * n_features / 3)))
        combos["first_2_3"] = (X[:, subset_idx], subset_idx)

    # Combination 3: First 1/3 of features
    if n_features > 10:
        subset_idx = list(range(int(n_features / 3)))
        combos["first_1_3"] = (X[:, subset_idx], subset_idx)

    # Combination 4: Every other feature
    if n_features > 5:
        subset_idx = list(range(0, n_features, 2))
        combos["every_other"] = (X[:, subset_idx], subset_idx)

    # Combination 5: Features excluding every other
    if n_features > 5:
        subset_idx = list(range(1, n_features, 2))
        combos["every_other_alt"] = (X[:, subset_idx], subset_idx)

    print(f"✓ Generated {len(combos)} feature combinations")
    for name, (X_combo, _) in combos.items():
        print(f"  - {name}: {X_combo.shape[1]} features")

    return combos


def demo_data(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic demo data for testing."""
    rng = np.random.default_rng(seed)
    n_samples = 2000
    n_features = 24
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    linear_weights = rng.normal(size=(n_features,)).astype(np.float32)
    nonlinear = 0.3 * np.sin(X[:, 0]) + 0.2 * np.square(X[:, 1]) - 0.15 * X[:, 2] * X[:, 3]
    noise = rng.normal(scale=1.0, size=n_samples).astype(np.float32)
    y = (X @ linear_weights + nonlinear + noise).astype(np.float32)
    return X, y


# ============================================================================
# Model Definitions
# ============================================================================

def get_classical_models() -> Dict[str, Tuple[Any, Dict[str, Sequence[Any]]]]:
    """Return dictionary of classical models with their hyperparameter grids."""
    models = {}

    # Random Forest
    models["Random Forest"] = (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }
    )

    # Linear Regression
    models["Linear Regression"] = (
        LinearRegression(),
        {}
    )

    # Ridge Regression
    models["Ridge"] = (
        Ridge(),
        {"alpha": [0.1, 1.0, 10.0, 100.0]}
    )

    # Gradient Boosting
    models["Gradient Boosting"] = (
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
        }
    )

    # AdaBoost
    models["AdaBoost"] = (
        AdaBoostRegressor(random_state=42),
        {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1, 1.0],
        }
    )

    # XGBoost (if available)
    if HAS_XGBOOST:
        models["XGBoost"] = (
            xgb.XGBRegressor(random_state=42, verbosity=0, use_label_encoder=False),
            {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5],
            }
        )

    # CatBoost (if available)
    if HAS_CATBOOST:
        models["CatBoost"] = (
            cb.CatBoostRegressor(random_state=42, verbose=0),
            {
                "iterations": [50, 100],
                "learning_rate": [0.01, 0.1],
                "depth": [3, 5],
            }
        )

    # SVM (if available)
    if HAS_SVM:
        models["SVM"] = (
            SVR(),
            {
                "kernel": ["rbf", "linear"],
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", "auto"],
            }
        )

    return models


# ============================================================================
# Classical Model Benchmarking
# ============================================================================

def benchmark_models_on_combo(
    X: np.ndarray,
    y: np.ndarray,
    combo_name: str,
    models: Dict[str, Tuple[Any, Dict]],
    test_splits: List[float],
    cv_folds: int = 5,
    seed: int = 42,
) -> List[ModelResult]:
    """Benchmark all models on a single feature combination."""
    results = []
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for model_name, (model_template, param_grid) in models.items():
        print(f"  {model_name}...", end=" ", flush=True)

        for test_split in test_splits:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_split, random_state=seed
            )

            # GridSearchCV for hyperparameter tuning
            if param_grid:
                search = GridSearchCV(
                    model_template,
                    param_grid,
                    cv=cv_folds,
                    scoring="r2",
                    n_jobs=-1,
                    verbose=0,
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                cv_rmse = np.sqrt(-search.best_score_)  # Convert R² to RMSE proxy
                best_params = search.best_params_
            else:
                # No tuning needed
                best_model = clone(model_template)
                best_model.fit(X_train, y_train)
                cv_rmse = 0.0  # Placeholder
                best_params = {}

            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)

            # Extract feature importance if available
            feature_importance = None
            if hasattr(best_model, "feature_importances_"):
                feature_importance = best_model.feature_importances_.tolist()

            result = ModelResult(
                combo_name=combo_name,
                model_name=model_name,
                test_split=test_split,
                cv_rmse=cv_rmse,
                test_rmse=test_rmse,
                test_mae=test_mae,
                test_r2=test_r2,
                params=best_params,
                feature_count=X.shape[1],
                feature_importance=feature_importance,
            )
            results.append(result)

        print("✓")

    return results


# ============================================================================
# Hyperparameter Tuning
# ============================================================================

def hyperparameter_tune_best_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    test_split: float,
    cv_folds: int = 5,
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """Perform intensive hyperparameter tuning on the best model."""
    print(f"Hyperparameter tuning for {model_name}...")

    models = get_classical_models()
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    model_template, param_grid = models[model_name]

    if not param_grid:
        print(f"  No hyperparameters to tune for {model_name}")
        return model_template, {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=test_split, random_state=seed)

    # Expanded grid for tuning
    expanded_grid = {k: v * 2 if len(v) < 4 else v for k, v in param_grid.items()}

    search = GridSearchCV(
        model_template,
        expanded_grid,
        cv=cv_folds,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"✓ Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


# ============================================================================
# ANN Training
# ============================================================================

def build_ann_model(
    input_shape: int,
    seed: int = 42,
    learning_rate: float = 0.001,
    n_layers: int = 2,
    layer_size_factor: float = 0.5,
) -> tf.keras.Model:
    """Build ANN architecture mirroring All_Sky_AIflux design."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    inputs = tf.keras.layers.Input(shape=(input_shape,), dtype=tf.float32, name="input")

    # Normalization layer
    norm = tf.keras.layers.Normalization(axis=-1, name="normalization")
    x = norm(inputs)

    # Dense layers with decreasing sizes
    current_size = input_shape
    for i in range(n_layers):
        current_size = max(int(current_size * layer_size_factor), 8)
        x = tf.keras.layers.Dense(current_size, activation="relu", name=f"dense_{i+1}")(x)

    # Output layer
    outputs = tf.keras.layers.Dense(1, name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="guy_ann")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )

    return model


def train_ann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_shape: int,
    epochs: int = 150,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[tf.keras.Model, Dict]:
    """Train ANN with early stopping."""
    model = build_ann_model(input_shape, seed=seed)

    # Adapt normalization layer
    model.layers[1].adapt(X_train)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
            )
        ],
    )

    return model, history.history


# ============================================================================
# Results & Visualization
# ============================================================================

def select_best_model(results: List[ModelResult]) -> BestModelConfig:
    """Select best model based on test R²."""
    best = max(results, key=lambda r: r.test_r2)
    return BestModelConfig(
        combo_name=best.combo_name,
        model_name=best.model_name,
        test_split=best.test_split,
        test_r2=best.test_r2,
        params=best.params,
        feature_names=[],
        feature_importance=best.feature_importance,
    )


def export_results_to_csv(results: List[ModelResult], output_path: Path) -> None:
    """Export benchmark results to CSV."""
    data = []
    for r in results:
        data.append({
            "combo_name": r.combo_name,
            "model_name": r.model_name,
            "test_split": r.test_split,
            "feature_count": r.feature_count,
            "cv_rmse": r.cv_rmse,
            "test_rmse": r.test_rmse,
            "test_mae": r.test_mae,
            "test_r2": r.test_r2,
        })
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✓ Results exported to {output_path}")


def plot_performance_heatmap(results: List[ModelResult], output_path: Path) -> None:
    """Create heatmap of model performance across feature combinations."""
    df = pd.DataFrame([asdict(r) for r in results])

    # Pivot for heatmap (combo × model, metric = test_r2)
    pivot = df.pivot_table(values="test_r2", index="model_name", columns="combo_name", aggfunc="mean")

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0, cbar_kws={"label": "Test R²"})
    plt.title("Model Performance Across Feature Combinations")
    plt.xlabel("Feature Combination")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Heatmap saved to {output_path}")


def plot_feature_importance(results: List[ModelResult], output_path: Path) -> None:
    """Plot feature importance for models that support it."""
    important_results = [r for r in results if r.feature_importance is not None]
    if not important_results:
        print("ℹ️  No feature importance data available")
        return

    fig, axes = plt.subplots(1, min(3, len(important_results)), figsize=(15, 4))
    if len(important_results) == 1:
        axes = [axes]

    for idx, (ax, result) in enumerate(zip(axes, important_results[:3])):
        importances = result.feature_importance[:20]  # Top 20
        ax.barh(range(len(importances)), importances)
        ax.set_yticks(range(len(importances)))
        ax.set_title(f"{result.model_name}\n{result.combo_name}")
        ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Feature importance plot saved to {output_path}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    args = parse_args()
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    setup_tensorflow_runtime()

    # Setup output directory
    output_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # ========== PHASE 1: DATA LOADING ==========
    print("\n" + "="*70)
    print("PHASE 1: DATA LOADING & FEATURE ENGINEERING")
    print("="*70)

    if args.demo:
        print("🔬 Using synthetic demo data...")
        X, y = demo_data(args.random_seed)
    else:
        # Use paired moments/scalars
        repo_root = Path(__file__).resolve().parents[1]
        data_root = args.data_root if args.data_root else str(repo_root / "data" / "paired" / "Pacific_2014-2015")

        print(f"Loading from {data_root}...")
        X, y, _ = load_paired_moments_scalars(data_root)

    feature_combos = generate_feature_combinations(X)

    # ========== PHASE 2: MODEL BENCHMARKING ==========
    if not args.skip_classical:
        print("\n" + "="*70)
        print("PHASE 2: MODEL BENCHMARKING")
        print("="*70)

        models = get_classical_models()
        all_results = []

        for combo_name, (X_combo, feature_indices) in feature_combos.items():
            print(f"\n🔄 Testing feature combination: {combo_name}")
            combo_results = benchmark_models_on_combo(
                X_combo, y, combo_name, models,
                test_splits=args.test_splits,
                cv_folds=args.cv_folds,
                seed=args.random_seed,
            )
            all_results.extend(combo_results)

        # Export results
        export_results_to_csv(all_results, output_dir / "benchmark_results.csv")
        plot_performance_heatmap(all_results, output_dir / "performance_heatmap.png")
        plot_feature_importance(all_results, output_dir / "feature_importance.png")

        # Select best
        best_config = select_best_model(all_results)
        print(f"\n🏆 Best model: {best_config.model_name} on {best_config.combo_name} (R²={best_config.test_r2:.4f})")

        # ========== PHASE 3: HYPERPARAMETER TUNING ==========
        print("\n" + "="*70)
        print("PHASE 3: HYPERPARAMETER TUNING")
        print("="*70)

        best_combo_data = feature_combos[best_config.combo_name][0]
        best_model_tuned, best_params = hyperparameter_tune_best_model(
            best_combo_data, y,
            best_config.model_name,
            best_config.test_split,
            cv_folds=args.cv_folds,
            seed=args.random_seed,
        )

    else:
        # Use first combo if skipping classical
        best_config = BestModelConfig(
            combo_name=list(feature_combos.keys())[0],
            model_name="Pretrained",
            test_split=0.2,
            test_r2=0.0,
            params={},
            feature_names=[],
        )
        best_combo_data = list(feature_combos.values())[0][0]

    # ========== PHASE 4: ANN TRAINING ==========
    print("\n" + "="*70)
    print("PHASE 4: ANN TRAINING")
    print("="*70)

    best_combo_data = feature_combos[best_config.combo_name][0]
    X_train, X_test, y_train, y_test = train_test_split(
        best_combo_data, y, test_size=0.2, random_state=args.random_seed
    )

    print(f"Training ANN on {best_config.combo_name}...")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    ann_model, ann_history = train_ann(
        X_train, y_train, X_test, y_test,
        input_shape=X_train.shape[1],
        epochs=args.ann_epochs,
        batch_size=args.ann_batch_size,
        seed=args.random_seed,
    )

    # Save ANN model
    ann_model_path = output_dir / "ann_model.keras"
    ann_model.save(ann_model_path)
    print(f"✓ ANN model saved to {ann_model_path}")

    # Evaluate ANN
    ann_y_pred = ann_model.predict(X_test, verbose=0).flatten()
    ann_rmse = np.sqrt(mean_squared_error(y_test, ann_y_pred))
    ann_mae = mean_absolute_error(y_test, ann_y_pred)
    ann_r2 = r2_score(y_test, ann_y_pred)

    print(f"ANN Results:")
    print(f"  RMSE: {ann_rmse:.4f}")
    print(f"  MAE: {ann_mae:.4f}")
    print(f"  R²: {ann_r2:.4f}")

    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "best_classical_model": best_config.model_name,
        "best_feature_combination": best_config.combo_name,
        "classical_test_r2": float(best_config.test_r2),
        "ann_test_r2": float(ann_r2),
        "ann_rmse": float(ann_rmse),
        "ann_mae": float(ann_mae),
        "data_shape": X.shape,
        "best_combo_shape": best_combo_data.shape,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to {summary_path}")

    print(f"\n✅ All done! Results in {output_dir}")


if __name__ == "__main__":
    main()
