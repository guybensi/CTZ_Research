#!/usr/bin/env python3
"""GPU-aware regression benchmarking pipeline for CTZ_Research.

This script keeps the original regression idea intact while making it easier to
compare multiple classical models, test several train/test splits, tune each
model with k-fold cross-validation, and benchmark a configurable ANN.

It is designed to work before the final dataset is available:
- --demo generates a realistic synthetic regression problem.
- CSV input uses the last column or a known target name as the label.
- NPZ input accepts common target keys such as Flux or target.

Outputs are written to the directory selected by --output-dir.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_CANDIDATES = ("Flux", "flux", "target", "Target", "y", "Y", "label", "Label")


@dataclass(frozen=True)
class ModelSummary:
    """Single benchmark result for a model, split, and hyperparameter set."""

    model_name: str
    split: float
    cv_rmse: float
    test_rmse: float
    test_mae: float
    test_r2: float
    params: Dict[str, Any]


@dataclass(frozen=True)
class BestRun:
    """Best configuration identified during the benchmark."""

    model_name: str
    split: float
    score: float
    params: Dict[str, Any]
    artifact_path: str


@dataclass(frozen=True)
class ClassicalSpec:
    model_name: str
    estimator: Pipeline
    param_grid: Dict[str, Sequence[Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark classical ML models and an ANN for CTZ data")
    parser.add_argument("--data-path", type=str, default="", help="CSV, NPZ, or directory containing data")
    parser.add_argument("--output-dir", type=str, default="benchmark_outputs", help="Directory for metrics and artifacts")
    parser.add_argument("--demo", action="store_true", help="Use a synthetic demo dataset")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for splits and models")
    parser.add_argument("--test-splits", type=float, nargs="+", default=[0.2, 0.3], help="One or more holdout fractions to evaluate")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of k-fold splits for hyperparameter search")
    parser.add_argument("--ann-epochs", type=int, default=80, help="Maximum ANN epochs per trial")
    parser.add_argument("--ann-batch-size", type=int, default=64, help="Batch size used for ANN training")
    parser.add_argument("--ann-trials", type=int, default=8, help="Number of ANN hyperparameter combinations to try")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if TensorFlow cannot see a GPU")
    return parser.parse_args()


def setup_tensorflow_runtime(require_gpu: bool) -> None:
    """Enable GPU execution when TensorFlow is available."""

    try:
        import tensorflow as tf
    except Exception:
        if require_gpu:
            raise RuntimeError("TensorFlow is required when GPU enforcement is enabled")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if require_gpu:
            raise RuntimeError("No GPU detected. Run inside the GPU-enabled Docker image or disable require_gpu.")
        return

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def _first_existing_file(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _demo_dataset(seed: int) -> Tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    n_samples = 1800
    n_features = 24
    x = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    linear_weights = rng.normal(size=(n_features,)).astype(np.float32)
    nonlinear_term = 0.35 * np.sin(x[:, 0]) + 0.25 * np.square(x[:, 1]) - 0.15 * x[:, 2] * x[:, 3]
    noise = rng.normal(scale=1.0, size=n_samples).astype(np.float32)
    y = (x @ linear_weights + nonlinear_term + noise).astype(np.float32)
    return x, y, "synthetic_demo"


def _infer_target_column(columns: Iterable[str]) -> Optional[str]:
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def _load_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    frame = pd.read_csv(path)
    target_name = _infer_target_column(frame.columns)
    if target_name is None:
        target_name = frame.columns[-1]
    y = frame[target_name].to_numpy(dtype=np.float32)
    x = frame.drop(columns=[target_name]).to_numpy(dtype=np.float32)
    return x, y, str(path)


def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=True) as data:
        target_key = None
        for key in TARGET_CANDIDATES:
            if key in data.files:
                target_key = key
                break

        if target_key is None:
            raise ValueError(f"Could not infer a target column in {path}. Expected one of: {', '.join(TARGET_CANDIDATES)}")

        y = np.asarray(data[target_key], dtype=np.float32).reshape(-1)
        feature_arrays: List[np.ndarray] = []

        for key in data.files:
            if key == target_key:
                continue
            values = data[key]
            if values.dtype == object or not np.issubdtype(values.dtype, np.number):
                continue
            arr = np.asarray(values, dtype=np.float32)
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            else:
                arr = arr.reshape(arr.shape[0], -1)
            feature_arrays.append(arr)

        if not feature_arrays:
            raise ValueError(f"No numeric feature arrays found in {path}")

        sample_count = min(array.shape[0] for array in feature_arrays)
        x = np.concatenate([array[:sample_count] for array in feature_arrays], axis=1).astype(np.float32)
        y = y[:sample_count]
        return x, y, str(path)


def load_dataset(data_path: str, demo: bool, seed: int) -> Tuple[np.ndarray, np.ndarray, str]:
    if demo or not data_path:
        return _demo_dataset(seed)

    path = Path(data_path)
    if path.is_dir():
        candidate = _first_existing_file(sorted(path.rglob("*.csv")) + sorted(path.rglob("*.npz")))
        if candidate is None:
            raise FileNotFoundError(f"No .csv or .npz files found under {path}")
        path = candidate

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".npz":
        return _load_npz(path)

    raise ValueError(f"Unsupported data source: {path}")


def build_classical_specs(random_seed: int) -> List[ClassicalSpec]:
    return [
        ClassicalSpec(
            model_name="LinearRegression",
            estimator=Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            param_grid={"model__fit_intercept": [True, False]},
        ),
        ClassicalSpec(
            model_name="Ridge",
            estimator=Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
            param_grid={"model__alpha": [0.1, 1.0, 10.0, 25.0]},
        ),
        ClassicalSpec(
            model_name="Lasso",
            estimator=Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=10000, random_state=random_seed))]),
            param_grid={"model__alpha": [0.0005, 0.001, 0.01, 0.05]},
        ),
        ClassicalSpec(
            model_name="ElasticNet",
            estimator=Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(max_iter=10000, random_state=random_seed))]),
            param_grid={"model__alpha": [0.0005, 0.001, 0.01], "model__l1_ratio": [0.2, 0.5, 0.8]},
        ),
        ClassicalSpec(
            model_name="RandomForest",
            estimator=Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(random_state=random_seed, n_jobs=-1))]),
            param_grid={"model__n_estimators": [100, 200], "model__max_depth": [None, 12, 24], "model__min_samples_leaf": [1, 3]},
        ),
        ClassicalSpec(
            model_name="GradientBoosting",
            estimator=Pipeline([("scaler", StandardScaler()), ("model", GradientBoostingRegressor(random_state=random_seed))]),
            param_grid={"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.05, 0.1], "model__max_depth": [2, 3]},
        ),
    ]


def benchmark_classical_models(
    x: np.ndarray,
    y: np.ndarray,
    splits: Sequence[float],
    cv_folds: int,
    random_seed: int,
    output_dir: Path,
) -> Tuple[List[ModelSummary], Optional[BestRun]]:
    results: List[ModelSummary] = []
    best_run: Optional[BestRun] = None
    best_spec: Optional[ClassicalSpec] = None

    specs = build_classical_specs(random_seed)

    for split in splits:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=random_seed, shuffle=True)
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

        for spec in specs:
            search = GridSearchCV(
                estimator=clone(spec.estimator),
                param_grid=spec.param_grid,
                scoring="neg_root_mean_squared_error",
                cv=kfold,
                n_jobs=-1,
                refit=True,
            )
            search.fit(x_train, y_train)
            predictions = search.predict(x_test)
            test_rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
            test_mae = float(mean_absolute_error(y_test, predictions))
            test_r2 = float(r2_score(y_test, predictions))
            cv_rmse = float(-search.best_score_)

            result = ModelSummary(
                model_name=spec.model_name,
                split=float(split),
                cv_rmse=cv_rmse,
                test_rmse=test_rmse,
                test_mae=test_mae,
                test_r2=test_r2,
                params=normalize_value(search.best_params_),
            )
            results.append(result)

            if best_run is None or result.cv_rmse < best_run.score:
                best_run = BestRun(
                    model_name=spec.model_name,
                    split=float(split),
                    score=result.cv_rmse,
                    params=normalize_value(search.best_params_),
                    artifact_path=str(output_dir / "best_classical_model.joblib"),
                )
                best_spec = spec

    if best_run is not None and best_spec is not None:
        final_model = clone(best_spec.estimator)
        final_model.set_params(**best_run.params)
        final_model.fit(x, y)
        save_joblib(final_model, output_dir / "best_classical_model.joblib")

    return results, best_run


def build_ann_specs(ann_trials: int, base_batch_size: int) -> List[Dict[str, Any]]:
    base_batch = max(8, base_batch_size)
    base_grid = [
        {"layers": (128, 64), "dropout": 0.15, "learning_rate": 1e-3, "batch_size": max(8, base_batch // 2)},
        {"layers": (192, 96), "dropout": 0.20, "learning_rate": 1e-3, "batch_size": base_batch},
        {"layers": (256, 128), "dropout": 0.25, "learning_rate": 5e-4, "batch_size": base_batch},
        {"layers": (128, 64, 32), "dropout": 0.20, "learning_rate": 5e-4, "batch_size": max(8, base_batch // 2)},
        {"layers": (256, 128, 64), "dropout": 0.30, "learning_rate": 5e-4, "batch_size": base_batch * 2},
        {"layers": (160, 80), "dropout": 0.10, "learning_rate": 2e-3, "batch_size": base_batch},
        {"layers": (224, 112), "dropout": 0.20, "learning_rate": 1e-3, "batch_size": base_batch * 2},
        {"layers": (96, 48), "dropout": 0.10, "learning_rate": 2e-3, "batch_size": max(8, base_batch // 2)},
    ]
    return base_grid[: max(1, min(ann_trials, len(base_grid)))]


def build_ann_model(input_dim: int, spec: Dict[str, Any], keras: Any):
    regularizer = keras.regularizers.l2(1e-5)
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    for units in spec["layers"]:
        model.add(keras.layers.Dense(units, activation="relu", kernel_regularizer=regularizer))
        model.add(keras.layers.Dropout(spec["dropout"]))
    model.add(keras.layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=spec["learning_rate"]),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def benchmark_ann(
    x: np.ndarray,
    y: np.ndarray,
    splits: Sequence[float],
    random_seed: int,
    ann_epochs: int,
    ann_batch_size: int,
    ann_trials: int,
    require_gpu: bool,
    output_dir: Path,
) -> Tuple[Optional[ModelSummary], Optional[BestRun]]:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception:
        return None, None

    setup_tensorflow_runtime(require_gpu=require_gpu)
    tf.keras.utils.set_random_seed(random_seed)

    ann_specs = build_ann_specs(ann_trials, ann_batch_size)
    best_result: Optional[ModelSummary] = None
    best_run: Optional[BestRun] = None
    best_model: Optional[Any] = None
    best_scaler: Optional[StandardScaler] = None

    for split in splits:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=random_seed, shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=random_seed, shuffle=True)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)

        for index, spec in enumerate(ann_specs):
            model = build_ann_model(input_dim=x.shape[1], spec=spec, keras=keras)
            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=6, factor=0.5, min_lr=1e-5),
            ]
            model.fit(
                x_train_scaled,
                y_train,
                validation_data=(x_val_scaled, y_val),
                epochs=ann_epochs,
                batch_size=spec["batch_size"],
                verbose=0,
                callbacks=callbacks,
            )

            val_predictions = model.predict(x_val_scaled, verbose=0).reshape(-1)
            test_predictions = model.predict(x_test_scaled, verbose=0).reshape(-1)
            val_rmse = float(np.sqrt(mean_squared_error(y_val, val_predictions)))
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_predictions)))
            test_mae = float(mean_absolute_error(y_test, test_predictions))
            test_r2 = float(r2_score(y_test, test_predictions))

            result = ModelSummary(
                model_name=f"ANN_{index + 1}",
                split=float(split),
                cv_rmse=val_rmse,
                test_rmse=test_rmse,
                test_mae=test_mae,
                test_r2=test_r2,
                params=normalize_value(spec),
            )

            if best_result is None or result.cv_rmse < best_result.cv_rmse:
                best_result = result
                best_run = BestRun(
                    model_name=result.model_name,
                    split=float(split),
                    score=result.cv_rmse,
                    params=normalize_value(spec),
                    artifact_path=str(output_dir / "best_ann_model.keras"),
                )
                best_model = model
                best_scaler = scaler

    if best_result is not None and best_model is not None and best_scaler is not None:
        best_model.save(output_dir / "best_ann_model.keras")
        save_joblib(best_scaler, output_dir / "best_ann_scaler.joblib")

    return best_result, best_run


def normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_joblib(obj: Any, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def write_outputs(
    classical_results: Sequence[ModelSummary],
    classical_best: Optional[BestRun],
    ann_result: Optional[ModelSummary],
    ann_best: Optional[BestRun],
    dataset_source: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    classical_frame = pd.DataFrame([asdict(item) for item in classical_results])
    classical_frame.to_csv(output_dir / "classical_results.csv", index=False)

    payload = {
        "dataset_source": dataset_source,
        "classical_results": [asdict(item) for item in classical_results],
        "classical_best": asdict(classical_best) if classical_best else None,
        "ann_result": asdict(ann_result) if ann_result else None,
        "ann_best": asdict(ann_best) if ann_best else None,
    }
    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    args = parse_args()
    np.random.seed(args.random_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    require_gpu = bool(args.require_gpu or os.environ.get("CTZ_REQUIRE_GPU") == "1")
    x, y, dataset_source = load_dataset(args.data_path, args.demo, args.random_seed)

    classical_results, classical_best = benchmark_classical_models(
        x=x,
        y=y,
        splits=args.test_splits,
        cv_folds=args.cv_folds,
        random_seed=args.random_seed,
        output_dir=output_dir,
    )

    ann_result, ann_best = benchmark_ann(
        x=x,
        y=y,
        splits=args.test_splits,
        random_seed=args.random_seed,
        ann_epochs=args.ann_epochs,
        ann_batch_size=args.ann_batch_size,
        ann_trials=args.ann_trials,
        require_gpu=require_gpu,
        output_dir=output_dir,
    )

    write_outputs(classical_results, classical_best, ann_result, ann_best, dataset_source, output_dir)

    print(f"Saved benchmark outputs to {output_dir}")
    if classical_best is not None:
        print(f"Best classical model: {classical_best.model_name} with cv_rmse={classical_best.score:.4f}")
    if ann_best is not None:
        print(f"Best ANN model: {ann_best.model_name} with cv_rmse={ann_best.score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())