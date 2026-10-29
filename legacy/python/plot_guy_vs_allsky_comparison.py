#!/usr/bin/env python3
"""Predicted-vs-true Fsw scatter/histogram plot, comparing the production
top-20-of-129 recipe against the all-37-allsky-features recipe (Ridge + ANN
each), styled after Eytan et al. 2025 AGU Advances Figure 3.

Reloads both datasets and applies the saved model artifacts from
benchmark_outputs/final_model_full/ and benchmark_outputs/final_model_allsky37/
to the same random_state=42 test split used at training time (guy_model.py),
so no retraining is needed.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guys_benchmark_models import load_allsky37_subset, load_paired_moments_scalars

DATA_ROOT = "data/paired/Pacific_2014-2015"
RANDOM_SEED = 42


def get_full_recipe_test_set():
    summary = json.load(open("benchmark_outputs/final_model_full/final_model_summary.json"))
    top_indices = summary["top_feature_indices"]
    X, y, _ = load_paired_moments_scalars(DATA_ROOT)
    X_top = X[:, top_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=RANDOM_SEED)
    return X_test, y_test


def get_allsky37_test_set():
    X, y, _ = load_allsky37_subset(DATA_ROOT)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    return X_test, y_test


def ridge_predict(model_dir, X_test):
    ridge = joblib.load(Path(model_dir) / "ridge_model.joblib")
    scaler = joblib.load(Path(model_dir) / "ridge_scaler.joblib")
    return ridge.predict(scaler.transform(X_test))


def ann_predict(model_dir, X_test):
    model = tf.keras.models.load_model(Path(model_dir) / "ann_model.keras")
    return model.predict(X_test, verbose=0).flatten()


def compute_stats(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    me = float(np.mean(y_pred - y_true))
    r2_xy = float(r2_score(y_true, y_pred))
    a, b = np.polyfit(y_true, y_pred, 1)
    # R^2 of the OLS fit itself = squared Pearson correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    fit_r2 = float(corr ** 2)
    return {"rmse": rmse, "mae": mae, "me": me, "r2_xy": r2_xy, "a": float(a), "b": float(b), "fit_r2": fit_r2}


def draw_panel(fig, gs_cell, y_true, y_pred, title, axis_range, unit="Wm$^{-2}$"):
    inner = gs_cell.subgridspec(2, 1, height_ratios=(1, 4), hspace=0.05)
    ax_hist = fig.add_subplot(inner[0])
    ax_main = fig.add_subplot(inner[1], sharex=ax_hist)
    ax_hist.tick_params(axis="x", labelbottom=False)

    stats = compute_stats(y_true, y_pred)

    lo, hi = axis_range
    bins = np.linspace(lo, hi, 60)

    ax_hist.hist(y_pred, bins=bins, color="#1f77b4", alpha=0.85, label="Model")
    ax_hist.hist(y_true, bins=bins, color="#ff7f0e", alpha=0.6, label="Truth")
    ax_hist.legend(fontsize=7, loc="upper right", frameon=True)
    ax_hist.set_title(title, fontsize=11, fontweight="bold", loc="left")

    idx = np.random.default_rng(0).choice(len(y_true), size=min(60000, len(y_true)), replace=False)
    ax_main.scatter(y_true[idx], y_pred[idx], s=4, color="#1f77b4", alpha=0.35, linewidths=0)
    ax_main.plot([lo, hi], [lo, hi], "k-", linewidth=1.2, label="X=Y")
    xs = np.linspace(lo, hi, 10)
    ax_main.plot(xs, stats["a"] * xs + stats["b"], "r--", linewidth=1.2)

    txt = (
        f"RMSE(Wm$^{{-2}}$)= {stats['rmse']:.2f}\n"
        f"MAE(Wm$^{{-2}}$)= {stats['mae']:.2f}\n"
        f"ME(Wm$^{{-2}}$)= {stats['me']:.2f}\n"
        f"$R^2_{{x=y}}$= {stats['r2_xy']:.2f}"
    )
    ax_main.text(0.03, 0.97, txt, transform=ax_main.transAxes, fontsize=8, va="top", ha="left", color="black")
    fit_txt = f"Y={stats['a']:.2f}X+{stats['b']:.2f}\n$R^2$={stats['fit_r2']:.2f}"
    ax_main.text(0.03, 0.62, fit_txt, transform=ax_main.transAxes, fontsize=8, va="top", ha="left", color="red")

    ax_main.set_xlim(lo, hi)
    ax_main.set_ylim(lo, hi)
    ax_main.set_xlabel(f"CERES-truth $F_{{sw}}$ ({unit})", fontsize=9)
    ax_main.set_ylabel(f"model $F_{{sw}}$ ({unit})", fontsize=9)
    ax_main.legend(loc="lower right", fontsize=7, frameon=True)
    return stats


def main():
    print("Loading full-129-feature recipe test set + models...")
    X_test_full, y_test_full = get_full_recipe_test_set()
    ridge_pred_full = ridge_predict("benchmark_outputs/final_model_full", X_test_full)
    ann_pred_full = ann_predict("benchmark_outputs/final_model_full", X_test_full)

    print("Loading allsky37 recipe test set + models...")
    X_test_37, y_test_37 = get_allsky37_test_set()
    ridge_pred_37 = ridge_predict("benchmark_outputs/final_model_allsky37", X_test_37)
    ann_pred_37 = ann_predict("benchmark_outputs/final_model_allsky37", X_test_37)

    # Shared axis range across all 4 panels, based on the true-flux distribution only
    # (a couple of wild outlier predictions from the restricted-feature Ridge model
    # would otherwise blow the axes out far beyond the physical Fsw range).
    lo = min(np.percentile(y_test_full, 0.1), np.percentile(y_test_37, 0.1))
    hi = max(np.percentile(y_test_full, 99.9), np.percentile(y_test_37, 99.9))
    pad = 0.03 * (hi - lo)
    axis_range = (lo - pad, hi + pad)

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    all_stats = {}
    all_stats["ann_full"] = draw_panel(fig, gs[0, 0], y_test_full, ann_pred_full, "(a) ANN - top-20 of 129 features", axis_range)
    all_stats["ann_37"] = draw_panel(fig, gs[0, 1], y_test_37, ann_pred_37, "(b) ANN - all 37 All_Sky features", axis_range)
    all_stats["ridge_full"] = draw_panel(fig, gs[1, 0], y_test_full, ridge_pred_full, "(c) Ridge - top-20 of 129 features", axis_range)
    all_stats["ridge_37"] = draw_panel(fig, gs[1, 1], y_test_37, ridge_pred_37, "(d) Ridge - all 37 All_Sky features", axis_range)

    fig.suptitle("guy_model.py: predicted vs. true $F_{sw}$ (Pacific_2014-2015 test set)", fontsize=13, fontweight="bold")
    out_path = "benchmark_outputs/guy_vs_allsky_scatter_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

    with open("benchmark_outputs/guy_vs_allsky_scatter_comparison_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print("Saved stats json")


if __name__ == "__main__":
    main()
