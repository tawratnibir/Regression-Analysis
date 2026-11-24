"""Shared evaluation metrics, artifact persistence, and common residual plots."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from config import RANDOM_SEED, RESULTS_DIR
from preprocessing.preprocess import PreparedDataset


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, MSE, RMSE, and R^2 on continuous targets."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_lpm_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute binary classification metrics (threshold 0.5) and [0,1] bounds diagnostics."""
    y_bin = y_true.astype(int)
    y_hat_bin = (y_pred >= 0.5).astype(int)
    cm = confusion_matrix(y_bin, y_hat_bin, labels=[0, 1])

    try:
        auc = float(roc_auc_score(y_bin, y_pred))
    except ValueError:
        auc = float("nan")

    below_zero = int(np.sum(y_pred < 0.0))
    above_one = int(np.sum(y_pred > 1.0))
    n = len(y_pred)

    return {
        "accuracy": float(accuracy_score(y_bin, y_hat_bin)),
        "precision": float(precision_score(y_bin, y_hat_bin, zero_division=0)),
        "recall": float(recall_score(y_bin, y_hat_bin, zero_division=0)),
        "f1": float(f1_score(y_bin, y_hat_bin, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "n_below_zero": below_zero,
        "n_above_one": above_one,
        "n_outside_unit_interval": below_zero + above_one,
        "pct_outside_unit_interval": float(100.0 * (below_zero + above_one) / n) if n else 0.0,
    }


def compute_all_metrics(
    dataset: PreparedDataset,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
) -> Dict[str, Any]:
    """Build full metrics dictionary for train and test splits."""
    metrics: Dict[str, Any] = {
        "dataset": dataset.config.key,
        "target": dataset.config.target,
        "target_units": dataset.config.target_units,
        "is_binary_classification": dataset.config.is_binary_classification,
        "n_train": int(len(dataset.y_train)),
        "n_test": int(len(dataset.y_test)),
        "train": compute_regression_metrics(dataset.y_train, y_train_pred),
        "test": compute_regression_metrics(dataset.y_test, y_test_pred),
    }
    if dataset.config.is_binary_classification:
        metrics["train_classification"] = compute_lpm_metrics(dataset.y_train, y_train_pred)
        metrics["test_classification"] = compute_lpm_metrics(dataset.y_test, y_test_pred)
    return metrics


def unscale_coefficients(
    coef_scaled: np.ndarray,
    intercept_scaled: float,
    scaler_means: np.ndarray,
    scaler_scales: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Convert coefficients from standardized predictors back to unstandardized predictor units."""
    safe_scales = np.where(scaler_scales == 0.0, 1.0, scaler_scales)
    coef_unscaled = coef_scaled / safe_scales
    intercept_unscaled = float(intercept_scaled - np.sum(coef_unscaled * scaler_means))
    return coef_unscaled, intercept_unscaled


def get_result_dir(model_key: str, dataset_key: str) -> Path:
    """Return results/<model>/<dataset> directory path."""
    return RESULTS_DIR / model_key / dataset_key


def save_model_artifacts(
    model_key: str,
    dataset: PreparedDataset,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
    coef_df: pd.DataFrame,
    metrics: Dict[str, Any],
    diagnostics: Dict[str, Any],
    objective_convention: str,
    penalty_settings: Dict[str, Any],
) -> Path:
    """Persist metrics, coefficients, predictions, diagnostics, and run metadata."""
    out_dir = get_result_dir(model_key, dataset.config.key)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    coef_df.to_csv(out_dir / "coefficients.csv", index=False)

    pd.DataFrame(
        {
            "raw_row_id": dataset.test_raw_indices,
            "observed": dataset.y_test,
            "predicted": y_test_pred,
            "residual": dataset.y_test - y_test_pred,
        }
    ).to_csv(out_dir / "predictions_test.csv", index=False)

    pd.DataFrame(
        {
            "raw_row_id": dataset.train_raw_indices,
            "observed": dataset.y_train,
            "predicted": y_train_pred,
            "residual": dataset.y_train - y_train_pred,
        }
    ).to_csv(out_dir / "predictions_train.csv", index=False)

    with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    run_metadata = {
        "dataset": dataset.config.key,
        "dataset_name": dataset.config.name,
        "uci_id": dataset.config.uci_id,
        "target": dataset.config.target,
        "target_units": dataset.config.target_units,
        "split_reference": str(dataset.split_path.relative_to(dataset.split_path.parents[2])),
        "split_strategy": dataset.config.split_strategy,
        "cv_strategy": dataset.cv_strategy_name,
        "seed": RANDOM_SEED,
        "preprocessing": dataset.preprocessing_info,
        "algorithm": model_key,
        "objective_convention": objective_convention,
        "penalty_settings": penalty_settings,
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    return out_dir


def plot_actual_vs_predicted(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    dataset: PreparedDataset,
    model_label: str,
    out_path: Path,
) -> None:
    """Save Actual vs Predicted scatter plot for train and test splits."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(y_train, y_train_pred, alpha=0.35, s=18, color="#4a90d9", label="Train")
    ax.scatter(y_test, y_test_pred, alpha=0.65, s=24, color="#d9534f", label="Test")

    lo = float(min(np.min(y_train), np.min(y_test), np.min(y_train_pred), np.min(y_test_pred)))
    hi = float(max(np.max(y_train), np.max(y_test), np.max(y_train_pred), np.max(y_test_pred)))
    pad = 0.03 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.2, label="y = x (ideal)")
    if dataset.config.is_binary_classification:
        ax.axhline(0.0, color="#718096", linestyle=":", lw=1.0)
        ax.axhline(1.0, color="#718096", linestyle=":", lw=1.0)

    ax.set_xlabel(f"Observed {dataset.config.target} ({dataset.config.target_units})")
    ax.set_ylabel(f"Predicted {dataset.config.target} ({dataset.config.target_units})")
    ax.set_title(f"{model_label} | {dataset.config.name}: Actual vs Predicted")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_residuals_vs_fitted(
    y_train_pred: np.ndarray,
    resid_train: np.ndarray,
    y_test_pred: np.ndarray,
    resid_test: np.ndarray,
    dataset: PreparedDataset,
    model_label: str,
    out_path: Path,
) -> None:
    """Save Residuals vs Fitted values plot with binned mean trend."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_train_pred, resid_train, alpha=0.35, s=16, color="#4a90d9", label="Train")
    ax.scatter(y_test_pred, resid_test, alpha=0.60, s=22, color="#d9534f", label="Test")
    ax.axhline(0.0, color="black", linestyle="--", lw=1.2)

    # Binned median/mean curve on training residuals for nonlinearity inspection
    if len(y_train_pred) >= 20:
        order = np.argsort(y_train_pred)
        bins = np.array_split(order, min(12, max(4, len(order) // 30)))
        bx = [float(np.mean(y_train_pred[b])) for b in bins if len(b) > 0]
        by = [float(np.mean(resid_train[b])) for b in bins if len(b) > 0]
        ax.plot(bx, by, color="#1a365d", lw=2.0, label="Train binned mean")

    ax.set_xlabel(f"Fitted {dataset.config.target} ({dataset.config.target_units})")
    ax.set_ylabel(f"Residual (Observed - Predicted, {dataset.config.target_units})")
    ax.set_title(f"{model_label} | {dataset.config.name}: Residuals vs Fitted")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_residual_distribution(
    resid_train: np.ndarray,
    resid_test: np.ndarray,
    dataset: PreparedDataset,
    model_label: str,
    out_path: Path,
) -> None:
    """Save histogram of train and test residuals with fitted normal density curve."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.hist(
        resid_train,
        bins=35,
        density=True,
        alpha=0.55,
        color="#4a90d9",
        edgecolor="white",
        label="Train residuals",
    )
    ax.hist(
        resid_test,
        bins=25,
        density=True,
        alpha=0.45,
        color="#d9534f",
        edgecolor="white",
        label="Test residuals",
    )

    mu = float(np.mean(resid_train))
    sigma = float(np.std(resid_train, ddof=1)) if len(resid_train) > 1 else 1.0
    if sigma > 0:
        xs = np.linspace(
            min(float(np.min(resid_train)), float(np.min(resid_test))),
            max(float(np.max(resid_train)), float(np.max(resid_test))),
            200,
        )
        pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
        ax.plot(xs, pdf, "k-", lw=1.6, label=f"Normal(0, {sigma:.3g})")

    ax.axvline(0.0, color="#2d3748", linestyle="--", lw=1.0)
    ax.set_xlabel(f"Residual ({dataset.config.target_units})")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_label} | {dataset.config.name}: Residual Distribution")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
