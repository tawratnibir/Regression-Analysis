"""Ridge-specific and standard diagnostic plots."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.base import (
    plot_actual_vs_predicted,
    plot_residual_distribution,
    plot_residuals_vs_fitted,
)
from models.ridge.model import RidgeFitResult
from preprocessing.preprocess import PreparedDataset


def plot_cv_curve(fit: RidgeFitResult, dataset: PreparedDataset, out_path: Path) -> None:
    """Plot cross-validated RMSE vs log10(alpha) with +/- 1 std band."""
    log_a = np.log10(fit.alphas)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(log_a, fit.cv_rmse_mean, color="#2b6cb0", lw=2.0, label="Mean CV RMSE")
    ax.fill_between(
        log_a,
        fit.cv_rmse_mean - fit.cv_rmse_std,
        fit.cv_rmse_mean + fit.cv_rmse_std,
        color="#4a90d9",
        alpha=0.22,
        label="±1 SD across CV folds",
    )
    best_log = float(np.log10(fit.best_alpha))
    best_rmse = float(fit.cv_rmse_mean[fit.best_alpha_idx])
    ax.axvline(
        best_log,
        color="#d9534f",
        linestyle="--",
        lw=1.4,
        label=f"Selected alpha={fit.best_alpha:.4g} (CV RMSE={best_rmse:.4g})",
    )
    ax.scatter([best_log], [best_rmse], color="#d9534f", zorder=5, s=45)
    ax.set_xlabel("log10(alpha)")
    ax.set_ylabel(f"CV RMSE ({dataset.config.target_units})")
    ax.set_title(f"Ridge | {dataset.config.name}: Cross-Validation Curve ({dataset.cv_strategy_name})")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_coefficient_path(
    fit: RidgeFitResult, dataset: PreparedDataset, out_path: Path
) -> None:
    """Plot standardized Ridge coefficient paths against log10(alpha)."""
    log_a = np.log10(fit.alphas)
    max_abs = np.max(np.abs(fit.coef_path), axis=0)
    top_k = min(10, len(dataset.feature_names))
    top_indices = set(np.argsort(max_abs)[-top_k:].tolist())

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for j, name in enumerate(dataset.feature_names):
        if j in top_indices:
            ax.plot(log_a, fit.coef_path[:, j], lw=1.7, label=name)
        else:
            ax.plot(log_a, fit.coef_path[:, j], lw=0.8, color="#cbd5e0", alpha=0.7)

    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.axvline(
        float(np.log10(fit.best_alpha)),
        color="#d9534f",
        linestyle="--",
        lw=1.4,
        label=f"Selected alpha={fit.best_alpha:.4g}",
    )
    ax.set_xlabel("log10(alpha)")
    ax.set_ylabel("Standardized Coefficient")
    ax.set_title(f"Ridge | {dataset.config.name}: Coefficient Shrinkage Path")
    ax.legend(fontsize=7.5, loc="best", ncol=2 if top_k > 6 else 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_ols_vs_ridge_coefficients(
    fit: RidgeFitResult,
    ols_coef_scaled: np.ndarray,
    dataset: PreparedDataset,
    out_path: Path,
) -> None:
    """Compare top OLS vs Ridge standardized coefficients side by side."""
    order = np.argsort(np.abs(ols_coef_scaled))[::-1][: min(15, len(dataset.feature_names))]
    names = [dataset.feature_names[i] for i in order][::-1]
    ols_vals = ols_coef_scaled[order][::-1]
    ridge_vals = fit.coef_scaled[order][::-1]

    y_pos = np.arange(len(names))
    height = 0.36
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.38 * len(names) + 1.5)))
    ax.barh(y_pos + height / 2, ols_vals, height=height, color="#718096", label="OLS (unpenalized)")
    ax.barh(y_pos - height / 2, ridge_vals, height=height, color="#2b6cb0", label=f"Ridge (alpha={fit.best_alpha:.4g})")
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Standardized Coefficient")
    ax.set_title(f"Ridge vs OLS | {dataset.config.name}: Top Standardized Coefficients")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def generate_ridge_plots(
    dataset: PreparedDataset,
    fit: RidgeFitResult,
    ols_coef_scaled: np.ndarray,
    plots_dir: Path,
) -> None:
    """Generate all standard and Ridge-specific plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_actual_vs_predicted(
        dataset.y_train,
        fit.y_train_pred,
        dataset.y_test,
        fit.y_test_pred,
        dataset,
        "Ridge",
        plots_dir / "actual_vs_predicted.png",
    )
    plot_residuals_vs_fitted(
        fit.y_train_pred,
        fit.resid_train,
        fit.y_test_pred,
        fit.resid_test,
        dataset,
        "Ridge",
        plots_dir / "residuals_vs_fitted.png",
    )
    plot_residual_distribution(
        fit.resid_train,
        fit.resid_test,
        dataset,
        "Ridge",
        plots_dir / "residual_distribution.png",
    )
    plot_cv_curve(fit, dataset, plots_dir / "cv_curve.png")
    plot_coefficient_path(fit, dataset, plots_dir / "coefficient_path.png")
    plot_ols_vs_ridge_coefficients(
        fit, ols_coef_scaled, dataset, plots_dir / "ols_vs_ridge_coefficients.png"
    )
