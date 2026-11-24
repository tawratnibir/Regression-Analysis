"""Lasso-specific and standard diagnostic plots."""

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
from models.lasso.model import ZERO_TOL, LassoFitResult
from preprocessing.preprocess import PreparedDataset


def plot_cv_curve(fit: LassoFitResult, dataset: PreparedDataset, out_path: Path) -> None:
    """Plot cross-validated RMSE vs log10(alpha) with nonzero feature count annotations."""
    log_a = np.log10(fit.alphas)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(log_a, fit.cv_rmse_mean, color="#2f855a", lw=2.0, label="Mean CV RMSE")
    ax.fill_between(
        log_a,
        fit.cv_rmse_mean - fit.cv_rmse_std,
        fit.cv_rmse_mean + fit.cv_rmse_std,
        color="#68d391",
        alpha=0.25,
        label="±1 SD across CV folds",
    )
    best_log = float(np.log10(fit.best_alpha))
    best_rmse = float(fit.cv_rmse_mean[fit.best_alpha_idx])
    best_nz = int(np.sum(np.abs(fit.coef_scaled) > ZERO_TOL))
    ax.axvline(
        best_log,
        color="#d9534f",
        linestyle="--",
        lw=1.4,
        label=f"Selected alpha={fit.best_alpha:.4g} (nonzero={best_nz}, CV RMSE={best_rmse:.4g})",
    )
    ax.scatter([best_log], [best_rmse], color="#d9534f", zorder=5, s=45)
    ax.set_xlabel("log10(alpha)")
    ax.set_ylabel(f"CV RMSE ({dataset.config.target_units})")
    ax.set_title(f"Lasso | {dataset.config.name}: Cross-Validation Curve ({dataset.cv_strategy_name})")
    ax.legend()
    ax.grid(alpha=0.25)

    # Top x-axis ticks showing active feature counts along the grid
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    tick_indices = np.linspace(0, len(log_a) - 1, 6, dtype=int)
    ax_top.set_xticks(log_a[tick_indices])
    ax_top.set_xticklabels([str(int(fit.nonzero_counts_path[i])) for i in tick_indices], fontsize=8.5)
    ax_top.set_xlabel("Number of Nonzero Predictors (Full Train)", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_coefficient_path(
    fit: LassoFitResult, dataset: PreparedDataset, out_path: Path
) -> None:
    """Plot Lasso coordinate-descent coefficient paths against log10(alpha)."""
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
    ax.set_title(f"Lasso | {dataset.config.name}: L1 Regularization Path")
    ax.legend(fontsize=7.5, loc="best", ncol=2 if top_k > 6 else 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_sparsity_vs_alpha(
    fit: LassoFitResult, dataset: PreparedDataset, out_path: Path
) -> None:
    """Plot number of retained nonzero predictors against log10(alpha)."""
    log_a = np.log10(fit.alphas)
    p = len(dataset.feature_names)
    best_log = float(np.log10(fit.best_alpha))
    best_nz = int(np.sum(np.abs(fit.coef_scaled) > ZERO_TOL))

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.step(log_a, fit.nonzero_counts_path, where="mid", color="#2f855a", lw=2.0, label="Retained predictors")
    ax.axvline(
        best_log,
        color="#d9534f",
        linestyle="--",
        lw=1.4,
        label=f"Selected alpha={fit.best_alpha:.4g} ({best_nz}/{p} retained)",
    )
    ax.scatter([best_log], [best_nz], color="#d9534f", zorder=5, s=45)
    ax.set_ylim(-0.5, p + 1)
    ax.set_xlabel("log10(alpha)")
    ax.set_ylabel("Nonzero Coefficients Count")
    ax.set_title(f"Lasso | {dataset.config.name}: Sparsity vs Penalty Strength")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def generate_lasso_plots(
    dataset: PreparedDataset,
    fit: LassoFitResult,
    plots_dir: Path,
) -> None:
    """Generate all standard and Lasso-specific plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_actual_vs_predicted(
        dataset.y_train,
        fit.y_train_pred,
        dataset.y_test,
        fit.y_test_pred,
        dataset,
        "Lasso",
        plots_dir / "actual_vs_predicted.png",
    )
    plot_residuals_vs_fitted(
        fit.y_train_pred,
        fit.resid_train,
        fit.y_test_pred,
        fit.resid_test,
        dataset,
        "Lasso",
        plots_dir / "residuals_vs_fitted.png",
    )
    plot_residual_distribution(
        fit.resid_train,
        fit.resid_test,
        dataset,
        "Lasso",
        plots_dir / "residual_distribution.png",
    )
    plot_cv_curve(fit, dataset, plots_dir / "cv_curve.png")
    plot_coefficient_path(fit, dataset, plots_dir / "coefficient_path.png")
    plot_sparsity_vs_alpha(fit, dataset, plots_dir / "sparsity_vs_alpha.png")
