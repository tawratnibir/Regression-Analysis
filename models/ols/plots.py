"""OLS diagnostic and residual plotting functions."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from models.base import (
    plot_actual_vs_predicted,
    plot_residual_distribution,
    plot_residuals_vs_fitted,
)
from models.ols.model import OLSFitResult
from preprocessing.preprocess import PreparedDataset


def plot_qq(resid_train: np.ndarray, dataset: PreparedDataset, out_path: Path) -> None:
    """Save Normal Q-Q plot of training residuals."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    (osm, osr), (slope, intercept, _) = stats.probplot(resid_train, dist="norm")
    ax.scatter(osm, osr, alpha=0.5, s=16, color="#2b6cb0", label="Observed residuals")
    xs = np.array([np.min(osm), np.max(osm)])
    ax.plot(xs, slope * xs + intercept, "r--", lw=1.5, label="Normal reference line")
    ax.set_xlabel("Theoretical Normal Quantiles")
    ax.set_ylabel(f"Ordered Training Residuals ({dataset.config.target_units})")
    ax.set_title(f"OLS | {dataset.config.name}: Normal Q-Q Plot")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_cooks_distance(
    cooks_d: np.ndarray, dataset: PreparedDataset, out_path: Path
) -> None:
    """Save Cook's distance plot across training observations."""
    n = len(cooks_d)
    threshold = 4.0 / n
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    idx = np.arange(n)
    ax.vlines(idx, 0, cooks_d, colors="#2b6cb0", alpha=0.65, lw=0.9)
    above = cooks_d > threshold
    if np.any(above):
        ax.scatter(idx[above], cooks_d[above], color="#d9534f", s=16, zorder=3, label=f"> 4/n ({int(np.sum(above))} pts)")
    ax.axhline(threshold, color="#d9534f", linestyle="--", lw=1.2, label=f"4/n threshold ({threshold:.4f})")
    ax.set_xlabel("Training Observation Position")
    ax.set_ylabel("Cook's Distance")
    ax.set_title(f"OLS | {dataset.config.name}: Cook's Distance Influence Check")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def generate_ols_plots(
    dataset: PreparedDataset,
    fit: OLSFitResult,
    cooks_d: np.ndarray,
    plots_dir: Path,
) -> None:
    """Generate all standard and OLS-specific plots into plots_dir."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_actual_vs_predicted(
        dataset.y_train,
        fit.y_train_pred,
        dataset.y_test,
        fit.y_test_pred,
        dataset,
        "OLS",
        plots_dir / "actual_vs_predicted.png",
    )
    plot_residuals_vs_fitted(
        fit.y_train_pred,
        fit.resid_train,
        fit.y_test_pred,
        fit.resid_test,
        dataset,
        "OLS",
        plots_dir / "residuals_vs_fitted.png",
    )
    plot_residual_distribution(
        fit.resid_train,
        fit.resid_test,
        dataset,
        "OLS",
        plots_dir / "residual_distribution.png",
    )
    plot_qq(fit.resid_train, dataset, plots_dir / "qq_plot.png")
    plot_cooks_distance(cooks_d, dataset, plots_dir / "cooks_distance.png")
