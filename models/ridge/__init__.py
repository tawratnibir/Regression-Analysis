"""Ridge regression package."""

from pathlib import Path

from models.base import compute_all_metrics, save_model_artifacts
from models.ridge.analysis import analyze_ridge
from models.ridge.model import build_ridge_coefficients_table, fit_ridge
from models.ridge.plots import generate_ridge_plots
from preprocessing.preprocess import PreparedDataset


def run_ridge(dataset: PreparedDataset) -> Path:
    """Tune and fit Ridge regression, compute diagnostics and plots, and save results."""
    fit = fit_ridge(dataset)
    diagnostics, ols_coef_scaled, ols_intercept_scaled = analyze_ridge(dataset, fit)
    coef_df = build_ridge_coefficients_table(
        dataset, fit, ols_coef_scaled, ols_intercept_scaled
    )
    metrics = compute_all_metrics(dataset, fit.y_train_pred, fit.y_test_pred)

    out_dir = save_model_artifacts(
        model_key="ridge",
        dataset=dataset,
        y_train_pred=fit.y_train_pred,
        y_test_pred=fit.y_test_pred,
        coef_df=coef_df,
        metrics=metrics,
        diagnostics=diagnostics,
        objective_convention="min_{beta0, beta} ||y - beta0 - X*beta||_2^2 + alpha * ||beta||_2^2 (intercept beta0 unpenalized)",
        penalty_settings={
            "penalty": "L2",
            "selected_alpha": fit.best_alpha,
            "alpha_grid_min": float(fit.alphas[0]),
            "alpha_grid_max": float(fit.alphas[-1]),
            "n_alphas": int(len(fit.alphas)),
            "cv_strategy": dataset.cv_strategy_name,
        },
    )
    generate_ridge_plots(dataset, fit, ols_coef_scaled, out_dir / "plots")
    return out_dir
