"""Lasso regression package."""

from pathlib import Path

from models.base import compute_all_metrics, save_model_artifacts
from models.lasso.analysis import analyze_lasso
from models.lasso.model import build_lasso_coefficients_table, fit_lasso
from models.lasso.plots import generate_lasso_plots
from preprocessing.preprocess import PreparedDataset


def run_lasso(dataset: PreparedDataset) -> Path:
    """Tune and fit Lasso regression, compute diagnostics and plots, and save results."""
    fit = fit_lasso(dataset)
    diagnostics, ols_coef_scaled, ols_intercept_scaled = analyze_lasso(dataset, fit)
    coef_df = build_lasso_coefficients_table(
        dataset, fit, ols_coef_scaled, ols_intercept_scaled
    )
    metrics = compute_all_metrics(dataset, fit.y_train_pred, fit.y_test_pred)

    out_dir = save_model_artifacts(
        model_key="lasso",
        dataset=dataset,
        y_train_pred=fit.y_train_pred,
        y_test_pred=fit.y_test_pred,
        coef_df=coef_df,
        metrics=metrics,
        diagnostics=diagnostics,
        objective_convention="min_{beta0, beta} (1 / (2*n)) * ||y - beta0 - X*beta||_2^2 + alpha * ||beta||_1 (intercept beta0 unpenalized)",
        penalty_settings={
            "penalty": "L1",
            "selected_alpha": fit.best_alpha,
            "alpha_max": fit.alpha_max,
            "alpha_grid_min": float(fit.alphas[-1]),
            "alpha_grid_max": float(fit.alphas[0]),
            "n_alphas": int(len(fit.alphas)),
            "cv_strategy": dataset.cv_strategy_name,
        },
    )
    generate_lasso_plots(dataset, fit, out_dir / "plots")
    return out_dir
