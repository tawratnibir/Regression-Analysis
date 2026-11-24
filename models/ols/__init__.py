"""Ordinary Least Squares (OLS) regression package."""

from pathlib import Path

from models.base import compute_all_metrics, get_result_dir, save_model_artifacts
from models.ols.analysis import analyze_ols
from models.ols.model import build_ols_coefficients_table, fit_ols
from models.ols.plots import generate_ols_plots
from preprocessing.preprocess import PreparedDataset


def run_ols(dataset: PreparedDataset) -> Path:
    """Fit OLS, compute diagnostics and plots, and save all results."""
    fit = fit_ols(dataset)
    diagnostics, _, cooks_d = analyze_ols(dataset, fit)
    coef_df = build_ols_coefficients_table(dataset, fit)
    metrics = compute_all_metrics(dataset, fit.y_train_pred, fit.y_test_pred)

    out_dir = save_model_artifacts(
        model_key="ols",
        dataset=dataset,
        y_train_pred=fit.y_train_pred,
        y_test_pred=fit.y_test_pred,
        coef_df=coef_df,
        metrics=metrics,
        diagnostics=diagnostics,
        objective_convention="min_{beta0, beta} ||y - beta0 - X*beta||_2^2 via numpy.linalg.lstsq (intercept unscaled/unpenalized)",
        penalty_settings={"penalty": "none"},
    )
    generate_ols_plots(dataset, fit, cooks_d, out_dir / "plots")
    return out_dir
