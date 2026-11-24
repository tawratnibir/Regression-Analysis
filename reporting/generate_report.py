"""Generate self-contained, explainable HTML reports for OLS, Ridge, and Lasso."""

import base64
import json
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, select_autoescape
import numpy as np
import pandas as pd

from config import DATASETS, REPORTS_DIR, RESULTS_DIR

JINJA_ENV = Environment(autoescape=select_autoescape(["html", "xml"]))

MODEL_TITLES = {
    "ols": "Ordinary Least Squares (OLS)",
    "ridge": "Ridge Regression (L2 Regularization)",
    "lasso": "Lasso Regression (L1 Regularization)",
}

PLOT_ORDER = [
    ("actual_vs_predicted.png", "Actual vs. Predicted"),
    ("residuals_vs_fitted.png", "Residuals vs. Fitted"),
    ("residual_distribution.png", "Residual Distribution"),
    ("qq_plot.png", "Normal Q-Q Plot"),
    ("cooks_distance.png", "Cook's Distance"),
    ("cv_curve.png", "Cross-Validation Error Curve"),
    ("coefficient_path.png", "Coefficient Regularization Path"),
    ("ols_vs_ridge_coefficients.png", "OLS vs. Ridge Coefficient Shrinkage"),
    ("sparsity_vs_alpha.png", "Active Predictor Count vs. Penalty"),
]


def _fmt(val: Any) -> str:
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, int):
        return f"{val:,}"
    if isinstance(val, float):
        if not np.isfinite(val):
            return str(val)
        if abs(val) > 0 and (abs(val) < 1e-4 or abs(val) >= 1e6):
            return f"{val:.4e}"
        return f"{val:.4g}"
    return str(val)


def build_figure_captions(
    model_key: str,
    metrics: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> Dict[str, str]:
    """Build concise, artifact-grounded captions for each diagnostic plot."""
    units = metrics["target_units"]
    tr_r2 = metrics["train"]["r2"]
    te_r2 = metrics["test"]["r2"]
    te_rmse = metrics["test"]["rmse"]

    captions: Dict[str, str] = {
        "actual_vs_predicted.png": (
            f"Train R² = {tr_r2:.4f} vs. Test R² = {te_r2:.4f} "
            f"(Test RMSE = {_fmt(te_rmse)} {units}; n_test = {metrics['n_test']})."
        ),
        "residuals_vs_fitted.png": (
            "Training and held-out residuals around zero; systematic curvature indicates unmodeled nonlinearity."
        ),
        "residual_distribution.png": (
            "Empirical training and test residual densities compared against a fitted Normal curve."
        ),
    }

    if model_key == "ols":
        het = diagnostics["heteroskedasticity_checks"]
        norm = diagnostics["residual_normality_checks"]
        inf = diagnostics["influence_diagnostics"]
        captions["residuals_vs_fitted.png"] = (
            f"Breusch-Pagan LM = {_fmt(het['breusch_pagan_lm_stat'])} (p = {_fmt(het['breusch_pagan_pvalue'])}); "
            f"Spearman(|e|, y_hat) r = {het['spearman_abs_resid_vs_fitted_r']:.3f}."
        )
        captions["residual_distribution.png"] = (
            f"Skewness = {norm['skewness']:.3f}, excess kurtosis = {norm['excess_kurtosis']:.3f}, "
            f"Jarque-Bera p = {_fmt(norm['jarque_bera_pvalue'])}."
        )
        captions["qq_plot.png"] = (
            f"Shapiro-Wilk W = {norm['shapiro_wilk_stat']:.4f} (p = {_fmt(norm['shapiro_wilk_pvalue'])}, "
            f"n = {norm['shapiro_wilk_n']})."
        )
        captions["cooks_distance.png"] = (
            f"{inf['n_high_cooks_d']} training points exceed 4/n = {inf['cooks_threshold_4_over_n']:.4f} "
            f"(max Cook's D = {_fmt(inf['max_cooks_d'])})."
        )

    elif model_key == "ridge":
        pt = diagnostics["penalty_tuning"]
        sd = diagnostics["shrinkage_and_degrees_of_freedom"]
        rc = diagnostics["residual_checks"]
        captions["residuals_vs_fitted.png"] = (
            f"Spearman(|e|, y_hat) r = {rc['spearman_abs_resid_vs_fitted_r']:.3f} "
            f"(p = {_fmt(rc['spearman_abs_resid_vs_fitted_pvalue'])}); DW = {_fmt(rc['durbin_watson'])}."
        )
        captions["cv_curve.png"] = (
            f"{pt['cv_strategy']} selected alpha = {_fmt(pt['selected_alpha'])} "
            f"(CV RMSE = {_fmt(pt['best_cv_rmse'])} +/- {_fmt(pt['best_cv_rmse_std'])} {units}, CV R² = {pt['best_cv_r2']:.4f})."
        )
        captions["coefficient_path.png"] = (
            f"Continuous L2 shrinkage across alpha; effective df = {sd['effective_df_predictors']:.2f} / {sd['n_predictors']} at selected alpha."
        )
        captions["ols_vs_ridge_coefficients.png"] = (
            f"L2 norm reduced from {_fmt(sd['ols_coef_l2_norm'])} (OLS) to {_fmt(sd['ridge_coef_l2_norm'])} "
            f"(ratio = {sd['l2_norm_ratio_ridge_over_ols']:.4f}); all {sd['n_predictors']} predictors remain nonzero."
        )

    elif model_key == "lasso":
        pt = diagnostics["penalty_tuning"]
        sp = diagnostics["sparsity_and_shrinkage"]
        rc = diagnostics["residual_checks"]
        captions["residuals_vs_fitted.png"] = (
            f"Spearman(|e|, y_hat) r = {rc['spearman_abs_resid_vs_fitted_r']:.3f} "
            f"(p = {_fmt(rc['spearman_abs_resid_vs_fitted_pvalue'])}); DW = {_fmt(rc['durbin_watson'])}."
        )
        captions["cv_curve.png"] = (
            f"{pt['cv_strategy']} selected alpha = {_fmt(pt['selected_alpha'])} "
            f"(CV RMSE = {_fmt(pt['best_cv_rmse'])} +/- {_fmt(pt['best_cv_rmse_std'])} {units}, CV R² = {pt['best_cv_r2']:.4f})."
        )
        captions["coefficient_path.png"] = (
            f"Piecewise-linear L1 paths from alpha_max = {_fmt(pt['alpha_max'])}; "
            f"L1 norm ratio vs OLS = {sp['l1_norm_ratio_lasso_over_ols']:.4f}."
        )
        captions["sparsity_vs_alpha.png"] = (
            f"Retains {sp['n_nonzero_features']} of {sp['n_total_features']} predictors "
            f"and zeros {sp['n_zero_features']} ({sp['sparsity_pct']:.1f}% sparse) at selected alpha."
        )

    return captions


def _load_plots(
    plots_dir: Path,
    model_key: str,
    metrics: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> List[Dict[str, str]]:
    captions = build_figure_captions(model_key, metrics, diagnostics)
    loaded: List[Dict[str, str]] = []
    for fname, title in PLOT_ORDER:
        p = plots_dir / fname
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            loaded.append(
                {
                    "filename": fname,
                    "title": title,
                    "caption": captions.get(fname, ""),
                    "b64": b64,
                }
            )
    return loaded


def _build_diagnostic_sections_and_narrative(
    model_key: str,
    dataset_key: str,
    metrics: Dict[str, Any],
    coef_df: pd.DataFrame,
    diagnostics: Dict[str, Any],
    metadata: Dict[str, Any],
) -> tuple[List[tuple[str, List[tuple[str, str]]]], List[str], List[str]]:
    """Convert saved JSON/CSV artifacts into structured diagnostic tables, findings, and limitations."""
    cfg = DATASETS[dataset_key]
    sections: List[tuple[str, List[tuple[str, str]]]] = []
    interpretations: List[str] = []
    limitations: List[str] = []

    train_r2 = metrics["train"]["r2"]
    test_r2 = metrics["test"]["r2"]
    test_rmse = metrics["test"]["rmse"]
    units = metrics["target_units"]

    pred_df = coef_df[coef_df["role"] == "predictor"]
    top_preds = pred_df.head(3)
    top_str = ", ".join(
        f"{r['feature']} ({r['coefficient_standardized']:+.4g})"
        for _, r in top_preds.iterrows()
    )
    interpretations.append(
        f"[Computed Fact] Held-out test performance (n={metrics['n_test']}): R² = {test_r2:.4f}, "
        f"RMSE = {_fmt(test_rmse)} {units} (Train R² = {train_r2:.4f}, Train RMSE = {_fmt(metrics['train']['rmse'])} {units})."
    )
    if len(top_preds) > 0:
        interpretations.append(
            f"[Computed Fact] Largest standardized coefficients by magnitude: {top_str}."
        )

    if model_key == "ols":
        ex = diagnostics["exact_algebraic_properties"]
        la = diagnostics["numerical_linear_algebra"]
        mc = diagnostics["multicollinearity"]
        het = diagnostics["heteroskedasticity_checks"]
        norm = diagnostics["residual_normality_checks"]
        inf = diagnostics["influence_diagnostics"]
        ac = diagnostics["autocorrelation_checks"]

        sections.append((
            "Exact OLS Algebraic Properties (Assumption-Free Consequences of Normal Equations)",
            [
                ("max |X_aug^T e| (residual orthogonality)", _fmt(ex["max_abs_Xt_residuals"])),
                ("Sum of training residuals", _fmt(ex["residual_sum"])),
                ("|mean(y_train) - mean(y_hat_train)|", _fmt(ex["mean_fitted_identity_abs_diff"])),
                ("|SST - (SSR + SSE)|", _fmt(ex["anova_identity_abs_error"])),
            ],
        ))
        sections.append((
            "Matrix Rank, Conditioning & Multicollinearity",
            [
                ("Matrix rank / augmented columns", f"{la['matrix_rank']} / {la['n_augmented_columns']}"),
                ("Rank deficient", _fmt(la["is_rank_deficient"])),
                ("Condition number kappa(X_aug) standardized", _fmt(la["condition_number_standardized_with_intercept"])),
                ("Condition number kappa(X_aug) unstandardized", _fmt(la["condition_number_unstandardized_with_intercept"])),
                ("Max finite VIF / features with VIF > 10", f"{_fmt(mc['max_vif'])} / {mc['n_vif_gt_10']}"),
                ("Max |pred diff| vs. sklearn LinearRegression", _fmt(la["sklearn_max_abs_test_pred_diff"])),
            ],
        ))
        sections.append((
            "Statistical Assumption Checks (Training Residuals)",
            [
                ("Breusch-Pagan LM stat / p-value", f"{_fmt(het['breusch_pagan_lm_stat'])} (p = {_fmt(het['breusch_pagan_pvalue'])})"),
                ("Spearman |e| vs. y_hat (r / p)", f"{_fmt(het['spearman_abs_resid_vs_fitted_r'])} (p = {_fmt(het['spearman_abs_resid_vs_fitted_pvalue'])})"),
                ("Skewness / Excess kurtosis", f"{_fmt(norm['skewness'])} / {_fmt(norm['excess_kurtosis'])}"),
                ("Jarque-Bera p-value / Shapiro-Wilk p-value", f"{_fmt(norm['jarque_bera_pvalue'])} / {_fmt(norm['shapiro_wilk_pvalue'])}"),
                ("High-leverage (> 2k/n) / High Cook's D (> 4/n)", f"{inf['n_high_leverage']} (max={_fmt(inf['max_leverage'])}) / {inf['n_high_cooks_d']} (max={_fmt(inf['max_cooks_d'])})"),
                ("Durbin-Watson / Lag-1 residual corr", f"{_fmt(ac['durbin_watson'])} / {_fmt(ac['lag1_residual_corr'])}"),
            ],
        ))

        interpretations.append(
            f"[Computed Fact] Exact normal-equation identities hold numerically: max |X_aug^T e| = {ex['max_abs_Xt_residuals']:.2e}, "
            f"|mean(y) - mean(y_hat)| = {ex['mean_fitted_identity_abs_diff']:.2e}, and predictions match scikit-learn within {la['sklearn_max_abs_test_pred_diff']:.2e}."
        )
        interpretations.append(
            f"[Interpretation] Predictor standardization changes the augmented condition number from {_fmt(la['condition_number_unstandardized_with_intercept'])} "
            f"(unstandardized) to {_fmt(la['condition_number_standardized_with_intercept'])} (standardized), with rank {la['matrix_rank']} of {la['n_augmented_columns']}."
        )

        if la["is_rank_deficient"]:
            limitations.append(
                f"[Assumption Violation — Full Rank] Design matrix is rank-deficient (rank {la['matrix_rank']} < {la['n_augmented_columns']}). Individual OLS coefficients are not uniquely identified; np.linalg.lstsq returns the minimum-norm pseudoinverse solution."
            )
        if mc["n_vif_gt_10"] > 0 or mc.get("has_infinite_vif"):
            limitations.append(
                f"[Multicollinearity] {mc['n_vif_gt_10']} predictor(s) have VIF > 10 (max finite VIF = {_fmt(mc['max_vif'])}), inflating OLS estimator variance sigma^2 (X^T X)^-1."
            )
        if het["breusch_pagan_pvalue"] < 0.05:
            limitations.append(
                f"[Assumption Violation — Homoskedasticity] Breusch-Pagan test rejects constant disturbance variance (p = {het['breusch_pagan_pvalue']:.2e}); classical standard errors would require White heteroskedasticity-consistent covariance."
            )
        if norm["jarque_bera_pvalue"] < 0.05:
            limitations.append(
                f"[Assumption Check — Normality] Residuals deviate from Gaussian normality (Jarque-Bera p = {norm['jarque_bera_pvalue']:.2e}, skewness = {norm['skewness']:.3f}, excess kurtosis = {norm['excess_kurtosis']:.3f})."
            )
        if inf["n_high_cooks_d"] > 0:
            limitations.append(
                f"[Influential Observations] {inf['n_high_cooks_d']} training row(s) exceed Cook's D > 4/n (max = {inf['max_cooks_d']:.4g})."
            )
        if ac["durbin_watson"] < 1.5 or ac["durbin_watson"] > 2.5:
            limitations.append(
                f"[Assumption Violation — No Autocorrelation] Durbin-Watson = {ac['durbin_watson']:.3f} (lag-1 corr = {ac['lag1_residual_corr']:.3f}), indicating correlated errors."
            )

    elif model_key == "ridge":
        pt = diagnostics["penalty_tuning"]
        sd = diagnostics["shrinkage_and_degrees_of_freedom"]
        ns = diagnostics["numerical_stability"]
        rc = diagnostics["residual_checks"]

        sections.append((
            "Training-Only Cross-Validation & SVD Verification",
            [
                ("CV strategy", pt["cv_strategy"]),
                ("Selected penalty alpha (lambda)", _fmt(pt["selected_alpha"])),
                ("Best CV RMSE (+/- SD)", f"{_fmt(pt['best_cv_rmse'])} (+/- {_fmt(pt['best_cv_rmse_std'])}) {units}"),
                ("Best CV R²", _fmt(pt["best_cv_r2"])),
                ("Max |coef diff| vs. closed-form SVD", _fmt(pt["svd_verification_max_abs_coef_diff"])),
            ],
        ))
        sections.append((
            "Shrinkage, Effective Degrees of Freedom & Conditioning",
            [
                ("Effective df (predictors / +intercept)", f"{_fmt(sd['effective_df_predictors'])} / {_fmt(sd['effective_df_with_intercept'])} (of {sd['n_predictors']} predictors)"),
                ("||beta_OLS||_2 / ||beta_Ridge||_2", f"{_fmt(sd['ols_coef_l2_norm'])} / {_fmt(sd['ridge_coef_l2_norm'])}"),
                ("L2 norm ratio (Ridge / OLS)", _fmt(sd["l2_norm_ratio_ridge_over_ols"])),
                ("kappa(X^T X) vs. kappa(X^T X + alpha I)", f"{_fmt(ns['x_centered_eig_condition_ols'])} -> {_fmt(ns['x_centered_eig_condition_ridge'])}"),
                ("Spearman |e| vs. y_hat (r / p)", f"{_fmt(rc['spearman_abs_resid_vs_fitted_r'])} (p = {_fmt(rc['spearman_abs_resid_vs_fitted_pvalue'])})"),
                ("Durbin-Watson", _fmt(rc["durbin_watson"])),
            ],
        ))

        interpretations.append(
            f"[Computed Fact] {pt['cv_strategy']} selected alpha = {pt['selected_alpha']:.4g} "
            f"(CV RMSE = {_fmt(pt['best_cv_rmse'])} {units}, CV R² = {pt['best_cv_r2']:.4f}), matching closed-form SVD within {pt['svd_verification_max_abs_coef_diff']:.2e}."
        )
        interpretations.append(
            f"[Interpretation] L2 regularization trades controlled bias for variance reduction, shrinking the standardized L2 norm from {_fmt(sd['ols_coef_l2_norm'])} (OLS) to {_fmt(sd['ridge_coef_l2_norm'])} "
            f"(ratio = {sd['l2_norm_ratio_ridge_over_ols']:.4f}) and reducing effective degrees of freedom from {sd['n_predictors']} to {sd['effective_df_predictors']:.2f}."
        )
        interpretations.append(
            f"[Computed Fact] Adding alpha*I stabilized the centered X^T X condition number from {_fmt(ns['x_centered_eig_condition_ols'])} to {_fmt(ns['x_centered_eig_condition_ridge'])}."
        )

        if sd["l2_norm_ratio_ridge_over_ols"] > 0.99:
            limitations.append(
                f"[Minimal Regularization] Selected alpha = {pt['selected_alpha']:.4g} yields L2 ratio {sd['l2_norm_ratio_ridge_over_ols']:.4f}, so Ridge stays close to OLS."
            )
        limitations.append(
            "[No Feature Elimination] Ridge shrinks coefficients smoothly toward zero via L2 regularization and never sets coefficients exactly to zero; all predictors remain in the model."
        )
        if rc["spearman_abs_resid_vs_fitted_pvalue"] < 0.05:
            limitations.append(
                f"[Heteroskedasticity / Nonlinearity] Spearman(|e|, y_hat) = {rc['spearman_abs_resid_vs_fitted_r']:.3f} (p = {rc['spearman_abs_resid_vs_fitted_pvalue']:.2e})."
            )

    elif model_key == "lasso":
        pt = diagnostics["penalty_tuning"]
        sp = diagnostics["sparsity_and_shrinkage"]
        st = diagnostics["selection_stability"]
        cvg = diagnostics["convergence"]
        rc = diagnostics["residual_checks"]

        sections.append((
            "Training-Only Cross-Validation & Coordinate Descent Convergence",
            [
                ("CV strategy", pt["cv_strategy"]),
                ("alpha_max / selected alpha (lambda)", f"{_fmt(pt['alpha_max'])} / {_fmt(pt['selected_alpha'])}"),
                ("Best CV RMSE (+/- SD)", f"{_fmt(pt['best_cv_rmse'])} (+/- {_fmt(pt['best_cv_rmse_std'])}) {units}"),
                ("Best CV R²", _fmt(pt["best_cv_r2"])),
                ("Final iterations / converged", f"{cvg['n_iter_final']} of {cvg['max_iter']} / {_fmt(cvg['converged_final'])}"),
                ("CV path convergence warnings", _fmt(cvg["cv_convergence_warnings"])),
            ],
        ))
        sections.append((
            "Sparsity, L1 Shrinkage & Active-Set Stability",
            [
                ("Active |A(alpha)| / Eliminated (zero)", f"{sp['n_nonzero_features']} / {sp['n_zero_features']} (of {sp['n_total_features']}, {sp['sparsity_pct']:.1f}% sparse)"),
                ("||beta_OLS||_1 / ||beta_Lasso||_1", f"{_fmt(sp['ols_coef_l1_norm'])} / {_fmt(sp['lasso_coef_l1_norm'])}"),
                ("L1 norm ratio (Lasso / OLS)", _fmt(sp["l1_norm_ratio_lasso_over_ols"])),
                ("Always / Never / Unstable across CV folds", f"{st['n_always_selected_in_cv']} / {st['n_never_selected_in_cv']} / {st['n_unstable_across_cv_folds']}"),
                ("Spearman |e| vs. y_hat (r / p)", f"{_fmt(rc['spearman_abs_resid_vs_fitted_r'])} (p = {_fmt(rc['spearman_abs_resid_vs_fitted_pvalue'])})"),
                ("Durbin-Watson", _fmt(rc["durbin_watson"])),
            ],
        ))

        interpretations.append(
            f"[Computed Fact] {pt['cv_strategy']} selected alpha = {pt['selected_alpha']:.4g} "
            f"(CV RMSE = {_fmt(pt['best_cv_rmse'])} {units}, CV R² = {pt['best_cv_r2']:.4f}), converging in {cvg['n_iter_final']} coordinate-descent iterations."
        )
        interpretations.append(
            f"[Interpretation] Soft-thresholding retained {sp['n_nonzero_features']} of {sp['n_total_features']} predictors and eliminated {sp['n_zero_features']} ({sp['sparsity_pct']:.1f}% sparse), "
            f"reducing the L1 norm from {_fmt(sp['ols_coef_l1_norm'])} (OLS) to {_fmt(sp['lasso_coef_l1_norm'])} (ratio = {sp['l1_norm_ratio_lasso_over_ols']:.4f})."
        )
        if sp["eliminated_features"]:
            elim_names = ", ".join(e["feature"] for e in sp["eliminated_features"][:5])
            max_elim_corr = max(e["max_abs_corr_with_retained"] for e in sp["eliminated_features"])
            interpretations.append(
                f"[Computed Fact] Eliminated predictors include {elim_names} (max |corr| with a retained predictor = {max_elim_corr:.3f})."
            )
        if st["n_unstable_across_cv_folds"] > 0:
            unstable_ex = ", ".join(
                f"{u['feature']} ({u['cv_selection_freq']:.2f})" for u in st["unstable_features"][:4]
            )
            limitations.append(
                f"[Selection Instability Under Collinearity] {st['n_unstable_across_cv_folds']} predictor(s) varied in selection across CV folds (e.g., {unstable_ex}), because L1 KKT pinning selects one representative among correlated features."
            )
        if not cvg["converged_final"] or cvg["cv_convergence_warnings"] > 0:
            limitations.append(
                f"[Solver Convergence] Coordinate descent emitted warning(s) (final converged = {cvg['converged_final']}, CV warnings = {cvg['cv_convergence_warnings']})."
            )

    if cfg.is_binary_classification and "test_classification" in metrics:
        tc = metrics["test_classification"]
        limitations.append(
            f"[Linear Probability Model Limitation] {tc['n_outside_unit_interval']} of {metrics['n_test']} test predictions ({tc['pct_outside_unit_interval']:.2f}%) lie outside [0, 1] (range [{tc['pred_min']:.3f}, {tc['pred_max']:.3f}])."
        )
    if cfg.is_ordinal_target:
        limitations.append(
            f"[Ordinal Target Assumption] '{cfg.target}' is an integer ordinal score modeled continuously."
        )
    if cfg.split_strategy in ("chronological", "chronological_date"):
        limitations.append(
            "[Temporal Structure] Static linear predictors omit explicit lag or seasonal terms across the chronological train-to-test boundary."
        )
    if cfg.split_strategy == "group":
        limitations.append(
            f"[Group Generalization] Evaluation tests unseen '{cfg.group_column}' clusters without subject-specific random intercepts."
        )
    if test_r2 < train_r2 - 0.10:
        limitations.append(
            f"[Generalization Gap] Test R² ({test_r2:.4f}) drops from Train R² ({train_r2:.4f}), indicating distribution shift or limited linear generalization."
        )
    limitations.append(
        "[Linearity Assumption] Linear models assume additive linear relationships between standardized predictors and conditional target expectation."
    )

    return sections, interpretations, limitations


MODEL_LATEX = {
    "ols": {
        "short": "Fits an unpenalized hyperplane minimizing the training residual sum of squares via SVD (`numpy.linalg.lstsq`).",
        "eq": r"$$\min_{\boldsymbol{\beta} \in \mathbb{R}^{p+1}} \|\mathbf{y} - \tilde{\mathbf{X}}_{\mathrm{aug}}\boldsymbol{\beta}\|_2^2, \qquad (\tilde{\mathbf{X}}_{\mathrm{aug}}^\top \tilde{\mathbf{X}}_{\mathrm{aug}})\hat{\boldsymbol{\beta}} = \tilde{\mathbf{X}}_{\mathrm{aug}}^\top \mathbf{y}$$",
    },
    "ridge": {
        "short": "Adds a squared L2 penalty on standardized slopes (unpenalized intercept) to stabilize ill-conditioned matrices and reduce estimator variance.",
        "eq": r"$$\min_{\beta_0 \in \mathbb{R},\, \boldsymbol{\beta} \in \mathbb{R}^p} \|\mathbf{y} - \beta_0\mathbf{1} - \tilde{\mathbf{X}}\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_2^2, \qquad \hat{\boldsymbol{\beta}}_{\alpha} = (\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} + \alpha \mathbf{I}_p)^{-1}\tilde{\mathbf{X}}^\top \tilde{\mathbf{y}} \quad (\lambda \equiv \alpha)$$",
    },
    "lasso": {
        "short": "Adds an L1 penalty to perform continuous shrinkage and automatic variable selection via coordinate-wise soft-thresholding.",
        "eq": r"$$\min_{\beta_0 \in \mathbb{R},\, \boldsymbol{\beta} \in \mathbb{R}^p} \frac{1}{2n}\|\mathbf{y} - \beta_0\mathbf{1} - \tilde{\mathbf{X}}\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_1, \qquad \hat{\beta}_j = \mathcal{S}\!\left(\frac{1}{n}\tilde{\mathbf{x}}_j^\top \mathbf{r}^{(-j)}, \alpha\right) \quad (\lambda \equiv \alpha)$$",
    },
}

REPORT_TEMPLATE = JINJA_ENV.from_string(r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ model_title }} — {{ dataset_name }}</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" crossorigin="anonymous"></script>
  <style>
    :root {
      --bg: #f8fafc;
      --card: #ffffff;
      --subtle: #f1f5f9;
      --text: #0f172a;
      --muted: #475569;
      --border: #e2e8f0;
      --accent: #1d4ed8;
      --accent-soft: #eff6ff;
      --navy: #1e3a5f;
    }
    * { box-sizing: border-box; }
    body {
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
      margin: 0;
      padding: 24px;
    }
    .container { max-width: 1120px; margin: 0 auto; }
    .topnav { display: flex; gap: 12px; margin-bottom: 14px; font-size: 0.85rem; flex-wrap: wrap; }
    .topnav a { color: var(--accent); text-decoration: none; font-weight: 600; }
    .topnav a:hover { text-decoration: underline; }
    header {
      background: var(--card);
      border: 1px solid var(--border);
      border-top: 4px solid var(--navy);
      border-radius: 8px;
      padding: 20px 24px;
      margin-bottom: 18px;
    }
    header h1 { margin: 6px 0; font-size: 1.55rem; color: var(--navy); }
    .subtitle { color: var(--muted); font-size: 0.92rem; }
    .badge {
      display: inline-block;
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid #bfdbfe;
      border-radius: 999px;
      padding: 2px 10px;
      font-size: 0.78rem;
      font-weight: 600;
      margin-right: 6px;
    }
    section {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px 24px;
      margin-bottom: 18px;
    }
    h2 {
      margin-top: 0;
      font-size: 1.15rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
      color: var(--navy);
    }
    h3 { font-size: 0.98rem; margin-top: 16px; color: var(--text); }
    table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.88rem; }
    th, td { border: 1px solid var(--border); padding: 8px 10px; text-align: left; }
    th { background: var(--subtle); font-weight: 600; color: var(--navy); }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    tr.intercept-row { background: var(--accent-soft); font-weight: 600; }
    tr.zero-row { color: #94a3b8; }
    .math-box {
      background: var(--subtle);
      border-left: 4px solid var(--accent);
      border-radius: 6px;
      padding: 12px 16px;
      margin: 12px 0;
      overflow-x: auto;
    }
    .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
    .plot-card { border: 1px solid var(--border); border-radius: 6px; padding: 12px; background: #fff; }
    .plot-card img { width: 100%; height: auto; display: block; border-radius: 4px; }
    .plot-title { font-weight: 600; margin-top: 8px; font-size: 0.92rem; color: var(--navy); }
    .plot-caption { color: var(--muted); font-size: 0.82rem; margin-top: 4px; }
    code { background: var(--subtle); padding: 1px 5px; border-radius: 4px; font-size: 0.86em; }
    footer { color: var(--muted); font-size: 0.82rem; text-align: center; margin-top: 16px; }
  </style>
</head>
<body>
<div class="container">
  <nav class="topnav" aria-label="Breadcrumb">
    <a href="../index.html">&larr; Dashboard</a>
    <a href="summary.html">{{ model_key | upper }} Summary</a>
    <a href="{{ uci_url }}" target="_blank" rel="noopener noreferrer">UCI Dataset {{ metadata.uci_id }} &nearr;</a>
  </nav>

  <header>
    <div>
      <span class="badge">{{ model_key | upper }}</span>
      <span class="badge">UCI ID {{ metadata.uci_id }}</span>
      <span class="badge">Seed {{ metadata.seed }}</span>
    </div>
    <h1>{{ model_title }} &mdash; {{ dataset_name }}</h1>
    <div class="subtitle">
      Target: <code>{{ metadata.target }}</code> ({{ metadata.target_units }}) &middot;
      Run UTC: {{ metadata.run_datetime_utc }}
    </div>
  </header>

  <section>
    <h2>1. Dataset and Task Description</h2>
    <p>{{ dataset_description }}</p>
    <table>
      <tr><th>Property</th><th>Observed Value</th></tr>
      <tr><td>Dataset Key / UCI ID</td><td><code>{{ metadata.dataset }}</code> (<a href="{{ uci_url }}" target="_blank" rel="noopener noreferrer">UCI {{ metadata.uci_id }} &nearr;</a>)</td></tr>
      <tr><td>Target Variable &amp; Units</td><td><code>{{ metadata.target }}</code> ({{ metadata.target_units }})</td></tr>
      <tr><td>Observations</td><td>{{ prep.n_raw }} raw &rarr; {{ prep.n_clean }} clean ({{ prep.n_train }} train / {{ prep.n_test }} test)</td></tr>
      <tr><td>Features</td><td>{{ prep.n_raw_features }} raw &rarr; {{ prep.n_encoded_features }} encoded model predictors</td></tr>
      <tr><td>Train Target Range</td><td>mean = {{ "%.4g"|format(prep.target_stats_train.mean) }}, sd = {{ "%.4g"|format(prep.target_stats_train.std) }}, [{{ "%.4g"|format(prep.target_stats_train.min) }}, {{ "%.4g"|format(prep.target_stats_train.max) }}] {{ metadata.target_units }}</td></tr>
      <tr><td>Test Target Range</td><td>mean = {{ "%.4g"|format(prep.target_stats_test.mean) }}, sd = {{ "%.4g"|format(prep.target_stats_test.std) }}, [{{ "%.4g"|format(prep.target_stats_test.min) }}, {{ "%.4g"|format(prep.target_stats_test.max) }}] {{ metadata.target_units }}</td></tr>
    </table>
    {% if dataset_notes %}
    <ul>{% for note in dataset_notes %}<li>{{ note }}</li>{% endfor %}</ul>
    {% endif %}
  </section>

  <section>
    <h2>2. Experimental Setup</h2>
    <table>
      <tr><th>Component</th><th>Configuration</th></tr>
      <tr><td>Split Strategy</td><td><code>{{ prep.split_strategy }}</code> (persisted at <code>{{ metadata.split_reference }}</code>)</td></tr>
      <tr><td>Cross-Validation</td><td><code>{{ prep.cv_strategy }}</code> (training data only)</td></tr>
      <tr><td>Excluded Columns</td><td>{% if prep.excluded_columns %}{{ prep.excluded_columns | join(", ") }}{% else %}None{% endif %}</td></tr>
      <tr><td>Categorical Encoding</td><td>{% if prep.categorical_columns %}{{ prep.categorical_columns | join(", ") }} (One-Hot drop-first, fit on train only){% else %}None{% endif %}</td></tr>
      <tr><td>Missing Value Imputation</td><td>{% if prep.missing_values_in_features %}Training-median imputation: {% for k, v in prep.missing_values_in_features.items() %}{{ k }} ({{ v }}){% if not loop.last %}, {% endif %}{% endfor %}{% else %}None required{% endif %}</td></tr>
      <tr><td>Standardization</td><td>Zero-mean, unit-variance <code>StandardScaler</code> fit strictly on training data (and separately within each CV fold). Intercept unscaled and unpenalized.</td></tr>
    </table>
  </section>

  <section>
    <h2>3. Mathematical Methodology</h2>
    <p>{{ model_short }}</p>
    <div class="math-box">{{ model_eq }}</div>
    <p>
      <strong>Implementation Note:</strong> <code>{{ metadata.objective_convention }}</code>
      {% if metadata.penalty_settings.selected_alpha is defined %}
      <br><strong>CV-Selected Hyperparameter:</strong> \(\alpha^* = {{ "%.6g"|format(metadata.penalty_settings.selected_alpha) }}\) over {{ metadata.penalty_settings.n_alphas }} grid points in \([{{ "%.4g"|format(metadata.penalty_settings.alpha_grid_min) }}, {{ "%.4g"|format(metadata.penalty_settings.alpha_grid_max) }}]\).
      {% endif %}
    </p>
  </section>

  <section>
    <h2>4. Model Results</h2>
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Units</th>
          <th class="num">Train ({{ metrics.n_train }})</th>
          <th class="num">Test ({{ metrics.n_test }})</th>
          <th class="num">Test &minus; Train</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>MAE</td><td>{{ metrics.target_units }}</td><td class="num">{{ "%.5g"|format(metrics.train.mae) }}</td><td class="num">{{ "%.5g"|format(metrics.test.mae) }}</td><td class="num">{{ "%+.4g"|format(metrics.test.mae - metrics.train.mae) }}</td></tr>
        <tr><td>MSE</td><td>({{ metrics.target_units }})&sup2;</td><td class="num">{{ "%.5g"|format(metrics.train.mse) }}</td><td class="num">{{ "%.5g"|format(metrics.test.mse) }}</td><td class="num">{{ "%+.4g"|format(metrics.test.mse - metrics.train.mse) }}</td></tr>
        <tr><td>RMSE</td><td>{{ metrics.target_units }}</td><td class="num">{{ "%.5g"|format(metrics.train.rmse) }}</td><td class="num">{{ "%.5g"|format(metrics.test.rmse) }}</td><td class="num">{{ "%+.4g"|format(metrics.test.rmse - metrics.train.rmse) }}</td></tr>
        <tr><td>R&sup2;</td><td>dimensionless</td><td class="num">{{ "%.5f"|format(metrics.train.r2) }}</td><td class="num">{{ "%.5f"|format(metrics.test.r2) }}</td><td class="num">{{ "%+.5f"|format(metrics.test.r2 - metrics.train.r2) }}</td></tr>
      </tbody>
    </table>

    {% if metrics.is_binary_classification %}
    <h3>Linear Probability Model (LPM) Classification &amp; Bounds</h3>
    <table>
      <thead><tr><th>Metric</th><th class="num">Train</th><th class="num">Test</th></tr></thead>
      <tbody>
        <tr><td>Accuracy (0.50 threshold)</td><td class="num">{{ "%.4f"|format(metrics.train_classification.accuracy) }}</td><td class="num">{{ "%.4f"|format(metrics.test_classification.accuracy) }}</td></tr>
        <tr><td>Precision / Recall / F1</td><td class="num">{{ "%.3f"|format(metrics.train_classification.precision) }} / {{ "%.3f"|format(metrics.train_classification.recall) }} / {{ "%.3f"|format(metrics.train_classification.f1) }}</td><td class="num">{{ "%.3f"|format(metrics.test_classification.precision) }} / {{ "%.3f"|format(metrics.test_classification.recall) }} / {{ "%.3f"|format(metrics.test_classification.f1) }}</td></tr>
        <tr><td>ROC AUC</td><td class="num">{{ "%.4f"|format(metrics.train_classification.roc_auc) }}</td><td class="num">{{ "%.4f"|format(metrics.test_classification.roc_auc) }}</td></tr>
        <tr><td>Predictions Outside [0, 1]</td><td class="num">{{ metrics.train_classification.n_outside_unit_interval }} ({{ "%.2f"|format(metrics.train_classification.pct_outside_unit_interval) }}%)</td><td class="num">{{ metrics.test_classification.n_outside_unit_interval }} ({{ "%.2f"|format(metrics.test_classification.pct_outside_unit_interval) }}%)</td></tr>
      </tbody>
    </table>
    {% endif %}
  </section>

  <section>
    <h2>5. Coefficient Interpretation</h2>
    <table>
      <thead>
        <tr>
          <th>Role</th>
          <th>Feature</th>
          <th class="num">Coef (Std)</th>
          <th class="num">Coef (Unstd)</th>
          {% if has_ols_ref %}<th class="num">OLS Coef (Std)</th>{% endif %}
          {% if has_shrinkage_ratio %}<th class="num">Shrinkage Ratio</th>{% endif %}
          {% if has_retained %}<th class="num">Active</th><th class="num">CV Freq</th>{% endif %}
        </tr>
      </thead>
      <tbody>
        {% for row in coef_rows %}
        <tr class="{% if row.role == 'intercept' %}intercept-row{% elif row.retained is defined and not row.retained %}zero-row{% endif %}">
          <td>{{ row.role }}</td>
          <td><code>{{ row.feature }}</code></td>
          <td class="num">{{ "%.6g"|format(row.coefficient_standardized) }}</td>
          <td class="num">{{ "%.6g"|format(row.coefficient_unstandardized) }}</td>
          {% if has_ols_ref %}<td class="num">{{ "%.6g"|format(row.ols_coefficient_standardized) }}</td>{% endif %}
          {% if has_shrinkage_ratio %}<td class="num">{% if row.shrinkage_ratio_vs_ols == row.shrinkage_ratio_vs_ols %}{{ "%.4f"|format(row.shrinkage_ratio_vs_ols) }}{% else %}&mdash;{% endif %}</td>{% endif %}
          {% if has_retained %}<td class="num">{{ "Yes" if row.retained else "No (0)" }}</td><td class="num">{{ "%.2f"|format(row.cv_selection_freq) }}</td>{% endif %}
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </section>

  <section>
    <h2>6. Diagnostics and Visualizations</h2>
    <div class="grid-2">
      {% for p in plots %}
      <div class="plot-card">
        <img src="data:image/png;base64,{{ p.b64 }}" alt="{{ p.title }}">
        <div class="plot-title">{{ p.title }}</div>
        <div class="plot-caption">{{ p.caption }}</div>
      </div>
      {% endfor %}
    </div>

    {% for group_name, items in diagnostic_sections %}
    <h3>{{ group_name }}</h3>
    <table>
      {% for k, v in items %}
      <tr><td>{{ k }}</td><td class="num">{{ v }}</td></tr>
      {% endfor %}
    </table>
    {% endfor %}
  </section>

  <section>
    <h2>7. Limitations and Conclusions</h2>
    <h3>Facts &amp; Interpretations</h3>
    <ul>{% for item in interpretations %}<li>{{ item }}</li>{% endfor %}</ul>
    <h3>Assumptions &amp; Limitations</h3>
    <ul>{% for lim in limitations %}<li>{{ lim }}</li>{% endfor %}</ul>
  </section>

  <footer>
    Generated strictly from saved artifacts in <code>results/{{ model_key }}/{{ metadata.dataset }}/</code> &middot;
    Python {{ metadata.dependencies.python }}, NumPy {{ metadata.dependencies.numpy }}, scikit-learn {{ metadata.dependencies.scikit_learn }}
  </footer>
</div>
<script>
document.addEventListener("DOMContentLoaded", function () {
  if (typeof renderMathInElement === "function") {
    renderMathInElement(document.body, {
      delimiters: [
        {left: "$$", right: "$$", display: true},
        {left: "\\(", right: "\\)", display: false}
      ],
      throwOnError: false
    });
  }
});
</script>
</body>
</html>
""")

SUMMARY_TEMPLATE = JINJA_ENV.from_string("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ model_title }} — Cross-Dataset Summary</title>
  <style>
    body { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 24px; line-height: 1.5; }
    .container { max-width: 1200px; margin: 0 auto; background: #fff; border: 1px solid #e2e8f0; border-top: 4px solid #1e3a5f; border-radius: 8px; padding: 24px; }
    h1 { margin-top: 6px; color: #1e3a5f; }
    .topnav a { color: #1d4ed8; text-decoration: none; font-weight: 600; font-size: 0.88rem; }
    .warning { background: #eff6ff; border-left: 4px solid #1d4ed8; padding: 12px 16px; border-radius: 4px; margin: 14px 0; font-size: 0.9rem; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 0.88rem; }
    th, td { border: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; }
    th { background: #f1f5f9; color: #1e3a5f; }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    a { color: #1d4ed8; text-decoration: none; font-weight: 600; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
<div class="container">
  <div class="topnav"><a href="../index.html">&larr; Grand Dashboard</a></div>
  <h1>{{ model_title }} &mdash; All-Dataset Summary</h1>
  <div class="warning">
    <strong>Unit Comparability:</strong> Raw RMSE and MAE carry each dataset's native target units and must not be compared across datasets with different units. Use dimensionless <strong>Test R&sup2;</strong> and unit-free <strong>Normalized RMSE (Test RMSE / Test Target SD)</strong> for cross-dataset comparisons.
  </div>
  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Split</th>
        <th class="num">Train / Test</th>
        <th class="num">Predictors</th>
        <th class="num">Train R&sup2;</th>
        <th class="num">Test R&sup2;</th>
        <th class="num">Test RMSE (Native Units)</th>
        <th class="num">Norm. RMSE</th>
        <th>Key Diagnostic</th>
        <th>Report</th>
      </tr>
    </thead>
    <tbody>
      {% for r in rows %}
      <tr>
        <td><strong>{{ r.dataset_name }}</strong><br><small><code>{{ r.dataset }}</code></small></td>
        <td><code>{{ r.split_strategy }}</code></td>
        <td class="num">{{ r.n_train }} / {{ r.n_test }}</td>
        <td class="num">{{ r.n_features }}</td>
        <td class="num">{{ "%.4f"|format(r.train_r2) }}</td>
        <td class="num">{{ "%.4f"|format(r.test_r2) }}</td>
        <td class="num">{{ "%.4g"|format(r.test_rmse) }} <small>{{ r.target_units }}</small></td>
        <td class="num">{{ "%.4f"|format(r.norm_rmse) }}</td>
        <td>{{ r.key_diag }}</td>
        <td><a href="{{ r.dataset }}.html">Report &nearr;</a></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>
</body>
</html>
""")


def generate_dataset_report(model_key: str, dataset_key: str) -> Path:
    """Generate standalone HTML report at reports/<model>/<dataset>.html from saved artifacts."""
    res_dir = RESULTS_DIR / model_key / dataset_key
    if not (res_dir / "metrics.json").exists():
        raise FileNotFoundError(f"Missing saved artifacts in {res_dir}")

    with open(res_dir / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(res_dir / "diagnostics.json", "r", encoding="utf-8") as f:
        diagnostics = json.load(f)
    with open(res_dir / "run_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    coef_df = pd.read_csv(res_dir / "coefficients.csv")

    cfg = DATASETS[dataset_key]
    plots = _load_plots(res_dir / "plots", model_key, metrics, diagnostics)
    diag_sections, interpretations, limitations = _build_diagnostic_sections_and_narrative(
        model_key, dataset_key, metrics, coef_df, diagnostics, metadata
    )

    html = REPORT_TEMPLATE.render(
        model_key=model_key,
        model_title=MODEL_TITLES[model_key],
        model_short=MODEL_LATEX[model_key]["short"],
        model_eq=MODEL_LATEX[model_key]["eq"],
        dataset_name=cfg.name,
        dataset_description=cfg.description,
        dataset_notes=cfg.notes,
        uci_url=f"https://archive.ics.uci.edu/dataset/{cfg.uci_id}",
        metadata=metadata,
        prep=metadata["preprocessing"],
        metrics=metrics,
        coef_rows=coef_df.to_dict(orient="records"),
        has_ols_ref="ols_coefficient_standardized" in coef_df.columns,
        has_shrinkage_ratio="shrinkage_ratio_vs_ols" in coef_df.columns,
        has_retained="retained" in coef_df.columns,
        plots=plots,
        diagnostic_sections=diag_sections,
        interpretations=interpretations,
        limitations=limitations,
    )

    out_dir = REPORTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_key}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_summary_report(model_key: str) -> Path:
    """Generate cross-dataset summary HTML report at reports/<model>/summary.html."""
    rows = []
    for dataset_key, cfg in DATASETS.items():
        res_dir = RESULTS_DIR / model_key / dataset_key
        if not (res_dir / "metrics.json").exists():
            continue
        with open(res_dir / "metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        with open(res_dir / "diagnostics.json", "r", encoding="utf-8") as f:
            diag = json.load(f)
        with open(res_dir / "run_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        prep = meta["preprocessing"]
        test_sd = prep["target_stats_test"]["std"]
        norm_rmse = metrics["test"]["rmse"] / test_sd if test_sd > 0 else float("nan")

        if model_key == "ols":
            la = diag["numerical_linear_algebra"]
            key_diag = f"rank {la['matrix_rank']}/{la['n_augmented_columns']}, kappa={_fmt(la['condition_number_standardized_with_intercept'])}"
        elif model_key == "ridge":
            pt = diag["penalty_tuning"]
            sd = diag["shrinkage_and_degrees_of_freedom"]
            key_diag = f"alpha={pt['selected_alpha']:.3g}, df={sd['effective_df_predictors']:.1f}/{sd['n_predictors']}, L2_ratio={sd['l2_norm_ratio_ridge_over_ols']:.3f}"
        else:
            pt = diag["penalty_tuning"]
            sp = diag["sparsity_and_shrinkage"]
            key_diag = f"alpha={pt['selected_alpha']:.3g}, active={sp['n_nonzero_features']}/{sp['n_total_features']}"

        rows.append(
            {
                "dataset": dataset_key,
                "dataset_name": cfg.name,
                "split_strategy": prep["split_strategy"],
                "n_train": metrics["n_train"],
                "n_test": metrics["n_test"],
                "n_features": prep["n_encoded_features"],
                "train_r2": metrics["train"]["r2"],
                "test_r2": metrics["test"]["r2"],
                "test_rmse": metrics["test"]["rmse"],
                "target_units": metrics["target_units"],
                "norm_rmse": norm_rmse,
                "key_diag": key_diag,
            }
        )

    out_dir = REPORTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.html"
    html = SUMMARY_TEMPLATE.render(model_title=MODEL_TITLES[model_key], rows=rows)
    out_path.write_text(html, encoding="utf-8")
    return out_path
