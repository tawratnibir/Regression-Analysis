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
    "ridge": "Ridge Regression (ℓ₂ Regularization)",
    "lasso": "Lasso Regression (ℓ₁ Regularization)",
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
            f"Breusch-Pagan \\(\\mathrm{{LM}} = {_fmt(het['breusch_pagan_lm_stat'])}\\) (\\(p = {_fmt(het['breusch_pagan_pvalue'])}\\)); "
            f"Spearman \\(r_s(|\\mathbf{{e}}|, \\hat{{\\mathbf{{y}}}}) = {het['spearman_abs_resid_vs_fitted_r']:.3f}\\)."
        )
        captions["residual_distribution.png"] = (
            f"Skewness = {norm['skewness']:.3f}, excess kurtosis = {norm['excess_kurtosis']:.3f}, "
            f"Jarque-Bera \\(p = {_fmt(norm['jarque_bera_pvalue'])}\\)."
        )
        captions["qq_plot.png"] = (
            f"Shapiro-Wilk \\(W = {norm['shapiro_wilk_stat']:.4f}\\) (\\(p = {_fmt(norm['shapiro_wilk_pvalue'])}\\), "
            f"\\(n = {norm['shapiro_wilk_n']}\\))."
        )
        captions["cooks_distance.png"] = (
            f"{inf['n_high_cooks_d']} training points exceed \\(4/n = {inf['cooks_threshold_4_over_n']:.4f}\\) "
            f"(\\(\\max D_i = {_fmt(inf['max_cooks_d'])}\\))."
        )

    elif model_key == "ridge":
        pt = diagnostics["penalty_tuning"]
        sd = diagnostics["shrinkage_and_degrees_of_freedom"]
        rc = diagnostics["residual_checks"]
        captions["residuals_vs_fitted.png"] = (
            f"Spearman \\(r_s(|\\mathbf{{e}}|, \\hat{{\\mathbf{{y}}}}) = {rc['spearman_abs_resid_vs_fitted_r']:.3f}\\) "
            f"(\\(p = {_fmt(rc['spearman_abs_resid_vs_fitted_pvalue'])}\\)); \\(\\mathrm{{DW}} = {_fmt(rc['durbin_watson'])}\\)."
        )
        captions["cv_curve.png"] = (
            f"{pt['cv_strategy']} selected \\(\\alpha^* = {_fmt(pt['selected_alpha'])}\\) "
            f"(CV RMSE = {_fmt(pt['best_cv_rmse'])} ± {_fmt(pt['best_cv_rmse_std'])} {units}, CV \\(R^2 = {pt['best_cv_r2']:.4f}\\))."
        )
        captions["coefficient_path.png"] = (
            f"Continuous \\(\\ell_2\\) shrinkage across \\(\\alpha\\); effective \\(\\operatorname{{df}}(\\alpha^*) = {sd['effective_df_predictors']:.2f} / {sd['n_predictors']}\\)."
        )
        captions["ols_vs_ridge_coefficients.png"] = (
            f"\\(\\ell_2\\) norm reduced from {_fmt(sd['ols_coef_l2_norm'])} (OLS) to {_fmt(sd['ridge_coef_l2_norm'])} "
            f"(ratio = {sd['l2_norm_ratio_ridge_over_ols']:.4f}); all {sd['n_predictors']} predictors remain nonzero."
        )

    elif model_key == "lasso":
        pt = diagnostics["penalty_tuning"]
        sp = diagnostics["sparsity_and_shrinkage"]
        rc = diagnostics["residual_checks"]
        captions["residuals_vs_fitted.png"] = (
            f"Spearman \\(r_s(|\\mathbf{{e}}|, \\hat{{\\mathbf{{y}}}}) = {rc['spearman_abs_resid_vs_fitted_r']:.3f}\\) "
            f"(\\(p = {_fmt(rc['spearman_abs_resid_vs_fitted_pvalue'])}\\)); \\(\\mathrm{{DW}} = {_fmt(rc['durbin_watson'])}\\)."
        )
        captions["cv_curve.png"] = (
            f"{pt['cv_strategy']} selected \\(\\alpha^* = {_fmt(pt['selected_alpha'])}\\) "
            f"(CV RMSE = {_fmt(pt['best_cv_rmse'])} ± {_fmt(pt['best_cv_rmse_std'])} {units}, CV \\(R^2 = {pt['best_cv_r2']:.4f}\\))."
        )
        captions["coefficient_path.png"] = (
            f"Piecewise-linear \\(\\ell_1\\) paths from \\(\\alpha_{{\\max}} = {_fmt(pt['alpha_max'])}\\); "
            f"\\(\\ell_1\\) norm ratio vs. OLS = {sp['l1_norm_ratio_lasso_over_ols']:.4f}."
        )
        captions["sparsity_vs_alpha.png"] = (
            f"Retains \\(|A(\\alpha^*)| = {sp['n_nonzero_features']}\\) of {sp['n_total_features']} predictors "
            f"and zeros {sp['n_zero_features']} ({sp['sparsity_pct']:.1f}% sparse) at \\(\\alpha^*\\)."
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
        f"[Computed Fact] Held-out test performance (\\(n_{{\\mathrm{{test}}}} = {metrics['n_test']}\\)): \\(R^2 = {test_r2:.4f}\\), "
        f"\\(\\mathrm{{RMSE}} = {_fmt(test_rmse)}\\) {units} (Train \\(R^2 = {train_r2:.4f}\\), Train \\(\\mathrm{{RMSE}} = {_fmt(metrics['train']['rmse'])}\\) {units})."
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
                (r"\(\max |\tilde{\mathbf{X}}_{\mathrm{aug}}^\top \mathbf{e}|\) (residual orthogonality)", _fmt(ex["max_abs_Xt_residuals"])),
                (r"\(\sum_{i=1}^n e_i\) (sum of training residuals)", _fmt(ex["residual_sum"])),
                (r"\(|\bar{y} - \bar{\hat{y}}|\) (sample-mean fitted identity)", _fmt(ex["mean_fitted_identity_abs_diff"])),
                (r"\(|\mathrm{SST} - (\mathrm{SSR} + \mathrm{SSE})|\) (ANOVA identity)", _fmt(ex["anova_identity_abs_error"])),
            ],
        ))
        sections.append((
            "Matrix Rank, Conditioning & Multicollinearity",
            [
                (r"\(\operatorname{rank}(\tilde{\mathbf{X}}_{\mathrm{aug}}) / k\) (\(k = p + 1\))", f"{la['matrix_rank']} / {la['n_augmented_columns']}"),
                ("Rank deficient", _fmt(la["is_rank_deficient"])),
                (r"\(\kappa(\tilde{\mathbf{X}}_{\mathrm{aug}})\) (standardized)", _fmt(la["condition_number_standardized_with_intercept"])),
                (r"\(\kappa(\mathbf{X}_{\mathrm{aug}})\) (unstandardized)", _fmt(la["condition_number_unstandardized_with_intercept"])),
                (r"\(\max \mathrm{VIF}_j\) / features with \(\mathrm{VIF}_j \gt 10\)", f"{_fmt(mc['max_vif'])} / {mc['n_vif_gt_10']}"),
                (r"\(\max |\hat{y} - \hat{y}_{\mathrm{sklearn}}|\) (test set)", _fmt(la["sklearn_max_abs_test_pred_diff"])),
            ],
        ))
        sections.append((
            "Statistical Assumption Checks (Training Residuals)",
            [
                (r"Breusch-Pagan \(\mathrm{LM}\) / \(p\)-value", f"{_fmt(het['breusch_pagan_lm_stat'])} (p = {_fmt(het['breusch_pagan_pvalue'])})"),
                (r"Spearman \(r_s(|\mathbf{e}|, \hat{\mathbf{y}})\) / \(p\)-value", f"{_fmt(het['spearman_abs_resid_vs_fitted_r'])} (p = {_fmt(het['spearman_abs_resid_vs_fitted_pvalue'])})"),
                ("Skewness / Excess kurtosis", f"{_fmt(norm['skewness'])} / {_fmt(norm['excess_kurtosis'])}"),
                (r"Jarque-Bera \(p\) / Shapiro-Wilk \(p\)", f"{_fmt(norm['jarque_bera_pvalue'])} / {_fmt(norm['shapiro_wilk_pvalue'])}"),
                (r"High leverage (\(h_{ii} \gt 2k/n\)) / Cook's \(D_i \gt 4/n\)", f"{inf['n_high_leverage']} (max={_fmt(inf['max_leverage'])}) / {inf['n_high_cooks_d']} (max={_fmt(inf['max_cooks_d'])})"),
                (r"Durbin-Watson \(\mathrm{DW}\) / lag-1 \(\hat{\rho}_1(e)\)", f"{_fmt(ac['durbin_watson'])} / {_fmt(ac['lag1_residual_corr'])}"),
            ],
        ))

        interpretations.append(
            f"[Computed Fact] Exact normal-equation identities hold numerically: \\(\\max |\\tilde{{\\mathbf{{X}}}}_{{\\mathrm{{aug}}}}^\\top \\mathbf{{e}}| = {ex['max_abs_Xt_residuals']:.2e}\\), "
            f"\\(|\\bar{{y}} - \\bar{{\\hat{{y}}}}| = {ex['mean_fitted_identity_abs_diff']:.2e}\\), and predictions match scikit-learn within {la['sklearn_max_abs_test_pred_diff']:.2e}."
        )
        interpretations.append(
            f"[Interpretation] Predictor standardization changes the augmented condition number from \\(\\kappa(\\mathbf{{X}}_{{\\mathrm{{aug}}}}) = {_fmt(la['condition_number_unstandardized_with_intercept'])}\\) "
            f"(unstandardized) to \\(\\kappa(\\tilde{{\\mathbf{{X}}}}_{{\\mathrm{{aug}}}}) = {_fmt(la['condition_number_standardized_with_intercept'])}\\) (standardized), with \\(\\operatorname{{rank}}(\\tilde{{\\mathbf{{X}}}}_{{\\mathrm{{aug}}}}) = {la['matrix_rank']}\\) of \\(k = {la['n_augmented_columns']}\\)."
        )

        if la["is_rank_deficient"]:
            limitations.append(
                f"[Assumption Violation — Full Rank] Design matrix is rank-deficient (\\(\\operatorname{{rank}}(\\tilde{{\\mathbf{{X}}}}_{{\\mathrm{{aug}}}}) = {la['matrix_rank']} \\lt {la['n_augmented_columns']}\\)). Individual OLS coefficients are not uniquely identified; np.linalg.lstsq returns the minimum-norm pseudoinverse \\(\\hat{{\\boldsymbol{{\\beta}}}} = \\tilde{{\\mathbf{{X}}}}_{{\\mathrm{{aug}}}}^{{+}}\\mathbf{{y}}\\)."
            )
        if mc["n_vif_gt_10"] > 0 or mc.get("has_infinite_vif"):
            limitations.append(
                f"[Multicollinearity] {mc['n_vif_gt_10']} predictor(s) have \\(\\mathrm{{VIF}}_j \\gt 10\\) (max finite VIF = {_fmt(mc['max_vif'])}), inflating OLS estimator variance \\(\\sigma^2(\\tilde{{\\mathbf{{X}}}}_{{\\mathrm{{aug}}}}^\\top \\tilde{{\\mathbf{{X}}}}_{{\\mathrm{{aug}}}})^{{-1}}\\)."
            )
        if het["breusch_pagan_pvalue"] < 0.05:
            limitations.append(
                f"[Assumption Violation — Homoskedasticity] Breusch-Pagan test rejects constant disturbance variance (\\(p = {het['breusch_pagan_pvalue']:.2e}\\)); classical standard errors require White's sandwich estimator \\((\\tilde{{\\mathbf{{X}}}}_{{\\mathrm{{aug}}}}^\\top \\tilde{{\\mathbf{{X}}}}_{{\\mathrm{{aug}}}})^{{-1}}\\tilde{{\\mathbf{{X}}}}_{{\\mathrm{{aug}}}}^\\top \\mathbf{{e}}\\mathbf{{e}}^\\top \\tilde{{\\mathbf{{X}}}}_{{\\mathrm{{aug}}}}(\\tilde{{\\mathbf{{X}}}}_{{\\mathrm{{aug}}}}^\\top \\tilde{{\\mathbf{{X}}}}_{{\\mathrm{{aug}}}})^{{-1}}\\)."
            )
        if norm["jarque_bera_pvalue"] < 0.05:
            limitations.append(
                f"[Assumption Check — Normality] Residuals deviate from Gaussian normality (Jarque-Bera \\(p = {norm['jarque_bera_pvalue']:.2e}\\), skewness = {norm['skewness']:.3f}, excess kurtosis = {norm['excess_kurtosis']:.3f})."
            )
        if inf["n_high_cooks_d"] > 0:
            limitations.append(
                f"[Influential Observations] {inf['n_high_cooks_d']} training row(s) exceed Cook's \\(D_i \\gt 4/n\\) (max = {inf['max_cooks_d']:.4g})."
            )
        if ac["durbin_watson"] < 1.5 or ac["durbin_watson"] > 2.5:
            limitations.append(
                f"[Assumption Violation — No Autocorrelation] Durbin-Watson \\(\\mathrm{{DW}} = {ac['durbin_watson']:.3f}\\) (lag-1 \\(\\hat{{\\rho}}_1(e) = {ac['lag1_residual_corr']:.3f}\\)), indicating correlated errors."
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
                (r"Selected penalty \(\alpha^*\) (\(\lambda \equiv \alpha\))", _fmt(pt["selected_alpha"])),
                (r"Best CV RMSE (\(\pm\) SD)", f"{_fmt(pt['best_cv_rmse'])} (± {_fmt(pt['best_cv_rmse_std'])}) {units}"),
                (r"Best CV \(R^2\)", _fmt(pt["best_cv_r2"])),
                (r"\(\max |\hat{\beta}_j - \hat{\beta}_{j,\mathrm{SVD}}|\)", _fmt(pt["svd_verification_max_abs_coef_diff"])),
            ],
        ))
        sections.append((
            "Shrinkage, Effective Degrees of Freedom & Conditioning",
            [
                (r"Effective \(\operatorname{df}(\alpha^*)\) (slopes / +intercept)", f"{_fmt(sd['effective_df_predictors'])} / {_fmt(sd['effective_df_with_intercept'])} (of {sd['n_predictors']} predictors)"),
                (r"\(\|\hat{\boldsymbol{\beta}}_{\mathrm{OLS}}\|_2\) / \(\|\hat{\boldsymbol{\beta}}_{\alpha^*}\|_2\)", f"{_fmt(sd['ols_coef_l2_norm'])} / {_fmt(sd['ridge_coef_l2_norm'])}"),
                (r"\(\ell_2\) norm ratio (\(\|\hat{\boldsymbol{\beta}}_{\alpha^*}\|_2 / \|\hat{\boldsymbol{\beta}}_{\mathrm{OLS}}\|_2\))", _fmt(sd["l2_norm_ratio_ridge_over_ols"])),
                (r"\(\kappa(\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}})\) vs. \(\kappa(\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} + \alpha^* \mathbf{I}_p)\)", f"{_fmt(ns['x_centered_eig_condition_ols'])} -> {_fmt(ns['x_centered_eig_condition_ridge'])}"),
                (r"Spearman \(r_s(|\mathbf{e}|, \hat{\mathbf{y}})\) / \(p\)-value", f"{_fmt(rc['spearman_abs_resid_vs_fitted_r'])} (p = {_fmt(rc['spearman_abs_resid_vs_fitted_pvalue'])})"),
                (r"Durbin-Watson \(\mathrm{DW}\)", _fmt(rc["durbin_watson"])),
            ],
        ))

        interpretations.append(
            f"[Computed Fact] {pt['cv_strategy']} selected \\(\\alpha^* = {pt['selected_alpha']:.4g}\\) "
            f"(CV RMSE = {_fmt(pt['best_cv_rmse'])} {units}, CV \\(R^2 = {pt['best_cv_r2']:.4f}\\)), matching closed-form SVD within {pt['svd_verification_max_abs_coef_diff']:.2e}."
        )
        interpretations.append(
            f"[Interpretation] \\(\\ell_2\\) regularization trades controlled bias for variance reduction, shrinking the standardized \\(\\ell_2\\) norm from {_fmt(sd['ols_coef_l2_norm'])} (OLS) to {_fmt(sd['ridge_coef_l2_norm'])} "
            f"(ratio = {sd['l2_norm_ratio_ridge_over_ols']:.4f}) and reducing effective degrees of freedom from \\(p = {sd['n_predictors']}\\) to \\(\\operatorname{{df}}(\\alpha^*) = {sd['effective_df_predictors']:.2f}\\)."
        )
        interpretations.append(
            f"[Computed Fact] Adding \\(\\alpha^* \\mathbf{{I}}_p\\) stabilized the centered Gram condition number from \\(\\kappa(\\tilde{{\\mathbf{{X}}}}^\\top\\tilde{{\\mathbf{{X}}}}) = {_fmt(ns['x_centered_eig_condition_ols'])}\\) to \\(\\kappa(\\tilde{{\\mathbf{{X}}}}^\\top\\tilde{{\\mathbf{{X}}}} + \\alpha^*\\mathbf{{I}}_p) = {_fmt(ns['x_centered_eig_condition_ridge'])}\\)."
        )

        if sd["l2_norm_ratio_ridge_over_ols"] > 0.99:
            limitations.append(
                f"[Minimal Regularization] Selected \\(\\alpha^* = {pt['selected_alpha']:.4g}\\) yields \\(\\ell_2\\) ratio {sd['l2_norm_ratio_ridge_over_ols']:.4f}, so Ridge stays close to OLS."
            )
        limitations.append(
            "[No Feature Elimination] Ridge shrinks coefficients smoothly toward zero via \\(\\ell_2\\) regularization and never sets coefficients exactly to zero; all \\(p\\) predictors remain active."
        )
        if rc["spearman_abs_resid_vs_fitted_pvalue"] < 0.05:
            limitations.append(
                f"[Heteroskedasticity / Nonlinearity] Spearman \\(r_s(|\\mathbf{{e}}|, \\hat{{\\mathbf{{y}}}}) = {rc['spearman_abs_resid_vs_fitted_r']:.3f}\\) (\\(p = {rc['spearman_abs_resid_vs_fitted_pvalue']:.2e}\\))."
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
                (r"\(\alpha_{\max}\) / selected \(\alpha^*\) (\(\lambda \equiv \alpha\))", f"{_fmt(pt['alpha_max'])} / {_fmt(pt['selected_alpha'])}"),
                (r"Best CV RMSE (\(\pm\) SD)", f"{_fmt(pt['best_cv_rmse'])} (± {_fmt(pt['best_cv_rmse_std'])}) {units}"),
                (r"Best CV \(R^2\)", _fmt(pt["best_cv_r2"])),
                ("Final iterations / converged", f"{cvg['n_iter_final']} of {cvg['max_iter']} / {_fmt(cvg['converged_final'])}"),
                ("CV path convergence warnings", _fmt(cvg["cv_convergence_warnings"])),
            ],
        ))
        sections.append((
            "Sparsity, L1 Shrinkage & Active-Set Stability",
            [
                (r"Active \(|A(\alpha^*)|\) / Eliminated (\(\hat{\beta}_j = 0\))", f"{sp['n_nonzero_features']} / {sp['n_zero_features']} (of {sp['n_total_features']}, {sp['sparsity_pct']:.1f}% sparse)"),
                (r"\(\|\hat{\boldsymbol{\beta}}_{\mathrm{OLS}}\|_1\) / \(\|\hat{\boldsymbol{\beta}}_{\alpha^*}\|_1\)", f"{_fmt(sp['ols_coef_l1_norm'])} / {_fmt(sp['lasso_coef_l1_norm'])}"),
                (r"\(\ell_1\) norm ratio (\(\|\hat{\boldsymbol{\beta}}_{\alpha^*}\|_1 / \|\hat{\boldsymbol{\beta}}_{\mathrm{OLS}}\|_1\))", _fmt(sp["l1_norm_ratio_lasso_over_ols"])),
                ("Always / Never / Unstable across CV folds", f"{st['n_always_selected_in_cv']} / {st['n_never_selected_in_cv']} / {st['n_unstable_across_cv_folds']}"),
                (r"Spearman \(r_s(|\mathbf{e}|, \hat{\mathbf{y}})\) / \(p\)-value", f"{_fmt(rc['spearman_abs_resid_vs_fitted_r'])} (p = {_fmt(rc['spearman_abs_resid_vs_fitted_pvalue'])})"),
                (r"Durbin-Watson \(\mathrm{DW}\)", _fmt(rc["durbin_watson"])),
            ],
        ))

        interpretations.append(
            f"[Computed Fact] {pt['cv_strategy']} selected \\(\\alpha^* = {pt['selected_alpha']:.4g}\\) "
            f"(CV RMSE = {_fmt(pt['best_cv_rmse'])} {units}, CV \\(R^2 = {pt['best_cv_r2']:.4f}\\)), converging in {cvg['n_iter_final']} coordinate-descent iterations."
        )
        interpretations.append(
            f"[Interpretation] Soft-thresholding retained \\(|A(\\alpha^*)| = {sp['n_nonzero_features']}\\) of \\(p = {sp['n_total_features']}\\) predictors and eliminated {sp['n_zero_features']} ({sp['sparsity_pct']:.1f}% sparse), "
            f"reducing the \\(\\ell_1\\) norm from {_fmt(sp['ols_coef_l1_norm'])} (OLS) to {_fmt(sp['lasso_coef_l1_norm'])} (ratio = {sp['l1_norm_ratio_lasso_over_ols']:.4f})."
        )
        if sp["eliminated_features"]:
            elim_names = ", ".join(e["feature"] for e in sp["eliminated_features"][:5])
            max_elim_corr = max(e["max_abs_corr_with_retained"] for e in sp["eliminated_features"])
            interpretations.append(
                f"[Computed Fact] Eliminated predictors include {elim_names} (max \\(|r|\\) with a retained predictor = {max_elim_corr:.3f})."
            )
        if st["n_unstable_across_cv_folds"] > 0:
            unstable_ex = ", ".join(
                f"{u['feature']} ({u['cv_selection_freq']:.2f})" for u in st["unstable_features"][:4]
            )
            limitations.append(
                f"[Selection Instability Under Collinearity] {st['n_unstable_across_cv_folds']} predictor(s) varied in selection across CV folds (e.g., {unstable_ex}), because \\(\\ell_1\\) KKT pinning selects one representative among correlated features."
            )
        if not cvg["converged_final"] or cvg["cv_convergence_warnings"] > 0:
            limitations.append(
                f"[Solver Convergence] Coordinate descent emitted warning(s) (final converged = {cvg['converged_final']}, CV warnings = {cvg['cv_convergence_warnings']})."
            )

    if cfg.is_binary_classification and "test_classification" in metrics:
        tc = metrics["test_classification"]
        limitations.append(
            f"[Linear Probability Model Limitation] {tc['n_outside_unit_interval']} of {metrics['n_test']} test predictions ({tc['pct_outside_unit_interval']:.2f}%) lie outside \\([0, 1]\\) (range [{tc['pred_min']:.3f}, {tc['pred_max']:.3f}])."
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
            f"[Generalization Gap] Test \\(R^2\\) ({test_r2:.4f}) drops from Train \\(R^2\\) ({train_r2:.4f}), indicating distribution shift or limited linear generalization."
        )
    limitations.append(
        "[Linearity Assumption] Linear models assume additive linear relationships between standardized predictors and conditional target expectation."
    )

    return sections, interpretations, limitations


MODEL_LATEX = {
    "ols": {
        "short": r"Fits an unpenalized hyperplane minimizing the training residual sum of squares via SVD (<code>numpy.linalg.lstsq</code>) on standardized predictors \(\tilde{\mathbf{X}}\) with an unscaled intercept column \(\mathbf{1}\).",
        "eq": r"$$\min_{\boldsymbol{\beta} \in \mathbb{R}^{k}} \|\mathbf{y} - \tilde{\mathbf{X}}_{\mathrm{aug}}\boldsymbol{\beta}\|_2^2 = \mathbf{e}^\top\mathbf{e}, \qquad \tilde{\mathbf{X}}_{\mathrm{aug}} = [\mathbf{1},\, \tilde{\mathbf{X}}] \in \mathbb{R}^{n \times k},\ k = p + 1, \qquad (\tilde{\mathbf{X}}_{\mathrm{aug}}^\top \tilde{\mathbf{X}}_{\mathrm{aug}})\hat{\boldsymbol{\beta}} = \tilde{\mathbf{X}}_{\mathrm{aug}}^\top \mathbf{y}$$"
    },
    "ridge": {
        "short": r"Adds a squared \(\ell_2\) penalty on standardized slopes (leaving \(\beta_0\) unpenalized) to stabilize ill-conditioned \(\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}\) and reduce estimator variance.",
        "eq": r"$$\min_{\beta_0 \in \mathbb{R},\, \boldsymbol{\beta} \in \mathbb{R}^p} \|\mathbf{y} - \beta_0\mathbf{1} - \tilde{\mathbf{X}}\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_2^2, \qquad \hat{\boldsymbol{\beta}}_{\alpha} = (\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} + \alpha \mathbf{I}_p)^{-1}\tilde{\mathbf{X}}^\top \tilde{\mathbf{y}} \quad (\lambda \equiv \alpha \ge 0)$$"
    },
    "lasso": {
        "short": r"Adds an \(\ell_1\) penalty on standardized slopes (leaving \(\beta_0\) unpenalized) to perform continuous shrinkage and automatic variable selection via coordinate-wise soft-thresholding.",
        "eq": r"$$\min_{\beta_0 \in \mathbb{R},\, \boldsymbol{\beta} \in \mathbb{R}^p} \frac{1}{2n}\|\mathbf{y} - \beta_0\mathbf{1} - \tilde{\mathbf{X}}\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_1, \qquad \hat{\beta}_j = \mathcal{S}\!\left(\frac{1}{n}\tilde{\mathbf{x}}_j^\top \mathbf{r}^{(-j)}, \alpha\right) \quad \left(\lambda \equiv \alpha \ge 0,\ \frac{1}{n}\|\tilde{\mathbf{x}}_j\|_2^2 = 1\right)$$"
    },
}

SHARED_DESIGN_CSS = r"""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {
      --bg: #f1f5f9;
      --card: #ffffff;
      --subtle: #f8fafc;
      --muted-surface: #e2e8f0;
      --text: #1e293b;
      --heading: #0f172a;
      --navy: #1e3a8a;
      --muted: #475569;
      --border: #e2e8f0;
      --border-strong: #cbd5e1;
      --accent: #1e40af;
      --accent-hover: #1e3a8a;
      --accent-soft: #eff6ff;
      --accent-border: #bfdbfe;
      --amber: #b45309;
      --amber-soft: #fffbeb;
      --amber-border: #fde68a;
      --green: #15803d;
      --green-soft: #f0fdf4;
      --green-border: #bbf7d0;
      --danger: #dc2626;
      --radius: 10px;
      --shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
      --header-h: 58px;
    }
    [data-theme="dark"] {
      --bg: #060b16;
      --card: #0f172a;
      --subtle: #162033;
      --muted-surface: #1e293b;
      --text: #e2e8f0;
      --heading: #f8fafc;
      --navy: #93c5fd;
      --muted: #94a3b8;
      --border: #1e293b;
      --border-strong: #334155;
      --accent: #3b82f6;
      --accent-hover: #60a5fa;
      --accent-soft: rgba(59, 130, 246, 0.14);
      --accent-border: rgba(96, 165, 250, 0.35);
      --amber: #fbbf24;
      --amber-soft: rgba(245, 158, 11, 0.14);
      --amber-border: rgba(251, 191, 36, 0.35);
      --green: #4ade80;
      --green-soft: rgba(34, 197, 94, 0.14);
      --green-border: rgba(74, 222, 128, 0.35);
      --shadow: 0 2px 6px rgba(0, 0, 0, 0.45);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; scroll-padding-top: calc(var(--header-h) + 16px); }
    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
      *, *::before, *::after { transition: none !important; animation: none !important; }
    }
    body {
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.58;
      margin: 0;
      padding: 0;
      -webkit-font-smoothing: antialiased;
    }
    h1, h2, h3, h4 { font-family: "Space Grotesk", "Inter", sans-serif; }
    code, pre, .mono { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }
    a:focus-visible, button:focus-visible, summary:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 60;
      background: var(--card);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(8px);
    }
    .topbar-inner {
      max-width: 1180px;
      margin: 0 auto;
      padding: 8px 24px;
      min-height: var(--header-h);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .topnav { display: flex; gap: 10px; align-items: center; font-size: 0.84rem; flex-wrap: wrap; }
    .topnav a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
      padding: 5px 10px;
      border-radius: 6px;
      transition: background-color 160ms ease, color 160ms ease;
    }
    .topnav a:hover { background: var(--accent-soft); }
    .brand-mini {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      font-family: "Space Grotesk", sans-serif;
      font-weight: 700;
      font-size: 0.92rem;
      color: var(--heading) !important;
      padding: 4px 8px !important;
    }
    .brand-mini img { width: 24px; height: 24px; object-fit: contain; display: block; background: #fff; border-radius: 6px; padding: 2px; border: 1px solid var(--border); }
    .hero-logo-pill {
      display: inline-flex;
      align-items: center;
      background: rgba(255, 255, 255, 0.96);
      padding: 5px 12px;
      border-radius: 9px;
      margin-bottom: 10px;
      box-shadow: 0 3px 12px rgba(0, 0, 0, 0.25);
    }
    .hero-logo-pill img { height: 26px; width: auto; display: block; }
    .theme-btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 7px;
      border: 1px solid var(--border-strong);
      background: var(--card);
      color: var(--heading);
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      transition: background-color 160ms ease, border-color 160ms ease;
    }
    .theme-btn:hover { background: var(--subtle); }
    .container { max-width: 1180px; margin: 0 auto; padding: 24px 24px 48px; }
    header.hero-card {
      position: relative;
      overflow: hidden;
      background: #060b16;
      border: 1px solid var(--border-strong);
      border-radius: 12px;
      padding: 24px 26px;
      margin-bottom: 18px;
      box-shadow: var(--shadow);
      color: #f8fafc;
    }
    header.hero-card .hero-bg {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      opacity: 0.35;
    }
    header.hero-card .hero-inner { position: relative; z-index: 2; }
    header.hero-card h1 { margin: 8px 0 6px; font-size: 1.58rem; color: #ffffff; letter-spacing: -0.02em; }
    .subtitle { color: #cbd5e1; font-size: 0.89rem; }
    .badge {
      display: inline-flex;
      align-items: center;
      background: rgba(59, 130, 246, 0.22);
      color: #93c5fd;
      border: 1px solid rgba(96, 165, 250, 0.38);
      border-radius: 999px;
      padding: 2px 10px;
      font-size: 0.75rem;
      font-weight: 600;
      margin-right: 6px;
    }
    section {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px 24px;
      margin-bottom: 18px;
      box-shadow: var(--shadow);
    }
    h2 {
      margin-top: 0;
      font-size: 1.16rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
      color: var(--heading);
      letter-spacing: -0.01em;
    }
    h3 { font-size: 0.98rem; margin-top: 16px; color: var(--navy); }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.86rem; }
    th, td { border-bottom: 1px solid var(--border); padding: 9px 10px; text-align: left; }
    th {
      background: var(--subtle);
      font-weight: 700;
      color: var(--navy);
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; font-family: "JetBrains Mono", monospace; font-size: 0.81rem; }
    tr.intercept-row { background: var(--accent-soft); font-weight: 600; }
    tr.zero-row { color: var(--muted); opacity: 0.68; }
    .math-box {
      background: var(--subtle);
      border-left: 4px solid var(--accent);
      border-radius: 8px;
      padding: 12px 16px;
      margin: 12px 0;
      overflow-x: auto;
    }
    .katex { text-transform: none !important; }
    .katex-display { max-width: 100%; overflow-x: auto; overflow-y: hidden; padding: 4px 0; margin: 0.35em 0; }
    .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
    .plot-card {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      background: var(--card);
      transition: border-color 160ms ease, box-shadow 160ms ease;
    }
    .plot-card:hover { border-color: var(--accent); }
    .plot-card img {
      width: 100%;
      height: auto;
      display: block;
      border-radius: 5px;
      background: #ffffff;
      cursor: zoom-in;
    }
    .plot-title { font-weight: 600; margin-top: 8px; font-size: 0.89rem; color: var(--heading); }
    .plot-caption { color: var(--muted); font-size: 0.79rem; margin-top: 4px; }
    code { background: var(--subtle); padding: 1px 5px; border-radius: 4px; font-size: 0.83em; }
    .notice {
      background: var(--accent-soft);
      border-left: 4px solid var(--accent);
      padding: 12px 16px;
      border-radius: 6px;
      margin: 14px 0;
      font-size: 0.87rem;
    }
    a.ext { color: var(--accent); text-decoration: none; font-weight: 600; }
    a.ext:hover { text-decoration: underline; }
    footer { color: var(--muted); font-size: 0.81rem; text-align: center; margin-top: 20px; }
    .lightbox {
      position: fixed;
      inset: 0;
      z-index: 100;
      background: rgba(6, 11, 22, 0.86);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .lightbox.open { display: flex; }
    .lightbox-inner {
      background: var(--card);
      border: 1px solid var(--border-strong);
      border-radius: 12px;
      max-width: min(1040px, 96vw);
      max-height: 92vh;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .lightbox-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
    .lightbox-head strong { color: var(--heading); font-size: 0.98rem; }
    .lightbox img { max-width: 100%; max-height: 74vh; object-fit: contain; background: #fff; border-radius: 6px; }
    @media (max-width: 640px) {
      .container { padding: 14px 12px 36px; }
      section, header.hero-card { padding: 16px 14px; }
      .grid-2 { grid-template-columns: 1fr; }
    }
"""

REPORT_TEMPLATE = JINJA_ENV.from_string(r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DataMorph | {{ model_title }} — {{ dataset_name }}</title>
  <link rel="icon" type="image/png" sizes="64x64" href="../assets/images/favicon.png">
  <link rel="shortcut icon" href="../favicon.ico">
  <link rel="apple-touch-icon" href="../assets/images/favicon.png">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" crossorigin="anonymous"></script>
  <style>""" + SHARED_DESIGN_CSS + r"""</style>
</head>
<body>
<header class="topbar">
  <div class="topbar-inner">
    <nav class="topnav" aria-label="Breadcrumb">
      <a class="brand-mini" href="../index.html"><img src="../assets/images/datamorph_mark.png" alt=""><span>DataMorph</span></a>
      <a href="../index.html">&larr; Studio</a>
      <a href="summary.html">{{ model_key | upper }} Summary</a>
      <a href="{{ uci_url }}" target="_blank" rel="noopener noreferrer">UCI Dataset {{ metadata.uci_id }} &nearr;</a>
    </nav>
    <button type="button" class="theme-btn" id="theme-toggle" aria-label="Toggle dark mode" onclick="toggleTheme()">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>
      <span id="theme-label">Dark</span>
    </button>
  </div>
</header>

<div class="container">
  <header class="hero-card">
    <img class="hero-bg" src="../assets/images/hero_banner.jpg" alt="" aria-hidden="true">
    <div class="hero-inner">
      <div class="hero-logo-pill"><img src="../assets/images/datamorph_logo.png" alt="DataMorph"></div>
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
    </div>
  </header>

  <section>
    <h2>1. Dataset and Task Description</h2>
    <p>{{ dataset_description }}</p>
    <div class="table-wrap">
      <table>
        <tr><th>Property</th><th>Observed Value</th></tr>
        <tr><td>Dataset Key / UCI ID</td><td><code>{{ metadata.dataset }}</code> (<a class="ext" href="{{ uci_url }}" target="_blank" rel="noopener noreferrer">UCI {{ metadata.uci_id }} &nearr;</a>)</td></tr>
        <tr><td>Target Variable &amp; Units</td><td><code>{{ metadata.target }}</code> ({{ metadata.target_units }})</td></tr>
        <tr><td>Observations</td><td>{{ prep.n_raw }} raw &rarr; {{ prep.n_clean }} clean (\(n = {{ prep.n_train }}\) train / \(n_{\mathrm{test}} = {{ prep.n_test }}\) test)</td></tr>
        <tr><td>Features</td><td>{{ prep.n_raw_features }} raw &rarr; \(p = {{ prep.n_encoded_features }}\) encoded model predictors</td></tr>
        <tr><td>Train Target Range</td><td>mean = {{ "%.4g"|format(prep.target_stats_train.mean) }}, sd = {{ "%.4g"|format(prep.target_stats_train.std) }}, [{{ "%.4g"|format(prep.target_stats_train.min) }}, {{ "%.4g"|format(prep.target_stats_train.max) }}] {{ metadata.target_units }}</td></tr>
        <tr><td>Test Target Range</td><td>mean = {{ "%.4g"|format(prep.target_stats_test.mean) }}, sd = {{ "%.4g"|format(prep.target_stats_test.std) }}, [{{ "%.4g"|format(prep.target_stats_test.min) }}, {{ "%.4g"|format(prep.target_stats_test.max) }}] {{ metadata.target_units }}</td></tr>
      </table>
    </div>
    {% if dataset_notes %}
    <ul>{% for note in dataset_notes %}<li>{{ note }}</li>{% endfor %}</ul>
    {% endif %}
  </section>

  <section>
    <h2>2. Experimental Setup</h2>
    <div class="table-wrap">
      <table>
        <tr><th>Component</th><th>Configuration</th></tr>
        <tr><td>Split Strategy</td><td><code>{{ prep.split_strategy }}</code> (persisted at <code>{{ metadata.split_reference }}</code>)</td></tr>
        <tr><td>Cross-Validation</td><td><code>{{ prep.cv_strategy }}</code> (training data only)</td></tr>
        <tr><td>Excluded Columns</td><td>{% if prep.excluded_columns %}{{ prep.excluded_columns | join(", ") }}{% else %}None{% endif %}</td></tr>
        <tr><td>Categorical Encoding</td><td>{% if prep.categorical_columns %}{{ prep.categorical_columns | join(", ") }} (One-Hot drop-first, fit on train only){% else %}None{% endif %}</td></tr>
        <tr><td>Missing Value Imputation</td><td>{% if prep.missing_values_in_features %}Training-median imputation: {% for k, v in prep.missing_values_in_features.items() %}{{ k }} ({{ v }}){% if not loop.last %}, {% endif %}{% endfor %}{% else %}None required{% endif %}</td></tr>
        <tr><td>Standardization</td><td>Zero-mean, unit-variance <code>StandardScaler</code> fit strictly on training data (and separately within each CV fold). Intercept \(\beta_0\) unscaled and unpenalized.</td></tr>
      </table>
    </div>
  </section>

  <section>
    <h2>3. Mathematical Methodology</h2>
    <p>{{ model_short | safe }}</p>
    <div class="math-box">{{ model_eq }}</div>
    <p>
      <strong>Implementation Note:</strong> <code>{{ metadata.objective_convention }}</code>
      {% if metadata.penalty_settings.selected_alpha is defined %}
      <br><strong>CV-Selected Hyperparameter:</strong> \(\alpha^* = {{ "%.6g"|format(metadata.penalty_settings.selected_alpha) }}\) (\(\lambda \equiv \alpha\)) over {{ metadata.penalty_settings.n_alphas }} grid points in \([{{ "%.4g"|format(metadata.penalty_settings.alpha_grid_min) }}, {{ "%.4g"|format(metadata.penalty_settings.alpha_grid_max) }}]\).
      {% endif %}
    </p>
  </section>

  <section>
    <h2>4. Model Results</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Units</th>
            <th class="num">Train (\(n = {{ metrics.n_train }}\))</th>
            <th class="num">Test (\(n_{\mathrm{test}} = {{ metrics.n_test }}\))</th>
            <th class="num">Test &minus; Train</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>\(\mathrm{MAE}\)</td><td>{{ metrics.target_units }}</td><td class="num">{{ "%.5g"|format(metrics.train.mae) }}</td><td class="num">{{ "%.5g"|format(metrics.test.mae) }}</td><td class="num">{{ "%+.4g"|format(metrics.test.mae - metrics.train.mae) }}</td></tr>
          <tr><td>\(\mathrm{MSE}\)</td><td>({{ metrics.target_units }})&sup2;</td><td class="num">{{ "%.5g"|format(metrics.train.mse) }}</td><td class="num">{{ "%.5g"|format(metrics.test.mse) }}</td><td class="num">{{ "%+.4g"|format(metrics.test.mse - metrics.train.mse) }}</td></tr>
          <tr><td>\(\mathrm{RMSE}\)</td><td>{{ metrics.target_units }}</td><td class="num">{{ "%.5g"|format(metrics.train.rmse) }}</td><td class="num">{{ "%.5g"|format(metrics.test.rmse) }}</td><td class="num">{{ "%+.4g"|format(metrics.test.rmse - metrics.train.rmse) }}</td></tr>
          <tr><td>\(R^2\)</td><td>dimensionless</td><td class="num">{{ "%.5f"|format(metrics.train.r2) }}</td><td class="num">{{ "%.5f"|format(metrics.test.r2) }}</td><td class="num">{{ "%+.5f"|format(metrics.test.r2 - metrics.train.r2) }}</td></tr>
        </tbody>
      </table>
    </div>

    {% if metrics.is_binary_classification %}
    <h3>Linear Probability Model (LPM) Classification &amp; Bounds</h3>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Metric</th><th class="num">Train</th><th class="num">Test</th></tr></thead>
        <tbody>
          <tr><td>Accuracy (0.50 threshold)</td><td class="num">{{ "%.4f"|format(metrics.train_classification.accuracy) }}</td><td class="num">{{ "%.4f"|format(metrics.test_classification.accuracy) }}</td></tr>
          <tr><td>Precision / Recall / F1</td><td class="num">{{ "%.3f"|format(metrics.train_classification.precision) }} / {{ "%.3f"|format(metrics.train_classification.recall) }} / {{ "%.3f"|format(metrics.train_classification.f1) }}</td><td class="num">{{ "%.3f"|format(metrics.test_classification.precision) }} / {{ "%.3f"|format(metrics.test_classification.recall) }} / {{ "%.3f"|format(metrics.test_classification.f1) }}</td></tr>
          <tr><td>ROC AUC</td><td class="num">{{ "%.4f"|format(metrics.train_classification.roc_auc) }}</td><td class="num">{{ "%.4f"|format(metrics.test_classification.roc_auc) }}</td></tr>
          <tr><td>Predictions Outside \([0, 1]\)</td><td class="num">{{ metrics.train_classification.n_outside_unit_interval }} ({{ "%.2f"|format(metrics.train_classification.pct_outside_unit_interval) }}%)</td><td class="num">{{ metrics.test_classification.n_outside_unit_interval }} ({{ "%.2f"|format(metrics.test_classification.pct_outside_unit_interval) }}%)</td></tr>
        </tbody>
      </table>
    </div>
    {% endif %}
  </section>

  <section>
    <h2>5. Coefficient Interpretation</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Role</th>
            <th>Feature</th>
            <th class="num">\(\hat{\beta}_{\mathrm{std}}\)</th>
            <th class="num">\(\hat{\beta}_{\mathrm{unstd}}\)</th>
            {% if has_ols_ref %}<th class="num">\(\hat{\beta}_{\mathrm{OLS, std}}\)</th>{% endif %}
            {% if has_shrinkage_ratio %}<th class="num">\(|\hat{\beta}| / |\hat{\beta}_{\mathrm{OLS}}|\)</th>{% endif %}
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
    </div>
  </section>

  <section>
    <h2>6. Diagnostics and Visualizations</h2>
    <div class="grid-2">
      {% for p in plots %}
      <div class="plot-card">
        <img src="data:image/png;base64,{{ p.b64 }}" alt="{{ p.title }}" tabindex="0" role="button" onclick="openLightbox(this)" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();openLightbox(this);}">
        <div class="plot-title">{{ p.title }}</div>
        <div class="plot-caption">{{ p.caption }}</div>
      </div>
      {% endfor %}
    </div>

    {% for group_name, items in diagnostic_sections %}
    <h3>{{ group_name }}</h3>
    <div class="table-wrap">
      <table>
        {% for k, v in items %}
        <tr><td>{{ k }}</td><td class="num">{{ v }}</td></tr>
        {% endfor %}
      </table>
    </div>
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
    DataMorph &middot; Generated strictly from saved artifacts in <code>results/{{ model_key }}/{{ metadata.dataset }}/</code> &middot;
    Python {{ metadata.dependencies.python }}, NumPy {{ metadata.dependencies.numpy }}, scikit-learn {{ metadata.dependencies.scikit_learn }}
  </footer>
</div>

<div class="lightbox" id="lightbox" role="dialog" aria-modal="true" aria-label="Plot preview" onclick="if(event.target===this)closeLightbox()">
  <div class="lightbox-inner">
    <div class="lightbox-head">
      <strong id="lb-title"></strong>
      <button type="button" class="theme-btn" onclick="closeLightbox()">Close (Esc)</button>
    </div>
    <img id="lb-img" src="" alt="">
    <div class="plot-caption" id="lb-caption"></div>
  </div>
</div>

<script>
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  const lbl = document.getElementById("theme-label");
  if (lbl) lbl.textContent = t === "dark" ? "Light" : "Dark";
  try { localStorage.setItem("uci_reg_theme", t); } catch (e) {}
}
function toggleTheme() {
  const cur = document.documentElement.getAttribute("data-theme") || "light";
  applyTheme(cur === "dark" ? "light" : "dark");
}
(function(){
  try {
    const saved = localStorage.getItem("uci_reg_theme");
    if (saved === "dark" || saved === "light") applyTheme(saved);
  } catch (e) {}
})();

let lastFocused = null;
function openLightbox(imgEl) {
  lastFocused = document.activeElement;
  const card = imgEl.closest(".plot-card");
  document.getElementById("lb-img").src = imgEl.src;
  document.getElementById("lb-img").alt = imgEl.alt;
  document.getElementById("lb-title").textContent = card.querySelector(".plot-title").textContent;
  document.getElementById("lb-caption").innerHTML = card.querySelector(".plot-caption").innerHTML;
  document.getElementById("lightbox").classList.add("open");
}
function closeLightbox() {
  document.getElementById("lightbox").classList.remove("open");
  if (lastFocused) lastFocused.focus();
}
document.addEventListener("keydown", function (e) {
  if (e.key === "Escape") closeLightbox();
});
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

SUMMARY_TEMPLATE = JINJA_ENV.from_string(r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DataMorph | {{ model_title }} — Cross-Dataset Summary</title>
  <link rel="icon" type="image/png" sizes="64x64" href="../assets/images/favicon.png">
  <link rel="shortcut icon" href="../favicon.ico">
  <link rel="apple-touch-icon" href="../assets/images/favicon.png">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" crossorigin="anonymous"></script>
  <style>""" + SHARED_DESIGN_CSS + r"""</style>
</head>
<body>
<header class="topbar">
  <div class="topbar-inner">
    <nav class="topnav" aria-label="Breadcrumb">
      <a class="brand-mini" href="../index.html"><img src="../assets/images/datamorph_mark.png" alt=""><span>DataMorph</span></a>
      <a href="../index.html">&larr; Studio</a>
      <a href="../ols/summary.html">OLS Summary</a>
      <a href="../ridge/summary.html">Ridge Summary</a>
      <a href="../lasso/summary.html">Lasso Summary</a>
    </nav>
    <button type="button" class="theme-btn" id="theme-toggle" aria-label="Toggle dark mode" onclick="toggleTheme()">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>
      <span id="theme-label">Dark</span>
    </button>
  </div>
</header>

<div class="container">
  <header class="hero-card">
    <img class="hero-bg" src="../assets/images/hero_banner.jpg" alt="" aria-hidden="true">
    <div class="hero-inner">
      <div class="hero-logo-pill"><img src="../assets/images/datamorph_logo.png" alt="DataMorph"></div>
      <span class="badge">{{ model_key | upper }}</span>
      <span class="badge">{{ rows | length }} UCI Datasets</span>
      <h1>{{ model_title }} &mdash; All-Dataset Summary</h1>
      <div class="subtitle">Cross-benchmark summary using persisted 80/20 splits and training-only preprocessing.</div>
    </div>
  </header>

  <section>
    <div class="notice">
      <strong>Unit Comparability:</strong> Raw \(\mathrm{RMSE}\) and \(\mathrm{MAE}\) carry each dataset&rsquo;s native target units and must not be compared across datasets with different units. Use dimensionless <strong>Test \(R^2\)</strong> and unit-free <strong>Normalized \(\mathrm{RMSE}\)</strong> (\(\mathrm{RMSE}_{\mathrm{test}} / s_{y,\mathrm{test}}\)) for cross-dataset comparisons.
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Split</th>
            <th class="num">\(n / n_{\mathrm{test}}\)</th>
            <th class="num">\(p\)</th>
            <th class="num">Train \(R^2\)</th>
            <th class="num">Test \(R^2\)</th>
            <th class="num">Test \(\mathrm{RMSE}\) (Units)</th>
            <th class="num">Norm. \(\mathrm{RMSE}\)</th>
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
            <td class="num"><strong>{{ "%.4f"|format(r.test_r2) }}</strong></td>
            <td class="num">{{ "%.4g"|format(r.test_rmse) }} <small>{{ r.target_units }}</small></td>
            <td class="num">{{ "%.4f"|format(r.norm_rmse) }}</td>
            <td>{{ r.key_diag }}</td>
            <td><a class="ext" href="{{ r.dataset }}.html">Report &nearr;</a></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </section>
</div>
<script>
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  const lbl = document.getElementById("theme-label");
  if (lbl) lbl.textContent = t === "dark" ? "Light" : "Dark";
  try { localStorage.setItem("uci_reg_theme", t); } catch (e) {}
}
function toggleTheme() {
  const cur = document.documentElement.getAttribute("data-theme") || "light";
  applyTheme(cur === "dark" ? "light" : "dark");
}
(function(){
  try {
    const saved = localStorage.getItem("uci_reg_theme");
    if (saved === "dark" || saved === "light") applyTheme(saved);
  } catch (e) {}
})();
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
            key_diag = (
                f"\\(\\operatorname{{rank}} = {la['matrix_rank']}/{la['n_augmented_columns']}\\), "
                f"\\(\\kappa(\\tilde{{\\mathbf{{X}}}}_{{\\mathrm{{aug}}}}) = {_fmt(la['condition_number_standardized_with_intercept'])}\\)"
            )
        elif model_key == "ridge":
            pt = diag["penalty_tuning"]
            sd = diag["shrinkage_and_degrees_of_freedom"]
            key_diag = (
                f"\\(\\alpha^* = {pt['selected_alpha']:.3g}\\), "
                f"\\(\\operatorname{{df}}(\\alpha^*) = {sd['effective_df_predictors']:.1f}/{sd['n_predictors']}\\), "
                f"\\(\\ell_2\\) ratio = {sd['l2_norm_ratio_ridge_over_ols']:.3f}"
            )
        else:
            pt = diag["penalty_tuning"]
            sp = diag["sparsity_and_shrinkage"]
            key_diag = (
                f"\\(\\alpha^* = {pt['selected_alpha']:.3g}\\), "
                f"\\(|A(\\alpha^*)| = {sp['n_nonzero_features']}/{sp['n_total_features']}\\)"
            )

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
    html = SUMMARY_TEMPLATE.render(model_key=model_key, model_title=MODEL_TITLES[model_key], rows=rows)
    out_path.write_text(html, encoding="utf-8")
    return out_path
