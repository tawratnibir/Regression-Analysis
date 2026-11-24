"""Ridge regression diagnostics: shrinkage, effective degrees of freedom, and stability."""

from typing import Any, Dict, Tuple

import numpy as np
from scipy import stats

from models.ridge.model import RidgeFitResult
from preprocessing.preprocess import PreparedDataset


def analyze_ridge(
    dataset: PreparedDataset, fit: RidgeFitResult
) -> Tuple[Dict[str, Any], np.ndarray, float]:
    """Compute Ridge shrinkage, effective df, condition stability, and reference OLS."""
    X_tr = dataset.X_train
    y_tr = dataset.y_train
    n, p = X_tr.shape

    # Reference OLS on exact same standardized training split
    X_aug = np.column_stack([np.ones(n), X_tr])
    beta_ols, _, _, _ = np.linalg.lstsq(X_aug, y_tr, rcond=None)
    ols_intercept_scaled = float(beta_ols[0])
    ols_coef_scaled = beta_ols[1:].copy()

    ols_l2 = float(np.linalg.norm(ols_coef_scaled, ord=2))
    ridge_l2 = float(np.linalg.norm(fit.coef_scaled, ord=2))
    l2_ratio = float(ridge_l2 / ols_l2) if ols_l2 > 0 else 1.0

    # Singular values of centered X_train for effective df & condition number stabilization
    X_c = X_tr - np.mean(X_tr, axis=0)
    s = np.linalg.svd(X_c, compute_uv=False)
    alpha = fit.best_alpha
    eff_df = float(np.sum((s ** 2) / (s ** 2 + alpha)))

    s_max_sq = float(s[0] ** 2) if len(s) > 0 else 0.0
    s_min_sq = float(s[-1] ** 2) if len(s) > 0 else 0.0
    ols_cond_sq = float(s_max_sq / s_min_sq) if s_min_sq > 1e-15 else float("inf")
    ridge_cond_sq = float((s_max_sq + alpha) / (s_min_sq + alpha))

    # Largest absolute reductions relative to OLS
    shrink_diffs = []
    for name, c_r, c_o in zip(dataset.feature_names, fit.coef_scaled, ols_coef_scaled):
        shrink_diffs.append(
            {
                "feature": name,
                "ols_coef_std": float(c_o),
                "ridge_coef_std": float(c_r),
                "abs_change": float(abs(c_o) - abs(c_r)),
            }
        )
    shrink_diffs.sort(key=lambda d: abs(d["abs_change"]), reverse=True)

    # Residual diagnostics
    resid = fit.resid_train
    sse = float(np.sum(resid ** 2))
    dw_stat = float(np.sum(np.diff(resid) ** 2) / sse) if n > 2 and sse > 0 else float("nan")
    sp_r, sp_p = stats.spearmanr(fit.y_train_pred, np.abs(resid))

    diagnostics: Dict[str, Any] = {
        "penalty_tuning": {
            "alpha_grid_min": float(fit.alphas[0]),
            "alpha_grid_max": float(fit.alphas[-1]),
            "n_alphas": int(len(fit.alphas)),
            "cv_strategy": dataset.cv_strategy_name,
            "selected_alpha": fit.best_alpha,
            "best_cv_rmse": float(fit.cv_rmse_mean[fit.best_alpha_idx]),
            "best_cv_rmse_std": float(fit.cv_rmse_std[fit.best_alpha_idx]),
            "best_cv_r2": float(fit.cv_r2_mean[fit.best_alpha_idx]),
            "svd_verification_max_abs_coef_diff": fit.svd_max_abs_coef_diff,
        },
        "shrinkage_and_degrees_of_freedom": {
            "n_predictors": int(p),
            "effective_df_predictors": eff_df,
            "effective_df_with_intercept": eff_df + 1.0,
            "ols_coef_l2_norm": ols_l2,
            "ridge_coef_l2_norm": ridge_l2,
            "l2_norm_ratio_ridge_over_ols": l2_ratio,
            "top_coefficient_changes_vs_ols": shrink_diffs[:10],
        },
        "numerical_stability": {
            "x_centered_eig_condition_ols": ols_cond_sq if np.isfinite(ols_cond_sq) else "inf",
            "x_centered_eig_condition_ridge": ridge_cond_sq,
            "min_singular_value_centered": float(s[-1]) if len(s) > 0 else 0.0,
            "max_singular_value_centered": float(s[0]) if len(s) > 0 else 0.0,
        },
        "residual_checks": {
            "spearman_abs_resid_vs_fitted_r": float(sp_r),
            "spearman_abs_resid_vs_fitted_pvalue": float(sp_p),
            "durbin_watson": dw_stat,
        },
    }

    return diagnostics, ols_coef_scaled, ols_intercept_scaled
