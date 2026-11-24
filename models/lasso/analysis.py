"""Lasso diagnostics: sparsity, retained/eliminated features, collinearity, and stability."""

from typing import Any, Dict, Tuple

import numpy as np
from scipy import stats

from models.lasso.model import ZERO_TOL, LassoFitResult
from preprocessing.preprocess import PreparedDataset


def analyze_lasso(
    dataset: PreparedDataset, fit: LassoFitResult
) -> Tuple[Dict[str, Any], np.ndarray, float]:
    """Compute Lasso sparsity, L1 shrinkage, CV selection stability, and collinearity context."""
    X_tr = dataset.X_train
    y_tr = dataset.y_train
    n, p = X_tr.shape

    # Reference OLS on standardized training split
    X_aug = np.column_stack([np.ones(n), X_tr])
    beta_ols, _, _, _ = np.linalg.lstsq(X_aug, y_tr, rcond=None)
    ols_intercept_scaled = float(beta_ols[0])
    ols_coef_scaled = beta_ols[1:].copy()

    ols_l1 = float(np.linalg.norm(ols_coef_scaled, ord=1))
    lasso_l1 = float(np.linalg.norm(fit.coef_scaled, ord=1))
    l1_ratio = float(lasso_l1 / ols_l1) if ols_l1 > 0 else 1.0

    nonzero_mask = np.abs(fit.coef_scaled) > ZERO_TOL
    n_nonzero = int(np.sum(nonzero_mask))
    n_zero = int(p - n_nonzero)
    sparsity_pct = float(100.0 * n_zero / p) if p > 0 else 0.0

    retained_indices = np.where(nonzero_mask)[0]
    eliminated_indices = np.where(~nonzero_mask)[0]

    retained_features = sorted(
        [
            {
                "feature": dataset.feature_names[i],
                "coef_std": float(fit.coef_scaled[i]),
                "cv_selection_freq": float(fit.cv_selection_freq_at_best[i]),
            }
            for i in retained_indices
        ],
        key=lambda d: abs(d["coef_std"]),
        reverse=True,
    )

    # Check collinearity of eliminated features with retained features
    corr_matrix = np.corrcoef(X_tr, rowvar=False) if p > 1 else np.ones((1, 1))
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    eliminated_features = []
    for i in eliminated_indices:
        if len(retained_indices) > 0:
            corrs = np.abs(corr_matrix[i, retained_indices])
            max_j = int(retained_indices[int(np.argmax(corrs))])
            max_corr = float(np.max(corrs))
            corr_partner = dataset.feature_names[max_j]
        else:
            max_corr = 0.0
            corr_partner = None
        eliminated_features.append(
            {
                "feature": dataset.feature_names[i],
                "ols_coef_std": float(ols_coef_scaled[i]),
                "cv_selection_freq": float(fit.cv_selection_freq_at_best[i]),
                "max_abs_corr_with_retained": max_corr,
                "most_correlated_retained_feature": corr_partner,
            }
        )

    # Identify features whose selection varied across CV folds (0 < freq < 1)
    unstable_features = [
        {
            "feature": dataset.feature_names[i],
            "cv_selection_freq": float(fit.cv_selection_freq_at_best[i]),
            "final_coef_std": float(fit.coef_scaled[i]),
        }
        for i in range(p)
        if 0.0 < fit.cv_selection_freq_at_best[i] < 1.0
    ]
    unstable_features.sort(key=lambda d: d["cv_selection_freq"], reverse=True)

    # Residual diagnostics
    resid = fit.resid_train
    sse = float(np.sum(resid ** 2))
    dw_stat = float(np.sum(np.diff(resid) ** 2) / sse) if n > 2 and sse > 0 else float("nan")
    sp_r, sp_p = stats.spearmanr(fit.y_train_pred, np.abs(resid))

    diagnostics: Dict[str, Any] = {
        "penalty_tuning": {
            "alpha_max": fit.alpha_max,
            "alpha_grid_min": float(fit.alphas[-1]),
            "alpha_grid_max": float(fit.alphas[0]),
            "n_alphas": int(len(fit.alphas)),
            "cv_strategy": dataset.cv_strategy_name,
            "selected_alpha": fit.best_alpha,
            "best_cv_rmse": float(fit.cv_rmse_mean[fit.best_alpha_idx]),
            "best_cv_rmse_std": float(fit.cv_rmse_std[fit.best_alpha_idx]),
            "best_cv_r2": float(fit.cv_r2_mean[fit.best_alpha_idx]),
        },
        "sparsity_and_shrinkage": {
            "n_total_features": int(p),
            "n_nonzero_features": n_nonzero,
            "n_zero_features": n_zero,
            "sparsity_pct": sparsity_pct,
            "ols_coef_l1_norm": ols_l1,
            "lasso_coef_l1_norm": lasso_l1,
            "l1_norm_ratio_lasso_over_ols": l1_ratio,
            "retained_features": retained_features,
            "eliminated_features": eliminated_features,
        },
        "selection_stability": {
            "n_always_selected_in_cv": int(np.sum(fit.cv_selection_freq_at_best >= 1.0 - 1e-12)),
            "n_never_selected_in_cv": int(np.sum(fit.cv_selection_freq_at_best <= 1e-12)),
            "n_unstable_across_cv_folds": len(unstable_features),
            "unstable_features": unstable_features[:15],
        },
        "convergence": {
            "n_iter_final": fit.n_iter_final,
            "max_iter": fit.max_iter,
            "converged_final": fit.converged_final,
            "cv_convergence_warnings": fit.cv_convergence_warnings,
        },
        "residual_checks": {
            "spearman_abs_resid_vs_fitted_r": float(sp_r),
            "spearman_abs_resid_vs_fitted_pvalue": float(sp_p),
            "durbin_watson": dw_stat,
        },
    }

    return diagnostics, ols_coef_scaled, ols_intercept_scaled
