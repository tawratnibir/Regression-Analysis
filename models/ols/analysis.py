"""OLS diagnostic checks distinguishing exact algebraic identities from statistical assumptions."""

from typing import Any, Dict, Tuple

import numpy as np
from scipy import stats

from models.ols.model import OLSFitResult
from preprocessing.preprocess import PreparedDataset


def _compute_leverage_and_cooks(
    X_aug: np.ndarray, resid: np.ndarray, rank: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute hat matrix diagonal and Cook's distance using thin SVD (avoids n x n matrix)."""
    n = len(resid)
    U, s, _ = np.linalg.svd(X_aug, full_matrices=False)
    tol = np.finfo(float).eps * max(X_aug.shape) * (s[0] if len(s) > 0 else 1.0)
    keep = s > tol
    U_r = U[:, keep]
    leverage = np.sum(U_r ** 2, axis=1)
    leverage = np.clip(leverage, 0.0, 0.999999999)

    dof = max(1, n - rank)
    mse_resid = float(np.sum(resid ** 2) / dof)
    if mse_resid <= 0.0 or rank <= 0:
        return leverage, np.zeros_like(resid)

    cooks_d = (resid ** 2 / (rank * mse_resid)) * (leverage / ((1.0 - leverage) ** 2))
    return leverage, cooks_d


def _compute_vif(X: np.ndarray, feature_names: list[str], is_rank_deficient: bool) -> Dict[str, Any]:
    """Compute Variance Inflation Factors via lstsq R^2 for each predictor."""
    n, p = X.shape
    if p <= 1:
        return {"max_vif": 1.0, "n_vif_gt_10": 0, "top_vifs": []}

    vifs = []
    for j in range(p):
        y_j = X[:, j]
        X_other = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
        beta, _, _, _ = np.linalg.lstsq(X_other, y_j, rcond=None)
        pred_j = X_other @ beta
        ss_tot = float(np.sum((y_j - np.mean(y_j)) ** 2))
        ss_res = float(np.sum((y_j - pred_j) ** 2))
        r2_j = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        vif = float("inf") if r2_j >= 1.0 - 1e-12 else float(1.0 / (1.0 - r2_j))
        vifs.append({"feature": feature_names[j], "vif": vif})

    vifs_sorted = sorted(vifs, key=lambda d: d["vif"], reverse=True)
    finite_vifs = [d["vif"] for d in vifs_sorted if np.isfinite(d["vif"])]
    return {
        "max_vif": float(max(finite_vifs)) if finite_vifs else float("inf"),
        "has_infinite_vif": bool(is_rank_deficient or any(not np.isfinite(d["vif"]) for d in vifs)),
        "n_vif_gt_10": int(sum(1 for d in vifs_sorted if d["vif"] > 10.0)),
        "top_vifs": vifs_sorted[:10],
    }


def analyze_ols(
    dataset: PreparedDataset, fit: OLSFitResult
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Run exact OLS algebraic checks and classical assumption diagnostics."""
    X_tr = dataset.X_train
    y_tr = dataset.y_train
    resid = fit.resid_train
    y_hat = fit.y_train_pred
    n, p = X_tr.shape
    X_aug = np.column_stack([np.ones(n), X_tr])

    # 1. Exact algebraic properties (guaranteed when intercept is included)
    max_abs_xt_resid = float(np.max(np.abs(X_aug.T @ resid)))
    resid_sum = float(np.sum(resid))
    mean_y = float(np.mean(y_tr))
    mean_y_hat = float(np.mean(y_hat))
    mean_fitted_diff = abs(mean_y - mean_y_hat)

    sst = float(np.sum((y_tr - mean_y) ** 2))
    ssr = float(np.sum((y_hat - mean_y) ** 2))
    sse = float(np.sum(resid ** 2))
    anova_identity_error = abs(sst - (ssr + sse))

    # 2. Heteroskedasticity (Breusch-Pagan auxiliary regression of r^2 on X_aug)
    r_sq = resid ** 2
    r_sq_centered = r_sq - np.mean(r_sq)
    ss_tot_bp = float(np.sum(r_sq_centered ** 2))
    if ss_tot_bp > 0:
        beta_bp, _, rank_bp, _ = np.linalg.lstsq(X_aug, r_sq, rcond=None)
        pred_bp = X_aug @ beta_bp
        ss_res_bp = float(np.sum((r_sq - pred_bp) ** 2))
        r2_bp = max(0.0, 1.0 - ss_res_bp / ss_tot_bp)
        bp_stat = float(n * r2_bp)
        bp_df = max(1, int(rank_bp) - 1)
        bp_pvalue = float(stats.chi2.sf(bp_stat, df=bp_df))
    else:
        bp_stat, bp_df, bp_pvalue = 0.0, max(1, p), 1.0

    abs_resid_spearman_r, abs_resid_spearman_p = stats.spearmanr(y_hat, np.abs(resid))

    # 3. Residual distribution / Normality (Jarque-Bera + Shapiro on <=5000 subsample)
    skew = float(stats.skew(resid))
    excess_kurt = float(stats.kurtosis(resid))
    jb_stat, jb_pvalue = stats.jarque_bera(resid)
    if n <= 5000:
        sw_stat, sw_pvalue = stats.shapiro(resid)
        sw_n = n
    else:
        rng = np.random.default_rng(42)
        sub = rng.choice(resid, size=5000, replace=False)
        sw_stat, sw_pvalue = stats.shapiro(sub)
        sw_n = 5000

    # 4. Leverage & Cook's distance
    leverage, cooks_d = _compute_leverage_and_cooks(X_aug, resid, fit.rank_augmented)
    lev_threshold = 2.0 * fit.rank_augmented / n
    cooks_threshold = 4.0 / n
    n_high_leverage = int(np.sum(leverage > lev_threshold))
    n_influential = int(np.sum(cooks_d > cooks_threshold))

    # 5. Multicollinearity (condition number & VIF)
    is_rank_deficient = fit.rank_augmented < fit.n_augmented_cols
    vif_info = _compute_vif(X_tr, dataset.feature_names, is_rank_deficient)

    # 6. Residual autocorrelation (Durbin-Watson & lag-1 corr; particularly meaningful for time series)
    if n > 2 and sse > 0:
        dw_stat = float(np.sum(np.diff(resid) ** 2) / sse)
        lag1_corr = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    else:
        dw_stat = float("nan")
        lag1_corr = float("nan")

    diagnostics: Dict[str, Any] = {
        "exact_algebraic_properties": {
            "note": "Exact mathematical consequences of least-squares projection with an unpenalized intercept (not statistical assumptions).",
            "max_abs_Xt_residuals": max_abs_xt_resid,
            "residual_sum": resid_sum,
            "mean_observed_train": mean_y,
            "mean_fitted_train": mean_y_hat,
            "mean_fitted_identity_abs_diff": mean_fitted_diff,
            "sst": sst,
            "ssr": ssr,
            "sse": sse,
            "anova_identity_abs_error": anova_identity_error,
        },
        "numerical_linear_algebra": {
            "n_train": int(n),
            "n_predictors": int(p),
            "n_augmented_columns": fit.n_augmented_cols,
            "matrix_rank": fit.rank_augmented,
            "is_rank_deficient": is_rank_deficient,
            "condition_number_standardized_with_intercept": (
                fit.condition_number_scaled if np.isfinite(fit.condition_number_scaled) else "inf"
            ),
            "condition_number_unstandardized_with_intercept": (
                fit.condition_number_unscaled if np.isfinite(fit.condition_number_unscaled) else "inf"
            ),
            "min_singular_value": float(fit.singular_values_augmented[-1]),
            "max_singular_value": float(fit.singular_values_augmented[0]),
            "sklearn_max_abs_test_pred_diff": fit.sklearn_max_abs_pred_diff,
        },
        "multicollinearity": vif_info,
        "heteroskedasticity_checks": {
            "note": "Empirical assumption checks on training residuals (not guaranteed by OLS algebra).",
            "breusch_pagan_lm_stat": bp_stat,
            "breusch_pagan_df": bp_df,
            "breusch_pagan_pvalue": bp_pvalue,
            "spearman_abs_resid_vs_fitted_r": float(abs_resid_spearman_r),
            "spearman_abs_resid_vs_fitted_pvalue": float(abs_resid_spearman_p),
        },
        "residual_normality_checks": {
            "skewness": skew,
            "excess_kurtosis": excess_kurt,
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_pvalue": float(jb_pvalue),
            "shapiro_wilk_stat": float(sw_stat),
            "shapiro_wilk_pvalue": float(sw_pvalue),
            "shapiro_wilk_n": int(sw_n),
        },
        "influence_diagnostics": {
            "leverage_threshold_2p_over_n": float(lev_threshold),
            "n_high_leverage": n_high_leverage,
            "max_leverage": float(np.max(leverage)),
            "cooks_threshold_4_over_n": float(cooks_threshold),
            "n_high_cooks_d": n_influential,
            "max_cooks_d": float(np.max(cooks_d)),
        },
        "autocorrelation_checks": {
            "durbin_watson": dw_stat,
            "lag1_residual_corr": lag1_corr,
        },
    }

    return diagnostics, leverage, cooks_d
