"""Lasso regression fitting with leak-free cross-validated alpha selection."""

from dataclasses import dataclass
from typing import Any, Dict, List
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, lasso_path
from sklearn.metrics import mean_squared_error, r2_score

from models.base import unscale_coefficients
from preprocessing.preprocess import PreparedDataset, build_preprocessor

ZERO_TOL = 1e-8


@dataclass
class LassoFitResult:
    """Container for cross-validated Lasso fit, path, sparsity, and fold stability."""

    alphas: np.ndarray
    alpha_max: float
    cv_mse_mean: np.ndarray
    cv_mse_std: np.ndarray
    cv_rmse_mean: np.ndarray
    cv_rmse_std: np.ndarray
    cv_r2_mean: np.ndarray
    cv_selection_freq_at_best: np.ndarray  # shape (n_features,)
    best_alpha: float
    best_alpha_idx: int
    intercept_scaled: float
    coef_scaled: np.ndarray
    intercept_unscaled: float
    coef_unscaled: np.ndarray
    y_train_pred: np.ndarray
    y_test_pred: np.ndarray
    resid_train: np.ndarray
    resid_test: np.ndarray
    coef_path: np.ndarray  # shape (n_alphas, n_features)
    nonzero_counts_path: np.ndarray  # shape (n_alphas,)
    n_iter_final: int
    max_iter: int
    converged_final: bool
    cv_convergence_warnings: int


def _compute_alpha_grid(X: np.ndarray, y: np.ndarray, n_alphas: int = 60) -> tuple[float, np.ndarray]:
    """Compute descending log-spaced alpha grid anchored at theoretical alpha_max."""
    n = len(y)
    X_c = X - np.mean(X, axis=0)
    y_c = y - np.mean(y)
    alpha_max = float(np.max(np.abs(X_c.T @ y_c)) / n)
    alpha_max = max(alpha_max, 1e-6)
    alpha_min = max(alpha_max * 1e-4, 1e-8)
    alphas = np.geomspace(alpha_max, alpha_min, n_alphas)
    return alpha_max, alphas


def fit_lasso(
    dataset: PreparedDataset,
    n_alphas: int = 60,
    max_iter: int = 20000,
    tol: float = 1e-4,
) -> LassoFitResult:
    """Tune Lasso penalty via leak-free training-only CV, then refit on full training split."""
    alpha_max, alphas = _compute_alpha_grid(dataset.X_train, dataset.y_train, n_alphas=n_alphas)
    n_folds = len(dataset.cv_splits)
    n_features = dataset.X_train.shape[1]

    fold_mse = np.zeros((n_folds, n_alphas), dtype=float)
    fold_r2 = np.zeros((n_folds, n_alphas), dtype=float)
    fold_nonzero_masks = np.zeros((n_folds, n_alphas, n_features), dtype=bool)
    cv_warnings = 0

    for f_idx, (tr_pos, val_pos) in enumerate(dataset.cv_splits):
        prep = build_preprocessor(
            dataset.numeric_columns, dataset.categorical_columns, standardize=True
        )
        X_tr_f = prep.fit_transform(dataset.df_train_raw.iloc[tr_pos])
        X_val_f = prep.transform(dataset.df_train_raw.iloc[val_pos])
        y_tr_f = dataset.y_train[tr_pos]
        y_val_f = dataset.y_train[val_pos]

        x_mean_f = np.mean(X_tr_f, axis=0)
        y_mean_f = float(np.mean(y_tr_f))
        X_tr_c = X_tr_f - x_mean_f
        y_tr_c = y_tr_f - y_mean_f

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            _, coefs_f, _ = lasso_path(
                X_tr_c,
                y_tr_c,
                alphas=alphas,
                max_iter=max_iter,
                tol=tol,
            )
            cv_warnings += sum(
                1 for w in caught if issubclass(w.category, ConvergenceWarning)
            )

        # coefs_f has shape (n_features, n_alphas)
        intercepts_f = y_mean_f - (x_mean_f @ coefs_f)
        preds_val = X_val_f @ coefs_f + intercepts_f  # shape (n_val, n_alphas)
        for a_idx in range(n_alphas):
            pred_val = preds_val[:, a_idx]
            fold_mse[f_idx, a_idx] = float(mean_squared_error(y_val_f, pred_val))
            fold_r2[f_idx, a_idx] = float(r2_score(y_val_f, pred_val))
            fold_nonzero_masks[f_idx, a_idx, :] = np.abs(coefs_f[:, a_idx]) > ZERO_TOL

    cv_mse_mean = np.mean(fold_mse, axis=0)
    cv_mse_std = np.std(fold_mse, axis=0, ddof=0)
    fold_rmse = np.sqrt(fold_mse)
    cv_rmse_mean = np.mean(fold_rmse, axis=0)
    cv_rmse_std = np.std(fold_rmse, axis=0, ddof=0)
    cv_r2_mean = np.mean(fold_r2, axis=0)

    best_idx = int(np.argmin(cv_mse_mean))
    best_alpha = float(alphas[best_idx])
    cv_selection_freq = np.mean(fold_nonzero_masks[:, best_idx, :], axis=0)

    # Full path on training split via Gram-precomputed lasso_path
    x_mean_tr = np.mean(dataset.X_train, axis=0)
    y_mean_tr = float(np.mean(dataset.y_train))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", ConvergenceWarning)
        _, coefs_full, _ = lasso_path(
            dataset.X_train - x_mean_tr,
            dataset.y_train - y_mean_tr,
            alphas=alphas,
            max_iter=max_iter,
            tol=tol,
        )
    coef_path = coefs_full.T  # shape (n_alphas, n_features)
    nonzero_counts_path = np.sum(np.abs(coef_path) > ZERO_TOL, axis=1)

    # Standalone final fit at best_alpha to capture exact convergence & n_iter_
    final_max_iter = 50000
    final_model = Lasso(
        alpha=best_alpha,
        fit_intercept=True,
        max_iter=final_max_iter,
        tol=1e-5,
    )
    with warnings.catch_warnings(record=True) as caught_final:
        warnings.simplefilter("always", ConvergenceWarning)
        final_model.fit(dataset.X_train, dataset.y_train)
        final_warn_count = sum(
            1 for w in caught_final if issubclass(w.category, ConvergenceWarning)
        )

    intercept_scaled = float(final_model.intercept_)
    coef_scaled = final_model.coef_.copy()
    coef_scaled[np.abs(coef_scaled) <= ZERO_TOL] = 0.0

    y_train_pred = final_model.predict(dataset.X_train)
    y_test_pred = final_model.predict(dataset.X_test)
    resid_train = dataset.y_train - y_train_pred
    resid_test = dataset.y_test - y_test_pred

    coef_unscaled, intercept_unscaled = unscale_coefficients(
        coef_scaled,
        intercept_scaled,
        dataset.scaler_means,
        dataset.scaler_scales,
    )

    n_iter_final = int(final_model.n_iter_)
    converged_final = (final_warn_count == 0) and (n_iter_final < final_max_iter)

    return LassoFitResult(
        alphas=alphas,
        alpha_max=alpha_max,
        cv_mse_mean=cv_mse_mean,
        cv_mse_std=cv_mse_std,
        cv_rmse_mean=cv_rmse_mean,
        cv_rmse_std=cv_rmse_std,
        cv_r2_mean=cv_r2_mean,
        cv_selection_freq_at_best=cv_selection_freq,
        best_alpha=best_alpha,
        best_alpha_idx=best_idx,
        intercept_scaled=intercept_scaled,
        coef_scaled=coef_scaled,
        intercept_unscaled=intercept_unscaled,
        coef_unscaled=coef_unscaled,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        resid_train=resid_train,
        resid_test=resid_test,
        coef_path=coef_path,
        nonzero_counts_path=nonzero_counts_path,
        n_iter_final=n_iter_final,
        max_iter=final_max_iter,
        converged_final=converged_final,
        cv_convergence_warnings=cv_warnings,
    )


def build_lasso_coefficients_table(
    dataset: PreparedDataset,
    fit: LassoFitResult,
    ols_coef_scaled: np.ndarray,
    ols_intercept_scaled: float,
) -> pd.DataFrame:
    """Build Lasso coefficients table noting sparsity status and CV fold selection frequency."""
    rows: List[Dict[str, Any]] = [
        {
            "role": "intercept",
            "feature": "INTERCEPT",
            "coefficient_standardized": fit.intercept_scaled,
            "coefficient_unstandardized": fit.intercept_unscaled,
            "ols_coefficient_standardized": float(ols_intercept_scaled),
            "retained": True,
            "cv_selection_freq": 1.0,
            "abs_coefficient_standardized": abs(fit.intercept_scaled),
        }
    ]
    for name, c_std, c_unstd, c_ols, freq in zip(
        dataset.feature_names,
        fit.coef_scaled,
        fit.coef_unscaled,
        ols_coef_scaled,
        fit.cv_selection_freq_at_best,
    ):
        rows.append(
            {
                "role": "predictor",
                "feature": name,
                "coefficient_standardized": float(c_std),
                "coefficient_unstandardized": float(c_unstd),
                "ols_coefficient_standardized": float(c_ols),
                "retained": bool(abs(c_std) > ZERO_TOL),
                "cv_selection_freq": float(freq),
                "abs_coefficient_standardized": float(abs(c_std)),
            }
        )
    df = pd.DataFrame(rows)
    df_predictors = df[df["role"] == "predictor"].sort_values(
        ["retained", "abs_coefficient_standardized"], ascending=[False, False]
    )
    return pd.concat([df[df["role"] == "intercept"], df_predictors], ignore_index=True)
