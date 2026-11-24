"""Ridge regression fitting with leak-free cross-validated alpha selection."""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from models.base import unscale_coefficients
from preprocessing.preprocess import PreparedDataset, build_preprocessor

RIDGE_ALPHAS = np.logspace(-4, 5, 60)


@dataclass
class RidgeFitResult:
    """Container for cross-validated Ridge fit, CV curve, and coefficient path."""

    alphas: np.ndarray
    cv_mse_mean: np.ndarray
    cv_mse_std: np.ndarray
    cv_rmse_mean: np.ndarray
    cv_rmse_std: np.ndarray
    cv_r2_mean: np.ndarray
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
    svd_max_abs_coef_diff: float


def fit_ridge(dataset: PreparedDataset, alphas: np.ndarray = RIDGE_ALPHAS) -> RidgeFitResult:
    """Tune Ridge penalty via leak-free training-only CV, then fit on full training split."""
    n_alphas = len(alphas)
    n_folds = len(dataset.cv_splits)
    fold_mse = np.zeros((n_folds, n_alphas), dtype=float)
    fold_r2 = np.zeros((n_folds, n_alphas), dtype=float)

    for f_idx, (tr_pos, val_pos) in enumerate(dataset.cv_splits):
        prep = build_preprocessor(
            dataset.numeric_columns, dataset.categorical_columns, standardize=True
        )
        X_tr_f = prep.fit_transform(dataset.df_train_raw.iloc[tr_pos])
        X_val_f = prep.transform(dataset.df_train_raw.iloc[val_pos])
        y_tr_f = dataset.y_train[tr_pos]
        y_val_f = dataset.y_train[val_pos]

        for a_idx, alpha in enumerate(alphas):
            reg = Ridge(alpha=float(alpha), fit_intercept=True)
            reg.fit(X_tr_f, y_tr_f)
            pred_val = reg.predict(X_val_f)
            fold_mse[f_idx, a_idx] = float(mean_squared_error(y_val_f, pred_val))
            fold_r2[f_idx, a_idx] = float(r2_score(y_val_f, pred_val))

    cv_mse_mean = np.mean(fold_mse, axis=0)
    cv_mse_std = np.std(fold_mse, axis=0, ddof=0)
    fold_rmse = np.sqrt(fold_mse)
    cv_rmse_mean = np.mean(fold_rmse, axis=0)
    cv_rmse_std = np.std(fold_rmse, axis=0, ddof=0)
    cv_r2_mean = np.mean(fold_r2, axis=0)

    best_idx = int(np.argmin(cv_mse_mean))
    best_alpha = float(alphas[best_idx])

    # Refit on full standardized training matrix
    final_model = Ridge(alpha=best_alpha, fit_intercept=True)
    final_model.fit(dataset.X_train, dataset.y_train)
    intercept_scaled = float(final_model.intercept_)
    coef_scaled = final_model.coef_.copy()

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

    # Full training coefficient path across alpha grid
    coef_path = np.zeros((n_alphas, dataset.X_train.shape[1]), dtype=float)
    for a_idx, alpha in enumerate(alphas):
        m = Ridge(alpha=float(alpha), fit_intercept=True)
        m.fit(dataset.X_train, dataset.y_train)
        coef_path[a_idx, :] = m.coef_

    # Verify against closed-form SVD solution on centered X_train
    X_c = dataset.X_train - np.mean(dataset.X_train, axis=0)
    y_c = dataset.y_train - np.mean(dataset.y_train)
    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    d = s / (s ** 2 + best_alpha)
    coef_svd = Vt.T @ (d * (U.T @ y_c))
    svd_diff = float(np.max(np.abs(coef_svd - coef_scaled)))

    return RidgeFitResult(
        alphas=alphas.copy(),
        cv_mse_mean=cv_mse_mean,
        cv_mse_std=cv_mse_std,
        cv_rmse_mean=cv_rmse_mean,
        cv_rmse_std=cv_rmse_std,
        cv_r2_mean=cv_r2_mean,
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
        svd_max_abs_coef_diff=svd_diff,
    )


def build_ridge_coefficients_table(
    dataset: PreparedDataset,
    fit: RidgeFitResult,
    ols_coef_scaled: np.ndarray,
    ols_intercept_scaled: float,
) -> pd.DataFrame:
    """Build Ridge coefficients table alongside OLS standardized reference and shrinkage ratio."""
    rows: List[Dict[str, Any]] = [
        {
            "role": "intercept",
            "feature": "INTERCEPT",
            "coefficient_standardized": fit.intercept_scaled,
            "coefficient_unstandardized": fit.intercept_unscaled,
            "ols_coefficient_standardized": float(ols_intercept_scaled),
            "shrinkage_ratio_vs_ols": 1.0,
            "abs_coefficient_standardized": abs(fit.intercept_scaled),
        }
    ]
    for name, c_std, c_unstd, c_ols in zip(
        dataset.feature_names, fit.coef_scaled, fit.coef_unscaled, ols_coef_scaled
    ):
        ratio = float(c_std / c_ols) if abs(c_ols) > 1e-12 else float("nan")
        rows.append(
            {
                "role": "predictor",
                "feature": name,
                "coefficient_standardized": float(c_std),
                "coefficient_unstandardized": float(c_unstd),
                "ols_coefficient_standardized": float(c_ols),
                "shrinkage_ratio_vs_ols": ratio,
                "abs_coefficient_standardized": float(abs(c_std)),
            }
        )
    df = pd.DataFrame(rows)
    df_predictors = df[df["role"] == "predictor"].sort_values(
        "abs_coefficient_standardized", ascending=False
    )
    return pd.concat([df[df["role"] == "intercept"], df_predictors], ignore_index=True)
