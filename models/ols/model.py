"""OLS model fitting via numpy.linalg.lstsq with sklearn verification."""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models.base import unscale_coefficients
from preprocessing.preprocess import PreparedDataset


@dataclass
class OLSFitResult:
    """Container for fitted OLS parameters and numerical SVD properties."""

    intercept_scaled: float
    coef_scaled: np.ndarray
    intercept_unscaled: float
    coef_unscaled: np.ndarray
    y_train_pred: np.ndarray
    y_test_pred: np.ndarray
    resid_train: np.ndarray
    resid_test: np.ndarray
    rank_augmented: int
    n_augmented_cols: int
    singular_values_augmented: np.ndarray
    condition_number_scaled: float
    condition_number_unscaled: float
    sklearn_max_abs_pred_diff: float


def fit_ols(dataset: PreparedDataset) -> OLSFitResult:
    """Fit OLS on standardized predictors + explicit unscaled intercept using numpy.linalg.lstsq."""
    X_tr = dataset.X_train
    X_te = dataset.X_test
    y_tr = dataset.y_train
    y_te = dataset.y_test

    X_tr_aug = np.column_stack([np.ones(len(X_tr)), X_tr])
    X_te_aug = np.column_stack([np.ones(len(X_te)), X_te])

    beta, _, rank_aug, s_aug = np.linalg.lstsq(X_tr_aug, y_tr, rcond=None)
    intercept_scaled = float(beta[0])
    coef_scaled = beta[1:].copy()

    y_train_pred = X_tr_aug @ beta
    y_test_pred = X_te_aug @ beta
    resid_train = y_tr - y_train_pred
    resid_test = y_te - y_test_pred

    coef_unscaled, intercept_unscaled = unscale_coefficients(
        coef_scaled,
        intercept_scaled,
        dataset.scaler_means,
        dataset.scaler_scales,
    )

    cond_scaled = (
        float(s_aug[0] / s_aug[-1]) if len(s_aug) > 0 and s_aug[-1] > 1e-15 else float("inf")
    )

    X_tr_unscaled_aug = np.column_stack([np.ones(len(dataset.X_train_unscaled)), dataset.X_train_unscaled])
    s_unscaled = np.linalg.svd(X_tr_unscaled_aug, compute_uv=False)
    cond_unscaled = (
        float(s_unscaled[0] / s_unscaled[-1])
        if len(s_unscaled) > 0 and s_unscaled[-1] > 1e-15
        else float("inf")
    )

    # Cross-check predictions against scikit-learn LinearRegression
    sk_lr = LinearRegression(fit_intercept=True)
    sk_lr.fit(X_tr, y_tr)
    sk_pred_test = sk_lr.predict(X_te)
    max_diff = float(np.max(np.abs(sk_pred_test - y_test_pred)))

    return OLSFitResult(
        intercept_scaled=intercept_scaled,
        coef_scaled=coef_scaled,
        intercept_unscaled=intercept_unscaled,
        coef_unscaled=coef_unscaled,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        resid_train=resid_train,
        resid_test=resid_test,
        rank_augmented=int(rank_aug),
        n_augmented_cols=int(X_tr_aug.shape[1]),
        singular_values_augmented=s_aug,
        condition_number_scaled=cond_scaled,
        condition_number_unscaled=cond_unscaled,
        sklearn_max_abs_pred_diff=max_diff,
    )


def build_ols_coefficients_table(
    dataset: PreparedDataset, fit: OLSFitResult
) -> pd.DataFrame:
    """Build coefficients DataFrame noting standardized, unstandardized, and intercept separately."""
    rows = [
        {
            "role": "intercept",
            "feature": "INTERCEPT",
            "coefficient_standardized": fit.intercept_scaled,
            "coefficient_unstandardized": fit.intercept_unscaled,
            "abs_coefficient_standardized": abs(fit.intercept_scaled),
        }
    ]
    for name, c_std, c_unstd in zip(
        dataset.feature_names, fit.coef_scaled, fit.coef_unscaled
    ):
        rows.append(
            {
                "role": "predictor",
                "feature": name,
                "coefficient_standardized": float(c_std),
                "coefficient_unstandardized": float(c_unstd),
                "abs_coefficient_standardized": float(abs(c_std)),
            }
        )
    df = pd.DataFrame(rows)
    df_predictors = df[df["role"] == "predictor"].sort_values(
        "abs_coefficient_standardized", ascending=False
    )
    return pd.concat([df[df["role"] == "intercept"], df_predictors], ignore_index=True)
