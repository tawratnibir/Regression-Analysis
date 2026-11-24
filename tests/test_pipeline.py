"""Lightweight verification suite for preprocessing, models, and HTML reports."""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from config import DATASETS, MODEL_KEYS, REPORTS_DIR, RESULTS_DIR
from models.lasso.model import fit_lasso
from models.ols.analysis import analyze_ols
from models.ols.model import fit_ols
from models.ridge.model import fit_ridge
from preprocessing.loader import load_raw_dataset
from preprocessing.preprocess import load_and_prepare_dataset


class TestPreprocessing(unittest.TestCase):
    """Verify leak-free preprocessing, split persistence, and no leakage across all 10 datasets."""

    def test_all_datasets_prepared_cleanly(self) -> None:
        for key, cfg in DATASETS.items():
            with self.subTest(dataset=key):
                d = load_and_prepare_dataset(key)
                self.assertEqual(d.X_train.shape[0], len(d.y_train))
                self.assertEqual(d.X_test.shape[0], len(d.y_test))
                self.assertEqual(d.X_train.shape[1], len(d.feature_names))
                self.assertEqual(d.X_test.shape[1], len(d.feature_names))

                # No NaN or Inf
                self.assertFalse(np.isnan(d.X_train).any())
                self.assertFalse(np.isnan(d.X_test).any())
                self.assertFalse(np.isinf(d.X_train).any())
                self.assertFalse(np.isinf(d.X_test).any())
                self.assertFalse(np.isnan(d.y_train).any())
                self.assertFalse(np.isnan(d.y_test).any())

                # No target or excluded column leakage
                self.assertNotIn(cfg.target, d.feature_names)
                for ex in cfg.exclude_columns:
                    self.assertNotIn(ex, d.feature_names)

                # Zero train/test index overlap
                tr_set = set(d.train_raw_indices.tolist())
                te_set = set(d.test_raw_indices.tolist())
                self.assertEqual(len(tr_set & te_set), 0)

                # Training predictors standardized (mean ~ 0)
                col_means = np.abs(np.mean(d.X_train, axis=0))
                self.assertLess(float(np.max(col_means)), 1e-10)

    def test_parkinsons_group_separation(self) -> None:
        raw = load_raw_dataset("parkinsons")
        d = load_and_prepare_dataset("parkinsons")
        train_subjects = set(raw.iloc[d.train_raw_indices]["subject#"].unique())
        test_subjects = set(raw.iloc[d.test_raw_indices]["subject#"].unique())
        self.assertEqual(len(train_subjects & test_subjects), 0)
        self.assertNotIn("motor_UPDRS", d.feature_names)
        self.assertNotIn("subject#", d.feature_names)

    def test_chronological_splits(self) -> None:
        raw_ae = load_raw_dataset("appliances_energy")
        d_ae = load_and_prepare_dataset("appliances_energy")
        t_ae = pd.to_datetime(raw_ae["date"], format="%Y-%m-%d%H:%M:%S")
        self.assertLess(
            t_ae.iloc[d_ae.train_raw_indices].max(),
            t_ae.iloc[d_ae.test_raw_indices].min(),
        )

        raw_w = load_raw_dataset("weather")
        d_w = load_and_prepare_dataset("weather")
        t_w = pd.to_datetime(raw_w["Date"], format="%Y-%m-%d")
        self.assertLess(
            t_w.iloc[d_w.train_raw_indices].max(),
            t_w.iloc[d_w.test_raw_indices].min(),
        )
        self.assertNotIn("Next_Tmin", d_w.feature_names)

    def test_split_reproducibility(self) -> None:
        d1 = load_and_prepare_dataset("auto_mpg")
        d2 = load_and_prepare_dataset("auto_mpg")
        np.testing.assert_array_equal(d1.train_raw_indices, d2.train_raw_indices)
        np.testing.assert_array_equal(d1.test_raw_indices, d2.test_raw_indices)
        np.testing.assert_allclose(d1.X_train, d2.X_train)
        np.testing.assert_allclose(d1.X_test, d2.X_test)


class TestModelsAndArtifacts(unittest.TestCase):
    """Verify OLS algebraic checks, Ridge/Lasso convergence, and saved artifacts/reports."""

    def test_ols_exact_properties_airfoil(self) -> None:
        d = load_and_prepare_dataset("airfoil")
        fit = fit_ols(d)
        diag, _, _ = analyze_ols(d, fit)
        ex = diag["exact_algebraic_properties"]
        self.assertLess(ex["max_abs_Xt_residuals"], 1e-8)
        self.assertLess(abs(ex["residual_sum"]), 1e-8)
        self.assertLess(ex["mean_fitted_identity_abs_diff"], 1e-8)
        self.assertLess(fit.sklearn_max_abs_pred_diff, 1e-8)

    def test_ridge_and_lasso_airfoil(self) -> None:
        d = load_and_prepare_dataset("airfoil")
        r_fit = fit_ridge(d)
        self.assertGreater(r_fit.best_alpha, 0.0)
        self.assertLess(r_fit.svd_max_abs_coef_diff, 1e-8)

        l_fit = fit_lasso(d)
        self.assertGreater(l_fit.best_alpha, 0.0)
        self.assertTrue(l_fit.converged_final)
        self.assertTrue(np.all(np.isfinite(l_fit.y_test_pred)))

    def test_saved_artifacts_and_reports_exist(self) -> None:
        for model_key in MODEL_KEYS:
            summary = REPORTS_DIR / model_key / "summary.html"
            self.assertTrue(summary.exists(), f"Missing summary report: {summary}")
            for dataset_key in DATASETS:
                res_dir = RESULTS_DIR / model_key / dataset_key
                with self.subTest(model=model_key, dataset=dataset_key):
                    for fname in (
                        "metrics.json",
                        "coefficients.csv",
                        "predictions_test.csv",
                        "diagnostics.json",
                        "run_metadata.json",
                    ):
                        self.assertTrue((res_dir / fname).exists(), f"Missing {res_dir / fname}")

                    with open(res_dir / "metrics.json", "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                    self.assertIn("train", metrics)
                    self.assertIn("test", metrics)
                    self.assertTrue(np.isfinite(metrics["test"]["rmse"]))
                    if dataset_key == "diabetes":
                        self.assertIn("test_classification", metrics)
                        self.assertIn("pct_outside_unit_interval", metrics["test_classification"])

                    report_path = REPORTS_DIR / model_key / f"{dataset_key}.html"
                    self.assertTrue(report_path.exists(), f"Missing {report_path}")
                    html = report_path.read_text(encoding="utf-8")
                    self.assertIn("data:image/png;base64,", html)
                    self.assertNotIn("{{", html)

    def test_grand_index_and_predictor_math(self) -> None:
        from reporting.grand_index import _build_predictor_payload

        index_path = REPORTS_DIR / "index.html"
        self.assertTrue(index_path.exists(), f"Missing grand index: {index_path}")
        html = index_path.read_text(encoding="utf-8")
        self.assertIn("PREDICTOR_DATA", html)
        self.assertIn("data:image/png;base64,", html)
        self.assertNotIn("{{", html)

        # Verify unstandardized JS prediction formula matches saved predictions_test.csv
        for dataset_key in DATASETS:
            payload = _build_predictor_payload(dataset_key)
            self.assertGreater(len(payload["samples"]), 0)
            sample = payload["samples"][0]
            # Extract row_id from label "Test row #<id> ..."
            raw_id = int(sample["label"].split("#")[1].split(" ")[0])

            x_map = {}
            for f in payload["fields"]:
                val = sample["values"][f["name"]]
                if f["kind"] == "categorical":
                    for opt in f["options"]:
                        x_map[f["encoded_prefix"] + opt["value"]] = 1.0 if str(val) == opt["value"] else 0.0
                else:
                    x_map[f["name"]] = float(val)

            for model_key in MODEL_KEYS:
                mw = payload["models"][model_key]
                pred_js = mw["intercept"] + sum(
                    w * x_map.get(feat, 0.0) for feat, w in mw["coefs"].items()
                )
                pred_df = pd.read_csv(RESULTS_DIR / model_key / dataset_key / "predictions_test.csv")
                saved_pred = float(pred_df.loc[pred_df["raw_row_id"] == raw_id, "predicted"].iloc[0])
                self.assertAlmostEqual(
                    pred_js,
                    saved_pred,
                    delta=max(1e-3, abs(saved_pred) * 1e-4),
                    msg=f"Predictor mismatch for {model_key}/{dataset_key}",
                )


if __name__ == "__main__":
    unittest.main()
