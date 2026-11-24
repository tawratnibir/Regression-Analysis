"""Generate self-contained, artifact-grounded HTML reports for OLS, Ridge, and Lasso."""

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
    ("actual_vs_predicted.png", "Actual vs Predicted", "Observed target values plotted against model predictions on training and held-out test splits."),
    ("residuals_vs_fitted.png", "Residuals vs Fitted", "Training and test residuals plotted against fitted values with binned training mean trend."),
    ("residual_distribution.png", "Residual Distribution", "Density histogram of training and test residuals overlaid with a fitted Normal curve."),
    ("qq_plot.png", "Normal Q-Q Plot", "Theoretical Normal quantiles vs ordered training residuals."),
    ("cooks_distance.png", "Cook's Distance", "Training observation influence measured via Cook's distance relative to the 4/n reference threshold."),
    ("cv_curve.png", "Cross-Validation Curve", "Training-only cross-validation RMSE (±1 SD across folds) versus log10(alpha)."),
    ("coefficient_path.png", "Regularization Coefficient Path", "Standardized coefficient trajectories across log10(alpha), highlighting the CV-selected penalty."),
    ("ols_vs_ridge_coefficients.png", "OLS vs Ridge Coefficients", "Side-by-side comparison of top standardized OLS and Ridge coefficients."),
    ("sparsity_vs_alpha.png", "Sparsity vs Penalty", "Number of retained nonzero predictors across log10(alpha) on the training split."),
]

REPORT_TEMPLATE = JINJA_ENV.from_string("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ model_title }} — {{ dataset_name }}</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #1e293b;
      --muted: #64748b;
      --border: #e2e8f0;
      --accent: #1d4ed8;
      --accent-soft: #eff6ff;
      --warn-bg: #fffbeb;
      --warn-border: #f59e0b;
    }
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
      margin: 0;
      padding: 24px;
    }
    .container { max-width: 1120px; margin: 0 auto; }
    header {
      background: var(--card);
      border: 1px solid var(--border);
      border-left: 5px solid var(--accent);
      border-radius: 8px;
      padding: 20px 24px;
      margin-bottom: 20px;
    }
    header h1 { margin: 0 0 6px 0; font-size: 1.65rem; }
    .subtitle { color: var(--muted); font-size: 0.95rem; }
    .badge {
      display: inline-block;
      background: var(--accent-soft);
      color: var(--accent);
      border-radius: 999px;
      padding: 2px 10px;
      font-size: 0.8rem;
      font-weight: 600;
      margin-right: 6px;
    }
    section {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px 24px;
      margin-bottom: 20px;
    }
    h2 {
      margin-top: 0;
      font-size: 1.2rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
      color: #0f172a;
    }
    h3 { font-size: 1rem; margin-top: 16px; color: #334155; }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0;
      font-size: 0.9rem;
    }
    th, td {
      border: 1px solid var(--border);
      padding: 8px 10px;
      text-align: left;
    }
    th { background: #f1f5f9; font-weight: 600; }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    tr.intercept-row { background: #eff6ff; font-weight: 600; }
    tr.zero-row { color: #94a3b8; }
    .callout {
      background: var(--accent-soft);
      border-left: 4px solid var(--accent);
      padding: 12px 14px;
      border-radius: 4px;
      margin: 12px 0;
      font-size: 0.92rem;
    }
    .warning {
      background: var(--warn-bg);
      border-left: 4px solid var(--warn-border);
      padding: 12px 14px;
      border-radius: 4px;
      margin: 12px 0;
      font-size: 0.92rem;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
    }
    .plot-card {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      background: #ffffff;
    }
    .plot-card img { width: 100%; height: auto; display: block; border-radius: 4px; }
    .plot-title { font-weight: 600; margin-top: 8px; font-size: 0.95rem; }
    .plot-caption { color: var(--muted); font-size: 0.85rem; margin-top: 4px; }
    ul { padding-left: 20px; }
    li { margin-bottom: 6px; }
    code {
      background: #f1f5f9;
      padding: 1px 5px;
      border-radius: 4px;
      font-size: 0.88em;
    }
    footer { color: var(--muted); font-size: 0.82rem; text-align: center; margin-top: 16px; }
  </style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <span class="badge">{{ model_key | upper }}</span>
      <span class="badge">UCI ID {{ metadata.uci_id }}</span>
      <span class="badge">Seed {{ metadata.seed }}</span>
    </div>
    <h1>{{ model_title }} — {{ dataset_name }}</h1>
    <div class="subtitle">
      Target: <code>{{ metadata.target }}</code> ({{ metadata.target_units }}) &middot;
      Run UTC: {{ metadata.run_datetime_utc }}
    </div>
  </header>

  <section>
    <h2>1. Dataset Description and Target</h2>
    <p>{{ dataset_description }}</p>
    <table>
      <tr><th>Property</th><th>Value</th></tr>
      <tr><td>Dataset Key / UCI ID</td><td><code>{{ metadata.dataset }}</code> (UCI ID {{ metadata.uci_id }})</td></tr>
      <tr><td>Target Variable &amp; Units</td><td><code>{{ metadata.target }}</code> ({{ metadata.target_units }})</td></tr>
      <tr><td>Raw / Clean Observations</td><td>{{ prep.n_raw }} raw rows &rarr; {{ prep.n_clean }} clean rows</td></tr>
      <tr><td>Train / Test Counts</td><td>{{ prep.n_train }} train / {{ prep.n_test }} test</td></tr>
      <tr><td>Raw / Encoded Predictors</td><td>{{ prep.n_raw_features }} raw features &rarr; {{ prep.n_encoded_features }} model columns</td></tr>
      <tr><td>Train Target Summary</td><td>mean={{ "%.4g"|format(prep.target_stats_train.mean) }}, sd={{ "%.4g"|format(prep.target_stats_train.std) }}, range=[{{ "%.4g"|format(prep.target_stats_train.min) }}, {{ "%.4g"|format(prep.target_stats_train.max) }}] {{ metadata.target_units }}</td></tr>
      <tr><td>Test Target Summary</td><td>mean={{ "%.4g"|format(prep.target_stats_test.mean) }}, sd={{ "%.4g"|format(prep.target_stats_test.std) }}, range=[{{ "%.4g"|format(prep.target_stats_test.min) }}, {{ "%.4g"|format(prep.target_stats_test.max) }}] {{ metadata.target_units }}</td></tr>
    </table>
    {% if dataset_notes %}
    <h3>Dataset-Specific Notes</h3>
    <ul>
      {% for note in dataset_notes %}<li>{{ note }}</li>{% endfor %}
    </ul>
    {% endif %}
  </section>

  <section>
    <h2>2. Preprocessing and Split Methodology</h2>
    <table>
      <tr><th>Setting</th><th>Details</th></tr>
      <tr><td>Split Strategy</td><td><code>{{ prep.split_strategy }}</code> (persisted at <code>{{ metadata.split_reference }}</code>)</td></tr>
      <tr><td>Cross-Validation Strategy</td><td><code>{{ prep.cv_strategy }}</code> (training split only)</td></tr>
      <tr><td>Excluded Columns</td><td>{% if prep.excluded_columns %}{{ prep.excluded_columns | join(", ") }}{% else %}None{% endif %}</td></tr>
      <tr><td>Categorical Columns</td><td>{% if prep.categorical_columns %}{{ prep.categorical_columns | join(", ") }} (One-Hot drop-first, fit on train only){% else %}None{% endif %}</td></tr>
      <tr><td>Missing Value Imputation</td><td>{% if prep.missing_values_in_features %}Median imputation fit strictly on training data for: {% for k, v in prep.missing_values_in_features.items() %}{{ k }} ({{ v }} missing){% if not loop.last %}, {% endif %}{% endfor %}{% else %}No missing predictor values in clean dataset{% endif %}</td></tr>
      <tr><td>Predictor Standardization</td><td>Zero-mean, unit-variance <code>StandardScaler</code> fit strictly on training data (and separately inside each CV fold). Intercept is neither scaled nor penalized.</td></tr>
    </table>
    {% if prep.dropped_summary_rows or prep.dropped_missing_target_rows %}
    <div class="callout">
      <strong>Row Cleaning Record:</strong> Dropped {{ prep.dropped_summary_rows }} trailing summary row(s) and {{ prep.dropped_missing_target_rows }} row(s) with missing target values before splitting.
    </div>
    {% endif %}
  </section>

  <section>
    <h2>3. Model Formulation and Fitted Coefficients</h2>
    <div class="callout">
      <strong>Objective Convention:</strong> <code>{{ metadata.objective_convention }}</code>
      {% if metadata.penalty_settings.selected_alpha is defined %}
      <br><strong>Selected Penalty (&alpha;):</strong> <code>{{ "%.6g"|format(metadata.penalty_settings.selected_alpha) }}</code>
      (grid [{{ "%.4g"|format(metadata.penalty_settings.alpha_grid_min) }}, {{ "%.4g"|format(metadata.penalty_settings.alpha_grid_max) }}], {{ metadata.penalty_settings.n_alphas }} points)
      {% endif %}
    </div>
    <p>Coefficients below are shown on both the standardized predictor scale (comparable across predictors and models) and back-transformed unstandardized predictor units.</p>
    <table>
      <thead>
        <tr>
          <th>Role</th>
          <th>Feature</th>
          <th class="num">Coef (Standardized)</th>
          <th class="num">Coef (Unstandardized)</th>
          {% if has_ols_ref %}<th class="num">OLS Coef (Std)</th>{% endif %}
          {% if has_shrinkage_ratio %}<th class="num">Shrinkage (Ridge / OLS)</th>{% endif %}
          {% if has_retained %}<th class="num">Retained</th><th class="num">CV Freq</th>{% endif %}
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
          {% if has_shrinkage_ratio %}<td class="num">{% if row.shrinkage_ratio_vs_ols == row.shrinkage_ratio_vs_ols %}{{ "%.4f"|format(row.shrinkage_ratio_vs_ols) }}{% else %}—{% endif %}</td>{% endif %}
          {% if has_retained %}<td class="num">{{ "Yes" if row.retained else "No (0)" }}</td><td class="num">{{ "%.2f"|format(row.cv_selection_freq) }}</td>{% endif %}
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </section>

  <section>
    <h2>4. Train and Test Metrics</h2>
    {% if metrics.is_binary_classification %}
    <div class="warning">
      <strong>Linear Probability Model (LPM) Notice:</strong> Because <code>{{ metadata.target }}</code> is binary (0/1), linear regression estimates conditional probabilities P(Positive | X), which are not constrained to [0, 1].
    </div>
    {% endif %}
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
    <h3>Linear Probability Model (LPM) Classification &amp; Probability Bounds</h3>
    <table>
      <thead>
        <tr><th>Classification / Bounds Metric</th><th class="num">Train</th><th class="num">Test</th></tr>
      </thead>
      <tbody>
        <tr><td>Accuracy (threshold 0.50)</td><td class="num">{{ "%.4f"|format(metrics.train_classification.accuracy) }}</td><td class="num">{{ "%.4f"|format(metrics.test_classification.accuracy) }}</td></tr>
        <tr><td>Precision</td><td class="num">{{ "%.4f"|format(metrics.train_classification.precision) }}</td><td class="num">{{ "%.4f"|format(metrics.test_classification.precision) }}</td></tr>
        <tr><td>Recall</td><td class="num">{{ "%.4f"|format(metrics.train_classification.recall) }}</td><td class="num">{{ "%.4f"|format(metrics.test_classification.recall) }}</td></tr>
        <tr><td>F1 Score</td><td class="num">{{ "%.4f"|format(metrics.train_classification.f1) }}</td><td class="num">{{ "%.4f"|format(metrics.test_classification.f1) }}</td></tr>
        <tr><td>ROC AUC</td><td class="num">{{ "%.4f"|format(metrics.train_classification.roc_auc) }}</td><td class="num">{{ "%.4f"|format(metrics.test_classification.roc_auc) }}</td></tr>
        <tr><td>Predicted Range [min, max]</td><td class="num">[{{ "%.4f"|format(metrics.train_classification.pred_min) }}, {{ "%.4f"|format(metrics.train_classification.pred_max) }}]</td><td class="num">[{{ "%.4f"|format(metrics.test_classification.pred_min) }}, {{ "%.4f"|format(metrics.test_classification.pred_max) }}]</td></tr>
        <tr><td>Predictions Outside [0, 1]</td><td class="num">{{ metrics.train_classification.n_outside_unit_interval }} ({{ "%.2f"|format(metrics.train_classification.pct_outside_unit_interval) }}%)</td><td class="num">{{ metrics.test_classification.n_outside_unit_interval }} ({{ "%.2f"|format(metrics.test_classification.pct_outside_unit_interval) }}%)</td></tr>
        <tr><td>Test Confusion Matrix (TN / FP / FN / TP)</td><td class="num">—</td><td class="num">{{ metrics.test_classification.confusion_matrix.tn }} / {{ metrics.test_classification.confusion_matrix.fp }} / {{ metrics.test_classification.confusion_matrix.fn }} / {{ metrics.test_classification.confusion_matrix.tp }}</td></tr>
      </tbody>
    </table>
    {% endif %}
  </section>

  <section>
    <h2>5. Diagnostic Plots and Quantitative Checks</h2>
    <div class="grid-2">
      {% for p in plots %}
      <div class="plot-card">
        <img src="data:image/png;base64,{{ p.b64 }}" alt="{{ p.title }}">
        <div class="plot-title">{{ p.title }}</div>
        <div class="plot-caption">{{ p.caption }}</div>
      </div>
      {% endfor %}
    </div>

    <h3>Computed Diagnostics Summary</h3>
    {% for group_name, items in diagnostic_sections %}
    <h4>{{ group_name }}</h4>
    <table>
      {% for k, v in items %}
      <tr><td>{{ k }}</td><td class="num">{{ v }}</td></tr>
      {% endfor %}
    </table>
    {% endfor %}
  </section>

  <section>
    <h2>6. Algorithm-Specific Interpretation</h2>
    <ul>
      {% for item in interpretations %}
      <li>{{ item }}</li>
      {% endfor %}
    </ul>
  </section>

  <section>
    <h2>7. Limitations and Potential Assumption Violations</h2>
    <ul>
      {% for lim in limitations %}
      <li>{{ lim }}</li>
      {% endfor %}
    </ul>
  </section>

  <footer>
    Generated strictly from saved artifacts in <code>results/{{ model_key }}/{{ metadata.dataset }}/</code> &middot;
    Python {{ metadata.dependencies.python }}, NumPy {{ metadata.dependencies.numpy }}, scikit-learn {{ metadata.dependencies.scikit_learn }}
  </footer>
</div>
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
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 24px; line-height: 1.5; }
    .container { max-width: 1200px; margin: 0 auto; background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 24px; }
    h1 { margin-top: 0; }
    .warning { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 4px; margin: 14px 0; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 0.88rem; }
    th, td { border: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; }
    th { background: #f1f5f9; }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    a { color: #1d4ed8; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
<div class="container">
  <h1>{{ model_title }} — All-Dataset Summary</h1>
  <div class="warning">
    <strong>Unit Comparability Warning:</strong> Raw RMSE and MAE values below carry each dataset's specific target units (e.g., dB, MW, K, mpg, Wh, deg C). Never compare raw RMSE across datasets with different units. Use dimensionless <strong>Test R&sup2;</strong> and unit-free <strong>Normalized RMSE (Test RMSE / Test Target SD)</strong> for cross-dataset context.
  </div>
  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Split</th>
        <th class="num">Train / Test</th>
        <th class="num">Features</th>
        <th class="num">Train R&sup2;</th>
        <th class="num">Test R&sup2;</th>
        <th class="num">Test RMSE (Native Units)</th>
        <th class="num">Norm. Test RMSE (RMSE/SD)</th>
        <th>Key Model Diagnostic</th>
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
        <td><a href="{{ r.dataset }}.html">View report</a></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>
</body>
</html>
""")


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
        return f"{val:.5g}"
    return str(val)


def _load_plots(plots_dir: Path) -> List[Dict[str, str]]:
    loaded: List[Dict[str, str]] = []
    for fname, title, caption in PLOT_ORDER:
        p = plots_dir / fname
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            loaded.append({"title": title, "caption": caption, "b64": b64})
    return loaded


def _build_diagnostic_sections_and_narrative(
    model_key: str,
    dataset_key: str,
    metrics: Dict[str, Any],
    coef_df: pd.DataFrame,
    diagnostics: Dict[str, Any],
    metadata: Dict[str, Any],
) -> tuple[List[tuple[str, List[tuple[str, str]]]], List[str], List[str]]:
    """Convert raw saved diagnostics into structured tables, interpretations, and limitations."""
    cfg = DATASETS[dataset_key]
    prep = metadata["preprocessing"]
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
        f"On the held-out test split (n={metrics['n_test']}), the model achieves R²={test_r2:.4f} "
        f"and RMSE={test_rmse:.4g} {units} (compared to Train R²={train_r2:.4f}, "
        f"Train RMSE={metrics['train']['rmse']:.4g} {units})."
    )
    if len(top_preds) > 0:
        interpretations.append(
            f"Largest standardized predictor effects in absolute magnitude: {top_str}."
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
            "Exact OLS Algebraic Properties (Guaranteed by Normal Equations with Intercept)",
            [
                ("max |X_aug^T r| (orthogonality check)", _fmt(ex["max_abs_Xt_residuals"])),
                ("Sum of training residuals", _fmt(ex["residual_sum"])),
                ("|mean(y_train) - mean(y_hat_train)|", _fmt(ex["mean_fitted_identity_abs_diff"])),
                ("|SST - (SSR + SSE)|", _fmt(ex["anova_identity_abs_error"])),
            ],
        ))
        sections.append((
            "Linear Algebra & Multicollinearity",
            [
                ("Matrix rank / augmented columns", f"{la['matrix_rank']} / {la['n_augmented_columns']}"),
                ("Rank deficient", _fmt(la["is_rank_deficient"])),
                ("Condition number (standardized + intercept)", _fmt(la["condition_number_standardized_with_intercept"])),
                ("Condition number (unstandardized + intercept)", _fmt(la["condition_number_unstandardized_with_intercept"])),
                ("Max finite VIF / features with VIF > 10", f"{_fmt(mc['max_vif'])} / {mc['n_vif_gt_10']}"),
                ("Max |pred diff| vs scikit-learn LinearRegression", _fmt(la["sklearn_max_abs_test_pred_diff"])),
            ],
        ))
        sections.append((
            "Empirical Statistical Assumption Checks (Training Residuals)",
            [
                ("Breusch-Pagan LM stat / p-value", f"{_fmt(het['breusch_pagan_lm_stat'])} (p={_fmt(het['breusch_pagan_pvalue'])})"),
                ("Spearman |residual| vs fitted (r / p)", f"{_fmt(het['spearman_abs_resid_vs_fitted_r'])} (p={_fmt(het['spearman_abs_resid_vs_fitted_pvalue'])})"),
                ("Skewness / Excess kurtosis", f"{_fmt(norm['skewness'])} / {_fmt(norm['excess_kurtosis'])}"),
                ("Jarque-Bera p-value / Shapiro-Wilk p-value", f"{_fmt(norm['jarque_bera_pvalue'])} / {_fmt(norm['shapiro_wilk_pvalue'])}"),
                ("High-leverage points (> 2p/n) / High Cook's D (> 4/n)", f"{inf['n_high_leverage']} (max={_fmt(inf['max_leverage'])}) / {inf['n_high_cooks_d']} (max={_fmt(inf['max_cooks_d'])})"),
                ("Durbin-Watson / Lag-1 residual corr", f"{_fmt(ac['durbin_watson'])} / {_fmt(ac['lag1_residual_corr'])}"),
            ],
        ))

        interpretations.append(
            f"Exact OLS identities hold to machine precision: max |X_aug^T r|={ex['max_abs_Xt_residuals']:.2e}, "
            f"|mean(y) - mean(y_hat)|={ex['mean_fitted_identity_abs_diff']:.2e}, and predictions match scikit-learn LinearRegression within {la['sklearn_max_abs_test_pred_diff']:.2e}."
        )
        interpretations.append(
            f"Standardizing predictors changes the augmented design matrix condition number from {_fmt(la['condition_number_unstandardized_with_intercept'])} "
            f"(unstandardized) to {_fmt(la['condition_number_standardized_with_intercept'])} (standardized), with rank {la['matrix_rank']} of {la['n_augmented_columns']}."
        )

        if la["is_rank_deficient"]:
            limitations.append(
                f"Rank-deficient design matrix (rank {la['matrix_rank']} < {la['n_augmented_columns']}): exact collinearity is present (e.g., identical features), so individual OLS coefficients are not uniquely identified even though lstsq returns the minimum-norm solution."
            )
        if mc["n_vif_gt_10"] > 0 or mc.get("has_infinite_vif"):
            limitations.append(
                f"Substantial multicollinearity detected: {mc['n_vif_gt_10']} predictor(s) have VIF > 10 (max finite VIF={_fmt(mc['max_vif'])}), inflating OLS coefficient variance."
            )
        if het["breusch_pagan_pvalue"] < 0.05:
            limitations.append(
                f"Heteroskedasticity check rejects constant error variance (Breusch-Pagan p={het['breusch_pagan_pvalue']:.2e}), meaning classical homoskedastic OLS standard errors would be unreliable."
            )
        if norm["jarque_bera_pvalue"] < 0.05:
            limitations.append(
                f"Training residuals deviate from Gaussian normality (Jarque-Bera p={norm['jarque_bera_pvalue']:.2e}, skewness={norm['skewness']:.3f}, excess kurtosis={norm['excess_kurtosis']:.3f})."
            )
        if inf["n_high_cooks_d"] > 0:
            limitations.append(
                f"{inf['n_high_cooks_d']} training observation(s) exceed the Cook's distance 4/n threshold (max Cook's D={inf['max_cooks_d']:.4g}), indicating potential sensitivity to influential points."
            )
        if ac["durbin_watson"] < 1.5 or ac["durbin_watson"] > 2.5:
            limitations.append(
                f"Residual autocorrelation is indicated (Durbin-Watson={ac['durbin_watson']:.3f}, lag-1 corr={ac['lag1_residual_corr']:.3f}), violating independent-error assumptions—especially notable for ordered/time-series data."
            )

    elif model_key == "ridge":
        pt = diagnostics["penalty_tuning"]
        sd = diagnostics["shrinkage_and_degrees_of_freedom"]
        ns = diagnostics["numerical_stability"]
        rc = diagnostics["residual_checks"]

        sections.append((
            "Cross-Validated Penalty Tuning & SVD Verification",
            [
                ("CV strategy", pt["cv_strategy"]),
                ("Selected alpha (L2)", _fmt(pt["selected_alpha"])),
                ("Best CV RMSE (±SD)", f"{_fmt(pt['best_cv_rmse'])} (±{_fmt(pt['best_cv_rmse_std'])}) {units}"),
                ("Best CV R²", _fmt(pt["best_cv_r2"])),
                ("Max |coef diff| vs closed-form SVD", _fmt(pt["svd_verification_max_abs_coef_diff"])),
            ],
        ))
        sections.append((
            "Shrinkage, Effective Degrees of Freedom & Stability",
            [
                ("Effective degrees of freedom (predictors / +intercept)", f"{_fmt(sd['effective_df_predictors'])} / {_fmt(sd['effective_df_with_intercept'])} (of {sd['n_predictors']} predictors)"),
                ("||beta_OLS||_2 / ||beta_Ridge||_2", f"{_fmt(sd['ols_coef_l2_norm'])} / {_fmt(sd['ridge_coef_l2_norm'])}"),
                ("L2 norm shrinkage ratio (Ridge / OLS)", _fmt(sd['l2_norm_ratio_ridge_over_ols'])),
                ("Centered X^T X condition ratio (OLS vs Ridge)", f"{_fmt(ns['x_centered_eig_condition_ols'])} vs {_fmt(ns['x_centered_eig_condition_ridge'])}"),
                ("Spearman |residual| vs fitted (r / p)", f"{_fmt(rc['spearman_abs_resid_vs_fitted_r'])} (p={_fmt(rc['spearman_abs_resid_vs_fitted_pvalue'])})"),
                ("Durbin-Watson", _fmt(rc["durbin_watson"])),
            ],
        ))

        interpretations.append(
            f"Training-only {pt['cv_strategy']} selected alpha={pt['selected_alpha']:.4g} "
            f"(CV RMSE={pt['best_cv_rmse']:.4g} {units}, CV R²={pt['best_cv_r2']:.4f}), matching the closed-form SVD solution within {pt['svd_verification_max_abs_coef_diff']:.2e}."
        )
        interpretations.append(
            f"L2 regularization shrunk the standardized coefficient L2 norm from {sd['ols_coef_l2_norm']:.4g} (OLS) to {sd['ridge_coef_l2_norm']:.4g} "
            f"(ratio={sd['l2_norm_ratio_ridge_over_ols']:.4f}), reducing effective predictor degrees of freedom from {sd['n_predictors']} to {sd['effective_df_predictors']:.2f}."
        )
        interpretations.append(
            f"Adding alpha*I stabilized the centered X^T X eigenvalue condition ratio from {_fmt(ns['x_centered_eig_condition_ols'])} to {_fmt(ns['x_centered_eig_condition_ridge'])}."
        )

        if sd["l2_norm_ratio_ridge_over_ols"] > 0.99:
            limitations.append(
                f"Selected alpha={pt['selected_alpha']:.4g} is very small (L2 shrinkage ratio={sd['l2_norm_ratio_ridge_over_ols']:.4f}), so Ridge behaves nearly identically to OLS here."
            )
        limitations.append(
            "Ridge regression shrinks all coefficients toward zero smoothly but never produces exact sparsity (all predictors remain active)."
        )
        if rc["spearman_abs_resid_vs_fitted_pvalue"] < 0.05:
            limitations.append(
                f"Residual magnitude correlates with fitted values (Spearman r={rc['spearman_abs_resid_vs_fitted_r']:.3f}, p={rc['spearman_abs_resid_vs_fitted_pvalue']:.2e}), indicating heteroskedasticity or remaining nonlinearity."
            )

    elif model_key == "lasso":
        pt = diagnostics["penalty_tuning"]
        sp = diagnostics["sparsity_and_shrinkage"]
        st = diagnostics["selection_stability"]
        cvg = diagnostics["convergence"]
        rc = diagnostics["residual_checks"]

        sections.append((
            "Cross-Validated Penalty Tuning & Convergence",
            [
                ("CV strategy", pt["cv_strategy"]),
                ("alpha_max / selected alpha (L1)", f"{_fmt(pt['alpha_max'])} / {_fmt(pt['selected_alpha'])}"),
                ("Best CV RMSE (±SD)", f"{_fmt(pt['best_cv_rmse'])} (±{_fmt(pt['best_cv_rmse_std'])}) {units}"),
                ("Best CV R²", _fmt(pt["best_cv_r2"])),
                ("Final coordinate descent iterations / converged", f"{cvg['n_iter_final']} of {cvg['max_iter']} / {_fmt(cvg['converged_final'])}"),
                ("CV convergence warnings", _fmt(cvg["cv_convergence_warnings"])),
            ],
        ))
        sections.append((
            "Sparsity, L1 Shrinkage & Fold Selection Stability",
            [
                ("Retained (nonzero) / Zeroed predictors", f"{sp['n_nonzero_features']} / {sp['n_zero_features']} (total {sp['n_total_features']}, {sp['sparsity_pct']:.1f}% sparse)"),
                ("||beta_OLS||_1 / ||beta_Lasso||_1", f"{_fmt(sp['ols_coef_l1_norm'])} / {_fmt(sp['lasso_coef_l1_norm'])}"),
                ("L1 norm ratio (Lasso / OLS)", _fmt(sp['l1_norm_ratio_lasso_over_ols'])),
                ("Always selected / Never selected / Unstable in CV", f"{st['n_always_selected_in_cv']} / {st['n_never_selected_in_cv']} / {st['n_unstable_across_cv_folds']}"),
                ("Spearman |residual| vs fitted (r / p)", f"{_fmt(rc['spearman_abs_resid_vs_fitted_r'])} (p={_fmt(rc['spearman_abs_resid_vs_fitted_pvalue'])})"),
                ("Durbin-Watson", _fmt(rc["durbin_watson"])),
            ],
        ))

        interpretations.append(
            f"Training-only {pt['cv_strategy']} selected alpha={pt['selected_alpha']:.4g} "
            f"(CV RMSE={pt['best_cv_rmse']:.4g} {units}, CV R²={pt['best_cv_r2']:.4f}), converging in {cvg['n_iter_final']} coordinate-descent iterations."
        )
        interpretations.append(
            f"Lasso retained {sp['n_nonzero_features']} of {sp['n_total_features']} predictors and set {sp['n_zero_features']} exactly to zero ({sp['sparsity_pct']:.1f}% sparsity), "
            f"shrinking the L1 norm from {sp['ols_coef_l1_norm']:.4g} (OLS) to {sp['lasso_coef_l1_norm']:.4g} (ratio={sp['l1_norm_ratio_lasso_over_ols']:.4f})."
        )
        if sp["eliminated_features"]:
            elim_names = ", ".join(e["feature"] for e in sp["eliminated_features"][:5])
            max_elim_corr = max(e["max_abs_corr_with_retained"] for e in sp["eliminated_features"])
            interpretations.append(
                f"Examples of eliminated predictors: {elim_names} (max |corr| with a retained predictor among eliminated features = {max_elim_corr:.3f})."
            )
        if st["n_unstable_across_cv_folds"] > 0:
            unstable_ex = ", ".join(
                f"{u['feature']} ({u['cv_selection_freq']:.2f})" for u in st["unstable_features"][:4]
            )
            limitations.append(
                f"Feature-selection instability across CV folds: {st['n_unstable_across_cv_folds']} feature(s) were selected in some folds but zeroed in others (e.g., {unstable_ex}), a known consequence of L1 regularization under correlated predictors."
            )
        if not cvg["converged_final"] or cvg["cv_convergence_warnings"] > 0:
            limitations.append(
                f"Coordinate descent raised convergence warning(s) (final converged={cvg['converged_final']}, CV warnings={cvg['cv_convergence_warnings']})."
            )

    # Shared structural / dataset limitations
    if cfg.is_binary_classification and "test_classification" in metrics:
        tc = metrics["test_classification"]
        limitations.append(
            f"Linear Probability Model limitation: {tc['n_outside_unit_interval']} of {metrics['n_test']} test predictions ({tc['pct_outside_unit_interval']:.2f}%) fall outside [0, 1] (test prediction range [{tc['pred_min']:.3f}, {tc['pred_max']:.3f}]), which cannot be interpreted literally as probabilities."
        )
    if cfg.is_ordinal_target:
        limitations.append(
            f"Ordinal target '{cfg.target}' is modeled as a continuous metric; predictions are real-valued rather than discrete ordered categories."
        )
    if cfg.split_strategy in ("chronological", "chronological_date"):
        limitations.append(
            "Purely linear static features without explicit lag or seasonal terms may miss temporal dynamics across the chronological train-to-test shift."
        )
    if cfg.split_strategy == "group":
        limitations.append(
            f"Group-aware evaluation tests generalization to completely unseen '{cfg.group_column}' clusters, where inter-subject baseline heterogeneity cannot be absorbed by subject IDs."
        )
    if test_r2 < train_r2 - 0.10:
        limitations.append(
            f"Noticeable train-to-test R² drop ({train_r2:.4f} -> {test_r2:.4f}), indicating distribution shift or limited out-of-sample generalization of linear terms."
        )
    limitations.append(
        "Linear models assume additive linear relationships between standardized predictors and the conditional target mean; nonlinear interactions require basis expansion or nonlinear models."
    )

    return sections, interpretations, limitations


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
    plots = _load_plots(res_dir / "plots")
    diag_sections, interpretations, limitations = _build_diagnostic_sections_and_narrative(
        model_key, dataset_key, metrics, coef_df, diagnostics, metadata
    )

    html = REPORT_TEMPLATE.render(
        model_key=model_key,
        model_title=MODEL_TITLES[model_key],
        dataset_name=cfg.name,
        dataset_description=cfg.description,
        dataset_notes=cfg.notes,
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
            key_diag = f"rank {la['matrix_rank']}/{la['n_augmented_columns']}, cond={_fmt(la['condition_number_standardized_with_intercept'])}"
        elif model_key == "ridge":
            pt = diag["penalty_tuning"]
            sd = diag["shrinkage_and_degrees_of_freedom"]
            key_diag = f"alpha={pt['selected_alpha']:.3g}, eff_df={sd['effective_df_predictors']:.1f}/{sd['n_predictors']}, L2_ratio={sd['l2_norm_ratio_ridge_over_ols']:.3f}"
        else:
            pt = diag["penalty_tuning"]
            sp = diag["sparsity_and_shrinkage"]
            key_diag = f"alpha={pt['selected_alpha']:.3g}, retained={sp['n_nonzero_features']}/{sp['n_total_features']}"

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
