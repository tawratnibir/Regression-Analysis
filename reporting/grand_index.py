"""Generate a self-contained grand dashboard (reports/index.html) with all results,
diagrams, explanations, and an interactive client-side predictor for OLS, Ridge, and Lasso.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, select_autoescape
import numpy as np
import pandas as pd

from config import DATASETS, MODEL_KEYS, REPORTS_DIR, RESULTS_DIR
from preprocessing.preprocess import load_and_prepare_dataset
from reporting.generate_report import (
    MODEL_TITLES,
    _build_diagnostic_sections_and_narrative,
    _load_plots,
)

JINJA_ENV = Environment(autoescape=select_autoescape(["html", "xml"]))


def _build_predictor_payload(dataset_key: str) -> Dict[str, Any]:
    """Build feature schema, sample test rows, and unstandardized model weights for JS."""
    prepared = load_and_prepare_dataset(dataset_key)
    cfg = prepared.config
    df_tr = prepared.df_train_raw
    df_te = prepared.df_test_raw

    # Build UI input schema from raw predictor columns
    fields: List[Dict[str, Any]] = []
    for col in df_tr.columns:
        if col in cfg.categorical_columns:
            uniq = sorted(df_tr[col].dropna().unique().tolist())
            label_map = {"1": "1 (USA)", "2": "2 (Europe)", "3": "3 (Japan)"} if col == "origin" else {}
            options = [{"value": str(u), "label": label_map.get(str(u), str(u))} for u in uniq]
            mode_val = str(df_tr[col].mode().iloc[0])
            fields.append(
                {
                    "name": col,
                    "kind": "categorical",
                    "options": options,
                    "default": mode_val,
                    "encoded_prefix": f"{col}_",
                }
            )
        elif col in cfg.binary_mappings:
            mapping = cfg.binary_mappings[col]
            options = [{"value": str(v), "label": f"{k} ({v})"} for k, v in mapping.items()]
            mode_val = str(int(round(float(df_tr[col].mode().iloc[0]))))
            fields.append(
                {
                    "name": col,
                    "kind": "binary",
                    "options": options,
                    "default": mode_val,
                }
            )
        else:
            s = pd.to_numeric(df_tr[col], errors="coerce").dropna()
            med = float(s.median())
            lo = float(s.min())
            hi = float(s.max())
            fields.append(
                {
                    "name": col,
                    "kind": "numeric",
                    "default": float(med),
                    "min": float(lo),
                    "max": float(hi),
                }
            )

    # Pick 3 representative test samples (low, median, high target) with valid raw values
    te_clean_mask = ~df_te.isna().any(axis=1)
    valid_pos = np.where(te_clean_mask.to_numpy())[0]
    samples: List[Dict[str, Any]] = []
    if len(valid_pos) > 0:
        order = valid_pos[np.argsort(prepared.y_test[valid_pos])]
        pick_indices = [
            int(order[0]),
            int(order[len(order) // 2]),
            int(order[-1]),
        ]
        # Deduplicate preserving order
        seen = set()
        for pos in pick_indices:
            if pos in seen:
                continue
            seen.add(pos)
            row_vals: Dict[str, Any] = {}
            for f in fields:
                v = df_te.iloc[pos][f["name"]]
                if f["kind"] == "categorical":
                    row_vals[f["name"]] = str(v)
                elif f["kind"] == "binary":
                    row_vals[f["name"]] = str(int(round(float(v))))
                else:
                    row_vals[f["name"]] = float(v)
            samples.append(
                {
                    "label": f"Test row #{int(prepared.test_raw_indices[pos])} (observed={prepared.y_test[pos]:.4g} {cfg.target_units})",
                    "observed": float(prepared.y_test[pos]),
                    "values": row_vals,
                }
            )

    # Unstandardized intercept & predictor coefficients per model
    models_weights: Dict[str, Dict[str, Any]] = {}
    for model_key in MODEL_KEYS:
        coef_path = RESULTS_DIR / model_key / dataset_key / "coefficients.csv"
        if not coef_path.exists():
            continue
        cdf = pd.read_csv(coef_path)
        intercept_row = cdf[cdf["role"] == "intercept"].iloc[0]
        pred_rows = cdf[cdf["role"] == "predictor"]
        models_weights[model_key] = {
            "intercept": float(intercept_row["coefficient_unstandardized"]),
            "coefs": {
                str(r["feature"]): float(r["coefficient_unstandardized"])
                for _, r in pred_rows.iterrows()
            },
        }

    return {
        "key": dataset_key,
        "name": cfg.name,
        "target": cfg.target,
        "target_units": cfg.target_units,
        "is_binary_classification": cfg.is_binary_classification,
        "is_ordinal_target": cfg.is_ordinal_target,
        "fields": fields,
        "samples": samples,
        "models": models_weights,
    }


GRAND_TEMPLATE = JINJA_ENV.from_string("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UCI Regression Analysis — Grand Dashboard &amp; Interactive Predictor</title>
  <style>
    :root {
      --bg: #f1f5f9;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --border: #cbd5e1;
      --accent: #1d4ed8;
      --accent-soft: #eff6ff;
      --green: #15803d;
      --green-soft: #f0fdf4;
      --warn-bg: #fffbeb;
      --warn-border: #f59e0b;
    }
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 20px;
      line-height: 1.5;
    }
    .container { max-width: 1320px; margin: 0 auto; }
    .hero {
      background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
      color: #ffffff;
      border-radius: 10px;
      padding: 24px 28px;
      margin-bottom: 20px;
    }
    .hero h1 { margin: 0 0 8px 0; font-size: 1.75rem; }
    .hero p { margin: 0; color: #dbeafe; font-size: 0.96rem; }
    .nav-links { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; }
    .nav-links a {
      background: rgba(255,255,255,0.14);
      color: #ffffff;
      text-decoration: none;
      padding: 5px 12px;
      border-radius: 999px;
      font-size: 0.85rem;
      border: 1px solid rgba(255,255,255,0.25);
    }
    .nav-links a:hover { background: rgba(255,255,255,0.25); }
    section.card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px 24px;
      margin-bottom: 20px;
    }
    h2 { margin-top: 0; font-size: 1.25rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }
    h3 { font-size: 1.05rem; margin-top: 16px; color: #1e293b; }
    table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.88rem; }
    th, td { border: 1px solid #e2e8f0; padding: 7px 9px; text-align: left; }
    th { background: #f8fafc; font-weight: 600; }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    tr.clickable-row { cursor: pointer; }
    tr.clickable-row:hover { background: #eff6ff; }
    td.best-cell { background: var(--green-soft); color: var(--green); font-weight: 700; }
    .warning {
      background: var(--warn-bg);
      border-left: 4px solid var(--warn-border);
      padding: 10px 14px;
      border-radius: 4px;
      margin: 10px 0;
      font-size: 0.9rem;
    }
    .callout {
      background: var(--accent-soft);
      border-left: 4px solid var(--accent);
      padding: 10px 14px;
      border-radius: 4px;
      margin: 10px 0;
      font-size: 0.9rem;
    }
    .ds-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 14px;
    }
    .ds-tab {
      border: 1px solid var(--border);
      background: #f8fafc;
      color: var(--text);
      padding: 7px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.86rem;
      font-weight: 600;
    }
    .ds-tab.active {
      background: var(--accent);
      color: #ffffff;
      border-color: var(--accent);
    }
    .model-tabs {
      display: flex;
      gap: 8px;
      margin: 14px 0;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 8px;
    }
    .model-tab {
      border: 1px solid var(--border);
      background: #f8fafc;
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.88rem;
    }
    .model-tab.active {
      background: #0f172a;
      color: #ffffff;
      border-color: #0f172a;
    }
    .ds-panel { display: none; }
    .ds-panel.active { display: block; }
    .model-panel { display: none; }
    .model-panel.active { display: block; }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 16px;
    }
    .plot-card {
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 10px;
      background: #ffffff;
    }
    .plot-card img { width: 100%; height: auto; display: block; border-radius: 4px; }
    .plot-title { font-weight: 600; margin-top: 6px; font-size: 0.92rem; }
    .plot-caption { color: var(--muted); font-size: 0.82rem; margin-top: 2px; }
    .scroll-table { max-height: 360px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 6px; }
    tr.intercept-row { background: #eff6ff; font-weight: 600; }
    tr.zero-row { color: #94a3b8; }

    /* Interactive Predictor Styles */
    .predictor-box {
      border: 2px solid #bfdbfe;
      background: #f8fafc;
      border-radius: 8px;
      padding: 16px;
      margin: 14px 0;
    }
    .pred-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
    }
    .btn {
      background: var(--accent);
      color: #ffffff;
      border: none;
      border-radius: 6px;
      padding: 8px 14px;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
    }
    .btn:hover { background: #1e40af; }
    .btn-secondary {
      background: #ffffff;
      color: var(--text);
      border: 1px solid var(--border);
    }
    .btn-secondary:hover { background: #f1f5f9; }
    select.ctrl-select, input.ctrl-input {
      padding: 7px 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 0.86rem;
      background: #ffffff;
    }
    .input-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 10px;
      max-height: 340px;
      overflow-y: auto;
      padding: 6px 2px;
    }
    .field-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px 10px;
    }
    .field-card label {
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      margin-bottom: 4px;
      word-break: break-word;
    }
    .field-card input, .field-card select {
      width: 100%;
      padding: 5px 7px;
      border: 1px solid #cbd5e1;
      border-radius: 4px;
      font-size: 0.86rem;
    }
    .field-hint { color: var(--muted); font-size: 0.72rem; margin-top: 3px; }
    .pred-results-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .pred-card {
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-top: 4px solid var(--accent);
      border-radius: 6px;
      padding: 12px;
    }
    .pred-card.ridge { border-top-color: #0369a1; }
    .pred-card.lasso { border-top-color: #15803d; }
    .pred-val { font-size: 1.35rem; font-weight: 700; margin: 4px 0; font-variant-numeric: tabular-nums; }
    .pred-meta { font-size: 0.8rem; color: var(--muted); }
    code { background: #f1f5f9; padding: 1px 4px; border-radius: 4px; font-size: 0.86em; }
  </style>
</head>
<body>
<div class="container">
  <div class="hero">
    <h1>UCI Regression Analysis — Grand Results &amp; Interactive Predictor</h1>
    <p>
      Unified comparison of <strong>Ordinary Least Squares (OLS)</strong>, <strong>Ridge (L2)</strong>, and <strong>Lasso (L1)</strong>
      across <strong>10 UCI datasets</strong> (30 experiments). Self-contained offline HTML with all metrics, diagrams, explanations, and live prediction.
    </p>
    <div class="nav-links">
      <a href="#master-overview">Master Comparison</a>
      <a href="#interactive-predictor">Interactive Predictor</a>
      <a href="#dataset-explorer">Dataset Explorer &amp; Diagrams</a>
      <a href="ols/summary.html">OLS Summary</a>
      <a href="ridge/summary.html">Ridge Summary</a>
      <a href="lasso/summary.html">Lasso Summary</a>
    </div>
  </div>

  <section class="card" id="master-overview">
    <h2>Master Cross-Dataset &amp; Cross-Model Overview</h2>
    <div class="warning">
      <strong>Unit Comparability Note:</strong> Raw RMSE is expressed in each dataset's native target units. Compare OLS, Ridge, and Lasso <em>within</em> the same dataset row, and use dimensionless <strong>Test R&sup2;</strong> across datasets. Click any row to jump directly to that dataset's diagrams, explanations, and predictor.
    </div>
    <table>
      <thead>
        <tr>
          <th>Dataset (UCI ID)</th>
          <th>Target (Units)</th>
          <th>Split / CV</th>
          <th class="num">Train / Test</th>
          <th class="num">OLS Test R&sup2; (RMSE)</th>
          <th class="num">Ridge Test R&sup2; (RMSE, &alpha;)</th>
          <th class="num">Lasso Test R&sup2; (RMSE, &alpha;, nz/p)</th>
        </tr>
      </thead>
      <tbody>
        {% for row in master_rows %}
        <tr class="clickable-row" onclick="selectDataset('{{ row.key }}'); document.getElementById('dataset-explorer').scrollIntoView({behavior:'smooth'});">
          <td><strong>{{ row.name }}</strong><br><small><code>{{ row.key }}</code> ({{ row.uci_id }})</small></td>
          <td><code>{{ row.target }}</code><br><small>{{ row.target_units }}</small></td>
          <td><code>{{ row.split_strategy }}</code><br><small>{{ row.cv_strategy }}</small></td>
          <td class="num">{{ row.n_train }} / {{ row.n_test }}</td>
          <td class="num {% if row.best_model == 'ols' %}best-cell{% endif %}">{{ "%.4f"|format(row.ols_r2) }}<br><small>{{ "%.4g"|format(row.ols_rmse) }} {{ row.target_units }}</small></td>
          <td class="num {% if row.best_model == 'ridge' %}best-cell{% endif %}">{{ "%.4f"|format(row.ridge_r2) }}<br><small>{{ "%.4g"|format(row.ridge_rmse) }} (&alpha;={{ "%.3g"|format(row.ridge_alpha) }})</small></td>
          <td class="num {% if row.best_model == 'lasso' %}best-cell{% endif %}">{{ "%.4f"|format(row.lasso_r2) }}<br><small>{{ "%.4f"|format(row.lasso_rmse) if row.lasso_rmse < 1 else "%.4g"|format(row.lasso_rmse) }} (&alpha;={{ "%.3g"|format(row.lasso_alpha) }}, {{ row.lasso_nz }}/{{ row.n_features }})</small></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </section>

  <section class="card" id="interactive-predictor">
    <h2>Interactive Prediction Playground (Input Prompt &rarr; OLS / Ridge / Lasso Output)</h2>
    <p>
      Choose a dataset, enter feature values (or load a held-out test sample or prompt sequential inputs), and compute real-time predictions from the fitted unstandardized equations
      <code>y_hat = intercept + sum(coef_j * x_j)</code> across OLS, Ridge, and Lasso.
    </p>
    <div class="predictor-box">
      <div class="pred-controls">
        <label for="pred-dataset-select"><strong>Dataset:</strong></label>
        <select id="pred-dataset-select" class="ctrl-select" onchange="initPredictor(this.value)">
          {% for row in master_rows %}
          <option value="{{ row.key }}">{{ row.name }} ({{ row.target }}, {{ row.target_units }})</option>
          {% endfor %}
        </select>

        <label for="pred-sample-select"><strong>Held-out Test Sample:</strong></label>
        <select id="pred-sample-select" class="ctrl-select" onchange="loadSelectedSample()"></select>

        <button type="button" class="btn" onclick="computePredictions()">Predict Output</button>
        <button type="button" class="btn btn-secondary" onclick="promptInputsSequential()">Prompt Inputs (Dialog)</button>
        <button type="button" class="btn btn-secondary" onclick="resetPredictorDefaults()">Reset to Train Medians</button>
        <input type="text" id="pred-filter" class="ctrl-input" placeholder="Filter features..." oninput="filterPredictorFields(this.value)">
      </div>

      <div id="pred-sample-banner" class="callout" style="display:none;"></div>
      <div id="pred-inputs-grid" class="input-grid"></div>

      <div class="pred-results-grid" id="pred-results-grid"></div>

      <details style="margin-top: 12px;">
        <summary style="cursor:pointer; font-weight:600;">Show Top Feature Contributions (OLS vs Ridge vs Lasso)</summary>
        <div id="pred-contrib-table" class="scroll-table" style="margin-top:8px;"></div>
      </details>
    </div>
  </section>

  <section class="card" id="dataset-explorer">
    <h2>Dataset Explorer: Metrics, Coefficients, Diagrams &amp; Explanations</h2>
    <div class="ds-tabs" id="ds-tabs">
      {% for ds in datasets_detail %}
      <button type="button" class="ds-tab {% if loop.first %}active{% endif %}" data-ds="{{ ds.key }}" onclick="selectDataset('{{ ds.key }}')">
        {{ ds.name }}
      </button>
      {% endfor %}
    </div>

    {% for ds in datasets_detail %}
    <div class="ds-panel {% if loop.first %}active{% endif %}" id="panel-{{ ds.key }}">
      <h3>{{ ds.name }} (<code>{{ ds.key }}</code>, UCI ID {{ ds.uci_id }})</h3>
      <p>{{ ds.description }}</p>
      <div class="callout">
        <strong>Target:</strong> <code>{{ ds.target }}</code> ({{ ds.target_units }}) &middot;
        <strong>Rows:</strong> {{ ds.prep.n_raw }} raw &rarr; {{ ds.prep.n_clean }} clean ({{ ds.prep.n_train }} train / {{ ds.prep.n_test }} test) &middot;
        <strong>Split:</strong> <code>{{ ds.prep.split_strategy }}</code> &middot;
        <strong>CV:</strong> <code>{{ ds.prep.cv_strategy }}</code>
      </div>
      {% if ds.notes %}
      <ul>
        {% for n in ds.notes %}<li>{{ n }}</li>{% endfor %}
      </ul>
      {% endif %}

      <h3>Model Comparison on {{ ds.name }}</h3>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th class="num">Train MAE</th>
            <th class="num">Train RMSE</th>
            <th class="num">Train R&sup2;</th>
            <th class="num">Test MAE</th>
            <th class="num">Test RMSE ({{ ds.target_units }})</th>
            <th class="num">Test R&sup2;</th>
            <th>Standalone Report</th>
          </tr>
        </thead>
        <tbody>
          {% for mkey in ['ols', 'ridge', 'lasso'] %}
          {% set m = ds.models[mkey] %}
          <tr>
            <td><strong>{{ mkey | upper }}</strong></td>
            <td class="num">{{ "%.4g"|format(m.metrics.train.mae) }}</td>
            <td class="num">{{ "%.4g"|format(m.metrics.train.rmse) }}</td>
            <td class="num">{{ "%.4f"|format(m.metrics.train.r2) }}</td>
            <td class="num">{{ "%.4g"|format(m.metrics.test.mae) }}</td>
            <td class="num">{{ "%.4g"|format(m.metrics.test.rmse) }}</td>
            <td class="num">{{ "%.4f"|format(m.metrics.test.r2) }}</td>
            <td><a href="{{ mkey }}/{{ ds.key }}.html" target="_blank">Open {{ mkey|upper }} HTML</a></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>

      {% if ds.is_binary_classification %}
      <div class="warning">
        <strong>Diabetes LPM Classification &amp; Bounds Comparison (Test Set, threshold 0.50):</strong>
        {% for mkey in ['ols', 'ridge', 'lasso'] %}
        {% set tc = ds.models[mkey].metrics.test_classification %}
        <br><strong>{{ mkey | upper }}:</strong> Accuracy={{ "%.4f"|format(tc.accuracy) }}, F1={{ "%.4f"|format(tc.f1) }}, ROC AUC={{ "%.4f"|format(tc.roc_auc) }}, Outside [0,1]={{ tc.n_outside_unit_interval }} ({{ "%.2f"|format(tc.pct_outside_unit_interval) }}%), Range=[{{ "%.3f"|format(tc.pred_min) }}, {{ "%.3f"|format(tc.pred_max) }}]
        {% endfor %}
      </div>
      {% endif %}

      <div class="model-tabs">
        {% for mkey in ['ols', 'ridge', 'lasso'] %}
        <button type="button" class="model-tab {% if loop.first %}active{% endif %}" data-ds="{{ ds.key }}" data-model="{{ mkey }}" onclick="selectModel('{{ ds.key }}', '{{ mkey }}')">
          {{ mkey | upper }} Diagnostics, Coefficients &amp; Plots
        </button>
        {% endfor %}
      </div>

      {% for mkey in ['ols', 'ridge', 'lasso'] %}
      {% set m = ds.models[mkey] %}
      <div class="model-panel {% if loop.first %}active{% endif %}" id="panel-{{ ds.key }}-{{ mkey }}">
        <div class="callout">
          <strong>Objective:</strong> <code>{{ m.metadata.objective_convention }}</code>
          {% if m.metadata.penalty_settings.selected_alpha is defined %}
          <br><strong>CV-Selected &alpha;:</strong> <code>{{ "%.6g"|format(m.metadata.penalty_settings.selected_alpha) }}</code>
          {% endif %}
        </div>

        <div class="grid-2">
          <div>
            <h4>Interpretation (Grounded in Saved Artifacts)</h4>
            <ul>
              {% for item in m.interpretations %}<li>{{ item }}</li>{% endfor %}
            </ul>
          </div>
          <div>
            <h4>Limitations &amp; Assumption Checks</h4>
            <ul>
              {% for lim in m.limitations %}<li>{{ lim }}</li>{% endfor %}
            </ul>
          </div>
        </div>

        <h4>Diagnostic Checks</h4>
        <div class="grid-2">
          {% for gname, items in m.diag_sections %}
          <div>
            <strong>{{ gname }}</strong>
            <table>
              {% for k, v in items %}
              <tr><td>{{ k }}</td><td class="num">{{ v }}</td></tr>
              {% endfor %}
            </table>
          </div>
          {% endfor %}
        </div>

        <h4>Fitted Coefficients ({{ m.coef_rows | length }} entries incl. intercept)</h4>
        <div class="scroll-table">
          <table>
            <thead>
              <tr>
                <th>Role</th>
                <th>Feature</th>
                <th class="num">Coef (Std)</th>
                <th class="num">Coef (Unstd)</th>
                {% if m.has_ols_ref %}<th class="num">OLS Coef (Std)</th>{% endif %}
                {% if m.has_shrinkage_ratio %}<th class="num">Shrinkage</th>{% endif %}
                {% if m.has_retained %}<th class="num">Retained</th><th class="num">CV Freq</th>{% endif %}
              </tr>
            </thead>
            <tbody>
              {% for r in m.coef_rows %}
              <tr class="{% if r.role == 'intercept' %}intercept-row{% elif r.retained is defined and not r.retained %}zero-row{% endif %}">
                <td>{{ r.role }}</td>
                <td><code>{{ r.feature }}</code></td>
                <td class="num">{{ "%.5g"|format(r.coefficient_standardized) }}</td>
                <td class="num">{{ "%.5g"|format(r.coefficient_unstandardized) }}</td>
                {% if m.has_ols_ref %}<td class="num">{{ "%.5g"|format(r.ols_coefficient_standardized) }}</td>{% endif %}
                {% if m.has_shrinkage_ratio %}<td class="num">{% if r.shrinkage_ratio_vs_ols == r.shrinkage_ratio_vs_ols %}{{ "%.4f"|format(r.shrinkage_ratio_vs_ols) }}{% else %}—{% endif %}</td>{% endif %}
                {% if m.has_retained %}<td class="num">{{ "Yes" if r.retained else "No (0)" }}</td><td class="num">{{ "%.2f"|format(r.cv_selection_freq) }}</td>{% endif %}
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>

        <h4>Diagrams ({{ mkey | upper }} &mdash; {{ ds.name }})</h4>
        <div class="grid-2">
          {% for p in m.plots %}
          <div class="plot-card">
            <img loading="lazy" src="data:image/png;base64,{{ p.b64 }}" alt="{{ p.title }}">
            <div class="plot-title">{{ p.title }}</div>
            <div class="plot-caption">{{ p.caption }}</div>
          </div>
          {% endfor %}
        </div>
      </div>
      {% endfor %}
    </div>
    {% endfor %}
  </section>
</div>

<script>
const PREDICTOR_DATA = {{ predictor_json | safe }};
let currentDatasetKey = "{{ master_rows[0].key }}";

function selectDataset(dsKey) {
  document.querySelectorAll('.ds-tab').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-ds') === dsKey);
  });
  document.querySelectorAll('.ds-panel').forEach(p => {
    p.classList.toggle('active', p.id === 'panel-' + dsKey);
  });
  const sel = document.getElementById('pred-dataset-select');
  if (sel && sel.value !== dsKey) {
    sel.value = dsKey;
    initPredictor(dsKey);
  }
}

function selectModel(dsKey, modelKey) {
  const panel = document.getElementById('panel-' + dsKey);
  panel.querySelectorAll('.model-tab').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-model') === modelKey);
  });
  panel.querySelectorAll('.model-panel').forEach(p => {
    p.classList.toggle('active', p.id === 'panel-' + dsKey + '-' + modelKey);
  });
}

function initPredictor(dsKey) {
  currentDatasetKey = dsKey;
  const ds = PREDICTOR_DATA[dsKey];
  const sampleSel = document.getElementById('pred-sample-select');
  sampleSel.innerHTML = '<option value="">-- Custom / Train Median --</option>';
  ds.samples.forEach((s, idx) => {
    const opt = document.createElement('option');
    opt.value = String(idx);
    opt.textContent = s.label;
    sampleSel.appendChild(opt);
  });

  const grid = document.getElementById('pred-inputs-grid');
  grid.innerHTML = '';
  document.getElementById('pred-filter').value = '';
  document.getElementById('pred-sample-banner').style.display = 'none';

  ds.fields.forEach(f => {
    const card = document.createElement('div');
    card.className = 'field-card';
    card.setAttribute('data-fname', f.name.toLowerCase());

    const lbl = document.createElement('label');
    lbl.textContent = f.name;
    lbl.setAttribute('for', 'inp-' + f.name);
    card.appendChild(lbl);

    if (f.kind === 'categorical' || f.kind === 'binary') {
      const sel = document.createElement('select');
      sel.id = 'inp-' + f.name;
      f.options.forEach(o => {
        const opt = document.createElement('option');
        opt.value = o.value;
        opt.textContent = o.label;
        if (o.value === String(f.default)) opt.selected = true;
        sel.appendChild(opt);
      });
      sel.addEventListener('change', computePredictions);
      card.appendChild(sel);
      const hint = document.createElement('div');
      hint.className = 'field-hint';
      hint.textContent = 'Default mode: ' + f.default;
      card.appendChild(hint);
    } else {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.step = 'any';
      inp.id = 'inp-' + f.name;
      inp.value = f.default;
      inp.addEventListener('input', computePredictions);
      card.appendChild(inp);
      const hint = document.createElement('div');
      hint.className = 'field-hint';
      hint.textContent = `Train median: ${f.default} | range [${f.min}, ${f.max}]`;
      card.appendChild(hint);
    }
    grid.appendChild(card);
  });

  computePredictions();
}

function filterPredictorFields(query) {
  const q = query.trim().toLowerCase();
  document.querySelectorAll('#pred-inputs-grid .field-card').forEach(card => {
    const fname = card.getAttribute('data-fname') || '';
    card.style.display = (!q || fname.includes(q)) ? '' : 'none';
  });
}

function resetPredictorDefaults() {
  const ds = PREDICTOR_DATA[currentDatasetKey];
  document.getElementById('pred-sample-select').value = '';
  document.getElementById('pred-sample-banner').style.display = 'none';
  ds.fields.forEach(f => {
    const el = document.getElementById('inp-' + f.name);
    if (el) el.value = String(f.default);
  });
  computePredictions();
}

function loadSelectedSample() {
  const idxStr = document.getElementById('pred-sample-select').value;
  const banner = document.getElementById('pred-sample-banner');
  if (idxStr === '') {
    resetPredictorDefaults();
    return;
  }
  const ds = PREDICTOR_DATA[currentDatasetKey];
  const sample = ds.samples[Number(idxStr)];
  ds.fields.forEach(f => {
    const el = document.getElementById('inp-' + f.name);
    if (el && sample.values[f.name] !== undefined) {
      el.value = String(sample.values[f.name]);
    }
  });
  banner.style.display = 'block';
  banner.innerHTML = `<strong>Loaded Held-Out Sample:</strong> ${sample.label} &mdash; Compare model predictions below against actual observed target <strong>${sample.observed}</strong> ${ds.target_units}.`;
  computePredictions();
}

function promptInputsSequential() {
  const ds = PREDICTOR_DATA[currentDatasetKey];
  if (ds.fields.length > 25) {
    const proceed = window.confirm(
      `${ds.name} has ${ds.fields.length} features. Do you want to step through prompts (Cancel leaves current form inputs)?`
    );
    if (!proceed) return;
  }
  for (const f of ds.fields) {
    const el = document.getElementById('inp-' + f.name);
    const current = el ? el.value : String(f.default);
    const hint = f.kind === 'numeric' ? `[range ${f.min} .. ${f.max}]` : f.options.map(o => o.value + '=' + o.label).join(', ');
    const ans = window.prompt(`Enter ${f.name} (${hint}):`, current);
    if (ans === null) break;
    if (el && ans.trim() !== '') el.value = ans.trim();
  }
  computePredictions();
}

function getEncodedFeatureMap() {
  const ds = PREDICTOR_DATA[currentDatasetKey];
  const xMap = {};
  ds.fields.forEach(f => {
    const el = document.getElementById('inp-' + f.name);
    const rawVal = el ? el.value : String(f.default);
    if (f.kind === 'categorical') {
      f.options.forEach(o => {
        const encName = f.encoded_prefix + o.value;
        xMap[encName] = (rawVal === o.value) ? 1.0 : 0.0;
      });
    } else if (f.kind === 'binary') {
      const num = parseFloat(rawVal);
      xMap[f.name] = Number.isFinite(num) ? num : parseFloat(f.default);
    } else {
      const num = parseFloat(rawVal);
      xMap[f.name] = Number.isFinite(num) ? num : parseFloat(f.default);
    }
  });
  return xMap;
}

function computePredictions() {
  const ds = PREDICTOR_DATA[currentDatasetKey];
  const xMap = getEncodedFeatureMap();
  const resGrid = document.getElementById('pred-results-grid');
  resGrid.innerHTML = '';

  const modelOrder = ['ols', 'ridge', 'lasso'];
  const contribRows = {};

  modelOrder.forEach(mkey => {
    const mw = ds.models[mkey];
    if (!mw) return;
    let yhat = mw.intercept;
    Object.entries(mw.coefs).forEach(([feat, w]) => {
      const xv = xMap[feat] !== undefined ? xMap[feat] : 0.0;
      const contrib = w * xv;
      yhat += contrib;
      if (!contribRows[feat]) contribRows[feat] = { feat, x: xv };
      contribRows[feat][mkey] = contrib;
    });

    const card = document.createElement('div');
    card.className = 'pred-card ' + mkey;
    let extraHtml = '';
    if (ds.is_binary_classification) {
      const cls = yhat >= 0.5 ? 'Positive (1)' : 'Negative (0)';
      const oob = (yhat < 0.0 || yhat > 1.0) ? ' &middot; <span style="color:#b45309;">Outside [0,1]</span>' : ' &middot; Within [0,1]';
      extraHtml = `<div class="pred-meta">Class (@ 0.50): <strong>${cls}</strong>${oob}</div>`;
    } else if (ds.is_ordinal_target) {
      extraHtml = `<div class="pred-meta">Rounded score: <strong>${Math.round(yhat)}</strong></div>`;
    }
    card.innerHTML = `
      <div style="font-weight:700; text-transform:uppercase; font-size:0.84rem;">${mkey} Prediction</div>
      <div class="pred-val">${yhat.toFixed(4)} <small style="font-size:0.75rem; font-weight:500;">${ds.target_units}</small></div>
      <div class="pred-meta">Target: <code>${ds.target}</code> (unstd intercept=${mw.intercept.toFixed(4)})</div>
      ${extraHtml}
    `;
    resGrid.appendChild(card);
  });

  // Render contribution breakdown sorted by |OLS contribution|
  const rows = Object.values(contribRows).sort((a, b) => Math.abs(b.ols || 0) - Math.abs(a.ols || 0));
  let tbl = '<table><thead><tr><th>Encoded Feature</th><th class="num">Input Value</th><th class="num">OLS Term (w&middot;x)</th><th class="num">Ridge Term</th><th class="num">Lasso Term</th></tr></thead><tbody>';
  rows.forEach(r => {
    tbl += `<tr><td><code>${r.feat}</code></td><td class="num">${Number(r.x).toPrecision(5)}</td><td class="num">${(r.ols || 0).toFixed(4)}</td><td class="num">${(r.ridge || 0).toFixed(4)}</td><td class="num">${(r.lasso || 0).toFixed(4)}</td></tr>`;
  });
  tbl += '</tbody></table>';
  document.getElementById('pred-contrib-table').innerHTML = tbl;
}

initPredictor(currentDatasetKey);
</script>
</body>
</html>
""")


def generate_grand_index() -> Path:
    """Build reports/index.html aggregating all 10 datasets × 3 models + live predictor."""
    master_rows: List[Dict[str, Any]] = []
    datasets_detail: List[Dict[str, Any]] = []
    predictor_payload: Dict[str, Any] = {}

    for dataset_key, cfg in DATASETS.items():
        model_bundles: Dict[str, Any] = {}
        for model_key in MODEL_KEYS:
            res_dir = RESULTS_DIR / model_key / dataset_key
            if not (res_dir / "metrics.json").exists():
                continue
            with open(res_dir / "metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)
            with open(res_dir / "diagnostics.json", "r", encoding="utf-8") as f:
                diagnostics = json.load(f)
            with open(res_dir / "run_metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)
            coef_df = pd.read_csv(res_dir / "coefficients.csv")

            diag_sections, interpretations, limitations = _build_diagnostic_sections_and_narrative(
                model_key, dataset_key, metrics, coef_df, diagnostics, metadata
            )
            plots = _load_plots(res_dir / "plots")

            model_bundles[model_key] = {
                "metrics": metrics,
                "diagnostics": diagnostics,
                "metadata": metadata,
                "coef_rows": coef_df.to_dict(orient="records"),
                "has_ols_ref": "ols_coefficient_standardized" in coef_df.columns,
                "has_shrinkage_ratio": "shrinkage_ratio_vs_ols" in coef_df.columns,
                "has_retained": "retained" in coef_df.columns,
                "diag_sections": diag_sections,
                "interpretations": interpretations,
                "limitations": limitations,
                "plots": plots,
            }

        if len(model_bundles) == 3:
            ols_m = model_bundles["ols"]["metrics"]
            ridge_m = model_bundles["ridge"]["metrics"]
            lasso_m = model_bundles["lasso"]["metrics"]
            ridge_d = model_bundles["ridge"]["diagnostics"]
            lasso_d = model_bundles["lasso"]["diagnostics"]
            prep = model_bundles["ols"]["metadata"]["preprocessing"]

            r2_map = {
                "ols": ols_m["test"]["r2"],
                "ridge": ridge_m["test"]["r2"],
                "lasso": lasso_m["test"]["r2"],
            }
            best_model = max(r2_map, key=lambda k: r2_map[k])

            master_rows.append(
                {
                    "key": dataset_key,
                    "name": cfg.name,
                    "uci_id": cfg.uci_id,
                    "target": cfg.target,
                    "target_units": cfg.target_units,
                    "split_strategy": prep["split_strategy"],
                    "cv_strategy": prep["cv_strategy"],
                    "n_train": ols_m["n_train"],
                    "n_test": ols_m["n_test"],
                    "n_features": prep["n_encoded_features"],
                    "ols_r2": ols_m["test"]["r2"],
                    "ols_rmse": ols_m["test"]["rmse"],
                    "ridge_r2": ridge_m["test"]["r2"],
                    "ridge_rmse": ridge_m["test"]["rmse"],
                    "ridge_alpha": ridge_d["penalty_tuning"]["selected_alpha"],
                    "lasso_r2": lasso_m["test"]["r2"],
                    "lasso_rmse": lasso_m["test"]["rmse"],
                    "lasso_alpha": lasso_d["penalty_tuning"]["selected_alpha"],
                    "lasso_nz": lasso_d["sparsity_and_shrinkage"]["n_nonzero_features"],
                    "best_model": best_model,
                }
            )
            datasets_detail.append(
                {
                    "key": dataset_key,
                    "name": cfg.name,
                    "uci_id": cfg.uci_id,
                    "description": cfg.description,
                    "target": cfg.target,
                    "target_units": cfg.target_units,
                    "is_binary_classification": cfg.is_binary_classification,
                    "notes": cfg.notes,
                    "prep": prep,
                    "models": model_bundles,
                }
            )
            predictor_payload[dataset_key] = _build_predictor_payload(dataset_key)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "index.html"
    html = GRAND_TEMPLATE.render(
        master_rows=master_rows,
        datasets_detail=datasets_detail,
        predictor_json=json.dumps(predictor_payload),
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path
