"""Generate a clean, executive-grade grand dashboard (reports/index.html)
with minimal text, visual KPIs, self-explanatory badges, diagrams, and live predictor.
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
    _build_diagnostic_sections_and_narrative,
    _fmt,
    _load_plots,
)

JINJA_ENV = Environment(autoescape=select_autoescape(["html", "xml"]))


def _build_predictor_payload(dataset_key: str) -> Dict[str, Any]:
    """Build feature schema, sample test rows, and unstandardized model weights for JS."""
    prepared = load_and_prepare_dataset(dataset_key)
    cfg = prepared.config
    df_tr = prepared.df_train_raw
    df_te = prepared.df_test_raw

    fields: List[Dict[str, Any]] = []
    for col in df_tr.columns:
        if col in cfg.categorical_columns:
            uniq = sorted(df_tr[col].dropna().unique().tolist())
            label_map = {"1": "USA (1)", "2": "Europe (2)", "3": "Japan (3)"} if col == "origin" else {}
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
            options = [{"value": str(v), "label": k} for k, v in mapping.items()]
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
            fields.append(
                {
                    "name": col,
                    "kind": "numeric",
                    "default": float(s.median()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                }
            )

    te_clean_mask = ~df_te.isna().any(axis=1)
    valid_pos = np.where(te_clean_mask.to_numpy())[0]
    samples: List[Dict[str, Any]] = []
    if len(valid_pos) > 0:
        order = valid_pos[np.argsort(prepared.y_test[valid_pos])]
        pick_indices = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
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
                    "label": f"Test row #{int(prepared.test_raw_indices[pos])} · obs {prepared.y_test[pos]:.4g} {cfg.target_units}",
                    "observed": float(prepared.y_test[pos]),
                    "values": row_vals,
                }
            )

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


def _build_visual_kpis_and_badges(
    model_key: str,
    metrics: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Create compact self-explanatory KPI cards and status pills for a model × dataset."""
    kpis: List[Dict[str, str]] = [
        {"label": "Test R²", "value": f"{metrics['test']['r2']:.4f}"},
        {"label": f"Test RMSE ({metrics['target_units']})", "value": _fmt(metrics["test"]["rmse"])},
    ]
    badges: List[Dict[str, str]] = []

    if model_key == "ols":
        la = diagnostics["numerical_linear_algebra"]
        ex = diagnostics["exact_algebraic_properties"]
        het = diagnostics["heteroskedasticity_checks"]
        norm = diagnostics["residual_normality_checks"]
        ac = diagnostics["autocorrelation_checks"]
        mc = diagnostics["multicollinearity"]

        kpis.append({"label": "Matrix Rank", "value": f"{la['matrix_rank']} / {la['n_augmented_columns']}"})
        kpis.append({"label": "Cond # (Std)", "value": _fmt(la["condition_number_standardized_with_intercept"])})
        kpis.append({"label": "Max |Xᵀr|", "value": f"{ex['max_abs_Xt_residuals']:.1e}"})

        badges.append({
            "kind": "warn" if la["is_rank_deficient"] else "ok",
            "text": "Rank Deficient" if la["is_rank_deficient"] else "Full Rank",
        })
        badges.append({
            "kind": "warn" if het["breusch_pagan_pvalue"] < 0.05 else "ok",
            "text": "Heteroskedastic (BP p<0.05)" if het["breusch_pagan_pvalue"] < 0.05 else "Homoskedastic",
        })
        badges.append({
            "kind": "warn" if norm["jarque_bera_pvalue"] < 0.05 else "ok",
            "text": "Non-Normal Residuals" if norm["jarque_bera_pvalue"] < 0.05 else "Normal Residuals",
        })
        if mc["n_vif_gt_10"] > 0 or mc.get("has_infinite_vif"):
            badges.append({"kind": "warn", "text": f"Multicollinear ({mc['n_vif_gt_10']} VIF>10)"})
        if ac["durbin_watson"] < 1.5 or ac["durbin_watson"] > 2.5:
            badges.append({"kind": "warn", "text": f"Autocorrelated (DW={ac['durbin_watson']:.2f})"})

    elif model_key == "ridge":
        pt = diagnostics["penalty_tuning"]
        sd = diagnostics["shrinkage_and_degrees_of_freedom"]
        ns = diagnostics["numerical_stability"]

        kpis.append({"label": "Selected α (L2)", "value": _fmt(pt["selected_alpha"])})
        kpis.append({"label": "Effective df", "value": f"{sd['effective_df_predictors']:.1f} / {sd['n_predictors']}"})
        kpis.append({"label": "L2 Ratio vs OLS", "value": f"{sd['l2_norm_ratio_ridge_over_ols']:.3f}"})

        badges.append({"kind": "info", "text": f"CV R² {pt['best_cv_r2']:.3f}"})
        badges.append({"kind": "ok", "text": f"Cond { _fmt(ns['x_centered_eig_condition_ols']) } → { _fmt(ns['x_centered_eig_condition_ridge']) }"})
        badges.append({"kind": "ok", "text": f"SVD Δ {pt['svd_verification_max_abs_coef_diff']:.1e}"})

    elif model_key == "lasso":
        pt = diagnostics["penalty_tuning"]
        sp = diagnostics["sparsity_and_shrinkage"]
        st = diagnostics["selection_stability"]
        cvg = diagnostics["convergence"]

        kpis.append({"label": "Selected α (L1)", "value": _fmt(pt["selected_alpha"])})
        kpis.append({"label": "Active Features", "value": f"{sp['n_nonzero_features']} / {sp['n_total_features']}"})
        kpis.append({"label": "L1 Ratio vs OLS", "value": f"{sp['l1_norm_ratio_lasso_over_ols']:.3f}"})

        badges.append({"kind": "info", "text": f"{sp['sparsity_pct']:.0f}% Sparse"})
        badges.append({
            "kind": "ok" if cvg["converged_final"] else "warn",
            "text": f"Converged ({cvg['n_iter_final']} iters)" if cvg["converged_final"] else "Max Iter Reached",
        })
        if st["n_unstable_across_cv_folds"] > 0:
            badges.append({"kind": "warn", "text": f"{st['n_unstable_across_cv_folds']} Unstable in CV"})
        else:
            badges.append({"kind": "ok", "text": "Stable CV Selection"})

    return kpis, badges


GRAND_TEMPLATE = JINJA_ENV.from_string("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UCI Regression Benchmark — OLS · Ridge · Lasso</title>
  <style>
    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-2: #f1f5f9;
      --text: #0f172a;
      --muted: #64748b;
      --border: #e2e8f0;
      --border-strong: #cbd5e1;
      --accent: #2563eb;
      --accent-dark: #1d4ed8;
      --accent-soft: #eff6ff;
      --green: #15803d;
      --green-soft: #dcfce7;
      --amber: #b45309;
      --amber-soft: #fef3c7;
      --red: #b91c1c;
      --shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      line-height: 1.45;
      -webkit-font-smoothing: antialiased;
    }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 50;
      background: rgba(15, 23, 42, 0.96);
      backdrop-filter: blur(8px);
      color: #ffffff;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .topbar-inner {
      max-width: 1360px;
      margin: 0 auto;
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-title { font-size: 1.05rem; font-weight: 700; letter-spacing: -0.01em; }
    .brand-sub { font-size: 0.78rem; color: #94a3b8; }
    .nav {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }
    .nav a {
      color: #cbd5e1;
      text-decoration: none;
      font-size: 0.82rem;
      font-weight: 500;
      padding: 6px 11px;
      border-radius: 6px;
      transition: background 0.15s, color 0.15s;
    }
    .nav a:hover { background: rgba(255, 255, 255, 0.1); color: #ffffff; }
    .container { max-width: 1360px; margin: 0 auto; padding: 22px 24px 48px; }

    /* Global KPI Strip */
    .kpi-strip {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }
    .kpi-tile {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 18px;
      box-shadow: var(--shadow);
    }
    .kpi-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
    .kpi-val { font-size: 1.55rem; font-weight: 700; margin: 2px 0; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
    .kpi-sub { font-size: 0.78rem; color: var(--muted); }

    /* Sections */
    section.panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 22px;
      box-shadow: var(--shadow);
    }
    .sec-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }
    .sec-head h2 { margin: 0; font-size: 1.12rem; font-weight: 700; letter-spacing: -0.01em; }
    .sec-note { font-size: 0.8rem; color: var(--muted); }

    /* Tables */
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th, td { border-bottom: 1px solid var(--border); padding: 9px 10px; text-align: left; }
    th {
      background: var(--surface-2);
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      font-weight: 700;
    }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    tr.clickable-row { cursor: pointer; transition: background 0.12s; }
    tr.clickable-row:hover { background: var(--accent-soft); }
    td.best-cell { background: #f0fdf4; color: var(--green); font-weight: 700; }
    .r2-bar-wrap { display: inline-flex; align-items: center; gap: 8px; justify-content: flex-end; }
    .r2-bar { width: 54px; height: 6px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
    .r2-fill { height: 100%; background: var(--accent); border-radius: 999px; }
    .best-cell .r2-fill { background: var(--green); }
    .sub-metric { display: block; font-size: 0.74rem; font-weight: 400; color: var(--muted); }

    /* Badges & Pills */
    .badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 12px; }
    .pill {
      display: inline-flex;
      align-items: center;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 3px 9px;
      border-radius: 999px;
      border: 1px solid transparent;
    }
    .pill.ok { background: var(--green-soft); color: var(--green); }
    .pill.warn { background: var(--amber-soft); color: var(--amber); }
    .pill.info { background: var(--accent-soft); color: var(--accent-dark); }
    .pill.neutral { background: var(--surface-2); color: var(--text); border-color: var(--border); }

    /* Tabs */
    .ds-tabs { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
    .ds-tab {
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--muted);
      padding: 6px 12px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 600;
      transition: all 0.15s;
    }
    .ds-tab:hover { border-color: var(--accent); color: var(--text); }
    .ds-tab.active { background: var(--accent); color: #ffffff; border-color: var(--accent); }

    .model-tabs {
      display: inline-flex;
      background: var(--surface-2);
      padding: 4px;
      border-radius: 8px;
      gap: 4px;
      margin: 14px 0;
    }
    .model-tab {
      border: none;
      background: transparent;
      color: var(--muted);
      padding: 6px 16px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.83rem;
    }
    .model-tab.active { background: var(--surface); color: var(--text); box-shadow: var(--shadow); }
    .ds-panel, .model-panel { display: none; }
    .ds-panel.active, .model-panel.active { display: block; }

    /* Dataset & Model Cards */
    .mini-kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin: 10px 0 14px;
    }
    .mini-kpi {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .mini-kpi .k-lbl { font-size: 0.72rem; color: var(--muted); font-weight: 600; text-transform: uppercase; }
    .mini-kpi .k-val { font-size: 1.1rem; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }

    /* Plots */
    .grid-plots {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
      margin-top: 10px;
    }
    .plot-card {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      background: var(--surface);
    }
    .plot-card img { width: 100%; height: auto; display: block; border-radius: 4px; }
    .plot-title { font-weight: 600; margin-top: 8px; font-size: 0.84rem; color: var(--text); text-align: center; }

    /* Predictor */
    .pred-toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 12px;
    }
    .btn {
      background: var(--accent);
      color: #ffffff;
      border: none;
      border-radius: 7px;
      padding: 7px 13px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
    }
    .btn:hover { background: var(--accent-dark); }
    .btn-ghost {
      background: var(--surface);
      color: var(--text);
      border: 1px solid var(--border-strong);
    }
    .btn-ghost:hover { background: var(--surface-2); }
    select.ctrl, input.ctrl {
      padding: 7px 10px;
      border: 1px solid var(--border-strong);
      border-radius: 7px;
      font-size: 0.83rem;
      background: var(--surface);
      color: var(--text);
    }
    .input-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(195px, 1fr));
      gap: 9px;
      max-height: 310px;
      overflow-y: auto;
      padding: 4px 2px;
    }
    .field-card {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
    }
    .field-top {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 6px;
      margin-bottom: 4px;
    }
    .field-top label {
      font-size: 0.78rem;
      font-weight: 600;
      word-break: break-word;
    }
    .field-range { font-size: 0.69rem; color: var(--muted); white-space: nowrap; }
    .field-card input, .field-card select {
      width: 100%;
      padding: 5px 8px;
      border: 1px solid var(--border-strong);
      border-radius: 5px;
      font-size: 0.84rem;
      background: var(--surface);
    }
    .pred-cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .pred-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-top: 3px solid var(--accent);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .pred-card.ridge { border-top-color: #0284c7; }
    .pred-card.lasso { border-top-color: var(--green); }
    .pred-val { font-size: 1.45rem; font-weight: 700; margin: 4px 0; font-variant-numeric: tabular-nums; }
    .pred-sub { font-size: 0.78rem; color: var(--muted); }

    .scroll-table { max-height: 340px; overflow-y: auto; border: 1px solid var(--border); border-radius: 8px; }
    tr.intercept-row { background: var(--accent-soft); font-weight: 600; }
    tr.zero-row { color: #94a3b8; }
    details { margin-top: 12px; border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; background: var(--bg); }
    summary { cursor: pointer; font-weight: 600; font-size: 0.84rem; color: var(--text); }
    .contrib-bar { display: inline-block; height: 7px; border-radius: 3px; vertical-align: middle; margin-right: 6px; }
    .contrib-pos { background: #16a34a; }
    .contrib-neg { background: #dc2626; }
    code { background: var(--surface-2); padding: 1px 5px; border-radius: 4px; font-size: 0.84em; }
    .link-btn {
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--accent);
      text-decoration: none;
    }
    .link-btn:hover { text-decoration: underline; }
    .ds-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      flex-wrap: wrap;
    }
  </style>
</head>
<body>
<header class="topbar">
  <div class="topbar-inner">
    <div class="brand">
      <div>
        <div class="brand-title">UCI Regression Benchmark</div>
        <div class="brand-sub">OLS &middot; Ridge (L2) &middot; Lasso (L1) across 10 Datasets</div>
      </div>
    </div>
    <nav class="nav">
      <a href="#overview">Overview</a>
      <a href="#predictor">Live Predictor</a>
      <a href="#explorer">Explorer &amp; Diagrams</a>
      <a href="ols/summary.html">OLS</a>
      <a href="ridge/summary.html">Ridge</a>
      <a href="lasso/summary.html">Lasso</a>
    </nav>
  </div>
</header>

<div class="container">
  <div class="kpi-strip">
    <div class="kpi-tile">
      <div class="kpi-label">Benchmarks</div>
      <div class="kpi-val">{{ master_rows | length }} Datasets</div>
      <div class="kpi-sub">30 runs &middot; Train-only preprocessing</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-label">Median Best Test R&sup2;</div>
      <div class="kpi-val">{{ "%.3f"|format(global_stats.median_best_r2) }}</div>
      <div class="kpi-sub">Max {{ "%.3f"|format(global_stats.max_best_r2) }} ({{ global_stats.max_dataset }})</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-label">Top Test R&sup2; Wins</div>
      <div class="kpi-val">{{ global_stats.wins_summary }}</div>
      <div class="kpi-sub">OLS / Ridge / Lasso</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-label">Diagnostics</div>
      <div class="kpi-val">{{ global_stats.total_plots }} Plots</div>
      <div class="kpi-sub">Residuals &middot; Paths &middot; CV Curves</div>
    </div>
  </div>

  <section class="panel" id="overview">
    <div class="sec-head">
      <h2>Cross-Model Performance</h2>
      <span class="sec-note">Compare R&sup2; across datasets; RMSE is in native target units &middot; Click any row to inspect</span>
    </div>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Target &amp; Split</th>
            <th class="num">Train / Test</th>
            <th class="num">OLS Test R&sup2;</th>
            <th class="num">Ridge Test R&sup2;</th>
            <th class="num">Lasso Test R&sup2;</th>
            <th class="num">Lasso Active</th>
          </tr>
        </thead>
        <tbody>
          {% for row in master_rows %}
          <tr class="clickable-row" onclick="selectDataset('{{ row.key }}'); document.getElementById('explorer').scrollIntoView({behavior:'smooth'});">
            <td><strong>{{ row.name }}</strong><span class="sub-metric"><code>{{ row.key }}</code> &middot; UCI {{ row.uci_id }}</span></td>
            <td><code>{{ row.target }}</code> ({{ row.target_units }})<span class="sub-metric">{{ row.split_strategy }} &middot; {{ row.cv_strategy }}</span></td>
            <td class="num">{{ row.n_train }} / {{ row.n_test }}<span class="sub-metric">{{ row.n_features }} features</span></td>
            <td class="num {% if row.best_model == 'ols' %}best-cell{% endif %}">
              <div class="r2-bar-wrap">
                <div class="r2-bar"><div class="r2-fill" style="width:{{ [0, [100, row.ols_r2 * 100] | min] | max }}%"></div></div>
                <span>{{ "%.4f"|format(row.ols_r2) }}</span>
              </div>
              <span class="sub-metric">RMSE {{ "%.4g"|format(row.ols_rmse) }}</span>
            </td>
            <td class="num {% if row.best_model == 'ridge' %}best-cell{% endif %}">
              <div class="r2-bar-wrap">
                <div class="r2-bar"><div class="r2-fill" style="width:{{ [0, [100, row.ridge_r2 * 100] | min] | max }}%"></div></div>
                <span>{{ "%.4f"|format(row.ridge_r2) }}</span>
              </div>
              <span class="sub-metric">RMSE {{ "%.4g"|format(row.ridge_rmse) }} &middot; &alpha;={{ "%.3g"|format(row.ridge_alpha) }}</span>
            </td>
            <td class="num {% if row.best_model == 'lasso' %}best-cell{% endif %}">
              <div class="r2-bar-wrap">
                <div class="r2-bar"><div class="r2-fill" style="width:{{ [0, [100, row.lasso_r2 * 100] | min] | max }}%"></div></div>
                <span>{{ "%.4f"|format(row.lasso_r2) }}</span>
              </div>
              <span class="sub-metric">RMSE {{ "%.4g"|format(row.lasso_rmse) }} &middot; &alpha;={{ "%.3g"|format(row.lasso_alpha) }}</span>
            </td>
            <td class="num"><strong>{{ row.lasso_nz }}</strong> / {{ row.n_features }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </section>

  <section class="panel" id="predictor">
    <div class="sec-head">
      <h2>Live Predictor</h2>
      <span class="sec-note">Edit inputs or load a held-out test row to compare OLS, Ridge, and Lasso predictions in real time</span>
    </div>
    <div class="pred-toolbar">
      <select id="pred-dataset-select" class="ctrl" onchange="initPredictor(this.value)">
        {% for row in master_rows %}
        <option value="{{ row.key }}">{{ row.name }} ({{ row.target_units }})</option>
        {% endfor %}
      </select>
      <select id="pred-sample-select" class="ctrl" onchange="loadSelectedSample()"></select>
      <button type="button" class="btn" onclick="computePredictions()">Predict</button>
      <button type="button" class="btn btn-ghost" onclick="promptInputsSequential()">Prompt Inputs</button>
      <button type="button" class="btn btn-ghost" onclick="resetPredictorDefaults()">Reset</button>
      <input type="text" id="pred-filter" class="ctrl" placeholder="Filter features..." oninput="filterPredictorFields(this.value)">
    </div>

    <div id="pred-inputs-grid" class="input-grid"></div>
    <div class="pred-cards" id="pred-results-grid"></div>

    <details>
      <summary>Feature Contributions (w &times; x)</summary>
      <div id="pred-contrib-table" class="scroll-table" style="margin-top:8px;"></div>
    </details>
  </section>

  <section class="panel" id="explorer">
    <div class="sec-head">
      <h2>Dataset Explorer &amp; Diagnostics</h2>
      <span class="sec-note">Select a dataset and model tab</span>
    </div>
    <div class="ds-tabs" id="ds-tabs">
      {% for ds in datasets_detail %}
      <button type="button" class="ds-tab {% if loop.first %}active{% endif %}" data-ds="{{ ds.key }}" onclick="selectDataset('{{ ds.key }}')">
        {{ ds.name }}
      </button>
      {% endfor %}
    </div>

    {% for ds in datasets_detail %}
    <div class="ds-panel {% if loop.first %}active{% endif %}" id="panel-{{ ds.key }}">
      <div class="ds-header">
        <div>
          <h3 style="margin:0;">{{ ds.name }}</h3>
          <div class="badge-row">
            <span class="pill neutral">Target: {{ ds.target }} ({{ ds.target_units }})</span>
            <span class="pill info">{{ ds.prep.split_strategy }} &middot; {{ ds.prep.cv_strategy }}</span>
            <span class="pill neutral">{{ ds.prep.n_train }} train / {{ ds.prep.n_test }} test</span>
            {% if ds.is_binary_classification %}<span class="pill warn">Linear Probability Model (0/1)</span>{% endif %}
            {% if ds.is_ordinal_target %}<span class="pill warn">Ordinal Score (3-8)</span>{% endif %}
          </div>
        </div>
      </div>

      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th class="num">Train RMSE</th>
            <th class="num">Train R&sup2;</th>
            <th class="num">Test MAE</th>
            <th class="num">Test RMSE ({{ ds.target_units }})</th>
            <th class="num">Test R&sup2;</th>
            <th class="num">Full Report</th>
          </tr>
        </thead>
        <tbody>
          {% for mkey in ['ols', 'ridge', 'lasso'] %}
          {% set m = ds.models[mkey] %}
          <tr>
            <td><strong>{{ mkey | upper }}</strong></td>
            <td class="num">{{ "%.4g"|format(m.metrics.train.rmse) }}</td>
            <td class="num">{{ "%.4f"|format(m.metrics.train.r2) }}</td>
            <td class="num">{{ "%.4g"|format(m.metrics.test.mae) }}</td>
            <td class="num">{{ "%.4g"|format(m.metrics.test.rmse) }}</td>
            <td class="num"><strong>{{ "%.4f"|format(m.metrics.test.r2) }}</strong></td>
            <td class="num"><a class="link-btn" href="{{ mkey }}/{{ ds.key }}.html" target="_blank">Open HTML &nearr;</a></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>

      {% if ds.is_binary_classification %}
      <div class="badge-row" style="margin-top:10px;">
        {% for mkey in ['ols', 'ridge', 'lasso'] %}
        {% set tc = ds.models[mkey].metrics.test_classification %}
        <span class="pill info">{{ mkey|upper }}: Acc {{ "%.3f"|format(tc.accuracy) }} &middot; AUC {{ "%.3f"|format(tc.roc_auc) }} &middot; Out[0,1] {{ "%.1f"|format(tc.pct_outside_unit_interval) }}%</span>
        {% endfor %}
      </div>
      {% endif %}

      <div class="model-tabs">
        {% for mkey in ['ols', 'ridge', 'lasso'] %}
        <button type="button" class="model-tab {% if loop.first %}active{% endif %}" data-ds="{{ ds.key }}" data-model="{{ mkey }}" onclick="selectModel('{{ ds.key }}', '{{ mkey }}')">
          {{ mkey | upper }}
        </button>
        {% endfor %}
      </div>

      {% for mkey in ['ols', 'ridge', 'lasso'] %}
      {% set m = ds.models[mkey] %}
      <div class="model-panel {% if loop.first %}active{% endif %}" id="panel-{{ ds.key }}-{{ mkey }}">
        <div class="mini-kpi-grid">
          {% for k in m.kpis %}
          <div class="mini-kpi">
            <div class="k-lbl">{{ k.label }}</div>
            <div class="k-val">{{ k.value }}</div>
          </div>
          {% endfor %}
        </div>

        <div class="badge-row">
          {% for b in m.badges %}
          <span class="pill {{ b.kind }}">{{ b.text }}</span>
          {% endfor %}
        </div>

        <div class="grid-plots">
          {% for p in m.plots %}
          <div class="plot-card">
            <img loading="lazy" src="data:image/png;base64,{{ p.b64 }}" alt="{{ p.title }}">
            <div class="plot-title">{{ p.title }}</div>
          </div>
          {% endfor %}
        </div>

        <details>
          <summary>Coefficients ({{ m.coef_rows | length }} entries)</summary>
          <div class="scroll-table" style="margin-top:8px;">
            <table>
              <thead>
                <tr>
                  <th>Role</th>
                  <th>Feature</th>
                  <th class="num">Coef (Std)</th>
                  <th class="num">Coef (Unstd)</th>
                  {% if m.has_ols_ref %}<th class="num">OLS (Std)</th>{% endif %}
                  {% if m.has_shrinkage_ratio %}<th class="num">Ratio</th>{% endif %}
                  {% if m.has_retained %}<th class="num">Active</th><th class="num">CV Freq</th>{% endif %}
                </tr>
              </thead>
              <tbody>
                {% for r in m.coef_rows %}
                <tr class="{% if r.role == 'intercept' %}intercept-row{% elif r.retained is defined and not r.retained %}zero-row{% endif %}">
                  <td>{{ r.role }}</td>
                  <td><code>{{ r.feature }}</code></td>
                  <td class="num">{{ "%.4g"|format(r.coefficient_standardized) }}</td>
                  <td class="num">{{ "%.4g"|format(r.coefficient_unstandardized) }}</td>
                  {% if m.has_ols_ref %}<td class="num">{{ "%.4g"|format(r.ols_coefficient_standardized) }}</td>{% endif %}
                  {% if m.has_shrinkage_ratio %}<td class="num">{% if r.shrinkage_ratio_vs_ols == r.shrinkage_ratio_vs_ols %}{{ "%.3f"|format(r.shrinkage_ratio_vs_ols) }}{% else %}—{% endif %}</td>{% endif %}
                  {% if m.has_retained %}<td class="num">{{ "Yes" if r.retained else "0" }}</td><td class="num">{{ "%.2f"|format(r.cv_selection_freq) }}</td>{% endif %}
                </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        </details>

        <details>
          <summary>Quantitative Diagnostics, Notes &amp; Caveats</summary>
          <div style="margin-top:10px;">
            {% if ds.notes %}
            <div style="margin-bottom:8px;">
              <strong>Dataset Notes:</strong>
              <ul style="margin:4px 0;">{% for n in ds.notes %}<li>{{ n }}</li>{% endfor %}</ul>
            </div>
            {% endif %}
            <div style="margin-bottom:8px;">
              <strong>Key Findings:</strong>
              <ul style="margin:4px 0;">{% for item in m.interpretations %}<li>{{ item }}</li>{% endfor %}</ul>
            </div>
            <div style="margin-bottom:8px;">
              <strong>Limitations:</strong>
              <ul style="margin:4px 0;">{% for lim in m.limitations %}<li>{{ lim }}</li>{% endfor %}</ul>
            </div>
            {% for gname, items in m.diag_sections %}
            <strong style="font-size:0.82rem;">{{ gname }}</strong>
            <table style="margin:4px 0 10px;">
              {% for k, v in items %}
              <tr><td>{{ k }}</td><td class="num">{{ v }}</td></tr>
              {% endfor %}
            </table>
            {% endfor %}
          </div>
        </details>
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

function fmtShort(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  if (n === 0) return '0';
  if (Math.abs(n) >= 1000 || Math.abs(n) < 0.01) return n.toPrecision(3);
  return Number(n.toFixed(2)).toString();
}

function initPredictor(dsKey) {
  currentDatasetKey = dsKey;
  const ds = PREDICTOR_DATA[dsKey];
  const sampleSel = document.getElementById('pred-sample-select');
  sampleSel.innerHTML = '<option value="">Train Median (Default)</option>';
  ds.samples.forEach((s, idx) => {
    const opt = document.createElement('option');
    opt.value = String(idx);
    opt.textContent = s.label;
    sampleSel.appendChild(opt);
  });

  const grid = document.getElementById('pred-inputs-grid');
  grid.innerHTML = '';
  document.getElementById('pred-filter').value = '';

  ds.fields.forEach(f => {
    const card = document.createElement('div');
    card.className = 'field-card';
    card.setAttribute('data-fname', f.name.toLowerCase());

    const top = document.createElement('div');
    top.className = 'field-top';
    const lbl = document.createElement('label');
    lbl.textContent = f.name;
    lbl.setAttribute('for', 'inp-' + f.name);
    top.appendChild(lbl);

    if (f.kind === 'numeric') {
      const rng = document.createElement('span');
      rng.className = 'field-range';
      rng.textContent = `${fmtShort(f.min)}–${fmtShort(f.max)}`;
      top.appendChild(rng);
    }
    card.appendChild(top);

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
    } else {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.step = 'any';
      inp.id = 'inp-' + f.name;
      inp.value = f.default;
      inp.addEventListener('input', computePredictions);
      card.appendChild(inp);
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
  ds.fields.forEach(f => {
    const el = document.getElementById('inp-' + f.name);
    if (el) el.value = String(f.default);
  });
  computePredictions();
}

function loadSelectedSample() {
  const idxStr = document.getElementById('pred-sample-select').value;
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
  computePredictions();
}

function promptInputsSequential() {
  const ds = PREDICTOR_DATA[currentDatasetKey];
  if (ds.fields.length > 25) {
    if (!window.confirm(`${ds.name} has ${ds.fields.length} features. Step through prompts?`)) return;
  }
  for (const f of ds.fields) {
    const el = document.getElementById('inp-' + f.name);
    const current = el ? el.value : String(f.default);
    const hint = f.kind === 'numeric' ? `${fmtShort(f.min)}..${fmtShort(f.max)}` : f.options.map(o => o.value + '=' + o.label).join(', ');
    const ans = window.prompt(`${f.name} (${hint}):`, current);
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
        xMap[f.encoded_prefix + o.value] = (rawVal === o.value) ? 1.0 : 0.0;
      });
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

  const sampleIdx = document.getElementById('pred-sample-select').value;
  const observed = sampleIdx !== '' ? ds.samples[Number(sampleIdx)].observed : null;

  const contribRows = {};
  ['ols', 'ridge', 'lasso'].forEach(mkey => {
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

    let pills = '';
    if (ds.is_binary_classification) {
      const isPos = yhat >= 0.5;
      pills += `<span class="pill ${isPos ? 'ok' : 'neutral'}">${isPos ? 'Positive (1)' : 'Negative (0)'}</span> `;
      if (yhat < 0.0 || yhat > 1.0) pills += `<span class="pill warn">Out of [0,1]</span>`;
    } else if (ds.is_ordinal_target) {
      pills += `<span class="pill info">Score ~ ${Math.round(yhat)}</span>`;
    }
    if (observed !== null) {
      const err = yhat - observed;
      pills += `<span class="pill neutral">Obs ${observed.toPrecision(4)} (&Delta; ${err >= 0 ? '+' : ''}${err.toFixed(3)})</span>`;
    }

    card.innerHTML = `
      <div class="kpi-label">${mkey.toUpperCase()} &rarr; ${ds.target}</div>
      <div class="pred-val">${yhat.toFixed(4)} <span style="font-size:0.76rem; font-weight:500; color:var(--muted);">${ds.target_units}</span></div>
      <div style="display:flex; flex-wrap:wrap; gap:5px; margin-top:6px;">${pills}</div>
    `;
    resGrid.appendChild(card);
  });

  const rows = Object.values(contribRows).sort((a, b) => Math.abs(b.ols || 0) - Math.abs(a.ols || 0));
  const maxAbs = Math.max(1e-9, ...rows.map(r => Math.abs(r.ols || 0)));
  let tbl = '<table><thead><tr><th>Feature</th><th class="num">Input</th><th class="num">OLS Term</th><th class="num">Ridge Term</th><th class="num">Lasso Term</th></tr></thead><tbody>';
  rows.forEach(r => {
    const val = r.ols || 0;
    const wPct = Math.min(48, Math.round((Math.abs(val) / maxAbs) * 48));
    const barCls = val >= 0 ? 'contrib-pos' : 'contrib-neg';
    tbl += `<tr><td><code>${r.feat}</code></td><td class="num">${Number(r.x).toPrecision(4)}</td><td class="num"><span class="contrib-bar ${barCls}" style="width:${wPct}px"></span>${val.toFixed(3)}</td><td class="num">${(r.ridge || 0).toFixed(3)}</td><td class="num">${(r.lasso || 0).toFixed(3)}</td></tr>`;
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
    best_r2_values: List[float] = []
    win_counts = {"ols": 0, "ridge": 0, "lasso": 0}
    max_best_r2 = -1e9
    max_dataset = ""
    total_plots = 0

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
            total_plots += len(plots)
            kpis, badges = _build_visual_kpis_and_badges(model_key, metrics, diagnostics)

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
                "kpis": kpis,
                "badges": badges,
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
            best_val = r2_map[best_model]
            best_r2_values.append(best_val)
            win_counts[best_model] += 1
            if best_val > max_best_r2:
                max_best_r2 = best_val
                max_dataset = cfg.name

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
                    "target": cfg.target,
                    "target_units": cfg.target_units,
                    "is_binary_classification": cfg.is_binary_classification,
                    "is_ordinal_target": cfg.is_ordinal_target,
                    "notes": cfg.notes,
                    "prep": prep,
                    "models": model_bundles,
                }
            )
            predictor_payload[dataset_key] = _build_predictor_payload(dataset_key)

    global_stats = {
        "median_best_r2": float(np.median(best_r2_values)) if best_r2_values else 0.0,
        "max_best_r2": float(max_best_r2) if best_r2_values else 0.0,
        "max_dataset": max_dataset,
        "wins_summary": f"{win_counts['ols']} / {win_counts['ridge']} / {win_counts['lasso']}",
        "total_plots": total_plots,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "index.html"
    html = GRAND_TEMPLATE.render(
        master_rows=master_rows,
        datasets_detail=datasets_detail,
        global_stats=global_stats,
        predictor_json=json.dumps(predictor_payload),
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path
