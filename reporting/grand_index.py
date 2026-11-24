"""Generate a monochrome, minimal-text grand dashboard (reports/index.html)
with UCI repository links, embedded diagrams, live predictor, and rigorous
matrix-notation theory grounded in the reference notes.
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
GITHUB_REPO_URL = "https://github.com/tawratnibir/Regression-Analysis"


def _uci_url(uci_id: int) -> str:
    return f"https://archive.ics.uci.edu/dataset/{uci_id}"


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
                    "label": f"Row #{int(prepared.test_raw_indices[pos])} ({prepared.y_test[pos]:.4g} {cfg.target_units})",
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
) -> tuple[List[Dict[str, str]], List[str]]:
    """Create compact monochrome KPI cards and status tags."""
    kpis: List[Dict[str, str]] = [
        {"label": "Test R²", "value": f"{metrics['test']['r2']:.4f}"},
        {"label": f"Test RMSE ({metrics['target_units']})", "value": _fmt(metrics["test"]["rmse"])},
    ]
    badges: List[str] = []

    if model_key == "ols":
        la = diagnostics["numerical_linear_algebra"]
        ex = diagnostics["exact_algebraic_properties"]
        het = diagnostics["heteroskedasticity_checks"]
        norm = diagnostics["residual_normality_checks"]
        ac = diagnostics["autocorrelation_checks"]
        mc = diagnostics["multicollinearity"]

        kpis.append({"label": "rank(X)", "value": f"{la['matrix_rank']} / {la['n_augmented_columns']}"})
        kpis.append({"label": "κ(X) Std", "value": _fmt(la["condition_number_standardized_with_intercept"])})
        kpis.append({"label": "‖Xᵀe‖∞", "value": f"{ex['max_abs_Xt_residuals']:.1e}"})

        badges.append("Rank Deficient" if la["is_rank_deficient"] else "Full Rank")
        badges.append("BP p < 0.05 (Heteroskedastic)" if het["breusch_pagan_pvalue"] < 0.05 else "Homoskedastic")
        badges.append("JB p < 0.05 (Non-Normal)" if norm["jarque_bera_pvalue"] < 0.05 else "Normal Residuals")
        if mc["n_vif_gt_10"] > 0 or mc.get("has_infinite_vif"):
            badges.append(f"{mc['n_vif_gt_10']} VIF > 10")
        if ac["durbin_watson"] < 1.5 or ac["durbin_watson"] > 2.5:
            badges.append(f"DW = {ac['durbin_watson']:.2f}")

    elif model_key == "ridge":
        pt = diagnostics["penalty_tuning"]
        sd = diagnostics["shrinkage_and_degrees_of_freedom"]
        ns = diagnostics["numerical_stability"]

        kpis.append({"label": "λ (alpha)", "value": _fmt(pt["selected_alpha"])})
        kpis.append({"label": "df(λ)", "value": f"{sd['effective_df_predictors']:.1f} / {sd['n_predictors']}"})
        kpis.append({"label": "‖β̂_λ‖₂ / ‖β̂_OLS‖₂", "value": f"{sd['l2_norm_ratio_ridge_over_ols']:.3f}"})

        badges.append(f"CV R² {pt['best_cv_r2']:.3f}")
        badges.append(f"κ: {_fmt(ns['x_centered_eig_condition_ols'])} → {_fmt(ns['x_centered_eig_condition_ridge'])}")
        badges.append(f"SVD Δ {pt['svd_verification_max_abs_coef_diff']:.1e}")

    elif model_key == "lasso":
        pt = diagnostics["penalty_tuning"]
        sp = diagnostics["sparsity_and_shrinkage"]
        st = diagnostics["selection_stability"]
        cvg = diagnostics["convergence"]

        kpis.append({"label": "λ (alpha)", "value": _fmt(pt["selected_alpha"])})
        kpis.append({"label": "|A(λ)|", "value": f"{sp['n_nonzero_features']} / {sp['n_total_features']}"})
        kpis.append({"label": "‖β̂‖₁ / ‖β̂_OLS‖₁", "value": f"{sp['l1_norm_ratio_lasso_over_ols']:.3f}"})

        badges.append(f"{sp['sparsity_pct']:.0f}% Sparse")
        badges.append(f"Converged ({cvg['n_iter_final']} iters)" if cvg["converged_final"] else "Max Iter Reached")
        if st["n_unstable_across_cv_folds"] > 0:
            badges.append(f"{st['n_unstable_across_cv_folds']} Unstable in CV")
        else:
            badges.append("Stable CV Active Set")

    return kpis, badges


GRAND_TEMPLATE = JINJA_ENV.from_string("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UCI Regression Analysis — OLS · Ridge · Lasso</title>
  <style>
    :root {
      --bg: #fafafa;
      --surface: #ffffff;
      --surface-2: #f4f4f5;
      --surface-3: #e4e4e7;
      --text: #09090b;
      --text-2: #27272a;
      --muted: #71717a;
      --border: #e4e4e7;
      --border-strong: #a1a1aa;
      --ink: #18181b;
      --shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }

    /* Monochrome Sticky Topbar */
    .topbar {
      position: sticky;
      top: 0;
      z-index: 50;
      background: #09090b;
      color: #fafafa;
      border-bottom: 1px solid #27272a;
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
    .brand-link {
      color: #ffffff;
      text-decoration: none;
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: -0.01em;
    }
    .brand-link:hover { text-decoration: underline; }
    .nav { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
    .nav a {
      color: #d4d4d8;
      text-decoration: none;
      font-size: 0.82rem;
      font-weight: 500;
      padding: 5px 10px;
      border-radius: 6px;
      border: 1px solid transparent;
    }
    .nav a:hover { background: #27272a; color: #ffffff; border-color: #3f3f46; }

    .container { max-width: 1360px; margin: 0 auto; padding: 22px 24px 56px; }

    /* KPI Strip */
    .kpi-strip {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }
    .kpi-tile {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
      box-shadow: var(--shadow);
    }
    .kpi-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
    .kpi-val { font-size: 1.5rem; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }

    /* Sections */
    section.panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px 24px;
      margin-bottom: 20px;
      box-shadow: var(--shadow);
    }
    .sec-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }
    .sec-head h2 { margin: 0; font-size: 1.08rem; font-weight: 700; letter-spacing: -0.01em; }

    /* Tables */
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th, td { border-bottom: 1px solid var(--border); padding: 9px 10px; text-align: left; vertical-align: middle; }
    th {
      background: var(--surface-2);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-2);
      font-weight: 700;
    }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    tr.clickable-row { cursor: pointer; }
    tr.clickable-row:hover { background: var(--surface-2); }
    td.best-cell { background: var(--surface-2); font-weight: 700; box-shadow: inset 2px 0 0 var(--ink); }
    .r2-bar-wrap { display: inline-flex; align-items: center; gap: 8px; justify-content: flex-end; }
    .r2-bar { width: 52px; height: 5px; background: var(--surface-3); border-radius: 999px; overflow: hidden; }
    .r2-fill { height: 100%; background: var(--ink); }

    /* Links & Pills */
    a.repo-link {
      color: var(--text);
      text-decoration: none;
      border-bottom: 1px solid var(--border-strong);
      font-weight: 600;
    }
    a.repo-link:hover { border-bottom-color: var(--ink); }
    .badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
    .pill {
      display: inline-flex;
      align-items: center;
      font-size: 0.74rem;
      font-weight: 600;
      padding: 3px 9px;
      border-radius: 999px;
      background: var(--surface-2);
      color: var(--text);
      border: 1px solid var(--border);
    }
    .pill-dark { background: var(--ink); color: #ffffff; border-color: var(--ink); }

    /* Tabs */
    .ds-tabs { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
    .ds-tab {
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text-2);
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 600;
    }
    .ds-tab:hover { border-color: var(--ink); }
    .ds-tab.active { background: var(--ink); color: #ffffff; border-color: var(--ink); }

    .model-tabs {
      display: inline-flex;
      background: var(--surface-2);
      padding: 3px;
      border-radius: 7px;
      gap: 3px;
      margin: 14px 0;
      border: 1px solid var(--border);
    }
    .model-tab {
      border: none;
      background: transparent;
      color: var(--muted);
      padding: 6px 16px;
      border-radius: 5px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.82rem;
    }
    .model-tab.active { background: var(--surface); color: var(--text); box-shadow: var(--shadow); }
    .ds-panel, .model-panel { display: none; }
    .ds-panel.active, .model-panel.active { display: block; }

    .mini-kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin: 10px 0;
    }
    .mini-kpi {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 9px 12px;
    }
    .mini-kpi .k-lbl { font-size: 0.72rem; color: var(--muted); font-weight: 600; }
    .mini-kpi .k-val { font-size: 1.08rem; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }

    /* Plots (images keep original colors) */
    .grid-plots {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
      margin-top: 12px;
    }
    .plot-card {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      background: var(--surface);
    }
    .plot-card img { width: 100%; height: auto; display: block; border-radius: 4px; }
    .plot-title { font-weight: 600; margin-top: 8px; font-size: 0.83rem; text-align: center; color: var(--text-2); }

    /* Predictor */
    .pred-toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }
    .btn {
      background: var(--ink);
      color: #ffffff;
      border: 1px solid var(--ink);
      border-radius: 6px;
      padding: 7px 13px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
    }
    .btn:hover { background: #27272a; }
    .btn-ghost { background: var(--surface); color: var(--text); border: 1px solid var(--border-strong); }
    .btn-ghost:hover { background: var(--surface-2); }
    select.ctrl, input.ctrl {
      padding: 7px 10px;
      border: 1px solid var(--border-strong);
      border-radius: 6px;
      font-size: 0.83rem;
      background: var(--surface);
      color: var(--text);
    }
    .input-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(195px, 1fr));
      gap: 8px;
      max-height: 300px;
      overflow-y: auto;
      padding: 2px;
    }
    .field-card { background: var(--bg); border: 1px solid var(--border); border-radius: 7px; padding: 8px 10px; }
    .field-top { display: flex; justify-content: space-between; align-items: baseline; gap: 6px; margin-bottom: 4px; }
    .field-top label { font-size: 0.77rem; font-weight: 600; word-break: break-word; }
    .field-range { font-size: 0.69rem; color: var(--muted); white-space: nowrap; }
    .field-card input, .field-card select {
      width: 100%;
      padding: 5px 7px;
      border: 1px solid var(--border-strong);
      border-radius: 5px;
      font-size: 0.83rem;
      background: var(--surface);
      color: var(--text);
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
      border-top: 3px solid var(--ink);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .pred-val { font-size: 1.45rem; font-weight: 700; margin: 4px 0; font-variant-numeric: tabular-nums; }

    .scroll-table { max-height: 340px; overflow-y: auto; border: 1px solid var(--border); border-radius: 7px; }
    tr.intercept-row { background: var(--surface-2); font-weight: 700; }
    tr.zero-row { color: var(--muted); }
    details { margin-top: 12px; border: 1px solid var(--border); border-radius: 7px; padding: 8px 12px; background: var(--bg); }
    summary { cursor: pointer; font-weight: 600; font-size: 0.84rem; }
    .contrib-bar { display: inline-block; height: 7px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
    .contrib-pos { background: #18181b; }
    .contrib-neg { background: #a1a1aa; border: 1px solid #52525b; }
    code { background: var(--surface-2); padding: 1px 5px; border-radius: 4px; font-size: 0.84em; }

    /* Mathematical Typesetting (100% offline monochrome CSS) */
    .math-block {
      background: var(--bg);
      border: 1px solid var(--border);
      border-left: 3px solid var(--ink);
      border-radius: 6px;
      padding: 12px 16px;
      margin: 10px 0;
      overflow-x: auto;
      font-family: "Cambria Math", "STIX Two Math", "Latin Modern Math", Georgia, serif;
      font-size: 1.02rem;
      line-height: 1.75;
    }
    .m { font-family: "Cambria Math", "STIX Two Math", Georgia, serif; }
    .mv { font-weight: 700; font-style: normal; } /* matrix / vector bold */
    .frac {
      display: inline-flex;
      flex-direction: column;
      align-items: center;
      vertical-align: middle;
      margin: 0 3px;
      font-size: 0.9em;
      line-height: 1.15;
    }
    .frac > span:first-child { border-bottom: 1px solid var(--text); padding: 0 4px 1px; }
    .frac > span:last-child { padding: 1px 4px 0; }
    .mat {
      display: inline-table;
      vertical-align: middle;
      border-left: 1.5px solid var(--text);
      border-right: 1.5px solid var(--text);
      padding: 2px 6px;
      margin: 0 4px;
      border-collapse: collapse;
    }
    .mat td { border: none; padding: 1px 6px; text-align: center; font-size: 0.92em; }
    .theory-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 18px;
    }
    .theory-card {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px 18px;
      background: var(--surface);
    }
    .theory-card h3 { margin: 0 0 10px; font-size: 1rem; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
    .theory-card h4 { margin: 14px 0 6px; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.03em; color: var(--text-2); }
    .theory-card ul { margin: 6px 0; padding-left: 18px; font-size: 0.87rem; }
    .theory-card li { margin-bottom: 5px; }
    .theory-card p { margin: 6px 0; font-size: 0.87rem; color: var(--text-2); }
  </style>
</head>
<body>
<header class="topbar">
  <div class="topbar-inner">
    <a class="brand-link" href="{{ github_repo_url }}" target="_blank" rel="noopener noreferrer">
      tawratnibir/Regression-Analysis &nearr;
    </a>
    <nav class="nav">
      <a href="#overview">Overview</a>
      <a href="#predictor">Predictor</a>
      <a href="#explorer">Explorer</a>
      <a href="#theory">Equations &amp; Methodology</a>
      <a href="ols/summary.html">OLS</a>
      <a href="ridge/summary.html">Ridge</a>
      <a href="lasso/summary.html">Lasso</a>
    </nav>
  </div>
</header>

<div class="container">
  <div class="kpi-strip">
    <div class="kpi-tile">
      <div class="kpi-label">UCI Datasets</div>
      <div class="kpi-val">{{ master_rows | length }}</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-label">Median Best Test R&sup2;</div>
      <div class="kpi-val">{{ "%.4f"|format(global_stats.median_best_r2) }}</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-label">Top R&sup2; Wins (OLS / Ridge / Lasso)</div>
      <div class="kpi-val">{{ global_stats.wins_summary }}</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-label">Saved Diagrams</div>
      <div class="kpi-val">{{ global_stats.total_plots }}</div>
    </div>
  </div>

  <section class="panel" id="overview">
    <div class="sec-head">
      <h2>Benchmark Matrix</h2>
    </div>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>UCI Repository</th>
            <th>Target (Units)</th>
            <th>Split / CV</th>
            <th class="num">n<sub>train</sub> / n<sub>test</sub> (p)</th>
            <th class="num">OLS Test R&sup2; (RMSE)</th>
            <th class="num">Ridge Test R&sup2; (RMSE)</th>
            <th class="num">Lasso Test R&sup2; (RMSE)</th>
            <th class="num">|A(&lambda;)| / p</th>
          </tr>
        </thead>
        <tbody>
          {% for row in master_rows %}
          <tr class="clickable-row" onclick="selectDataset('{{ row.key }}'); document.getElementById('explorer').scrollIntoView({behavior:'smooth'});">
            <td><strong>{{ row.name }}</strong></td>
            <td onclick="event.stopPropagation();"><a class="repo-link" href="{{ row.uci_url }}" target="_blank" rel="noopener noreferrer">UCI {{ row.uci_id }} &nearr;</a></td>
            <td><code>{{ row.target }}</code> ({{ row.target_units }})</td>
            <td><code>{{ row.split_strategy }}</code></td>
            <td class="num">{{ row.n_train }} / {{ row.n_test }} ({{ row.n_features }})</td>
            <td class="num {% if row.best_model == 'ols' %}best-cell{% endif %}">
              <div class="r2-bar-wrap">
                <div class="r2-bar"><div class="r2-fill" style="width:{{ [0, [100, row.ols_r2 * 100] | min] | max }}%"></div></div>
                <span>{{ "%.4f"|format(row.ols_r2) }}</span>
              </div>
              ({{ "%.4g"|format(row.ols_rmse) }})
            </td>
            <td class="num {% if row.best_model == 'ridge' %}best-cell{% endif %}">
              <div class="r2-bar-wrap">
                <div class="r2-bar"><div class="r2-fill" style="width:{{ [0, [100, row.ridge_r2 * 100] | min] | max }}%"></div></div>
                <span>{{ "%.4f"|format(row.ridge_r2) }}</span>
              </div>
              ({{ "%.4g"|format(row.ridge_rmse) }})
            </td>
            <td class="num {% if row.best_model == 'lasso' %}best-cell{% endif %}">
              <div class="r2-bar-wrap">
                <div class="r2-bar"><div class="r2-fill" style="width:{{ [0, [100, row.lasso_r2 * 100] | min] | max }}%"></div></div>
                <span>{{ "%.4f"|format(row.lasso_r2) }}</span>
              </div>
              ({{ "%.4g"|format(row.lasso_rmse) }})
            </td>
            <td class="num">{{ row.lasso_nz }} / {{ row.n_features }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </section>

  <section class="panel" id="predictor">
    <div class="sec-head">
      <h2>Predictor</h2>
    </div>
    <div class="pred-toolbar">
      <select id="pred-dataset-select" class="ctrl" onchange="initPredictor(this.value)">
        {% for row in master_rows %}
        <option value="{{ row.key }}">{{ row.name }} ({{ row.target_units }})</option>
        {% endfor %}
      </select>
      <select id="pred-sample-select" class="ctrl" onchange="loadSelectedSample()"></select>
      <button type="button" class="btn" onclick="computePredictions()">Predict</button>
      <button type="button" class="btn btn-ghost" onclick="promptInputsSequential()">Prompt</button>
      <button type="button" class="btn btn-ghost" onclick="resetPredictorDefaults()">Reset</button>
      <input type="text" id="pred-filter" class="ctrl" placeholder="Filter features..." oninput="filterPredictorFields(this.value)">
    </div>

    <div id="pred-inputs-grid" class="input-grid"></div>
    <div class="pred-cards" id="pred-results-grid"></div>

    <details>
      <summary>Feature Contributions (<span class="m">&beta;&#770;<sub>j</sub> x<sub>j</sub></span>)</summary>
      <div id="pred-contrib-table" class="scroll-table" style="margin-top:8px;"></div>
    </details>
  </section>

  <section class="panel" id="explorer">
    <div class="sec-head">
      <h2>Explorer</h2>
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
      <div class="badge-row">
        <a class="pill pill-dark" style="text-decoration:none;" href="{{ ds.uci_url }}" target="_blank" rel="noopener noreferrer">UCI {{ ds.uci_id }} &nearr;</a>
        <span class="pill">{{ ds.target }} ({{ ds.target_units }})</span>
        <span class="pill">{{ ds.prep.split_strategy }} &middot; {{ ds.prep.cv_strategy }}</span>
        <span class="pill">{{ ds.prep.n_train }} / {{ ds.prep.n_test }}</span>
        {% if ds.is_binary_classification %}<span class="pill">LPM (0/1)</span>{% endif %}
        {% if ds.is_ordinal_target %}<span class="pill">Ordinal (3–8)</span>{% endif %}
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
            <th class="num">HTML</th>
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
            <td class="num"><a class="repo-link" href="{{ mkey }}/{{ ds.key }}.html" target="_blank">Report &nearr;</a></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>

      {% if ds.is_binary_classification %}
      <div class="badge-row">
        {% for mkey in ['ols', 'ridge', 'lasso'] %}
        {% set tc = ds.models[mkey].metrics.test_classification %}
        <span class="pill">{{ mkey|upper }}: Acc {{ "%.3f"|format(tc.accuracy) }} &middot; AUC {{ "%.3f"|format(tc.roc_auc) }} &middot; Out[0,1] {{ "%.1f"|format(tc.pct_outside_unit_interval) }}%</span>
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
          <span class="pill">{{ b }}</span>
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
          <summary>Coefficients ({{ m.coef_rows | length }})</summary>
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
                  {% if m.has_shrinkage_ratio %}<td class="num">{% if r.shrinkage_ratio_vs_ols == r.shrinkage_ratio_vs_ols %}{{ "%.3f"|format(r.shrinkage_ratio_vs_ols) }}{% else %}&mdash;{% endif %}</td>{% endif %}
                  {% if m.has_retained %}<td class="num">{{ "Yes" if r.retained else "0" }}</td><td class="num">{{ "%.2f"|format(r.cv_selection_freq) }}</td>{% endif %}
                </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        </details>

        <details>
          <summary>Diagnostics &amp; Notes</summary>
          <div style="margin-top:10px;">
            {% if ds.notes %}
            <ul style="margin:4px 0 8px;">{% for n in ds.notes %}<li>{{ n }}</li>{% endfor %}</ul>
            {% endif %}
            <ul style="margin:4px 0 8px;">{% for item in m.interpretations %}<li>{{ item }}</li>{% endfor %}</ul>
            <ul style="margin:4px 0 8px;">{% for lim in m.limitations %}<li>{{ lim }}</li>{% endfor %}</ul>
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

  <!-- THOROUGH MATHEMATICAL FORMULATION & METHODOLOGY SECTION -->
  <section class="panel" id="theory">
    <div class="sec-head">
      <h2>Mathematical Formulation, Matrix Notation &amp; Methodology</h2>
    </div>

    <!-- 1. Symbol & Matrix Glossary -->
    <div class="theory-card" style="margin-bottom:18px;">
      <h3>1. Matrix Notation &amp; Symbol Glossary</h3>
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Dimensions / Space</th>
            <th>Mathematical Meaning</th>
            <th>Dashboard / Code Mapping</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td class="m"><span class="mv">y</span>, <span class="mv">&epsilon;</span>, <span class="mv">e</span></td>
            <td class="m">&Ropf;<sup><i>n</i>&times;1</sup></td>
            <td>Observed response vector <span class="mv">y</span>, unobservable population disturbances <span class="mv">&epsilon;</span>, and observable sample residuals <span class="mv">e</span> = <span class="mv">y</span> &minus; <span class="mv">X</span><span class="mv">&beta;&#770;</span> (<span class="mv">&epsilon;</span> &ne; <span class="mv">e</span>).</td>
            <td><code>y_train</code>, <code>resid_train = y_train - y_train_pred</code></td>
          </tr>
          <tr>
            <td class="m"><span class="mv">X</span><sub>aug</sub> vs <span class="mv">X</span></td>
            <td class="m">&Ropf;<sup><i>n</i>&times;<i>k</i></sup> vs &Ropf;<sup><i>n</i>&times;<i>p</i></sup></td>
            <td>In OLS matrix notes, <span class="mv">X</span><sub>aug</sub> = [<span class="mv">1</span>, <span class="mv">X</span>] &isin; &Ropf;<sup><i>n</i>&times;<i>k</i></sup> prepends a column of ones <span class="mv">1</span> so <i>k</i> = <i>p</i> + 1. In Ridge &amp; LASSO, <span class="mv">X</span> &isin; &Ropf;<sup><i>n</i>&times;<i>p</i></sup> holds the <i>p</i> predictor columns <span class="mv">x</span><sub>1</sub>,&dots;,<span class="mv">x</span><sub><i>p</i></sub> while the intercept &alpha; (or &beta;<sub>0</sub>) is unpenalized.</td>
            <td><code>X_tr_aug</code> (OLS), <code>X_train</code> (Ridge/Lasso)</td>
          </tr>
          <tr>
            <td class="m"><span class="mv">X&#771;</span>, <span class="mv">y&#771;</span></td>
            <td class="m">&Ropf;<sup><i>n</i>&times;<i>p</i></sup>, &Ropf;<sup><i>n</i>&times;1</sup></td>
            <td>Centered response <span class="mv">y&#771;</span> = <span class="mv">y</span> &minus; <span class="mv">1</span><i>y&#772;</i> and standardized design matrix <span class="mv">X&#771;</span> = (<span class="mv">X</span> &minus; <span class="mv">1</span><span class="mv">x&#772;</span><sup>&top;</sup>)diag(1/<span class="mv">s</span>) with column mean 0 and variance <i>s<sub>j</sub></i><sup>2</sup> = (1/<i>n</i>)&sum;<sub><i>i</i>=1</sub><sup><i>n</i></sup>(<i>x<sub>ij</sub></i> &minus; <i>x&#772;<sub>j</sub></i>)<sup>2</sup> = 1.</td>
            <td><code>StandardScaler</code> fit on training split only</td>
          </tr>
          <tr>
            <td class="m"><span class="mv">&beta;</span>, <span class="mv">&beta;&#770;</span>, <span class="mv">&beta;&#771;</span></td>
            <td class="m">&Ropf;<sup><i>k</i>&times;1</sup> or &Ropf;<sup><i>p</i>&times;1</sup></td>
            <td>Unknown population parameters <span class="mv">&beta;</span>, standardized-scale estimates <span class="mv">&beta;&#771;</span>, and unstandardized estimates <span class="mv">&beta;&#770;</span> = diag(1/<span class="mv">s</span>)<span class="mv">&beta;&#771;</span> with &alpha;&#770; = <i>y&#772;</i> &minus; <span class="mv">x&#772;</span><sup>&top;</sup><span class="mv">&beta;&#770;</span>.</td>
            <td><code>coefficient_standardized</code>, <code>coefficient_unstandardized</code></td>
          </tr>
          <tr>
            <td class="m"><span class="mv">H</span>, <span class="mv">M</span></td>
            <td class="m">&Ropf;<sup><i>n</i>&times;<i>n</i></sup></td>
            <td>Hat matrix <span class="mv">H</span> = <span class="mv">X</span>(<span class="mv">X</span><sup>&top;</sup><span class="mv">X</span>)<sup>&minus;1</sup><span class="mv">X</span><sup>&top;</sup> (<span class="mv">y&#770;</span> = <span class="mv">H</span><span class="mv">y</span>) and Residual Maker <span class="mv">M</span> = <span class="mv">I</span><sub><i>n</i></sub> &minus; <span class="mv">H</span> (<span class="mv">e</span> = <span class="mv">M</span><span class="mv">y</span>). Both are symmetric and idempotent (<span class="mv">M</span><sup>2</sup> = <span class="mv">M</span>, <span class="mv">M</span><span class="mv">X</span> = <span class="mv">0</span>).</td>
            <td>Leverage <i>h<sub>i</sub></i> = <i>H<sub>ii</sub></i> via thin SVD</td>
          </tr>
          <tr>
            <td class="m"><span class="mv">U</span><span class="mv">D</span><span class="mv">V</span><sup>&top;</sup></td>
            <td class="m"><i>n</i>&times;<i>n</i>, <i>n</i>&times;<i>p</i>, <i>p</i>&times;<i>p</i></td>
            <td>Singular Value Decomposition of <span class="mv">X</span>: orthogonal <span class="mv">U</span>, <span class="mv">V</span> (<span class="mv">U</span><sup>&top;</sup><span class="mv">U</span> = <span class="mv">I</span>, <span class="mv">V</span><sup>&top;</sup><span class="mv">V</span> = <span class="mv">I</span><sub><i>p</i></sub>), and diagonal <span class="mv">D</span> with singular values <i>d</i><sub>1</sub> &ge; &hellip; &ge; <i>d<sub>m</sub></i> &ge; 0, <i>m</i> = min(<i>n</i>, <i>p</i>).</td>
            <td>Used in <code>lstsq</code>, SVD verification, and df(&lambda;)</td>
          </tr>
          <tr>
            <td class="m">&kappa;(<span class="mv">X</span><sup>&top;</sup><span class="mv">X</span>)</td>
            <td class="m">&Ropf;<sub>&ge;1</sub> &cup; {&infin;}</td>
            <td>Condition number <i>d</i><sub>max</sub><sup>2</sup> / <i>d</i><sub>min</sub><sup>2</sup> (or <i>d</i><sub>max</sub> / <i>d</i><sub>min</sub> for <span class="mv">X</span>). Large values indicate collinearity / ill-conditioning.</td>
            <td><code>Cond # (Std)</code>, <code>&kappa;</code> pills</td>
          </tr>
          <tr>
            <td class="m">&lambda; (<code>alpha</code>)</td>
            <td class="m">&lambda; &ge; 0</td>
            <td>Regularization multiplier controlling penalty strength (denoted <code>alpha</code> in scikit-learn <code>Ridge</code> and <code>Lasso</code>). Tuned exclusively by training-fold CV.</td>
            <td><code>selected_alpha</code>, CV curves</td>
          </tr>
          <tr>
            <td class="m">&parallel;<span class="mv">&beta;</span>&parallel;<sub><i>q</i></sub></td>
            <td class="m"><i>q</i> &isin; {0, 1, 2}</td>
            <td>&parallel;<span class="mv">&beta;</span>&parallel;<sub>0</sub> = &sum; 1{&beta;<sub><i>j</i></sub> &ne; 0}, &parallel;<span class="mv">&beta;</span>&parallel;<sub>1</sub> = &sum; |&beta;<sub><i>j</i></sub>| (LASSO), &parallel;<span class="mv">&beta;</span>&parallel;<sub>2</sub> = (&sum; &beta;<sub><i>j</i></sub><sup>2</sup>)<sup>1/2</sup> (Ridge).</td>
            <td>L1/L2 shrinkage ratios &amp; active set |<i>A</i>(&lambda;)|</td>
          </tr>
          <tr>
            <td class="m"><i>S</i>(<i>z</i>, &lambda;), <span class="mv">s&#770;</span></td>
            <td class="m">&Ropf;, [&minus;1, 1]<sup><i>p</i></sup></td>
            <td>Soft-thresholding operator <i>S</i>(<i>z</i>, &lambda;) = sign(<i>z</i>)max(|<i>z</i>| &minus; &lambda;, 0) and subgradient vector <span class="mv">s&#770;</span> &isin; &part;&parallel;<span class="mv">&beta;&#770;</span>&parallel;<sub>1</sub> in LASSO KKT conditions.</td>
            <td>Coordinate descent sparsity &amp; convergence</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="theory-grid">
      <!-- 2. OLS -->
      <div class="theory-card">
        <h3>2. Ordinary Least Squares (OLS)</h3>
        <h4>Model, Objective &amp; Normal Equations</h4>
        <p>Given <span class="m"><span class="mv">X</span> &isin; &Ropf;<sup><i>n</i>&times;<i>k</i></sup></span> (first column <span class="m"><span class="mv">1</span></span>) and <span class="m"><span class="mv">y</span> = <span class="mv">X</span><span class="mv">&beta;</span> + <span class="mv">&epsilon;</span></span>, OLS minimizes the Residual Sum of Squares (RSS):</p>
        <div class="math-block">
          <span class="mv">e</span><sup>&top;</sup><span class="mv">e</span> = (<span class="mv">y</span> &minus; <span class="mv">X</span><span class="mv">&beta;&#770;</span>)<sup>&top;</sup>(<span class="mv">y</span> &minus; <span class="mv">X</span><span class="mv">&beta;&#770;</span>) = <span class="mv">y</span><sup>&top;</sup><span class="mv">y</span> &minus; 2<span class="mv">&beta;&#770;</span><sup>&top;</sup><span class="mv">X</span><sup>&top;</sup><span class="mv">y</span> + <span class="mv">&beta;&#770;</span><sup>&top;</sup><span class="mv">X</span><sup>&top;</sup><span class="mv">X</span><span class="mv">&beta;&#770;</span>
          <br>
          <span class="frac"><span>&part;(<span class="mv">e</span><sup>&top;</sup><span class="mv">e</span>)</span><span>&part;<span class="mv">&beta;&#770;</span></span></span> = &minus;2<span class="mv">X</span><sup>&top;</sup><span class="mv">y</span> + 2<span class="mv">X</span><sup>&top;</sup><span class="mv">X</span><span class="mv">&beta;&#770;</span> = <span class="mv">0</span> &nbsp;&rArr;&nbsp; (<span class="mv">X</span><sup>&top;</sup><span class="mv">X</span>)<span class="mv">&beta;&#770;</span> = <span class="mv">X</span><sup>&top;</sup><span class="mv">y</span>
        </div>
        <p>When <span class="m">rank(<span class="mv">X</span>) = <i>k</i></span>, the Hessian <span class="m">2<span class="mv">X</span><sup>&top;</sup><span class="mv">X</span></span> is positive definite and <span class="m"><span class="mv">&beta;&#770;</span> = (<span class="mv">X</span><sup>&top;</sup><span class="mv">X</span>)<sup>&minus;1</sup><span class="mv">X</span><sup>&top;</sup><span class="mv">y</span></span>. When rank-deficient (e.g., <code>appliances_energy</code> where <code>rv1 == rv2</code>), <code>np.linalg.lstsq</code> uses SVD to return the unique minimum-norm solution.</p>

        <h4>Exact Algebraic Identities (Assumption-Free)</h4>
        <p>Substituting <span class="m"><span class="mv">y</span> = <span class="mv">X</span><span class="mv">&beta;&#770;</span> + <span class="mv">e</span></span> into the normal equations gives <span class="m"><span class="mv">X</span><sup>&top;</sup><span class="mv">e</span> = <span class="mv">0</span></span>, which guarantees unconditionally:</p>
        <ul>
          <li>Every column <span class="m"><span class="mv">x</span><sub><i>j</i></sub></span> is orthogonal to residuals: <span class="m"><span class="mv">x</span><sub><i>j</i></sub><sup>&top;</sup><span class="mv">e</span> = 0</span>.</li>
          <li>With intercept column <span class="m"><span class="mv">1</span></span>: <span class="m">&sum;<sub><i>i</i></sub> <i>e<sub>i</sub></i> = 0</span>, <span class="m"><i>e&#772;</i> = 0</span>, <span class="m"><i>y&#772;</i> = <span class="mv">x&#772;</span><sup>&top;</sup><span class="mv">&beta;&#770;</span></span>, <span class="m"><span class="mv">y&#770;</span><sup>&top;</sup><span class="mv">e</span> = 0</span>, and <span class="m"><i>y&#770;&#772;</i> = <i>y&#772;</i></span>.</li>
          <li>Hat matrix <span class="m"><span class="mv">H</span> = <span class="mv">X</span>(<span class="mv">X</span><sup>&top;</sup><span class="mv">X</span>)<sup>&minus;1</sup><span class="mv">X</span><sup>&top;</sup></span> yields leverage <span class="m"><i>h<sub>i</sub></i> = <i>H<sub>ii</sub></i></span>, and Cook&rsquo;s distance <span class="m"><i>D<sub>i</sub></i> = [<i>e<sub>i</sub></i><sup>2</sup> / (<i>r</i>&sigma;&#770;<sup>2</sup>)] &middot; [<i>h<sub>i</sub></i> / (1 &minus; <i>h<sub>i</sub></i>)<sup>2</sup>]</span> flags influential rows.</li>
        </ul>

        <h4>Gauss-Markov Assumptions &amp; Diagnostics</h4>
        <p>Under (1) linearity, (2) full rank, (3) <span class="m">E[<span class="mv">&epsilon;</span>|<span class="mv">X</span>] = <span class="mv">0</span></span>, (4) spherical errors <span class="m">E[<span class="mv">&epsilon;</span><span class="mv">&epsilon;</span><sup>&top;</sup>|<span class="mv">X</span>] = &sigma;<sup>2</sup><span class="mv">I</span><sub><i>n</i></sub></span> (homoskedasticity + no autocorrelation), and (5) exogenous <span class="m"><span class="mv">X</span></span>, <span class="m"><span class="mv">&beta;&#770;</span></span> is BLUE with <span class="m">Var(<span class="mv">&beta;&#770;</span>) = &sigma;<sup>2</sup>(<span class="mv">X</span><sup>&top;</sup><span class="mv">X</span>)<sup>&minus;1</sup></span> and <span class="m">&sigma;&#770;<sup>2</sup> = <span class="mv">e</span><sup>&top;</sup><span class="mv">e</span>/(<i>n</i> &minus; <i>k</i>)</span>. Adding (6) <span class="m"><span class="mv">&epsilon;</span>|<span class="mv">X</span> ~ N(<span class="mv">0</span>, &sigma;<sup>2</sup><span class="mv">I</span><sub><i>n</i></sub>)</span> gives <span class="m"><span class="mv">&beta;&#770;</span> ~ N(<span class="mv">&beta;</span>, &sigma;<sup>2</sup>(<span class="mv">X</span><sup>&top;</sup><span class="mv">X</span>)<sup>&minus;1</sup>)</span>. When <span class="m">E[<span class="mv">&epsilon;</span><span class="mv">&epsilon;</span><sup>&top;</sup>|<span class="mv">X</span>] = &Omega; &ne; &sigma;<sup>2</sup><span class="mv">I</span></span> (checked via Breusch-Pagan &amp; Durbin-Watson), standard errors require White&rsquo;s sandwich <span class="m">(<span class="mv">X</span><sup>&top;</sup><span class="mv">X</span>)<sup>&minus;1</sup><span class="mv">X</span><sup>&top;</sup><span class="mv">e</span><span class="mv">e</span><sup>&top;</sup><span class="mv">X</span>(<span class="mv">X</span><sup>&top;</sup><span class="mv">X</span>)<sup>&minus;1</sup></span>.</p>
      </div>

      <!-- 3. Ridge -->
      <div class="theory-card">
        <h3>3. Ridge Regression (&ell;<sub>2</sub> Regularization)</h3>
        <h4>Tikhonov Normal Equations &amp; Conditioning</h4>
        <p>When <span class="m"><span class="mv">X</span><sup>&top;</sup><span class="mv">X</span></span> is singular or ill-conditioned (<span class="m">&kappa;(<span class="mv">X</span><sup>&top;</sup><span class="mv">X</span>) = <i>d</i><sub>max</sub><sup>2</sup> / <i>d</i><sub>min</sub><sup>2</sup> &gg; 1</span>), adding <span class="m">&lambda;<span class="mv">I</span><sub><i>p</i></sub></span> (&lambda; &gt; 0) makes the system strictly convex and invertible:</p>
        <div class="math-block">
          min<sub>&alpha;,&beta;</sub> &parallel;<span class="mv">y</span> &minus; <span class="mv">1</span>&alpha; &minus; <span class="mv">X</span><span class="mv">&beta;</span>&parallel;<sub>2</sub><sup>2</sup> + &lambda;&parallel;<span class="mv">&beta;</span>&parallel;<sub>2</sub><sup>2</sup>
          &nbsp;&hArr;&nbsp;
          min<sub>&beta;</sub> &parallel;<span class="mv">y&#771;</span> &minus; <span class="mv">X&#771;</span><span class="mv">&beta;</span>&parallel;<sub>2</sub><sup>2</sup> &nbsp;s.t.&nbsp; &parallel;<span class="mv">&beta;</span>&parallel;<sub>2</sub> &le; <i>c</i>
          <br>
          (<span class="mv">X&#771;</span><sup>&top;</sup><span class="mv">X&#771;</span> + &lambda;<span class="mv">I</span><sub><i>p</i></sub>)<span class="mv">&beta;&#771;</span><sub>&lambda;</sub> = <span class="mv">X&#771;</span><sup>&top;</sup><span class="mv">y&#771;</span>,
          &nbsp;&nbsp;
          &kappa;(<span class="mv">X&#771;</span><sup>&top;</sup><span class="mv">X&#771;</span> + &lambda;<span class="mv">I</span><sub><i>p</i></sub>) = <span class="frac"><span><i>d</i><sub>max</sub><sup>2</sup> + &lambda;</span><span><i>d</i><sub>min</sub><sup>2</sup> + &lambda;</span></span> &lt; &infin;
        </div>
        <p>Unstandardized parameters are recovered via <span class="m"><span class="mv">&beta;&#770;</span><sub>&lambda;</sub> = diag(1/<span class="mv">s</span>)<span class="mv">&beta;&#771;</span><sub>&lambda;</sub></span> and <span class="m">&alpha;&#770; = <i>y&#772;</i> &minus; <span class="mv">x&#772;</span><sup>&top;</sup><span class="mv">&beta;&#770;</span><sub>&lambda;</sub></span>.</p>

        <h4>SVD Shrinkage &amp; Effective Degrees of Freedom</h4>
        <p>With SVD <span class="m"><span class="mv">X&#771;</span> = <span class="mv">U</span><span class="mv">D</span><span class="mv">V</span><sup>&top;</sup></span>, Ridge shrinks small-singular-value directions smoothly without thresholding any coefficient to zero:</p>
        <div class="math-block">
          <span class="mv">&beta;&#771;</span><sub>&lambda;</sub> = &sum;<sub><i>d<sub>j</sub></i>&gt;0</sub> <span class="mv">v</span><sub><i>j</i></sub> <span class="frac"><span><i>d<sub>j</sub></i></span><span><i>d<sub>j</sub></i><sup>2</sup> + &lambda;</span></span> &lang;<span class="mv">u</span><sub><i>j</i></sub>, <span class="mv">y&#771;</span>&rang;,
          &nbsp;&nbsp;
          <span class="mv">y&#770;</span><sub>&lambda;</sub> = <span class="mv">R</span><sup>&lambda;</sup><span class="mv">y</span>,
          &nbsp;&nbsp;
          df(&lambda;) = tr(<span class="mv">R</span><sup>&lambda;</sup>) = &sum;<sub><i>j</i>=1</sub><sup><i>p</i></sup> <span class="frac"><span><i>d<sub>j</sub></i><sup>2</sup></span><span><i>d<sub>j</sub></i><sup>2</sup> + &lambda;</span></span>
        </div>

        <h4>Bias-Variance Trade-off &amp; Bayesian Prior</h4>
        <div class="math-block">
          Bias(<span class="mv">&beta;&#770;</span><sub>&lambda;</sub>) = &minus;&sum;<sub><i>j</i>=1</sub><sup><i>p</i></sup> <span class="mv">v</span><sub><i>j</i></sub> <span class="frac"><span>&lambda;</span><span><i>d<sub>j</sub></i><sup>2</sup> + &lambda;</span></span> &lang;<span class="mv">v</span><sub><i>j</i></sub>, <span class="mv">&beta;</span>&rang;,
          &nbsp;&nbsp;
          Var(<span class="mv">&beta;&#770;</span><sub>&lambda;</sub>) = &sigma;<sup>2</sup> &sum;<sub><i>j</i>=1</sub><sup><i>p</i></sup> <span class="frac"><span><i>d<sub>j</sub></i><sup>2</sup></span><span>(<i>d<sub>j</sub></i><sup>2</sup> + &lambda;)<sup>2</sup></span></span> <span class="mv">v</span><sub><i>j</i></sub><span class="mv">v</span><sub><i>j</i></sub><sup>&top;</sup> &preceq; &sigma;<sup>2</sup>(<span class="mv">X</span><sup>&top;</sup><span class="mv">X</span>)<sup>&minus;1</sup>
        </div>
        <p>By Theobald&rsquo;s theorem, there exists <span class="m">&lambda; &gt; 0</span> with <span class="m">MSE(<span class="mv">&beta;&#770;</span><sub>&lambda;</sub>) = tr[Var(<span class="mv">&beta;&#770;</span><sub>&lambda;</sub>)] + &parallel;Bias(<span class="mv">&beta;&#770;</span><sub>&lambda;</sub>)&parallel;<sub>2</sub><sup>2</sup> &lt; MSE(<span class="mv">&beta;&#770;</span><sub>OLS</sub>)</span>. Equivalently, Ridge is the posterior mean/mode under <span class="m"><span class="mv">&beta;</span> ~ N(<span class="mv">0</span>, &sigma;<sub>&beta;</sub><sup>2</sup><span class="mv">I</span><sub><i>p</i></sub>)</span> with <span class="m">&lambda; = &sigma;<sub>&epsilon;</sub><sup>2</sup> / &sigma;<sub>&beta;</sub><sup>2</sup></span>.</p>
      </div>

      <!-- 4. LASSO & Sparsity -->
      <div class="theory-card">
        <h3>4. LASSO Regression (&ell;<sub>1</sub> Sparsity &amp; KKT Derivation)</h3>
        <h4>Convexity &amp; Sparsity Geometry</h4>
        <p>Among <span class="m">&parallel;<span class="mv">&beta;</span>&parallel;<sub><i>q</i></sub> = (&sum; |&beta;<sub><i>j</i></sub>|<sup><i>q</i></sup>)<sup>1/<i>q</i></sup></span>, sparsity requires <span class="m"><i>q</i> &le; 1</span> and convexity requires <span class="m"><i>q</i> &ge; 1</span>; hence <span class="m"><i>q</i> = 1</span> (LASSO) is the unique convex sparse penalty (unlike NP-complete &ell;<sub>0</sub> Best Subset Selection):</p>
        <div class="math-block">
          L(<span class="mv">&beta;</span>; &lambda;) = <span class="frac"><span>1</span><span>2<i>n</i></span></span>&parallel;<span class="mv">y&#771;</span> &minus; <span class="mv">X&#771;</span><span class="mv">&beta;</span>&parallel;<sub>2</sub><sup>2</sup> + &lambda;&parallel;<span class="mv">&beta;</span>&parallel;<sub>1</sub>
          &nbsp;&hArr;&nbsp;
          min<sub>&beta;</sub> <span class="frac"><span>1</span><span>2<i>n</i></span></span>&parallel;<span class="mv">y&#771;</span> &minus; <span class="mv">X&#771;</span><span class="mv">&beta;</span>&parallel;<sub>2</sub><sup>2</sup> &nbsp;s.t.&nbsp; &sum;<sub><i>j</i>=1</sub><sup><i>p</i></sup> |&beta;<sub><i>j</i></sub>| &le; <i>t</i>
        </div>

        <h4>Subdifferential &amp; Vector KKT Conditions</h4>
        <p>Because <span class="m">|&beta;<sub><i>j</i></sub>|</span> is non-differentiable at 0, the subdifferential <span class="m">&part;|&beta;<sub><i>j</i></sub>|</span> equals <span class="m">{+1}</span> for <span class="m">&beta;<sub><i>j</i></sub> &gt; 0</span>, <span class="m">{&minus;1}</span> for <span class="m">&beta;<sub><i>j</i></sub> &lt; 0</span>, and <span class="m">[&minus;1, +1]</span> for <span class="m">&beta;<sub><i>j</i></sub> = 0</span>. By Moreau&ndash;Rockafellar, optimality <span class="m"><span class="mv">0</span> &isin; &part;L(<span class="mv">&beta;&#770;</span>; &lambda;)</span> holds iff <span class="m">&exist; <span class="mv">s&#770;</span> &isin; &part;&parallel;<span class="mv">&beta;&#770;</span>&parallel;<sub>1</sub></span>:</p>
        <div class="math-block">
          <span class="frac"><span>1</span><span><i>n</i></span></span><span class="mv">X&#771;</span><sup>&top;</sup>(<span class="mv">y&#771;</span> &minus; <span class="mv">X&#771;</span><span class="mv">&beta;&#770;</span>) = &lambda;<span class="mv">s&#770;</span>
          &nbsp;&rArr;&nbsp;
          { <span class="m">|<span class="mv">x&#771;</span><sub><i>j</i></sub><sup>&top;</sup><span class="mv">r&#770;</span> / <i>n</i>| = &lambda;</span> if <span class="m">&beta;&#770;<sub><i>j</i></sub> &ne; 0</span> (active); &nbsp; <span class="m">|<span class="mv">x&#771;</span><sub><i>j</i></sub><sup>&top;</sup><span class="mv">r&#770;</span> / <i>n</i>| &le; &lambda;</span> if <span class="m">&beta;&#770;<sub><i>j</i></sub> = 0</span> (zeroed) }
        </div>

        <h4>Coordinate Descent &amp; Soft-Thresholding</h4>
        <p>Holding <span class="m"><span class="mv">&beta;</span><sub>&minus;<i>j</i></sub></span> fixed, define partial residual <span class="m"><span class="mv">r</span><sup>(&minus;<i>j</i>)</sup> = <span class="mv">y&#771;</span> &minus; &sum;<sub><i>k</i>&ne;<i>j</i></sub> <span class="mv">x&#771;</span><sub><i>k</i></sub>&beta;<sub><i>k</i></sub></span>, partial covariance <span class="m"><i>z<sub>j</sub></i> = (1/<i>n</i>)<span class="mv">x&#771;</span><sub><i>j</i></sub><sup>&top;</sup><span class="mv">r</span><sup>(&minus;<i>j</i>)</sup></span>, and <span class="m"><i>d<sub>j</sub></i> = (1/<i>n</i>)&parallel;<span class="mv">x&#771;</span><sub><i>j</i></sub>&parallel;<sub>2</sub><sup>2</sup></span> (<i>d<sub>j</sub></i> = 1 when standardized):</p>
        <div class="math-block">
          &beta;&#770;<sub><i>j</i></sub> = <span class="frac"><span>1</span><span><i>d<sub>j</sub></i></span></span> <i>S</i>(<i>z<sub>j</sub></i>, &lambda;),
          &nbsp;&nbsp;
          <i>S</i>(<i>z</i>, &lambda;) = sign(<i>z</i>) max(|<i>z</i>| &minus; &lambda;, 0)
          <br>
          <span style="font-size:0.88em;">Orthonormal (<span class="mv">X&#771;</span><sup>&top;</sup><span class="mv">X&#771;</span>/<i>n</i> = <span class="mv">I</span><sub><i>p</i></sub>): &beta;&#770;<sub><i>j</i></sub><sup>lasso</sup> = <i>S</i>(&beta;&#770;<sub><i>j</i></sub><sup>OLS</sup>, &lambda;) &nbsp;|&nbsp; Null threshold: &lambda;<sub>max</sub> = <span class="frac"><span>1</span><span><i>n</i></span></span>&parallel;<span class="mv">X&#771;</span><sup>&top;</sup><span class="mv">y&#771;</span>&parallel;<sub>&infin;</sub></span>
        </div>
        <p>Effective degrees of freedom equal the active set cardinality <span class="m">df(&lambda;) = |<i>A</i>(&lambda;)|</span>, and LASSO is the posterior mode under a Laplace prior <span class="m">(1/2&tau;)exp(&minus;&parallel;<span class="mv">&beta;</span>&parallel;<sub>1</sub>/&tau;)</span> (<span class="m">&tau; = 1/&lambda;</span>). Under strongly correlated predictors, KKT pinning allows equivalent subsets, causing CV fold selection instability.</p>
      </div>

      <!-- 5. Pipeline Methodology & Evaluation Metrics -->
      <div class="theory-card">
        <h3>5. Pipeline Methodology &amp; Evaluation Metrics</h3>
        <h4>Leak-Free Preprocessing &amp; Splitting</h4>
        <ul>
          <li><strong>Persisted Splits:</strong> 80/20 splits (seed 42) are stored in <code>data/processed/&lt;key&gt;/split.json</code> so OLS, Ridge, and LASSO evaluate identical test rows.</li>
          <li><strong>Domain-Specific Splits:</strong> <code>parkinsons</code> uses <code>GroupShuffleSplit</code> &amp; <code>GroupKFold</code> by <code>subject#</code> (and drops co-target <code>motor_UPDRS</code>); <code>appliances_energy</code> &amp; <code>weather</code> use chronological / date-grouped <code>TimeSeriesSplit</code>; <code>diabetes</code> uses stratified folds.</li>
          <li><strong>Fold-Isolated Transforms:</strong> Median imputation, one-hot encoding (<code>auto_mpg.origin</code>), and <code>StandardScaler</code> are fit strictly inside training data (and separately inside every CV fold).</li>
        </ul>

        <h4>Evaluation Metrics &amp; LPM Treatment</h4>
        <div class="math-block">
          RMSE = &radic;<span style="text-decoration:overline;">(1/<i>n</i>)&sum;(<i>y<sub>i</sub></i> &minus; <i>y&#770;<sub>i</sub></i>)<sup>2</sup></span>,
          &nbsp;&nbsp;
          MAE = <span class="frac"><span>1</span><span><i>n</i></span></span>&sum;|<i>y<sub>i</sub></i> &minus; <i>y&#770;<sub>i</sub></i>|,
          &nbsp;&nbsp;
          <i>R</i><sup>2</sup> = 1 &minus; <span class="frac"><span>&sum;(<i>y<sub>i</sub></i> &minus; <i>y&#770;<sub>i</sub></i>)<sup>2</sup></span><span>&sum;(<i>y<sub>i</sub></i> &minus; <i>y&#772;</i>)<sup>2</sup></span></span>
        </div>
        <ul>
          <li><strong>Unit Discipline:</strong> RMSE and MAE inherit target units and cannot be compared across datasets; cross-dataset comparison uses dimensionless <span class="m"><i>R</i><sup>2</sup></span>.</li>
          <li><strong>Linear Probability Model (<code>diabetes</code>):</strong> Fitting linear regression on <span class="m"><i>y</i> &isin; {0, 1}</span> estimates <span class="m">P(<i>y</i> = 1|<span class="mv">x</span>)</span>, evaluated with Accuracy/F1 (threshold 0.50), ROC AUC, and % predictions outside <span class="m">[0, 1]</span>.</li>
        </ul>
      </div>
    </div>
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
  sampleSel.innerHTML = '<option value="">Train Median</option>';
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
    card.className = 'pred-card';

    let pills = '';
    if (ds.is_binary_classification) {
      const isPos = yhat >= 0.5;
      pills += `<span class="pill ${isPos ? 'pill-dark' : ''}">${isPos ? 'Positive (1)' : 'Negative (0)'}</span> `;
      if (yhat < 0.0 || yhat > 1.0) pills += `<span class="pill">Out of [0,1]</span>`;
    } else if (ds.is_ordinal_target) {
      pills += `<span class="pill">Score ~ ${Math.round(yhat)}</span>`;
    }
    if (observed !== null) {
      const err = yhat - observed;
      pills += `<span class="pill">Obs ${observed.toPrecision(4)} (&Delta; ${err >= 0 ? '+' : ''}${err.toFixed(3)})</span>`;
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
  let tbl = '<table><thead><tr><th>Feature</th><th class="num">x<sub>j</sub></th><th class="num">OLS (&beta;&#770;<sub>j</sub>x<sub>j</sub>)</th><th class="num">Ridge</th><th class="num">Lasso</th></tr></thead><tbody>';
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
            best_r2_values.append(r2_map[best_model])
            win_counts[best_model] += 1

            master_rows.append(
                {
                    "key": dataset_key,
                    "name": cfg.name,
                    "uci_id": cfg.uci_id,
                    "uci_url": _uci_url(cfg.uci_id),
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
                    "uci_url": _uci_url(cfg.uci_id),
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
        "wins_summary": f"{win_counts['ols']} / {win_counts['ridge']} / {win_counts['lasso']}",
        "total_plots": total_plots,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "index.html"
    html = GRAND_TEMPLATE.render(
        github_repo_url=GITHUB_REPO_URL,
        master_rows=master_rows,
        datasets_detail=datasets_detail,
        global_stats=global_stats,
        predictor_json=json.dumps(predictor_payload),
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path
