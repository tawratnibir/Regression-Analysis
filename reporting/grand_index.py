"""Generate a lightweight, responsive academic research dashboard (reports/index.html)
with KaTeX equations, lazy-loaded per-dataset bundles (reports/assets/data/<ds>.js),
lazy-loaded plot assets (reports/assets/plots/), downloadable CSV/JSON exports,
and an interactive OLS/Ridge/Lasso prediction playground.
"""

import json
from pathlib import Path
import shutil
from typing import Any, Dict, List

from jinja2 import Environment, select_autoescape
import numpy as np
import pandas as pd

from config import DATASETS, MODEL_KEYS, REPORTS_DIR, RESULTS_DIR
from preprocessing.preprocess import load_and_prepare_dataset
from reporting.generate_report import (
    PLOT_ORDER,
    _build_diagnostic_sections_and_narrative,
    _fmt,
    build_figure_captions,
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
                    "label": f"Test row #{int(prepared.test_raw_indices[pos])} (observed = {prepared.y_test[pos]:.4g} {cfg.target_units})",
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


def _sync_plots_and_build_dataset_bundle(dataset_key: str) -> Dict[str, Any]:
    """Copy plots to reports/assets/plots/ and build lazy-loaded JSON payload for one dataset."""
    cfg = DATASETS[dataset_key]
    assets_plots_root = REPORTS_DIR / "assets" / "plots"
    models_data: Dict[str, Any] = {}

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

        # Clean NaN -> None so json.dumps produces strict RFC-8259 JSON (no bare NaN!)
        coef_clean = coef_df.replace({np.nan: None})
        coef_csv_text = coef_df.to_csv(index=False)

        diag_sections, interpretations, limitations = _build_diagnostic_sections_and_narrative(
            model_key, dataset_key, metrics, coef_df, diagnostics, metadata
        )
        captions = build_figure_captions(model_key, metrics, diagnostics)

        dst_plot_dir = assets_plots_root / model_key / dataset_key
        dst_plot_dir.mkdir(parents=True, exist_ok=True)
        plot_list: List[Dict[str, str]] = []
        for fname, title in PLOT_ORDER:
            src_png = res_dir / "plots" / fname
            if src_png.exists():
                dst_png = dst_plot_dir / fname
                shutil.copy2(src_png, dst_png)
                plot_list.append(
                    {
                        "title": title,
                        "caption": captions.get(fname, ""),
                        "src": f"assets/plots/{model_key}/{dataset_key}/{fname}",
                    }
                )

        models_data[model_key] = {
            "metrics": metrics,
            "diagnostics": diagnostics,
            "metadata": metadata,
            "coef_rows": coef_clean.to_dict(orient="records"),
            "coef_csv": coef_csv_text,
            "diag_sections": [[g, items] for g, items in diag_sections],
            "interpretations": interpretations,
            "limitations": limitations,
            "plots": plot_list,
        }

    predictor = _build_predictor_payload(dataset_key)
    prep = models_data["ols"]["metadata"]["preprocessing"] if "ols" in models_data else {}

    return {
        "key": dataset_key,
        "uci_id": cfg.uci_id,
        "uci_url": _uci_url(cfg.uci_id),
        "name": cfg.name,
        "description": cfg.description,
        "target": cfg.target,
        "target_units": cfg.target_units,
        "is_binary_classification": cfg.is_binary_classification,
        "is_ordinal_target": cfg.is_ordinal_target,
        "notes": cfg.notes,
        "preprocessing": prep,
        "models": models_data,
        "predictor": predictor,
    }


DASHBOARD_TEMPLATE = JINJA_ENV.from_string(r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UCI Regression Analysis — OLS, Ridge &amp; Lasso Benchmark</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" crossorigin="anonymous"></script>
  <style>
    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --subtle: #f1f5f9;
      --border: #e2e8f0;
      --border-strong: #cbd5e1;
      --navy: #0f172a;
      --navy-2: #1e3a5f;
      --accent: #1d4ed8;
      --accent-hover: #1e40af;
      --accent-soft: #eff6ff;
      --text: #1e293b;
      --muted: #475569;
      --green: #15803d;
      --green-soft: #f0fdf4;
      --amber: #b45309;
      --amber-soft: #fffbeb;
      --radius: 8px;
      --shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      line-height: 1.55;
      -webkit-font-smoothing: antialiased;
    }
    a:focus-visible, button:focus-visible, select:focus-visible, input:focus-visible, summary:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }

    /* Sticky Top Navigation */
    .topbar {
      position: sticky;
      top: 0;
      z-index: 60;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
    }
    .topbar-inner {
      max-width: 1320px;
      margin: 0 auto;
      padding: 10px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      flex-wrap: wrap;
    }
    .brand {
      font-size: 0.98rem;
      font-weight: 700;
      color: var(--navy);
      text-decoration: none;
      letter-spacing: -0.01em;
    }
    .nav {
      display: flex;
      align-items: center;
      gap: 4px;
      flex-wrap: wrap;
    }
    .nav a {
      color: var(--muted);
      text-decoration: none;
      font-size: 0.82rem;
      font-weight: 600;
      padding: 6px 10px;
      border-radius: 6px;
    }
    .nav a:hover { background: var(--subtle); color: var(--navy); }
    .nav a.nav-repo { color: var(--accent); border: 1px solid #bfdbfe; background: var(--accent-soft); }

    .container { max-width: 1320px; margin: 0 auto; padding: 24px 24px 56px; }

    /* Sections & Cards */
    section.sec {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 22px 24px;
      margin-bottom: 22px;
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
    .sec-head h1, .sec-head h2 {
      margin: 0;
      color: var(--navy);
      letter-spacing: -0.01em;
    }
    .sec-head h1 { font-size: 1.4rem; }
    .sec-head h2 { font-size: 1.15rem; }
    .lead { color: var(--muted); font-size: 0.92rem; margin: 0 0 14px; }

    /* KPI Strip */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
      gap: 12px;
      margin: 12px 0 16px;
    }
    .kpi-card {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 12px 14px;
    }
    .kpi-lbl { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
    .kpi-val { font-size: 1.32rem; font-weight: 700; color: var(--navy); margin-top: 2px; font-variant-numeric: tabular-nums; }

    /* Tables */
    .table-wrap { overflow-x: auto; }
    .scroll-table { max-height: 360px; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th, td { border-bottom: 1px solid var(--border); padding: 8px 10px; text-align: left; vertical-align: middle; }
    th {
      background: var(--subtle);
      color: var(--navy-2);
      font-size: 0.74rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    tr.clickable-row { cursor: pointer; }
    tr.clickable-row:hover { background: var(--accent-soft); }
    td.best-cell { background: var(--green-soft); color: var(--green); font-weight: 700; }
    tr.intercept-row { background: var(--accent-soft); font-weight: 600; }
    tr.zero-row { color: #94a3b8; }

    /* Equations (responsive KaTeX) */
    .math-box {
      background: var(--subtle);
      border-left: 4px solid var(--accent);
      border-radius: 6px;
      padding: 12px 16px;
      margin: 12px 0;
      overflow-x: auto;
      max-width: 100%;
    }
    .katex-display { overflow-x: auto; overflow-y: hidden; padding: 4px 0; margin: 0.4em 0; }
    .symbol-note { font-size: 0.84rem; color: var(--muted); margin: 6px 0 0; }

    /* Controls & Badges */
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    select.ctrl, input.ctrl {
      padding: 7px 10px;
      border: 1px solid var(--border-strong);
      border-radius: 6px;
      font-size: 0.84rem;
      background: var(--surface);
      color: var(--text);
    }
    .btn {
      background: var(--accent);
      color: #ffffff;
      border: 1px solid var(--accent);
      border-radius: 6px;
      padding: 7px 12px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
    }
    .btn:hover { background: var(--accent-hover); }
    .btn-ghost {
      background: var(--surface);
      color: var(--navy);
      border: 1px solid var(--border-strong);
    }
    .btn-ghost:hover { background: var(--subtle); }
    .pill-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
    .pill {
      display: inline-flex;
      align-items: center;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 3px 9px;
      border-radius: 999px;
      background: var(--subtle);
      color: var(--navy);
      border: 1px solid var(--border);
    }
    .pill-accent { background: var(--accent-soft); color: var(--accent); border-color: #bfdbfe; }
    .pill-warn { background: var(--amber-soft); color: var(--amber); border-color: #fde68a; }
    .pill-ok { background: var(--green-soft); color: var(--green); border-color: #bbf7d0; }

    /* Plots */
    .plots-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 14px;
      margin-top: 12px;
    }
    .plot-card {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      background: #ffffff;
    }
    .plot-card img { width: 100%; height: auto; display: block; border-radius: 4px; }
    .plot-title { font-weight: 600; font-size: 0.86rem; color: var(--navy); margin-top: 8px; }
    .plot-caption { font-size: 0.8rem; color: var(--muted); margin-top: 3px; }

    /* Predictor */
    .input-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(195px, 1fr));
      gap: 8px;
      max-height: 290px;
      overflow-y: auto;
      padding: 2px;
      margin: 10px 0;
    }
    .field-card { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; }
    .field-top { display: flex; justify-content: space-between; gap: 6px; margin-bottom: 4px; }
    .field-top label { font-size: 0.77rem; font-weight: 600; word-break: break-word; }
    .field-range { font-size: 0.69rem; color: var(--muted); white-space: nowrap; }
    .field-card input, .field-card select {
      width: 100%; padding: 5px 7px; border: 1px solid var(--border-strong); border-radius: 4px; font-size: 0.83rem;
    }
    .pred-cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .pred-card {
      background: var(--bg);
      border: 1px solid var(--border);
      border-top: 3px solid var(--accent);
      border-radius: 6px;
      padding: 12px 14px;
    }
    .pred-val { font-size: 1.4rem; font-weight: 700; color: var(--navy); margin: 4px 0; font-variant-numeric: tabular-nums; }

    details {
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
      background: var(--bg);
    }
    summary { cursor: pointer; font-weight: 600; font-size: 0.85rem; color: var(--navy); }
    .two-col { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; margin-top: 12px; }
    code { background: var(--subtle); padding: 1px 5px; border-radius: 4px; font-size: 0.84em; }
    a.ext { color: var(--accent); text-decoration: none; font-weight: 600; }
    a.ext:hover { text-decoration: underline; }
    .notice {
      background: var(--accent-soft);
      border-left: 4px solid var(--accent);
      padding: 10px 14px;
      border-radius: 4px;
      font-size: 0.86rem;
      margin: 10px 0;
    }
    @media (max-width: 640px) {
      .container { padding: 14px 12px 40px; }
      section.sec { padding: 16px 14px; }
      .plots-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<header class="topbar">
  <div class="topbar-inner">
    <a class="brand" href="#overview">UCI Regression Research Dashboard</a>
    <nav class="nav" aria-label="Primary sections">
      <a href="#overview">1. Overview</a>
      <a href="#datasets">2. Datasets</a>
      <a href="#ols">3. OLS</a>
      <a href="#ridge">4. Ridge</a>
      <a href="#lasso">5. Lasso</a>
      <a href="#comparison">6. Comparison</a>
      <a href="#methodology">7. Methodology &amp; References</a>
      <a class="nav-repo" href="{{ github_repo_url }}" target="_blank" rel="noopener noreferrer">GitHub &nearr;</a>
    </nav>
  </div>
</header>

<main class="container">
  <!-- 1. OVERVIEW -->
  <section class="sec" id="overview">
    <div class="sec-head">
      <h1>1. Overview</h1>
      <div class="toolbar">
        <a class="ext" href="ols/summary.html">OLS Summary &nearr;</a>
        <a class="ext" href="ridge/summary.html">Ridge Summary &nearr;</a>
        <a class="ext" href="lasso/summary.html">Lasso Summary &nearr;</a>
      </div>
    </div>
    <p class="lead">
      Empirical comparison of Ordinary Least Squares (OLS), Ridge Regression (\(\ell_2\)), and Lasso Regression (\(\ell_1\)) across 10 UCI datasets (30 experiments) using persisted 80/20 splits and training-only preprocessing and hyperparameter tuning.
    </p>

    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-lbl">Datasets</div><div class="kpi-val">{{ master_rows | length }} UCI</div></div>
      <div class="kpi-card"><div class="kpi-lbl">Model Runs</div><div class="kpi-val">30 (3 &times; 10)</div></div>
      <div class="kpi-card"><div class="kpi-lbl">Median Best Test R&sup2;</div><div class="kpi-val">{{ "%.4f"|format(global_stats.median_best_r2) }}</div></div>
      <div class="kpi-card"><div class="kpi-lbl">Best Test R&sup2; Wins (OLS/Ridge/Lasso)</div><div class="kpi-val">{{ global_stats.wins_summary }}</div></div>
      <div class="kpi-card"><div class="kpi-lbl">Diagnostic Figures</div><div class="kpi-val">{{ global_stats.total_plots }}</div></div>
    </div>

    <div class="notice">
      <strong>Unit Discipline:</strong> Raw RMSE and MAE inherit each dataset&rsquo;s target units and must not be compared across different datasets. Compare algorithms within the same dataset row, or use dimensionless Test \(R^2\) across benchmarks.
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>UCI Repo</th>
            <th>Target (Units)</th>
            <th>Split / CV</th>
            <th class="num">Train / Test (p)</th>
            <th class="num">OLS Test R&sup2; (RMSE)</th>
            <th class="num">Ridge Test R&sup2; (RMSE, &alpha;*)</th>
            <th class="num">Lasso Test R&sup2; (RMSE, &alpha;*, Active)</th>
          </tr>
        </thead>
        <tbody>
          {% for row in master_rows %}
          <tr class="clickable-row" tabindex="0" onclick="jumpToDataset('{{ row.key }}')" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();jumpToDataset('{{ row.key }}');}">
            <td><strong>{{ row.name }}</strong></td>
            <td onclick="event.stopPropagation();"><a class="ext" href="{{ row.uci_url }}" target="_blank" rel="noopener noreferrer">UCI {{ row.uci_id }} &nearr;</a></td>
            <td><code>{{ row.target }}</code> ({{ row.target_units }})</td>
            <td><code>{{ row.split_strategy }}</code> / {{ row.cv_strategy }}</td>
            <td class="num">{{ row.n_train }} / {{ row.n_test }} ({{ row.n_features }})</td>
            <td class="num {% if row.best_model == 'ols' %}best-cell{% endif %}">{{ "%.4f"|format(row.ols_r2) }}<br><small>{{ "%.4g"|format(row.ols_rmse) }}</small></td>
            <td class="num {% if row.best_model == 'ridge' %}best-cell{% endif %}">{{ "%.4f"|format(row.ridge_r2) }}<br><small>{{ "%.4g"|format(row.ridge_rmse) }} (&alpha;={{ "%.3g"|format(row.ridge_alpha) }})</small></td>
            <td class="num {% if row.best_model == 'lasso' %}best-cell{% endif %}">{{ "%.4f"|format(row.lasso_r2) }}<br><small>{{ "%.4g"|format(row.lasso_rmse) }} (&alpha;={{ "%.3g"|format(row.lasso_alpha) }}, {{ row.lasso_nz }}/{{ row.n_features }})</small></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </section>

  <!-- 2. DATASETS -->
  <section class="sec" id="datasets">
    <div class="sec-head">
      <h2>2. Datasets &amp; Experimental Splits</h2>
    </div>
    <p class="lead">
      All transformations (median imputation, one-hot encoding, and predictor standardization) are fit strictly on training observations and separately within each cross-validation fold.
    </p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>UCI Link</th>
            <th>Target (Units)</th>
            <th class="num">Clean Rows</th>
            <th class="num">Predictors</th>
            <th>Split &amp; CV Strategy</th>
            <th>Domain Caveats &amp; Exclusions</th>
          </tr>
        </thead>
        <tbody>
          {% for ds in dataset_inventory %}
          <tr>
            <td><strong>{{ ds.name }}</strong><br><small><code>{{ ds.key }}</code></small></td>
            <td><a class="ext" href="{{ ds.uci_url }}" target="_blank" rel="noopener noreferrer">UCI {{ ds.uci_id }} &nearr;</a></td>
            <td><code>{{ ds.target }}</code> ({{ ds.target_units }})</td>
            <td class="num">{{ ds.n_clean }}</td>
            <td class="num">{{ ds.n_encoded }}</td>
            <td><code>{{ ds.split_strategy }}</code><br><small>{{ ds.cv_strategy }}</small></td>
            <td>{{ ds.caveat_summary }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </section>

  <!-- 3. ORDINARY LEAST SQUARES -->
  <section class="sec" id="ols">
    <div class="sec-head">
      <h2>3. Ordinary Least Squares (OLS)</h2>
      <div class="toolbar">
        <label for="sel-ols"><strong>Dataset:</strong></label>
        <select id="sel-ols" class="ctrl" onchange="selectGlobalDataset(this.value)">
          {% for row in master_rows %}<option value="{{ row.key }}">{{ row.name }}</option>{% endfor %}
        </select>
        <button type="button" class="btn btn-ghost" onclick="downloadCoefficientsCSV('ols')">Download Coefficients CSV</button>
        <button type="button" class="btn btn-ghost" onclick="downloadRunJSON('ols')">Download JSON</button>
      </div>
    </div>
    <p class="lead">
      OLS fits an unpenalized linear hyperplane minimizing the training residual sum of squares via SVD-based least squares (<code>numpy.linalg.lstsq</code>) on standardized predictors \(\tilde{\mathbf{X}}\) with an unscaled intercept column \(\mathbf{1}\).
    </p>
    <div class="math-box">
      $$\min_{\boldsymbol{\beta} \in \mathbb{R}^{p+1}} \|\mathbf{y} - \tilde{\mathbf{X}}_{\mathrm{aug}}\boldsymbol{\beta}\|_2^2 = \mathbf{e}^\top\mathbf{e}, \qquad \tilde{\mathbf{X}}_{\mathrm{aug}} = [\mathbf{1},\, \tilde{\mathbf{X}}] \in \mathbb{R}^{n \times k}, \quad k = p + 1 \tag{1}$$
      $$(\tilde{\mathbf{X}}_{\mathrm{aug}}^\top \tilde{\mathbf{X}}_{\mathrm{aug}})\hat{\boldsymbol{\beta}} = \tilde{\mathbf{X}}_{\mathrm{aug}}^\top \mathbf{y} \quad \Longrightarrow \quad \tilde{\mathbf{X}}_{\mathrm{aug}}^\top \mathbf{e} = \mathbf{0} \tag{2}$$
      <p class="symbol-note">
        \(\mathbf{y} \in \mathbb{R}^n\): training targets; \(\mathbf{e} = \mathbf{y} - \tilde{\mathbf{X}}_{\mathrm{aug}}\hat{\boldsymbol{\beta}}\): observable sample residuals (distinct from unobservable disturbances \(\boldsymbol{\varepsilon}\)). When \(\operatorname{rank}(\tilde{\mathbf{X}}_{\mathrm{aug}}) < k\), SVD returns the minimum-norm pseudoinverse solution \(\hat{\boldsymbol{\beta}} = \tilde{\mathbf{X}}_{\mathrm{aug}}^{+}\mathbf{y}\).
      </p>
    </div>
    <details>
      <summary>Expand Matrix Derivation: Residual Maker, Hat Matrix &amp; Gauss-Markov Assumptions</summary>
      <div class="math-box">
        $$\mathbf{H} = \tilde{\mathbf{X}}_{\mathrm{aug}}(\tilde{\mathbf{X}}_{\mathrm{aug}}^\top\tilde{\mathbf{X}}_{\mathrm{aug}})^{-1}\tilde{\mathbf{X}}_{\mathrm{aug}}^\top, \quad \mathbf{M} = \mathbf{I}_n - \mathbf{H}, \quad \hat{\mathbf{y}} = \mathbf{H}\mathbf{y}, \quad \mathbf{e} = \mathbf{M}\mathbf{y}$$
        $$\mathbb{E}[\boldsymbol{\varepsilon}\mid \mathbf{X}] = \mathbf{0},\ \mathbb{E}[\boldsymbol{\varepsilon}\boldsymbol{\varepsilon}^\top \mid \mathbf{X}] = \sigma^2 \mathbf{I}_n \implies \operatorname{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}, \quad \hat{\sigma}^2 = \frac{\mathbf{e}^\top\mathbf{e}}{n - k}$$
      </div>
      <p class="symbol-note">Under homoskedastic, uncorrelated errors and full rank, OLS is BLUE (Gauss-Markov Theorem). Under heteroskedasticity \(\boldsymbol{\Omega} = \operatorname{diag}(\sigma_1^2,\dots,\sigma_n^2)\), classical standard errors require White&rsquo;s sandwich estimator \((\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \mathbf{e}\mathbf{e}^\top \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\).</p>
    </details>
    <div id="mount-ols"></div>
  </section>

  <!-- 4. RIDGE REGRESSION -->
  <section class="sec" id="ridge">
    <div class="sec-head">
      <h2>4. Ridge Regression (\(\ell_2\) Regularization)</h2>
      <div class="toolbar">
        <label for="sel-ridge"><strong>Dataset:</strong></label>
        <select id="sel-ridge" class="ctrl" onchange="selectGlobalDataset(this.value)">
          {% for row in master_rows %}<option value="{{ row.key }}">{{ row.name }}</option>{% endfor %}
        </select>
        <button type="button" class="btn btn-ghost" onclick="downloadCoefficientsCSV('ridge')">Download Coefficients CSV</button>
        <button type="button" class="btn btn-ghost" onclick="downloadRunJSON('ridge')">Download JSON</button>
      </div>
    </div>
    <p class="lead">
      Ridge regression adds a squared \(\ell_2\) penalty on standardized slopes (leaving \(\beta_0\) unpenalized) to shrink variance and stabilize ill-conditioned \(\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}}\). <strong>Ridge shrinks all coefficients smoothly toward zero and does not perform automatic feature elimination.</strong>
    </p>
    <div class="math-box">
      $$\min_{\beta_0 \in \mathbb{R},\, \boldsymbol{\beta} \in \mathbb{R}^p} \|\mathbf{y} - \beta_0\mathbf{1} - \tilde{\mathbf{X}}\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_2^2 \quad (\lambda \equiv \alpha \ge 0\text{ in }\texttt{scikit-learn}) \tag{3}$$
      $$\hat{\boldsymbol{\beta}}_\alpha = (\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}} + \alpha \mathbf{I}_p)^{-1}\tilde{\mathbf{X}}^\top\tilde{\mathbf{y}}, \qquad \hat{\beta}_0 = \bar{y}, \qquad \kappa(\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}} + \alpha\mathbf{I}_p) = \frac{d_{\max}^2 + \alpha}{d_{\min}^2 + \alpha} \tag{4}$$
      <p class="symbol-note">
        \(\tilde{\mathbf{y}} = \mathbf{y} - \bar{y}\mathbf{1}\); \(d_1 \ge \dots \ge d_p \ge 0\) are singular values of \(\tilde{\mathbf{X}}\). Note \(\texttt{sklearn.linear\_model.Ridge}\) omits the \(\frac{1}{2n}\) factor, matching Eq. (3) directly.
      </p>
    </div>
    <details>
      <summary>Expand SVD Shrinkage, Effective Degrees of Freedom &amp; Bias-Variance Trade-off</summary>
      <div class="math-box">
        $$\tilde{\mathbf{X}} = \mathbf{U}\mathbf{D}\mathbf{V}^\top \implies \hat{\boldsymbol{\beta}}_\alpha = \sum_{d_j > 0} \mathbf{v}_j \frac{d_j}{d_j^2 + \alpha}\mathbf{u}_j^\top\tilde{\mathbf{y}}, \qquad \operatorname{df}(\alpha) = \operatorname{tr}(\mathbf{R}^\alpha) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \alpha}$$
        $$\operatorname{Bias}(\hat{\boldsymbol{\beta}}_\alpha) = -\sum_{j=1}^p \mathbf{v}_j \frac{\alpha}{d_j^2 + \alpha}\langle\mathbf{v}_j,\boldsymbol{\beta}\rangle, \qquad \operatorname{Var}(\hat{\boldsymbol{\beta}}_\alpha) = \sigma^2\sum_{j=1}^p \frac{d_j^2}{(d_j^2+\alpha)^2}\mathbf{v}_j\mathbf{v}_j^\top \preceq \sigma^2(\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}})^{-1}$$
      </div>
      <p class="symbol-note">Unstandardized slopes are recovered via \(\hat{\beta}_{\mathrm{unstd}, j} = \hat{\beta}_{\alpha, j} / s_j\) and \(\hat{\beta}_{0,\mathrm{unstd}} = \bar{y} - \sum_{j=1}^p \hat{\beta}_{\mathrm{unstd}, j}\bar{x}_j\). By Theobald&rsquo;s theorem, \(\exists \alpha &gt; 0\) with strictly lower parameter MSE than OLS.</p>
    </details>
    <div id="mount-ridge"></div>
  </section>

  <!-- 5. LASSO REGRESSION -->
  <section class="sec" id="lasso">
    <div class="sec-head">
      <h2>5. Lasso Regression (\(\ell_1\) Regularization)</h2>
      <div class="toolbar">
        <label for="sel-lasso"><strong>Dataset:</strong></label>
        <select id="sel-lasso" class="ctrl" onchange="selectGlobalDataset(this.value)">
          {% for row in master_rows %}<option value="{{ row.key }}">{{ row.name }}</option>{% endfor %}
        </select>
        <button type="button" class="btn btn-ghost" onclick="downloadCoefficientsCSV('lasso')">Download Coefficients CSV</button>
        <button type="button" class="btn btn-ghost" onclick="downloadRunJSON('lasso')">Download JSON</button>
      </div>
    </div>
    <p class="lead">
      Lasso penalizes the \(\ell_1\) norm \(\|\boldsymbol{\beta}\|_1 = \sum_{j=1}^p |\beta_j|\), yielding a convex objective that sets weak or redundant predictors exactly to zero via soft-thresholding.
    </p>
    <div class="math-box">
      $$\min_{\beta_0 \in \mathbb{R},\, \boldsymbol{\beta} \in \mathbb{R}^p} \frac{1}{2n}\|\mathbf{y} - \beta_0\mathbf{1} - \tilde{\mathbf{X}}\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_1 \quad (\lambda \equiv \alpha \ge 0\text{ in }\texttt{scikit-learn}) \tag{5}$$
      $$\hat{\beta}_j = \frac{1}{d_j}\mathcal{S}\!\left(\frac{1}{n}\tilde{\mathbf{x}}_j^\top \mathbf{r}^{(-j)},\, \alpha\right), \quad \mathcal{S}(z, \alpha) = \operatorname{sign}(z)\max(|z| - \alpha,\, 0), \quad \alpha_{\max} = \frac{1}{n}\|\tilde{\mathbf{X}}^\top\tilde{\mathbf{y}}\|_\infty \tag{6}$$
      <p class="symbol-note">
        Here \(d_j = \frac{1}{n}\|\tilde{\mathbf{x}}_j\|_2^2 = 1\) on standardized training columns, \(\mathbf{r}^{(-j)} = \tilde{\mathbf{y}} - \sum_{k \ne j}\tilde{\mathbf{x}}_k \hat{\beta}_k\) is the partial residual, and \(\texttt{sklearn.linear\_model.Lasso}\) includes the \(\frac{1}{2n}\) scaling matching Eq. (5).
      </p>
    </div>
    <details>
      <summary>Expand Subdifferential KKT Optimality Conditions &amp; Feature Selection</summary>
      <div class="math-box">
        $$\frac{1}{n}\tilde{\mathbf{X}}^\top(\tilde{\mathbf{y}} - \tilde{\mathbf{X}}\hat{\boldsymbol{\beta}}) = \alpha \hat{\mathbf{s}}, \qquad \hat{s}_j \in \partial|\hat{\beta}_j| = \begin{cases} \{+1\}, & \hat{\beta}_j > 0 \\ [-1, +1], & \hat{\beta}_j = 0 \\ \{-1\}, & \hat{\beta}_j < 0 \end{cases} \tag{7}$$
        $$\left|\frac{1}{n}\tilde{\mathbf{x}}_j^\top \hat{\mathbf{r}}\right| = \alpha \text{ for } \hat{\beta}_j \ne 0, \qquad \left|\frac{1}{n}\tilde{\mathbf{x}}_j^\top \hat{\mathbf{r}}\right| \le \alpha \text{ for } \hat{\beta}_j = 0, \qquad \operatorname{df}(\alpha) = |A(\alpha)|$$
      </div>
      <p class="symbol-note">When predictors are highly correlated, multiple subsets can satisfy the KKT correlation bound \(\alpha\), explaining active-set selection variability across CV folds.</p>
    </details>
    <div id="mount-lasso"></div>
  </section>

  <!-- 6. ALGORITHM COMPARISON + INTERACTIVE PREDICTOR -->
  <section class="sec" id="comparison">
    <div class="sec-head">
      <h2>6. Algorithm Comparison &amp; Interactive Predictor</h2>
      <div class="toolbar">
        <label for="sel-comp"><strong>Dataset:</strong></label>
        <select id="sel-comp" class="ctrl" onchange="selectGlobalDataset(this.value)">
          {% for row in master_rows %}<option value="{{ row.key }}">{{ row.name }}</option>{% endfor %}
        </select>
      </div>
    </div>
    <div id="mount-comparison"></div>

    <h3 style="margin-top:20px; color:var(--navy);">Interactive Prediction Playground</h3>
    <p class="lead">Computes live predictions \(\hat{y} = \hat{\beta}_{0,\mathrm{unstd}} + \sum_j \hat{\beta}_{\mathrm{unstd}, j} x_j\) using the fitted unstandardized coefficients for OLS, Ridge, and Lasso.</p>
    <div class="toolbar">
      <label for="pred-sample-select"><strong>Sample:</strong></label>
      <select id="pred-sample-select" class="ctrl" onchange="loadSelectedSample()"></select>
      <button type="button" class="btn" onclick="computePredictions()">Predict</button>
      <button type="button" class="btn btn-ghost" onclick="promptInputsSequential()">Prompt Inputs</button>
      <button type="button" class="btn btn-ghost" onclick="resetPredictorDefaults()">Reset Medians</button>
      <input type="text" id="pred-filter" class="ctrl" placeholder="Filter predictors..." oninput="filterPredictorFields(this.value)">
    </div>
    <div id="pred-inputs-grid" class="input-grid"></div>
    <div class="pred-cards" id="pred-results-grid"></div>
    <details>
      <summary>Per-Feature Contributions (\(\hat{\beta}_{\mathrm{unstd}, j} x_j\))</summary>
      <div id="pred-contrib-table" class="scroll-table" style="margin-top:8px;"></div>
    </details>
  </section>

  <!-- 7. METHODOLOGY AND REFERENCES -->
  <section class="sec" id="methodology">
    <div class="sec-head">
      <h2>7. Methodology &amp; References</h2>
    </div>

    <div class="two-col">
      <div>
        <h3 style="margin-top:0; color:var(--navy);">Experimental Protocol</h3>
        <ul>
          <li><strong>Persisted Splits:</strong> 80/20 splits (seed 42) are stored in <code>data/processed/&lt;dataset&gt;/split.json</code> so OLS, Ridge, and Lasso evaluate identical test observations.</li>
          <li><strong>Zero Leakage:</strong> Imputation, one-hot encoding, and standardization are fit only on training data and separately within each CV fold.</li>
          <li><strong>Penalty Parameter Mapping:</strong> Mathematical \(\lambda\) corresponds to scikit-learn&rsquo;s <code>alpha</code> argument. Note that <code>Ridge</code> uses unnormalized SSE \(\|\mathbf{y}-\hat{\mathbf{y}}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_2^2\), whereas <code>Lasso</code> uses \(\frac{1}{2n}\|\mathbf{y}-\hat{\mathbf{y}}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_1\).</li>
          <li><strong>Evaluation Metrics:</strong> \(\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}\), \(\mathrm{MAE} = \frac{1}{n}\sum|y_i - \hat{y}_i|\), \(R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}\). For <code>diabetes</code> (LPM), Accuracy/Precision/Recall/F1 (0.50 cutoff), ROC AUC, and out-of-\([0,1]\) rates are reported.</li>
        </ul>
      </div>
      <div>
        <h3 style="margin-top:0; color:var(--navy);">References</h3>
        <ol>
          <li><em>OLS in Matrix Form</em> (Course lecture notes on matrix OLS, normal equations, residual maker/hat matrices, Frisch-Waugh-Lovell theorem, Gauss-Markov theorem, and White robust standard errors).</li>
          <li>Aldo Solari, <em>Ridge Regression</em>, Statistical Learning, CLAMSES &mdash; University of Milano-Bicocca.</li>
          <li>Aldo Solari, <em>Sparse Modeling: Best Subset and the Lasso</em>, Statistical Learning, CLAMSES &mdash; University of Milano-Bicocca.</li>
          <li><em>Derivation of the General LASSO Estimator</em> (Statistical Learning &amp; Convex Optimization notes on Lagrangian duality, subdifferentials, KKT conditions, and coordinate-wise soft-thresholding).</li>
        </ol>
      </div>
    </div>
  </section>
</main>

<script>
const MASTER_SUMMARY = {{ master_json | safe }};
window.__DATASET_STORE__ = window.__DATASET_STORE__ || {};
let currentDatasetKey = "{{ master_rows[0].key }}";
const pendingLoads = {};

function triggerMathRender(el) {
  if (typeof renderMathInElement === "function") {
    renderMathInElement(el || document.body, {
      delimiters: [
        {left: "$$", right: "$$", display: true},
        {left: "\\(", right: "\\)", display: false}
      ],
      throwOnError: false
    });
  }
}

function loadDatasetBundle(dsKey) {
  if (window.__DATASET_STORE__[dsKey]) {
    return Promise.resolve(window.__DATASET_STORE__[dsKey]);
  }
  if (pendingLoads[dsKey]) return pendingLoads[dsKey];
  pendingLoads[dsKey] = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "assets/data/" + encodeURIComponent(dsKey) + ".js";
    s.async = true;
    s.onload = () => {
      if (window.__DATASET_STORE__[dsKey]) resolve(window.__DATASET_STORE__[dsKey]);
      else reject(new Error("Missing bundle payload for " + dsKey));
    };
    s.onerror = () => reject(new Error("Failed to load assets/data/" + dsKey + ".js"));
    document.head.appendChild(s);
  });
  return pendingLoads[dsKey];
}

function escHtml(val) {
  return String(val)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmtNum(v) {
  if (v === null || v === undefined || v === "inf") return String(v ?? "—");
  if (typeof v === "boolean") return v ? "Yes" : "No";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    if (v === 0) return "0";
    if (Math.abs(v) < 1e-4 || Math.abs(v) >= 1e6) return v.toExponential(3);
    return Number(v.toPrecision(5)).toString();
  }
  return String(v);
}

function renderMetricsTable(m, modelKey, d) {
  const units = escHtml(m.target_units);
  let cvRow = "";
  if (modelKey === "ridge" || modelKey === "lasso") {
    const pt = d.penalty_tuning;
    cvRow = `<tr><td>Training CV (5-fold)</td><td>${escHtml(pt.cv_strategy)}</td><td class="num">&mdash;</td><td class="num">${fmtNum(pt.best_cv_rmse)} &plusmn; ${fmtNum(pt.best_cv_rmse_std)}</td><td class="num">${pt.best_cv_r2.toFixed(4)}</td></tr>`;
  }
  let lpmHtml = "";
  if (m.is_binary_classification && m.test_classification) {
    const trc = m.train_classification;
    const tec = m.test_classification;
    lpmHtml = `
      <div class="pill-row">
        <span class="pill pill-accent">Test Acc: ${tec.accuracy.toFixed(4)}</span>
        <span class="pill pill-accent">Test F1: ${tec.f1.toFixed(4)}</span>
        <span class="pill pill-accent">Test ROC AUC: ${tec.roc_auc.toFixed(4)}</span>
        <span class="pill pill-warn">Test Outside [0,1]: ${tec.n_outside_unit_interval}/${m.n_test} (${tec.pct_outside_unit_interval.toFixed(2)}%)</span>
      </div>`;
  }
  return `
    <div class="table-wrap">
      <table>
        <thead><tr><th>Split</th><th>Detail</th><th class="num">MAE (${units})</th><th class="num">RMSE (${units})</th><th class="num">R&sup2;</th></tr></thead>
        <tbody>
          <tr><td>Train</td><td>n = ${m.n_train}</td><td class="num">${fmtNum(m.train.mae)}</td><td class="num">${fmtNum(m.train.rmse)}</td><td class="num">${m.train.r2.toFixed(4)}</td></tr>
          ${cvRow}
          <tr><td><strong>Test (Held-out)</strong></td><td>n = ${m.n_test}</td><td class="num"><strong>${fmtNum(m.test.mae)}</strong></td><td class="num"><strong>${fmtNum(m.test.rmse)}</strong></td><td class="num"><strong>${m.test.r2.toFixed(4)}</strong></td></tr>
        </tbody>
      </table>
    </div>
    ${lpmHtml}
  `;
}

function renderAlgorithmWorkspace(bundle, modelKey) {
  const mount = document.getElementById("mount-" + modelKey);
  const mb = bundle.models[modelKey];
  if (!mb) {
    mount.innerHTML = "<p>No artifacts available.</p>";
    return;
  }
  const m = mb.metrics;
  const d = mb.diagnostics;
  const prep = bundle.preprocessing;

  // Config & summary KPIs
  let kpis = [
    {lbl: "Test R²", val: m.test.r2.toFixed(4)},
    {lbl: `Test RMSE (${m.target_units})`, val: fmtNum(m.test.rmse)},
  ];
  let configPills = [
    `Split: ${prep.split_strategy}`,
    `Features: ${prep.n_encoded_features}`,
    `Scaling: Train-only StandardScaler`,
    `Intercept: Unscaled & Unpenalized`,
  ];

  if (modelKey === "ols") {
    const la = d.numerical_linear_algebra;
    const ex = d.exact_algebraic_properties;
    kpis.push({lbl: "Matrix Rank", val: `${la.matrix_rank} / ${la.n_augmented_columns}`});
    kpis.push({lbl: "Condition κ (Std)", val: fmtNum(la.condition_number_standardized_with_intercept)});
    kpis.push({lbl: "Max |Xᵀe|", val: ex.max_abs_Xt_residuals.toExponential(2)});
    configPills.push("Solver: np.linalg.lstsq (SVD)");
  } else if (modelKey === "ridge") {
    const pt = d.penalty_tuning;
    const sd = d.shrinkage_and_degrees_of_freedom;
    const ns = d.numerical_stability;
    kpis.push({lbl: "Selected α (λ)", val: fmtNum(pt.selected_alpha)});
    kpis.push({lbl: "Effective df(α)", val: `${sd.effective_df_predictors.toFixed(2)} / ${sd.n_predictors}`});
    kpis.push({lbl: "L2 Ratio vs. OLS", val: sd.l2_norm_ratio_ridge_over_ols.toFixed(4)});
    kpis.push({lbl: "κ Before → After", val: `${fmtNum(ns.x_centered_eig_condition_ols)} → ${fmtNum(ns.x_centered_eig_condition_ridge)}`});
    configPills.push(`CV: ${pt.cv_strategy}`, `Grid: 60 alphas [1e-4, 1e5]`);
  } else if (modelKey === "lasso") {
    const pt = d.penalty_tuning;
    const sp = d.sparsity_and_shrinkage;
    const cvg = d.convergence;
    kpis.push({lbl: "Selected α (λ)", val: fmtNum(pt.selected_alpha)});
    kpis.push({lbl: "Selected / Total", val: `${sp.n_nonzero_features} / ${sp.n_total_features}`});
    kpis.push({lbl: "L1 Ratio vs. OLS", val: sp.l1_norm_ratio_lasso_over_ols.toFixed(4)});
    kpis.push({lbl: "Converged", val: `${cvg.converged_final ? "Yes" : "No"} (${cvg.n_iter_final} iters)`});
    configPills.push(`CV: ${pt.cv_strategy}`, `Grid: 60 alphas [α_max, 1e-4·α_max]`);
  }

  const kpiHtml = kpis.map(k => `<div class="kpi-card"><div class="kpi-lbl">${escHtml(k.lbl)}</div><div class="kpi-val">${escHtml(k.val)}</div></div>`).join("");
  const pillsHtml = configPills.map(p => `<span class="pill">${escHtml(p)}</span>`).join("");

  // Plots with grounded captions
  const plotsHtml = mb.plots.map(p => `
    <div class="plot-card">
      <img loading="lazy" src="${escHtml(p.src)}" alt="${escHtml(p.title)}">
      <div class="plot-title">${escHtml(p.title)}</div>
      <div class="plot-caption">${escHtml(p.caption)}</div>
    </div>
  `).join("");

  // Coefficients table
  const hasOls = mb.coef_rows.length > 0 && mb.coef_rows[0].ols_coefficient_standardized !== undefined;
  const hasShrink = mb.coef_rows.length > 0 && mb.coef_rows[0].shrinkage_ratio_vs_ols !== undefined;
  const hasRetained = mb.coef_rows.length > 0 && mb.coef_rows[0].retained !== undefined;

  const coefRowsHtml = mb.coef_rows.map(r => {
    const cls = r.role === "intercept" ? "intercept-row" : (hasRetained && !r.retained ? "zero-row" : "");
    return `<tr class="${cls}">
      <td>${escHtml(r.role)}</td>
      <td><code>${escHtml(r.feature)}</code></td>
      <td class="num">${fmtNum(r.coefficient_standardized)}</td>
      <td class="num">${fmtNum(r.coefficient_unstandardized)}</td>
      ${hasOls ? `<td class="num">${fmtNum(r.ols_coefficient_standardized)}</td>` : ""}
      ${hasShrink ? `<td class="num">${r.shrinkage_ratio_vs_ols === null ? "—" : Number(r.shrinkage_ratio_vs_ols).toFixed(4)}</td>` : ""}
      ${hasRetained ? `<td class="num">${r.retained ? "Active" : "Zero (0)"}</td><td class="num">${Number(r.cv_selection_freq).toFixed(2)}</td>` : ""}
    </tr>`;
  }).join("");

  // Diagnostics & narrative
  const diagTables = mb.diag_sections.map(([title, items]) => `
    <h4 style="margin:12px 0 6px; color:var(--navy-2);">${escHtml(title)}</h4>
    <table>${items.map(([k, v]) => `<tr><td>${escHtml(k)}</td><td class="num">${escHtml(v)}</td></tr>`).join("")}</table>
  `).join("");

  const interpHtml = mb.interpretations.map(x => `<li>${escHtml(x)}</li>`).join("");
  const limitHtml = mb.limitations.map(x => `<li>${escHtml(x)}</li>`).join("");

  let lassoSelectionHtml = "";
  if (modelKey === "lasso") {
    const sp = d.sparsity_and_shrinkage;
    const retained = sp.retained_features.map(x => `${x.feature} (${x.coef_std >= 0 ? "+" : ""}${x.coef_std.toFixed(3)})`).join(", ") || "None";
    const eliminated = sp.eliminated_features.map(x => x.feature).join(", ") || "None";
    lassoSelectionHtml = `
      <div class="notice">
        <strong>Selected Features (${sp.n_nonzero_features}):</strong> ${escHtml(retained)}<br>
        <strong>Eliminated Features (${sp.n_zero_features}):</strong> ${escHtml(eliminated)}
      </div>`;
  }

  mount.innerHTML = `
    <div class="pill-row">
      <a class="pill pill-accent" style="text-decoration:none;" href="${escHtml(bundle.uci_url)}" target="_blank" rel="noopener noreferrer">UCI ${bundle.uci_id} &nearr;</a>
      <a class="pill pill-accent" style="text-decoration:none;" href="${modelKey}/${escHtml(bundle.key)}.html" target="_blank">Standalone HTML Report &nearr;</a>
      ${pillsHtml}
    </div>
    <div class="kpi-grid">${kpiHtml}</div>
    ${renderMetricsTable(m, modelKey, d)}
    ${lassoSelectionHtml}
    <div class="plots-grid">${plotsHtml}</div>
    <details>
      <summary>Coefficient Analysis (${mb.coef_rows.length} entries incl. unpenalized intercept)</summary>
      <div class="scroll-table" style="margin-top:8px;">
        <table>
          <thead><tr>
            <th>Role</th><th>Feature</th><th class="num">Coef (Std)</th><th class="num">Coef (Unstd)</th>
            ${hasOls ? `<th class="num">OLS Coef (Std)</th>` : ""}
            ${hasShrink ? `<th class="num">Shrinkage Ratio</th>` : ""}
            ${hasRetained ? `<th class="num">Status</th><th class="num">CV Freq</th>` : ""}
          </tr></thead>
          <tbody>${coefRowsHtml}</tbody>
        </table>
      </div>
    </details>
    <details>
      <summary>Experiment Interpretation, Assumptions, Limitations &amp; Full Diagnostics</summary>
      <div class="two-col">
        <div><h4>Computed Facts &amp; Interpretations</h4><ul>${interpHtml}</ul></div>
        <div><h4>Assumptions &amp; Limitations</h4><ul>${limitHtml}</ul></div>
      </div>
      ${diagTables}
    </details>
  `;
}

function renderComparisonSection(bundle) {
  const mount = document.getElementById("mount-comparison");
  const ols = bundle.models.ols;
  const ridge = bundle.models.ridge;
  const lasso = bundle.models.lasso;
  if (!ols || !ridge || !lasso) return;

  const units = escHtml(bundle.target_units);
  const compTable = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Algorithm</th>
            <th class="num">Train RMSE</th>
            <th class="num">Train R&sup2;</th>
            <th class="num">Test MAE</th>
            <th class="num">Test RMSE (${units})</th>
            <th class="num">Test R&sup2;</th>
            <th class="num">Active / df</th>
            <th class="num">Selected &alpha;</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>OLS</strong></td>
            <td class="num">${fmtNum(ols.metrics.train.rmse)}</td>
            <td class="num">${ols.metrics.train.r2.toFixed(4)}</td>
            <td class="num">${fmtNum(ols.metrics.test.mae)}</td>
            <td class="num">${fmtNum(ols.metrics.test.rmse)}</td>
            <td class="num"><strong>${ols.metrics.test.r2.toFixed(4)}</strong></td>
            <td class="num">${ols.diagnostics.numerical_linear_algebra.matrix_rank - 1} (rank-1)</td>
            <td class="num">0 (none)</td>
          </tr>
          <tr>
            <td><strong>Ridge (L2)</strong></td>
            <td class="num">${fmtNum(ridge.metrics.train.rmse)}</td>
            <td class="num">${ridge.metrics.train.r2.toFixed(4)}</td>
            <td class="num">${fmtNum(ridge.metrics.test.mae)}</td>
            <td class="num">${fmtNum(ridge.metrics.test.rmse)}</td>
            <td class="num"><strong>${ridge.metrics.test.r2.toFixed(4)}</strong></td>
            <td class="num">df = ${ridge.diagnostics.shrinkage_and_degrees_of_freedom.effective_df_predictors.toFixed(2)} (all ${ridge.diagnostics.shrinkage_and_degrees_of_freedom.n_predictors})</td>
            <td class="num">${fmtNum(ridge.diagnostics.penalty_tuning.selected_alpha)}</td>
          </tr>
          <tr>
            <td><strong>Lasso (L1)</strong></td>
            <td class="num">${fmtNum(lasso.metrics.train.rmse)}</td>
            <td class="num">${lasso.metrics.train.r2.toFixed(4)}</td>
            <td class="num">${fmtNum(lasso.metrics.test.mae)}</td>
            <td class="num">${fmtNum(lasso.metrics.test.rmse)}</td>
            <td class="num"><strong>${lasso.metrics.test.r2.toFixed(4)}</strong></td>
            <td class="num">${lasso.diagnostics.sparsity_and_shrinkage.n_nonzero_features} / ${lasso.diagnostics.sparsity_and_shrinkage.n_total_features} active</td>
            <td class="num">${fmtNum(lasso.diagnostics.penalty_tuning.selected_alpha)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  `;

  // Side-by-side standardized coefficients aligned by feature
  const mapByFeat = {};
  const addCoefs = (rows, key) => {
    rows.forEach(r => {
      if (!mapByFeat[r.feature]) mapByFeat[r.feature] = {role: r.role, feature: r.feature};
      mapByFeat[r.feature][key] = r.coefficient_standardized;
    });
  };
  addCoefs(ols.coef_rows, "ols");
  addCoefs(ridge.coef_rows, "ridge");
  addCoefs(lasso.coef_rows, "lasso");
  const aligned = Object.values(mapByFeat).sort((a, b) => {
    if (a.role === "intercept") return -1;
    if (b.role === "intercept") return 1;
    return Math.abs(b.ols || 0) - Math.abs(a.ols || 0);
  });

  const coefCompRows = aligned.map(r => `
    <tr class="${r.role === "intercept" ? "intercept-row" : ""}">
      <td><code>${escHtml(r.feature)}</code></td>
      <td class="num">${fmtNum(r.ols)}</td>
      <td class="num">${fmtNum(r.ridge)}</td>
      <td class="num">${fmtNum(r.lasso)}</td>
    </tr>
  `).join("");

  mount.innerHTML = `
    ${compTable}
    <details>
      <summary>Side-by-Side Standardized Coefficients (OLS vs. Ridge vs. Lasso)</summary>
      <div class="scroll-table" style="margin-top:8px;">
        <table>
          <thead><tr><th>Feature</th><th class="num">OLS (Std)</th><th class="num">Ridge (Std)</th><th class="num">Lasso (Std)</th></tr></thead>
          <tbody>${coefCompRows}</tbody>
        </table>
      </div>
    </details>
  `;
}

// Predictor logic
function fmtShort(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  if (n === 0) return "0";
  if (Math.abs(n) >= 1000 || Math.abs(n) < 0.01) return n.toPrecision(3);
  return Number(n.toFixed(2)).toString();
}

function renderPredictorForBundle(bundle) {
  const ds = bundle.predictor;
  const sampleSel = document.getElementById("pred-sample-select");
  sampleSel.innerHTML = '<option value="">Train Median (Default)</option>';
  ds.samples.forEach((s, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = s.label;
    sampleSel.appendChild(opt);
  });

  const grid = document.getElementById("pred-inputs-grid");
  grid.innerHTML = "";
  document.getElementById("pred-filter").value = "";

  ds.fields.forEach(f => {
    const card = document.createElement("div");
    card.className = "field-card";
    card.setAttribute("data-fname", f.name.toLowerCase());

    const top = document.createElement("div");
    top.className = "field-top";
    const lbl = document.createElement("label");
    lbl.textContent = f.name;
    lbl.setAttribute("for", "inp-" + f.name);
    top.appendChild(lbl);
    if (f.kind === "numeric") {
      const rng = document.createElement("span");
      rng.className = "field-range";
      rng.textContent = `${fmtShort(f.min)}–${fmtShort(f.max)}`;
      top.appendChild(rng);
    }
    card.appendChild(top);

    if (f.kind === "categorical" || f.kind === "binary") {
      const sel = document.createElement("select");
      sel.id = "inp-" + f.name;
      f.options.forEach(o => {
        const opt = document.createElement("option");
        opt.value = o.value;
        opt.textContent = o.label;
        if (o.value === String(f.default)) opt.selected = true;
        sel.appendChild(opt);
      });
      sel.addEventListener("change", computePredictions);
      card.appendChild(sel);
    } else {
      const inp = document.createElement("input");
      inp.type = "number";
      inp.step = "any";
      inp.id = "inp-" + f.name;
      inp.value = f.default;
      inp.addEventListener("input", computePredictions);
      card.appendChild(inp);
    }
    grid.appendChild(card);
  });

  computePredictions();
}

function filterPredictorFields(query) {
  const q = query.trim().toLowerCase();
  document.querySelectorAll("#pred-inputs-grid .field-card").forEach(card => {
    const fname = card.getAttribute("data-fname") || "";
    card.style.display = (!q || fname.includes(q)) ? "" : "none";
  });
}

function resetPredictorDefaults() {
  const bundle = window.__DATASET_STORE__[currentDatasetKey];
  if (!bundle) return;
  document.getElementById("pred-sample-select").value = "";
  bundle.predictor.fields.forEach(f => {
    const el = document.getElementById("inp-" + f.name);
    if (el) el.value = String(f.default);
  });
  computePredictions();
}

function loadSelectedSample() {
  const idxStr = document.getElementById("pred-sample-select").value;
  if (idxStr === "") { resetPredictorDefaults(); return; }
  const bundle = window.__DATASET_STORE__[currentDatasetKey];
  if (!bundle) return;
  const sample = bundle.predictor.samples[Number(idxStr)];
  bundle.predictor.fields.forEach(f => {
    const el = document.getElementById("inp-" + f.name);
    if (el && sample.values[f.name] !== undefined) el.value = String(sample.values[f.name]);
  });
  computePredictions();
}

function promptInputsSequential() {
  const bundle = window.__DATASET_STORE__[currentDatasetKey];
  if (!bundle) return;
  const ds = bundle.predictor;
  if (ds.fields.length > 25 && !window.confirm(`${ds.name} has ${ds.fields.length} features. Step through prompts?`)) return;
  for (const f of ds.fields) {
    const el = document.getElementById("inp-" + f.name);
    const current = el ? el.value : String(f.default);
    const hint = f.kind === "numeric" ? `${fmtShort(f.min)}..${fmtShort(f.max)}` : f.options.map(o => o.value + "=" + o.label).join(", ");
    const ans = window.prompt(`${f.name} (${hint}):`, current);
    if (ans === null) break;
    if (el && ans.trim() !== "") el.value = ans.trim();
  }
  computePredictions();
}

function getEncodedFeatureMap(ds) {
  const xMap = {};
  ds.fields.forEach(f => {
    const el = document.getElementById("inp-" + f.name);
    const rawVal = el ? el.value : String(f.default);
    if (f.kind === "categorical") {
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
  const bundle = window.__DATASET_STORE__[currentDatasetKey];
  if (!bundle) return;
  const ds = bundle.predictor;
  const xMap = getEncodedFeatureMap(ds);
  const resGrid = document.getElementById("pred-results-grid");
  resGrid.innerHTML = "";

  const sampleIdx = document.getElementById("pred-sample-select").value;
  const observed = sampleIdx !== "" ? ds.samples[Number(sampleIdx)].observed : null;

  const contribRows = {};
  ["ols", "ridge", "lasso"].forEach(mkey => {
    const mw = ds.models[mkey];
    if (!mw) return;
    let yhat = mw.intercept;
    Object.entries(mw.coefs).forEach(([feat, w]) => {
      const xv = xMap[feat] !== undefined ? xMap[feat] : 0.0;
      const contrib = w * xv;
      yhat += contrib;
      if (!contribRows[feat]) contribRows[feat] = {feat, x: xv};
      contribRows[feat][mkey] = contrib;
    });

    const card = document.createElement("div");
    card.className = "pred-card";
    let pills = "";
    if (ds.is_binary_classification) {
      const isPos = yhat >= 0.5;
      pills += `<span class="pill ${isPos ? "pill-ok" : ""}">${isPos ? "Positive (1)" : "Negative (0)"}</span>`;
      if (yhat < 0.0 || yhat > 1.0) pills += `<span class="pill pill-warn">Outside [0,1]</span>`;
    } else if (ds.is_ordinal_target) {
      pills += `<span class="pill pill-accent">Rounded: ${Math.round(yhat)}</span>`;
    }
    if (observed !== null) {
      const err = yhat - observed;
      pills += `<span class="pill">Observed: ${observed.toPrecision(4)} (&Delta; = ${err >= 0 ? "+" : ""}${err.toFixed(3)})</span>`;
    }
    card.innerHTML = `
      <div class="kpi-lbl">${mkey.toUpperCase()} &rarr; ${escHtml(ds.target)}</div>
      <div class="pred-val">${yhat.toFixed(4)} <small style="font-size:0.76rem; color:var(--muted); font-weight:500;">${escHtml(ds.target_units)}</small></div>
      <div class="pill-row" style="margin:4px 0 0;">${pills}</div>
    `;
    resGrid.appendChild(card);
  });

  const rows = Object.values(contribRows).sort((a, b) => Math.abs(b.ols || 0) - Math.abs(a.ols || 0));
  let tbl = "<table><thead><tr><th>Encoded Feature</th><th class='num'>x_j</th><th class='num'>OLS Term</th><th class='num'>Ridge Term</th><th class='num'>Lasso Term</th></tr></thead><tbody>";
  rows.forEach(r => {
    tbl += `<tr><td><code>${escHtml(r.feat)}</code></td><td class="num">${Number(r.x).toPrecision(5)}</td><td class="num">${(r.ols || 0).toFixed(4)}</td><td class="num">${(r.ridge || 0).toFixed(4)}</td><td class="num">${(r.lasso || 0).toFixed(4)}</td></tr>`;
  });
  tbl += "</tbody></table>";
  document.getElementById("pred-contrib-table").innerHTML = tbl;
}

// Downloads
function downloadBlob(filename, content, mime) {
  const blob = new Blob([content], {type: mime});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function downloadCoefficientsCSV(modelKey) {
  const bundle = window.__DATASET_STORE__[currentDatasetKey];
  if (!bundle || !bundle.models[modelKey]) return;
  downloadBlob(`${modelKey}_${currentDatasetKey}_coefficients.csv`, bundle.models[modelKey].coef_csv, "text/csv;charset=utf-8");
}

function downloadRunJSON(modelKey) {
  const bundle = window.__DATASET_STORE__[currentDatasetKey];
  if (!bundle || !bundle.models[modelKey]) return;
  const mb = bundle.models[modelKey];
  const payload = {metrics: mb.metrics, diagnostics: mb.diagnostics, metadata: mb.metadata};
  downloadBlob(`${modelKey}_${currentDatasetKey}_results.json`, JSON.stringify(payload, null, 2), "application/json");
}

function selectGlobalDataset(dsKey) {
  currentDatasetKey = dsKey;
  ["sel-ols", "sel-ridge", "sel-lasso", "sel-comp"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = dsKey;
  });
  loadDatasetBundle(dsKey).then(bundle => {
    renderAlgorithmWorkspace(bundle, "ols");
    renderAlgorithmWorkspace(bundle, "ridge");
    renderAlgorithmWorkspace(bundle, "lasso");
    renderComparisonSection(bundle);
    renderPredictorForBundle(bundle);
    triggerMathRender(document.body);
  }).catch(err => {
    console.error(err);
  });
}

function jumpToDataset(dsKey) {
  selectGlobalDataset(dsKey);
  document.getElementById("comparison").scrollIntoView({behavior: "smooth"});
}

document.addEventListener("DOMContentLoaded", () => {
  selectGlobalDataset(currentDatasetKey);
  triggerMathRender(document.body);
});
</script>
</body>
</html>
""")


def generate_grand_index() -> Path:
    """Build lightweight reports/index.html + lazy-loaded assets/data/*.js + assets/plots/."""
    master_rows: List[Dict[str, Any]] = []
    dataset_inventory: List[Dict[str, Any]] = []
    best_r2_values: List[float] = []
    win_counts = {"ols": 0, "ridge": 0, "lasso": 0}
    total_plots = 0

    assets_data_dir = REPORTS_DIR / "assets" / "data"
    assets_data_dir.mkdir(parents=True, exist_ok=True)

    for dataset_key, cfg in DATASETS.items():
        bundle = _sync_plots_and_build_dataset_bundle(dataset_key)
        # Write lazy-loaded JS bundle (works over HTTPS, HTTP, and local file://)
        js_path = assets_data_dir / f"{dataset_key}.js"
        js_content = (
            f"window.__DATASET_STORE__ = window.__DATASET_STORE__ || {{}};\n"
            f"window.__DATASET_STORE__[{json.dumps(dataset_key)}] = {json.dumps(bundle)};\n"
        )
        js_path.write_text(js_content, encoding="utf-8")

        models = bundle["models"]
        for mb in models.values():
            total_plots += len(mb["plots"])

        if len(models) == 3:
            ols_m = models["ols"]["metrics"]
            ridge_m = models["ridge"]["metrics"]
            lasso_m = models["lasso"]["metrics"]
            ridge_d = models["ridge"]["diagnostics"]
            lasso_d = models["lasso"]["diagnostics"]
            prep = bundle["preprocessing"]

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
            dataset_inventory.append(
                {
                    "key": dataset_key,
                    "name": cfg.name,
                    "uci_id": cfg.uci_id,
                    "uci_url": _uci_url(cfg.uci_id),
                    "target": cfg.target,
                    "target_units": cfg.target_units,
                    "n_clean": prep["n_clean"],
                    "n_encoded": prep["n_encoded_features"],
                    "split_strategy": prep["split_strategy"],
                    "cv_strategy": prep["cv_strategy"],
                    "caveat_summary": cfg.notes[0] if cfg.notes else "Standard numeric regression.",
                }
            )

    global_stats = {
        "median_best_r2": float(np.median(best_r2_values)) if best_r2_values else 0.0,
        "wins_summary": f"{win_counts['ols']} / {win_counts['ridge']} / {win_counts['lasso']}",
        "total_plots": total_plots,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "index.html"
    html = DASHBOARD_TEMPLATE.render(
        github_repo_url=GITHUB_REPO_URL,
        master_rows=master_rows,
        dataset_inventory=dataset_inventory,
        global_stats=global_stats,
        master_json=json.dumps(master_rows),
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path
