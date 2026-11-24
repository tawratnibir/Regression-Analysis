# UCI Regression Analysis: OLS, Ridge, and Lasso

A reproducible Python benchmark comparing **Ordinary Least Squares (OLS)**, **Ridge Regression (L2)**, and **Lasso Regression (L1)** across **10 fixed UCI datasets**, with dataset-specific preprocessing, persisted train/test splits, leak-free cross-validation, numerical diagnostics, and self-contained HTML reports.

## Project Structure

```text
Regression-Analysis/
├── data/
│   ├── raw/                  # Unmodified raw UCI CSVs and variable metadata
│   └── processed/            # Persisted split indices (split.json) and summaries
├── preprocessing/
│   ├── loader.py             # Raw CSV loading & optional UCI download helper
│   ├── preprocess.py         # Shared leak-free preprocessing & CV fold builder
│   └── eda.py                # Matplotlib exploratory plots & summary statistics
├── models/
│   ├── base.py               # Shared metrics, artifact I/O, and standard plots
│   ├── ols/                  # OLS via numpy.linalg.lstsq, algebraic & assumption checks
│   ├── ridge/                # Ridge via training-only CV, SVD verification, shrinkage
│   └── lasso/                # Lasso via Gram-precomputed lasso_path CV, sparsity, stability
├── reporting/
│   └── generate_report.py    # Standalone HTML reports (per model×dataset + summary)
├── results/                  # Saved metrics, coefficients, predictions, diagnostics, plots
├── reports/                  # Standalone HTML reports viewable locally via file://
├── tests/
│   └── test_pipeline.py      # Unit tests for splits, leakage, OLS identities, artifacts
├── config.py                 # Central dataset definitions and split strategies
├── main.py                   # CLI entrypoint
└── requirements.txt
```

## Setup

Using the project virtual environment (`.venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Core runtime dependencies used by the pipeline: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `jinja2`.)*

## CLI Usage

```bash
python main.py --model ols --dataset airfoil
python main.py --model ridge --dataset parkinsons
python main.py --model lasso --dataset all
python main.py --model all --dataset all
```

### Additional Flags
- `--overwrite`: Re-run and overwrite existing outputs in `results/` and `reports/` (by default existing runs are skipped cleanly).
- `--force-resplit`: Recompute and overwrite persisted split indices under `data/processed/<dataset>/split.json`.
- `--run-eda`: Generate EDA statistics and plots under `results/eda/<dataset>/`.

## Verification Tests

```bash
python -m unittest discover -s tests -v
```

## Dataset-Specific Caveats & Split Methodology

All preprocessing transformers (`SimpleImputer`, `OneHotEncoder`, `StandardScaler`) are fit **only on training data** (and separately inside each cross-validation fold). The intercept is never scaled or penalized.

| Dataset Key | UCI ID | Target (Units) | Split & CV Strategy | Key Caveats & Exclusions |
|---|---:|---|---|---|
| `airfoil` | 291 | `scaled-sound-pressure` (dB) | Random 80/20 (seed 42), 5-fold KFold | Predictors span wide scales; standardization reduces condition number from ~5.3e5 to 3.52. |
| `power_plant` | 294 | `PE` (MW) | Random 80/20 (seed 42), 5-fold KFold | 41 exact duplicate rows preserved as recorded. |
| `real_estate` | 477 | `Y house price of unit area` (10,000 NTD/Ping) | Random 80/20 (seed 42), 5-fold KFold | Record identifier `No` excluded. |
| `wine_quality_red` | 186 | `quality` (ordinal 0–10) | Random 80/20 (seed 42), 5-fold KFold | Red wine only (1,599 rows); 240 duplicate rows preserved; target is ordinal sensory score modeled continuously. |
| `parkinsons` | 189 | `total_UPDRS` (UPDRS points) | Group 80/20 by `subject#`, 5-fold GroupKFold | `subject#` excluded from predictors; co-target `motor_UPDRS` excluded to prevent target leakage; extreme collinearity among jitter/shimmer measures. |
| `superconductivity` | 464 | `critical_temp` (K) | Random 80/20 (seed 42), 5-fold KFold | 81 strongly collinear elemental summary predictors; 66 duplicate rows preserved. |
| `auto_mpg` | 9 | `mpg` (mpg) | Random 80/20 (seed 42), 5-fold KFold | `car_name` excluded; `horsepower` (6 missing) median-imputed on train; `origin` one-hot encoded (drop first). |
| `appliances_energy` | 374 | `Appliances` (Wh) | Chronological 80/20, 5-fold TimeSeriesSplit | Sorted by `date`; `date` excluded from predictors; `rv1 == rv2` exactly across all rows causing rank deficiency in OLS. |
| `diabetes` | 529 | `class` (Positive=1 / Negative=0) | Stratified 80/20 (seed 42), 5-fold StratifiedKFold | Modeled as a **Linear Probability Model (LPM)**; reports both regression and classification metrics (threshold 0.50, ROC AUC) plus out-of-`[0,1]` prediction rates. |
| `weather` | 514 | `Next_Tmax` (deg C) | Chronological date-level 80/20, 5-fold Date-Grouped TimeSeriesSplit | 2 trailing min/max summary rows and 27 missing-target rows dropped; `Date`, `station`, and co-target `Next_Tmin` excluded; missing predictors median-imputed on train. |

## Outputs & Reports

- Per-run artifacts: `results/<model>/<dataset>/` (`metrics.json`, `coefficients.csv`, `predictions_test.csv`, `predictions_train.csv`, `diagnostics.json`, `run_metadata.json`, `plots/*.png`).
- Standalone HTML reports: `reports/<model>/<dataset>.html` and `reports/<model>/summary.html` (all plots embedded directly as base64 images so reports open offline via `file://` without a web server).
- Grand interactive dashboard & live predictor: `reports/index.html` combines all 10 datasets × 3 models, all embedded diagrams, artifact-grounded explanations, and a client-side input prompt predictor for OLS, Ridge, and Lasso.
