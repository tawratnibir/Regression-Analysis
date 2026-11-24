# AI Agent Prompt: UCI Regression Analysis Project

You are implementing a simple, reproducible Python project for a university assignment comparing **Ordinary Least Squares (OLS), Ridge, and Lasso** on **10 fixed UCI datasets**. Implement the project in the existing `Regression-Analysis/` directory. **Do not discard working files or re-download valid raw data unnecessarily.** First inspect the current repository, then reuse or carefully refactor existing scripts such as `utils/dataset_loader.py`, `utils/dataset_inspector.py`, and `utils/eda.py`.

## Goal

The project should do three things:
1. **Preprocess datasets** with dataset-specific handling and reusable train/test splits.
2. **Run models** from separate per-algorithm folders under `models/` (currently `ols`, `ridge`, `lasso`, with clean extensibility for future models).
3. **Generate explainable HTML reports** with metrics, coefficients/diagnostics, plots, and interpretations based only on actual results, plus a unified **grand dashboard** (`reports/index.html`) with an interactive input-to-prediction playground.

Keep it simple. Do not introduce abstract factories, registries, cloud services, databases, or unnecessary frameworks.

## Fixed datasets

| Key | UCI ID | Dataset | Target |
|---|---:|---|---|
| `airfoil` | 291 | Airfoil Self-Noise | `scaled-sound-pressure` |
| `power_plant` | 294 | Combined Cycle Power Plant | Electrical power output (confirm column name) |
| `real_estate` | 477 | Real Estate Valuation | House price per unit area (confirm column name) |
| `wine_quality_red` | 186 | Wine Quality (red wine only) | `quality` |
| `parkinsons` | 189 | Parkinsons Telemonitoring | `total_UPDRS` |
| `superconductivity` | 464 | Superconductivity Data | Critical temperature (confirm column name) |
| `auto_mpg` | 9 | Auto MPG | `mpg` |
| `appliances_energy` | 374 | Appliances Energy Prediction | `Appliances` |
| `diabetes` | 529 | Early Stage Diabetes Risk Prediction | Binary positive/negative classification label (confirm column name) |
| `weather` | 514 | Bias Correction of Numerical Prediction Model Temperature Forecast | Next-day maximum temperature (confirm exact target column) |

**Important:** Inspect actual `data/raw/<key>/data.csv` headers before setting column names. Do not guess column names. Dataset 186 contains red and white wine versions; use **red only**. Weather 514 may require manual ZIP-based loading since the dataset is not importable with `ucimlrepo`.

## Required project structure

```text
Regression-Analysis/
├── data/
│   ├── raw/                  # Do not modify original downloads
│   └── processed/            # Reusable split indices and processed artifacts
├── preprocessing/
│   ├── loader.py
│   ├── preprocess.py
│   └── eda.py
├── models/
│   ├── base.py               # Shared metrics, artifact I/O, and standard residual plots
│   ├── ols/
│   │   ├── model.py
│   │   ├── analysis.py
│   │   └── plots.py
│   ├── ridge/
│   │   ├── model.py
│   │   ├── analysis.py
│   │   └── plots.py
│   └── lasso/                # Additional future models follow this same folder pattern
│       ├── model.py
│       ├── analysis.py
│       └── plots.py
├── reporting/
│   ├── generate_report.py    # Per-model×dataset reports + per-model summary.html
│   └── grand_index.py        # Grand dashboard reports/index.html + live predictor
├── results/
│   ├── eda/
│   ├── ols/
│   ├── ridge/
│   └── lasso/
├── reports/
│   ├── index.html            # Grand unified dashboard across all models & datasets
│   ├── ols/
│   ├── ridge/
│   └── lasso/
├── tests/
│   └── test_pipeline.py      # Verification suite for splits, leakage, models, reports
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

Add `__init__.py` files only where needed. Minor adjustments to this layout are allowed if they meaningfully simplify execution.

## Preprocessing requirements

- Build one dataset configuration in `config.py` with target, excluded columns, split strategy, and preprocessing options.
- Inspect missing values and types. Convert numerical strings safely, impute missing values when necessary, and encode categorical inputs correctly.
- Never include the target in predictors. Exclude row IDs, patient IDs, target-derived columns, or features not available at prediction time.
- **Parkinsons:** Exclude `subject#` as predictor; use group-aware split by subject. Do not automatically include `motor_UPDRS` when predicting `total_UPDRS` because of leakage/target overlap concerns.
- **Weather and Appliances Energy:** Use chronological train/test splits and preserve ordering; confirm date/time fields and target availability.
- **Diabetes:** Encode binary inputs/target numerically; treat OLS/Ridge/Lasso results as *linear probability models*, not standard continuous-outcome regression. Report classification metrics and predicted values outside [0,1] as well as regression metrics where useful.
- **Wine Quality:** Use red wine only; document the ordinal nature of its score.
- **Auto MPG:** Handle missing horsepower, exclude car name, and encode origin as categorical rather than treating arbitrary codes as continuous.
- **Real Estate:** Exclude record number.
- Preserve all valid measurements, including box-plot outliers, unless a specific exclusion is justified and recorded.
- Use a reproducible baseline split, usually 80/20 with seed 42 where random splitting is appropriate. Persist actual split row indices or stable record keys in `data/processed/`, so every algorithm evaluates on identical held-out observations.
- Fit transformations **only on training data**; during cross-validation fit them separately within each fold. Standardize predictors for Ridge and Lasso; standardizing for OLS is permitted. Do not scale the intercept or penalize it.
- Keep one shared preprocessing API that returns appropriately processed training/test matrices, targets, feature names, and split metadata. Never fit preprocessing on the full dataset before splitting.

## Algorithms

### OLS (`models/ols/`)
- Use `numpy.linalg.lstsq`, not explicit inversion of `X.T @ X`.
- Include an intercept.
- Calculate training/test predictions and residuals.
- Record coefficients, matrix rank, condition number, and training properties including `X.T @ residuals ≈ 0`, residual sum ≈ 0, and sample-mean fitted-value identity (when intercept is included).
- Provide useful diagnostics: residuals versus fitted, Q-Q plot, heteroskedasticity checks, influential points when appropriate. Distinguish exact OLS properties from untestable assumptions.

### Ridge (`models/ridge/`)
- Optimize squared-error loss plus an L2 coefficient penalty. Clearly specify the objective convention and the meaning of `alpha`/`lambda`.
- Tune regularization through cross-validation **within training data only**, with suitable folds for grouped/time-series datasets.
- Explain coefficient shrinkage relative to OLS, coefficient paths versus penalty, selected penalty, CV performance, and numerical stability where relevant.
- Do not penalize the intercept.

### Lasso (`models/lasso/`)
- Optimize squared-error loss plus an L1 coefficient penalty, using coordinate descent or a trusted numerical implementation; specify objective scaling.
- Tune penalty through training-only cross-validation using appropriate folds.
- Explain zero coefficients, retained/removed predictors, coefficient paths, sparsity versus penalty, convergence and feature-selection instability when correlated predictors are present.
- Do not penalize the intercept.

Use NumPy for OLS, and scikit-learn for Ridge/Lasso and preprocessing where practical. Verify numerical results against trusted implementations. Algorithms must live in separate directories but reuse shared preprocessing and evaluation functions.

## Results and reports

For each `model × dataset`, save outputs under `results/<model>/<dataset>/`:

- `metrics.json`: train/test MAE, MSE, RMSE, R² when applicable, plus binary classification metrics for diabetes.
- `coefficients.csv`: feature names, numerical coefficient values, intercept noted separately.
- `predictions_test.csv`: record identifier/index, observed value, prediction, residual; include training predictions where useful.
- `diagnostics.json`: algorithm-specific checks and findings.
- `plots/`: actual vs predicted, residuals vs fitted, residual distribution, plus algorithm-specific visualizations.
- `run_metadata.json`: dataset, target, split reference, seed, preprocessing, algorithm, penalty settings, dependencies, run date/time.

Generate readable standalone reports in `reports/<model>/<dataset>.html`, with:
1. Dataset description and target
2. Preprocessing and split methodology
3. Model formulation and fitted coefficients
4. Train/test metrics with correct units
5. Relevant plots and diagnostics
6. Algorithm-specific explanations grounded in actual computed outputs
7. Limitations and potential assumption violations

Optionally generate one summary report per algorithm across all ten datasets (`reports/<model>/summary.html`). Always regenerate the grand unified dashboard at `reports/index.html` (via `reporting/grand_index.py`), containing:
- A master cross-dataset and cross-model comparison table (highlighting best Test R² per dataset while warning against cross-unit raw RMSE comparisons).
- A dataset explorer with per-model tabs showing metrics, coefficients/feature summaries, diagnostic checks, artifact-grounded explanations, limitations, and all embedded diagrams.
- An interactive prediction playground (`Prompt` / form inputs, training-median defaults, held-out test sample loader, and live output comparison across all supported models).

Do not compare raw RMSE across datasets with different target units as though they are directly comparable. Never invent outputs, statistical significance, or explanations unsupported by artifacts. Make all HTML files (including `reports/index.html`) self-contained with base64-embedded plots so they are viewable locally via `file://` without a web server.

## CLI behavior

Implement these commands in `main.py`:

```bash
python main.py --model ols --dataset airfoil
python main.py --model ridge --dataset parkinsons
python main.py --model lasso --dataset all
python main.py --model all --dataset all
```

Each run should preprocess as needed, train/evaluate the chosen model(s), create results and write corresponding HTML reports. Print concise status and errors. If a dataset fails, report the cause and continue with other datasets when `all` is selected. Avoid overwriting valuable prior runs without making that behavior explicit.

## Execution and validation

1. Inspect existing project and files; list what can be reused.
2. Implement dataset configuration and a basic preprocessing pipeline.
3. Implement OLS end-to-end on **Airfoil** first and verify the results.
4. Implement Ridge and Lasso on Airfoil, including penalty tuning and algorithm-specific plots.
5. Extend to the other nine datasets with their special split/preprocessing rules.
6. Generate the HTML reports and confirm all plotted or narrated findings are derived from saved results.
7. Add lightweight tests or checks for shapes, missing values, no target leakage, split overlap, model convergence, predictions, and reproducibility.
8. Document setup, commands, and dataset-specific caveats in `README.md`.

**Definition of done:** The four CLI examples work (plus any newly added `--model <key>` flags); each applicable algorithm runs on all ten datasets, produces structured results, its own explainable HTML report, and updates `reports/index.html` (with all models' results, diagrams, explanations, and interactive predictor support); uses valid consistent splits; passes `tests/test_pipeline.py`; and makes no unsupported statistical claims. Keep the code clear and teachable. Avoid unnecessary complexity.

## Adding future models (extension contract)

When asked to add a new regression/LPM model `<new_model>` in the future, follow the exact same workflow and conventions established for `ols`, `ridge`, and `lasso`:
1. **Model folder:** Create `models/<new_model>/` with `__init__.py` (exposing `run_<new_model>(dataset)`), `model.py`, `analysis.py`, and `plots.py`, reusing `models/base.py` (`compute_all_metrics`, `save_model_artifacts`, standard residual plots) and `preprocessing/preprocess.py` (`PreparedDataset`, `build_preprocessor`, `cv_splits`).
2. **Leak-free tuning & evaluation:** Use the existing persisted splits in `data/processed/<dataset>/split.json` and tune hyperparameters strictly within `dataset.cv_splits` (fitting preprocessing separately inside each CV fold). Use fast vectorized/Cython solvers (e.g., `lasso_path` style for path methods) where available.
3. **Artifacts & per-model reports:** Save `metrics.json`, `coefficients.csv` (or model-appropriate parameter/importance CSV), `predictions_test.csv`, `predictions_train.csv`, `diagnostics.json`, `run_metadata.json`, and `plots/*.png` under `results/<new_model>/<dataset>/`, and update `reporting/generate_report.py` so `reports/<new_model>/<dataset>.html` and `reports/<new_model>/summary.html` render cleanly with grounded interpretations and limitations.
4. **CLI & config integration:** Register `<new_model>` in `config.py` (`MODEL_KEYS`) and `main.py` (`MODEL_RUNNERS`) so `--model <new_model> --dataset <key|all>` and `--model all --dataset all` work with `--overwrite` protection.
5. **Grand dashboard (`reports/index.html`):** Update `reporting/grand_index.py` so `<new_model>` appears in the Master Comparison Table, each dataset's Model Comparison Table and tabbed Diagnostics/Coefficients/Plots explorer, and the Interactive Prediction Playground (for linear models via unstandardized weights, or documented appropriately if non-linear).
6. **Verification & docs:** Extend `tests/test_pipeline.py` to cover the new model's outputs, reports, and `reports/index.html` integration, run `python -m unittest discover -s tests -v` (using `.venv/bin/python`), and update `README.md`.
