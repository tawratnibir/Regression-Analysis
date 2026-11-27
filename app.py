"""FastAPI web service for serving UCI Regression Analysis reports and live predictions."""

from functools import lru_cache
import json
import os
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache_reg_analysis")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel, Field
import uvicorn

from config import DATASETS, MODEL_KEYS, REPORTS_DIR, RESULTS_DIR

app = FastAPI(
    title="DataMorph — UCI Regression Analysis API & Dashboard",
    description="DataMorph web service serving standalone OLS, Ridge, and Lasso reports plus live Python prediction endpoints.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    """Request payload for live regression / LPM prediction."""

    dataset: str = Field(..., description="Dataset key, e.g. 'airfoil', 'auto_mpg', 'diabetes'.")
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of raw predictor names to values.",
    )
    models: Optional[List[str]] = Field(
        default=None,
        description="Optional subset of models ('ols', 'ridge', 'lasso'). Defaults to all.",
    )


@lru_cache(maxsize=64)
def _load_model_weights(model_key: str, dataset_key: str) -> Dict[str, Any]:
    """Load and cache unstandardized intercept and coefficients from saved CSV."""
    coef_path = RESULTS_DIR / model_key / dataset_key / "coefficients.csv"
    if not coef_path.exists():
        raise FileNotFoundError(f"Saved coefficients not found: {coef_path}")
    cdf = pd.read_csv(coef_path)
    intercept = float(cdf.loc[cdf["role"] == "intercept", "coefficient_unstandardized"].iloc[0])
    pred_rows = cdf[cdf["role"] == "predictor"]
    coefs = {
        str(r["feature"]): float(r["coefficient_unstandardized"])
        for _, r in pred_rows.iterrows()
    }
    return {"intercept": intercept, "coefs": coefs}


@lru_cache(maxsize=16)
def _load_default_raw_inputs(dataset_key: str) -> Dict[str, Any]:
    """Compute and cache training medians/modes lazily if a caller omits optional fields."""
    from preprocessing.preprocess import load_and_prepare_dataset

    prepared = load_and_prepare_dataset(dataset_key)
    cfg = prepared.config
    defaults: Dict[str, Any] = {}
    for col in prepared.df_train_raw.columns:
        if col in cfg.categorical_columns:
            defaults[col] = str(prepared.df_train_raw[col].mode().iloc[0])
        elif col in cfg.binary_mappings:
            defaults[col] = int(round(float(prepared.df_train_raw[col].mode().iloc[0])))
        else:
            s = pd.to_numeric(prepared.df_train_raw[col], errors="coerce").dropna()
            defaults[col] = float(s.median())
    return defaults


def _encode_raw_inputs(dataset_key: str, raw_inputs: Dict[str, Any]) -> Dict[str, float]:
    """Convert raw user-provided feature values into the model's encoded feature dictionary."""
    cfg = DATASETS[dataset_key]
    merged: Dict[str, Any] = {}
    # Only load defaults from raw CSV when at least one predictor might be omitted
    defaults: Optional[Dict[str, Any]] = None

    encoded: Dict[str, float] = {}
    # Collect expected encoded feature names from OLS coefficients
    expected_features = list(_load_model_weights("ols", dataset_key)["coefs"].keys())

    if set(raw_inputs.keys()) >= set(expected_features) and not cfg.categorical_columns:
        merged = raw_inputs
    else:
        defaults = _load_default_raw_inputs(dataset_key)
        merged = {**defaults, **raw_inputs}

    for col, val in merged.items():
        if col in cfg.categorical_columns:
            val_str = str(val)
            for feat in expected_features:
                if feat.startswith(f"{col}_"):
                    suffix = feat[len(col) + 1 :]
                    encoded[feat] = 1.0 if val_str == suffix else 0.0
        elif col in cfg.binary_mappings:
            mapping = cfg.binary_mappings[col]
            if isinstance(val, str) and val in mapping:
                encoded[col] = float(mapping[val])
            else:
                try:
                    encoded[col] = float(val)
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Invalid binary value '{val}' for '{col}'. Allowed: {list(mapping)} or 0/1.",
                    ) from exc
        else:
            try:
                encoded[col] = float(val)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid numeric value '{val}' for '{col}'.",
                ) from exc

    return encoded


@app.get("/health")
def health() -> Dict[str, Any]:
    """Render health-check endpoint."""
    return {
        "status": "ok",
        "models": list(MODEL_KEYS),
        "datasets": list(DATASETS.keys()),
        "grand_index_ready": (REPORTS_DIR / "index.html").exists(),
    }


@app.get("/api/datasets")
def list_datasets() -> List[Dict[str, Any]]:
    """Return metadata for all 10 configured UCI datasets."""
    return [
        {
            "key": cfg.key,
            "uci_id": cfg.uci_id,
            "name": cfg.name,
            "target": cfg.target,
            "target_units": cfg.target_units,
            "split_strategy": cfg.split_strategy,
            "is_binary_classification": cfg.is_binary_classification,
            "is_ordinal_target": cfg.is_ordinal_target,
        }
        for cfg in DATASETS.values()
    ]


@app.get("/api/results/{model_key}/{dataset_key}")
def get_results(model_key: str, dataset_key: str) -> Dict[str, Any]:
    """Return saved metrics, diagnostics, and metadata for a model × dataset run."""
    if model_key not in MODEL_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_key}'")
    if dataset_key not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset_key}'")

    res_dir = RESULTS_DIR / model_key / dataset_key
    metrics_path = res_dir / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"No results for {model_key}/{dataset_key}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(res_dir / "diagnostics.json", "r", encoding="utf-8") as f:
        diagnostics = json.load(f)
    with open(res_dir / "run_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "metadata": metadata,
    }


@app.post("/api/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    """Predict target output in Python using saved unstandardized model coefficients."""
    if req.dataset not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{req.dataset}'")

    target_models = req.models if req.models else list(MODEL_KEYS)
    for m in target_models:
        if m not in MODEL_KEYS:
            raise HTTPException(status_code=400, detail=f"Unknown model '{m}'")

    cfg = DATASETS[req.dataset]
    encoded_x = _encode_raw_inputs(req.dataset, req.inputs)

    predictions: Dict[str, Any] = {}
    for model_key in target_models:
        try:
            weights = _load_model_weights(model_key, req.dataset)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        yhat = weights["intercept"] + sum(
            w * encoded_x.get(feat, 0.0) for feat, w in weights["coefs"].items()
        )
        entry: Dict[str, Any] = {"predicted": float(yhat)}
        if cfg.is_binary_classification:
            entry["predicted_class"] = "Positive" if yhat >= 0.5 else "Negative"
            entry["outside_unit_interval"] = bool(yhat < 0.0 or yhat > 1.0)
        elif cfg.is_ordinal_target:
            entry["rounded_score"] = int(round(yhat))
        predictions[model_key] = entry

    return {
        "dataset": cfg.key,
        "target": cfg.target,
        "target_units": cfg.target_units,
        "predictions": predictions,
    }


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Serve the grand dashboard at root URL."""
    index_path = REPORTS_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="reports/index.html has not been generated yet.")
    return FileResponse(index_path)


# Serve all standalone HTML reports under reports/ (e.g. /ols/airfoil.html, /ridge/summary.html)
if REPORTS_DIR.exists():
    app.mount("/", StaticFiles(directory=str(REPORTS_DIR), html=True), name="reports")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
