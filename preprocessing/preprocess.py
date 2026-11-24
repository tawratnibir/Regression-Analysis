"""Shared leak-free preprocessing pipeline and persistent train/test splits."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    DATASETS,
    N_CV_FOLDS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    DatasetConfig,
)
from preprocessing.loader import load_raw_dataset


@dataclass
class PreparedDataset:
    """Container holding raw splits, leak-free transformed arrays, and CV folds."""

    config: DatasetConfig
    df_train_raw: pd.DataFrame
    df_test_raw: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    train_raw_indices: np.ndarray
    test_raw_indices: np.ndarray
    numeric_columns: List[str]
    categorical_columns: List[str]
    X_train: np.ndarray
    X_test: np.ndarray
    X_train_unscaled: np.ndarray
    X_test_unscaled: np.ndarray
    feature_names: List[str]
    scaler_means: np.ndarray
    scaler_scales: np.ndarray
    cv_splits: List[Tuple[np.ndarray, np.ndarray]]
    cv_strategy_name: str
    split_path: Path
    preprocessing_info: Dict[str, Any]


def build_preprocessor(
    numeric_columns: List[str],
    categorical_columns: List[str],
    standardize: bool = True,
) -> ColumnTransformer:
    """Create an unfitted ColumnTransformer for training-only / CV-fold fitting."""
    transformers = []

    if numeric_columns:
        num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if standardize:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), list(numeric_columns)))

    if categorical_columns:
        cat_steps: List[Tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
            ),
        ]
        if standardize:
            cat_steps.append(("scaler", StandardScaler()))
        transformers.append(("cat", Pipeline(cat_steps), list(categorical_columns)))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _clean_and_order_dataframe(
    raw_df: pd.DataFrame, cfg: DatasetConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply dataset-specific cleaning, type conversion, and chronological ordering."""
    df = raw_df.copy()
    df["_raw_row_id"] = np.arange(len(df), dtype=int)
    n_raw = len(df)
    dropped_summary_count = 0
    dropped_missing_target_count = 0

    if cfg.drop_missing_id_time_summary_rows and cfg.time_column:
        mask_summary = df[cfg.time_column].isna()
        if cfg.secondary_sort_column and cfg.secondary_sort_column in df.columns:
            mask_summary = mask_summary & df[cfg.secondary_sort_column].isna()
        dropped_summary_count = int(mask_summary.sum())
        df = df.loc[~mask_summary].copy()

    if cfg.target not in df.columns:
        raise KeyError(f"Target column '{cfg.target}' not found in {cfg.key}")

    missing_target_mask = df[cfg.target].isna()
    dropped_missing_target_count = int(missing_target_mask.sum())
    if dropped_missing_target_count > 0:
        df = df.loc[~missing_target_mask].copy()

    # Chronological sorting before splitting for time-ordered datasets
    if cfg.time_column and cfg.time_column in df.columns:
        parsed_time = pd.to_datetime(
            df[cfg.time_column], format=cfg.time_format, errors="raise"
        )
        df["_parsed_time"] = parsed_time
        sort_cols = ["_parsed_time"]
        if cfg.secondary_sort_column and cfg.secondary_sort_column in df.columns:
            sort_cols.append(cfg.secondary_sort_column)
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Encode target if categorical binary
    if cfg.target_mapping is not None:
        mapped_y = df[cfg.target].map(cfg.target_mapping)
        if mapped_y.isna().any():
            bad_vals = df.loc[mapped_y.isna(), cfg.target].unique().tolist()
            raise ValueError(f"Unmapped target values in {cfg.key}: {bad_vals}")
        df[cfg.target] = mapped_y.astype(float)
    else:
        df[cfg.target] = pd.to_numeric(df[cfg.target], errors="raise").astype(float)

    # Encode binary string predictors
    for col, mapping in cfg.binary_mappings.items():
        if col in df.columns:
            mapped = df[col].map(mapping)
            if mapped.isna().any():
                bad_vals = df.loc[mapped.isna(), col].unique().tolist()
                raise ValueError(f"Unmapped values in {cfg.key}.{col}: {bad_vals}")
            df[col] = mapped.astype(float)

    excluded_set = set(cfg.exclude_columns) | {cfg.target, "_raw_row_id", "_parsed_time"}
    feature_cols = [c for c in df.columns if c not in excluded_set]
    categorical_cols = [c for c in cfg.categorical_columns if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # Safely coerce numeric predictors and stringify categorical predictors
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    info = {
        "n_raw": n_raw,
        "n_clean": len(df),
        "dropped_summary_rows": dropped_summary_count,
        "dropped_missing_target_rows": dropped_missing_target_count,
        "duplicate_rows_in_clean": int(df.drop(columns=["_raw_row_id"]).duplicated().sum()),
        "missing_values_in_features": {
            col: int(df[col].isna().sum()) for col in feature_cols if df[col].isna().any()
        },
    }
    return df, info


def _compute_split_indices(
    df: pd.DataFrame, cfg: DatasetConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute train/test raw row indices according to dataset split strategy."""
    raw_ids = df["_raw_row_id"].to_numpy(dtype=int)
    n = len(df)

    if cfg.split_strategy == "random":
        train_ids, test_ids = train_test_split(
            raw_ids, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True
        )
        return np.sort(train_ids), np.sort(test_ids)

    if cfg.split_strategy == "stratified":
        train_ids, test_ids = train_test_split(
            raw_ids,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=df[cfg.target].to_numpy(),
        )
        return np.sort(train_ids), np.sort(test_ids)

    if cfg.split_strategy == "group":
        if not cfg.group_column or cfg.group_column not in df.columns:
            raise ValueError(f"Missing group column for {cfg.key}")
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        tr_pos, te_pos = next(splitter.split(df, groups=df[cfg.group_column]))
        return np.sort(raw_ids[tr_pos]), np.sort(raw_ids[te_pos])

    if cfg.split_strategy == "chronological":
        cutoff = int(round((1.0 - TEST_SIZE) * n))
        # Preserve chronological order
        return raw_ids[:cutoff], raw_ids[cutoff:]

    if cfg.split_strategy == "chronological_date":
        if "_parsed_time" not in df.columns:
            raise ValueError(f"Missing parsed time column for {cfg.key}")
        unique_dates = np.sort(df["_parsed_time"].unique())
        cutoff_idx = int(round((1.0 - TEST_SIZE) * len(unique_dates)))
        train_dates = set(unique_dates[:cutoff_idx])
        is_train = df["_parsed_time"].isin(train_dates).to_numpy()
        return raw_ids[is_train], raw_ids[~is_train]

    raise ValueError(f"Unsupported split strategy: {cfg.split_strategy}")


def _build_cv_splits(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    cfg: DatasetConfig,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    """Create training-only CV fold index pairs matching the dataset structure."""
    if cfg.split_strategy == "group":
        groups = df_train[cfg.group_column].to_numpy()
        splitter = GroupKFold(n_splits=N_CV_FOLDS)
        return list(splitter.split(df_train, y_train, groups=groups)), f"GroupKFold({N_CV_FOLDS}, group={cfg.group_column})"

    if cfg.split_strategy == "stratified":
        splitter = StratifiedKFold(
            n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
        )
        return list(splitter.split(df_train, y_train)), f"StratifiedKFold({N_CV_FOLDS}, seed={RANDOM_SEED})"

    if cfg.split_strategy == "chronological":
        splitter = TimeSeriesSplit(n_splits=N_CV_FOLDS)
        return list(splitter.split(df_train)), f"TimeSeriesSplit({N_CV_FOLDS})"

    if cfg.split_strategy == "chronological_date":
        # Expanding-window date-level splits so same-day station records stay together
        unique_dates = np.sort(df_train["_parsed_time"].unique())
        date_splitter = TimeSeriesSplit(n_splits=N_CV_FOLDS)
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        train_dates_col = df_train["_parsed_time"].to_numpy()
        for tr_d_idx, val_d_idx in date_splitter.split(unique_dates):
            tr_dates = set(unique_dates[tr_d_idx])
            val_dates = set(unique_dates[val_d_idx])
            tr_pos = np.where(np.isin(train_dates_col, list(tr_dates)))[0]
            val_pos = np.where(np.isin(train_dates_col, list(val_dates)))[0]
            folds.append((tr_pos, val_pos))
        return folds, f"DateGroupedTimeSeriesSplit({N_CV_FOLDS})"

    splitter = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    return list(splitter.split(df_train)), f"KFold({N_CV_FOLDS}, shuffle=True, seed={RANDOM_SEED})"


def _extract_scaler_params(
    preprocessor: ColumnTransformer,
    numeric_columns: List[str],
    categorical_columns: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature-aligned means and standard deviations from a fitted ColumnTransformer."""
    means = []
    scales = []
    if numeric_columns:
        num_scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
        means.append(num_scaler.mean_)
        scales.append(num_scaler.scale_)
    if categorical_columns:
        cat_scaler = preprocessor.named_transformers_["cat"].named_steps["scaler"]
        means.append(cat_scaler.mean_)
        scales.append(cat_scaler.scale_)
    return np.concatenate(means), np.concatenate(scales)


def load_and_prepare_dataset(
    dataset_key: str,
    force_resplit: bool = False,
) -> PreparedDataset:
    """Load raw data, persist/reuse split indices, and fit train-only preprocessing."""
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_key}'")
    cfg = DATASETS[dataset_key]
    raw_df = load_raw_dataset(dataset_key)
    clean_df, clean_info = _clean_and_order_dataframe(raw_df, cfg)

    excluded_set = set(cfg.exclude_columns) | {cfg.target, "_raw_row_id", "_parsed_time"}
    feature_cols = [c for c in clean_df.columns if c not in excluded_set]
    categorical_cols = [c for c in cfg.categorical_columns if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    processed_dir = PROCESSED_DATA_DIR / dataset_key
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_path = processed_dir / "split.json"

    if split_path.exists() and not force_resplit:
        with open(split_path, "r", encoding="utf-8") as f:
            split_payload = json.load(f)
        train_raw_indices = np.array(split_payload["train_raw_indices"], dtype=int)
        test_raw_indices = np.array(split_payload["test_raw_indices"], dtype=int)
    else:
        train_raw_indices, test_raw_indices = _compute_split_indices(clean_df, cfg)
        split_payload = {
            "dataset": dataset_key,
            "uci_id": cfg.uci_id,
            "target": cfg.target,
            "split_strategy": cfg.split_strategy,
            "seed": RANDOM_SEED,
            "test_size": TEST_SIZE,
            "n_raw": clean_info["n_raw"],
            "n_clean": clean_info["n_clean"],
            "n_train": int(len(train_raw_indices)),
            "n_test": int(len(test_raw_indices)),
            "train_raw_indices": train_raw_indices.tolist(),
            "test_raw_indices": test_raw_indices.tolist(),
        }
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_payload, f, indent=2)

    # Index clean_df using _raw_row_id while preserving chronological order when applicable
    train_set = set(train_raw_indices.tolist())
    test_set = set(test_raw_indices.tolist())
    if train_set & test_set:
        raise RuntimeError(f"Train/test split overlap detected for {dataset_key}!")

    df_train_full = clean_df[clean_df["_raw_row_id"].isin(train_set)].reset_index(drop=True)
    df_test_full = clean_df[clean_df["_raw_row_id"].isin(test_set)].reset_index(drop=True)

    if cfg.group_column and cfg.group_column in clean_df.columns:
        train_groups = set(df_train_full[cfg.group_column].unique().tolist())
        test_groups = set(df_test_full[cfg.group_column].unique().tolist())
        if train_groups & test_groups:
            raise RuntimeError(
                f"Group overlap on '{cfg.group_column}' in {dataset_key}: {train_groups & test_groups}"
            )

    y_train = df_train_full[cfg.target].to_numpy(dtype=float)
    y_test = df_test_full[cfg.target].to_numpy(dtype=float)
    train_ids_ordered = df_train_full["_raw_row_id"].to_numpy(dtype=int)
    test_ids_ordered = df_test_full["_raw_row_id"].to_numpy(dtype=int)

    # Fit preprocessors strictly on training split
    scaled_prep = build_preprocessor(numeric_cols, categorical_cols, standardize=True)
    X_train = scaled_prep.fit_transform(df_train_full[feature_cols])
    X_test = scaled_prep.transform(df_test_full[feature_cols])

    unscaled_prep = build_preprocessor(numeric_cols, categorical_cols, standardize=False)
    X_train_unscaled = unscaled_prep.fit_transform(df_train_full[feature_cols])
    X_test_unscaled = unscaled_prep.transform(df_test_full[feature_cols])

    feature_names = [str(name) for name in scaled_prep.get_feature_names_out()]
    scaler_means, scaler_scales = _extract_scaler_params(
        scaled_prep, numeric_cols, categorical_cols
    )

    cv_splits, cv_strategy_name = _build_cv_splits(df_train_full, y_train, cfg)

    prep_info: Dict[str, Any] = {
        **clean_info,
        "n_train": int(len(df_train_full)),
        "n_test": int(len(df_test_full)),
        "n_raw_features": len(feature_cols),
        "n_encoded_features": len(feature_names),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "excluded_columns": list(cfg.exclude_columns),
        "feature_names": feature_names,
        "split_strategy": cfg.split_strategy,
        "cv_strategy": cv_strategy_name,
        "target_stats_train": {
            "mean": float(np.mean(y_train)),
            "std": float(np.std(y_train, ddof=0)),
            "min": float(np.min(y_train)),
            "max": float(np.max(y_train)),
        },
        "target_stats_test": {
            "mean": float(np.mean(y_test)),
            "std": float(np.std(y_test, ddof=0)),
            "min": float(np.min(y_test)),
            "max": float(np.max(y_test)),
        },
    }
    if cfg.time_column and "_parsed_time" in df_train_full.columns:
        prep_info["time_range_train"] = [
            str(df_train_full["_parsed_time"].min()),
            str(df_train_full["_parsed_time"].max()),
        ]
        prep_info["time_range_test"] = [
            str(df_test_full["_parsed_time"].min()),
            str(df_test_full["_parsed_time"].max()),
        ]
    if cfg.group_column and cfg.group_column in df_train_full.columns:
        prep_info["n_groups_train"] = int(df_train_full[cfg.group_column].nunique())
        prep_info["n_groups_test"] = int(df_test_full[cfg.group_column].nunique())

    with open(processed_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(prep_info, f, indent=2)

    return PreparedDataset(
        config=cfg,
        df_train_raw=df_train_full[feature_cols].copy(),
        df_test_raw=df_test_full[feature_cols].copy(),
        y_train=y_train,
        y_test=y_test,
        train_raw_indices=train_ids_ordered,
        test_raw_indices=test_ids_ordered,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        X_train=X_train,
        X_test=X_test,
        X_train_unscaled=X_train_unscaled,
        X_test_unscaled=X_test_unscaled,
        feature_names=feature_names,
        scaler_means=scaler_means,
        scaler_scales=scaler_scales,
        cv_splits=cv_splits,
        cv_strategy_name=cv_strategy_name,
        split_path=split_path,
        preprocessing_info=prep_info,
    )
