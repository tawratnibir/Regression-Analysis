"""Lightweight EDA script refactored from utils/eda.py using Matplotlib."""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATASETS, RESULTS_DIR
from preprocessing.loader import load_raw_dataset


def perform_eda(dataset_key: str, overwrite: bool = False) -> Path:
    """Generate summary CSVs and key exploratory plots under results/eda/<key>/."""
    cfg = DATASETS[dataset_key]
    output = RESULTS_DIR / "eda" / dataset_key
    stats_path = output / "statistics.csv"
    if stats_path.exists() and not overwrite:
        print(f"Skipping EDA for {dataset_key}: existing results at {output}")
        return output

    output.mkdir(parents=True, exist_ok=True)
    df = load_raw_dataset(dataset_key)

    # 1. Descriptive statistics
    df.describe(include="all").to_csv(stats_path)

    # 2. Correlation matrix on numeric columns
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        corr = num_df.corr()
        corr.to_csv(output / "correlation.csv")

        n_cols = min(len(corr.columns), 20)
        fig_size = max(7, min(14, 0.55 * n_cols + 2))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
        sub_corr = corr.iloc[:n_cols, :n_cols]
        im = ax.imshow(sub_corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(sub_corr.columns)))
        ax.set_yticks(range(len(sub_corr.index)))
        ax.set_xticklabels(sub_corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(sub_corr.index, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{cfg.name} - Correlation Matrix")
        fig.tight_layout()
        fig.savefig(output / "correlation.png", dpi=160)
        plt.close(fig)

    # 3. Target distribution
    target_series = df[cfg.target].dropna()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if pd.api.types.is_numeric_dtype(target_series):
        ax.hist(target_series.values, bins=30, color="#2b6cb0", edgecolor="white", alpha=0.85)
    else:
        counts = target_series.value_counts()
        ax.bar(counts.index.astype(str), counts.values, color="#2b6cb0", alpha=0.85)
    ax.set_title(f"{cfg.name} - Target Distribution ({cfg.target})")
    ax.set_xlabel(f"{cfg.target} ({cfg.target_units})")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output / "target_distribution.png", dpi=160)
    plt.close(fig)

    print(f"EDA completed for {dataset_key}: {output}")
    return output


def run_all_eda(overwrite: bool = False) -> None:
    """Run EDA across all configured datasets."""
    for key in DATASETS:
        perform_eda(key, overwrite=overwrite)
