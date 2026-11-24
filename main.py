"""CLI entrypoint for preprocessing, training, evaluation, and HTML reporting."""

import argparse
import os
import sys
from typing import Callable, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache_reg_analysis")

from config import DATASETS, MODEL_KEYS, REPORTS_DIR, RESULTS_DIR
from models.lasso import run_lasso
from models.ols import run_ols
from models.ridge import run_ridge
from preprocessing.eda import perform_eda
from preprocessing.preprocess import PreparedDataset, load_and_prepare_dataset
from reporting.generate_report import generate_dataset_report, generate_summary_report
from reporting.grand_index import generate_grand_index

MODEL_RUNNERS: Dict[str, Callable[[PreparedDataset], object]] = {
    "ols": run_ols,
    "ridge": run_ridge,
    "lasso": run_lasso,
}


def _run_exists(model_key: str, dataset_key: str) -> bool:
    metrics_path = RESULTS_DIR / model_key / dataset_key / "metrics.json"
    report_path = REPORTS_DIR / model_key / f"{dataset_key}.html"
    return metrics_path.exists() and report_path.exists()


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OLS, Ridge, and Lasso regression on 10 UCI datasets."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[*MODEL_KEYS, "all"],
        help="Model to run (ols, ridge, lasso, or all).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[*DATASETS.keys(), "all"],
        help="Dataset key to process, or 'all'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly overwrite existing results and reports instead of skipping.",
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="Recompute persisted train/test splits under data/processed/.",
    )
    parser.add_argument(
        "--run-eda",
        action="store_true",
        help="Also generate EDA summaries for selected datasets.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    target_models = list(MODEL_KEYS) if args.model == "all" else [args.model]
    target_datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    failures: List[str] = []
    prepared_cache: Dict[str, PreparedDataset] = {}

    for dataset_key in target_datasets:
        if args.run_eda:
            try:
                perform_eda(dataset_key, overwrite=args.overwrite)
            except Exception as err:
                print(f"[ERROR] EDA failed for {dataset_key}: {err}")

        needs_run = any(
            args.overwrite or not _run_exists(m, dataset_key) for m in target_models
        )
        if not needs_run:
            for m in target_models:
                print(
                    f"[SKIP] {m}/{dataset_key}: prior run exists "
                    "(pass --overwrite to regenerate)."
                )
            continue

        try:
            dataset = load_and_prepare_dataset(
                dataset_key, force_resplit=args.force_resplit
            )
            prepared_cache[dataset_key] = dataset
            print(
                f"[DATA] {dataset_key}: train={dataset.X_train.shape}, "
                f"test={dataset.X_test.shape}, split={dataset.config.split_strategy}, "
                f"cv={dataset.cv_strategy_name}"
            )
        except Exception as err:
            msg = f"preprocessing/{dataset_key}: {err}"
            print(f"[ERROR] {msg}")
            failures.append(msg)
            continue

        for model_key in target_models:
            if not args.overwrite and _run_exists(model_key, dataset_key):
                print(
                    f"[SKIP] {model_key}/{dataset_key}: prior run exists "
                    "(pass --overwrite to regenerate)."
                )
                continue

            if _run_exists(model_key, dataset_key) and args.overwrite:
                print(f"[OVERWRITE] Regenerating {model_key}/{dataset_key}...")

            try:
                res_dir = MODEL_RUNNERS[model_key](dataset)
                report_path = generate_dataset_report(model_key, dataset_key)
                print(f"[OK] {model_key}/{dataset_key} -> {res_dir} | {report_path}")
            except Exception as err:
                msg = f"{model_key}/{dataset_key}: {err}"
                print(f"[ERROR] {msg}")
                failures.append(msg)

    for model_key in target_models:
        try:
            summary_path = generate_summary_report(model_key)
            print(f"[SUMMARY] {model_key} -> {summary_path}")
        except Exception as err:
            msg = f"summary/{model_key}: {err}"
            print(f"[ERROR] {msg}")
            failures.append(msg)

    try:
        index_path = generate_grand_index()
        print(f"[INDEX] grand dashboard -> {index_path}")
    except Exception as err:
        msg = f"grand_index: {err}"
        print(f"[ERROR] {msg}")
        failures.append(msg)

    if failures:
        print(f"\nCompleted with {len(failures)} error(s):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nAll requested runs completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
