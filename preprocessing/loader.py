"""Dataset download and raw CSV loading utilities.

Refactored from utils/dataset_loader.py so raw downloads are preserved and
optional network dependencies (ucimlrepo) are only imported if downloading.
"""

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from config import DATASETS, RAW_DATA_DIR


def download_zip_csv(url: str, filename: str, sep: str = ",") -> pd.DataFrame:
    """Download a ZIP archive from URL and read a specific CSV inside it."""
    with urlopen(url, timeout=60) as response:
        archive = ZipFile(BytesIO(response.read()))

    with archive.open(filename) as file:
        return pd.read_csv(file, sep=sep)


def download_datasets(force: bool = False) -> None:
    """Download any missing raw UCI datasets into data/raw/<key>/data.csv."""
    for name, cfg in DATASETS.items():
        folder = RAW_DATA_DIR / name
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / "data.csv"

        if output.exists() and not force:
            print(f"Skipping {name}: already downloaded.")
            continue

        print(f"Downloading {name} (UCI ID {cfg.uci_id})...")
        try:
            if cfg.uci_id == 186:
                df = download_zip_csv(
                    "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
                    "winequality-red.csv",
                    sep=";",
                )
            elif cfg.uci_id == 514:
                df = download_zip_csv(
                    "https://archive.ics.uci.edu/static/public/514/bias+correction+of+numerical+prediction+model+temperature+forecast.zip",
                    "Bias_correction_ucl.csv",
                )
            else:
                from ucimlrepo import fetch_ucirepo

                dataset = fetch_ucirepo(id=cfg.uci_id)
                df = dataset.data.original
                dataset.variables.to_csv(folder / "variables.csv", index=False)

            df.to_csv(output, index=False)
            print(f"Downloaded {name}: {df.shape}")
        except Exception as error:
            print(f"Failed {name}: {error}")


def load_raw_dataset(dataset_key: str) -> pd.DataFrame:
    """Load the unmodified raw CSV for a configured dataset key."""
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Valid keys: {list(DATASETS)}")
    path = Path(RAW_DATA_DIR) / dataset_key / "data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")
    return pd.read_csv(path)
