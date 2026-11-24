import pandas as pd
from pathlib import Path
from ucimlrepo import fetch_ucirepo
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

DATASETS = {
    "airfoil": 291,
    "power_plant": 294,
    "real_estate": 477,
    "wine_quality_red": 186,
    "parkinsons": 189,
    "superconductivity": 464,
    "auto_mpg": 9,
    "appliances_energy": 374,
    "diabetes": 529,
    "weather": 514
}


def download_zip_csv(url, filename, sep=","):
    with urlopen(url, timeout=60) as response:
        archive = ZipFile(BytesIO(response.read()))

    with archive.open(filename) as file:
        return pd.read_csv(file, sep=sep)


def download_datasets():
    for name, dataset_id in DATASETS.items():

        folder = DATA_DIR / name
        folder.mkdir(parents=True, exist_ok=True)

        output = folder / "data.csv"

        if output.exists():
            print(f"Skipping {name}: already downloaded.")
            continue

        print(f"Downloading {name}...")

        try:
            if dataset_id == 186:
                df = download_zip_csv(
                    "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
                    "winequality-red.csv",
                    sep=";"
                )

            elif dataset_id == 514:
                df = download_zip_csv(
                    "https://archive.ics.uci.edu/static/public/514/bias+correction+of+numerical+prediction+model+temperature+forecast.zip",
                    "Bias_correction_ucl.csv"
                )

            else:
                dataset = fetch_ucirepo(id=dataset_id)
                df = dataset.data.original

                dataset.variables.to_csv(
                    folder / "variables.csv",
                    index=False
                )

            df.to_csv(output, index=False)

            print(f"Downloaded: {df.shape}")

        except Exception as error:
            print(f"Failed: {name}: {error}")


if __name__ == "__main__":
    download_datasets()