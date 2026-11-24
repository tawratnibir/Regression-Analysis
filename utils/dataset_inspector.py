import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def inspect_dataset(name):
    df = pd.read_csv(
        DATA_DIR / name / "data.csv"
    )

    print("\nDataset:", name)

    print("\nDimensions:")
    print(df.shape)

    print("\nFirst five observations:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nDescriptive statistics:")
    print(df.describe())

    print("\nNon-numeric columns:")
    print(
        df.select_dtypes(
            exclude="number"
        ).columns.tolist()
    )


if __name__ == "__main__":
    inspect_dataset("airfoil")