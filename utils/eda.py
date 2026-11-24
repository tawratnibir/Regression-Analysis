import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RESULT_DIR = ROOT / "results" / "eda"


def perform_eda(name, target):
    df = pd.read_csv(DATA_DIR / name / "data.csv")

    output = RESULT_DIR / name
    output.mkdir(parents=True, exist_ok=True)

    # 1. Descriptive statistics
    df.describe().to_csv(output / "statistics.csv")

    # 2. Feature distributions
    df.hist(figsize=(15, 10), bins=30)
    plt.tight_layout()
    plt.savefig(output / "distributions.png", dpi=200)
    plt.close()

    # 3. Correlation matrix
    corr = df.corr(numeric_only=True)
    corr.to_csv(output / "correlation.csv")

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f"
    )

    plt.tight_layout()
    plt.savefig(output / "correlation.png", dpi=200)
    plt.close()

    # 4. Feature vs target scatter plots
    features = df.drop(columns=[target]).columns

    for feature in features:
        plt.figure(figsize=(7, 5))

        sns.scatterplot(
            data=df,
            x=feature,
            y=target,
            alpha=0.5
        )

        plt.tight_layout()
        plt.savefig(
            output / f"{feature}_vs_target.png",
            dpi=200
        )
        plt.close()

    # 5. Box plots for outlier inspection
    for column in df.select_dtypes(include="number"):
        plt.figure(figsize=(7, 4))
        sns.boxplot(x=df[column])
        plt.tight_layout()
        plt.savefig(
            output / f"{column}_boxplot.png",
            dpi=200
        )
        plt.close()

    print(f"EDA completed: {name}")
    print(f"Results saved in: {output}")


if __name__ == "__main__":
    # perform_eda("airfoil", "scaled-sound-pressure")
    perform_eda("parkinsons", "total_UPDRS")
