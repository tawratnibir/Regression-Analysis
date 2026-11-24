"""Central configuration for all 10 UCI datasets and shared experiment settings."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
REPORTS_DIR = ROOT_DIR / "reports"

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5

MODEL_KEYS = ("ols", "ridge", "lasso")

DIABETES_BINARY_FEATURES = [
    "polyuria",
    "polydipsia",
    "sudden_weight_loss",
    "weakness",
    "polyphagia",
    "genital_thrush",
    "visual_blurring",
    "itching",
    "irritability",
    "delayed_healing",
    "partial_paresis",
    "muscle_stiffness",
    "alopecia",
    "obesity",
]


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration defining target, exclusions, splitting, and caveats for a dataset."""

    key: str
    uci_id: int
    name: str
    description: str
    target: str
    target_units: str
    split_strategy: str  # "random", "stratified", "group", "chronological", "chronological_date"
    exclude_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    binary_mappings: Dict[str, Dict[str, int]] = field(default_factory=dict)
    target_mapping: Optional[Dict[str, int]] = None
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    time_format: Optional[str] = None
    secondary_sort_column: Optional[str] = None
    is_binary_classification: bool = False
    is_ordinal_target: bool = False
    drop_missing_id_time_summary_rows: bool = False
    notes: List[str] = field(default_factory=list)


DATASETS: Dict[str, DatasetConfig] = {
    "airfoil": DatasetConfig(
        key="airfoil",
        uci_id=291,
        name="Airfoil Self-Noise",
        description=(
            "NASA wind tunnel aerodynamic and acoustic tests of two- and three-dimensional "
            "airfoil blade sections at various speeds and angles of attack."
        ),
        target="scaled-sound-pressure",
        target_units="dB",
        split_strategy="random",
        notes=[
            "All 5 predictors and target are numeric with no missing values.",
            "Predictors span very different scales (e.g., frequency in Hz up to 20,000 vs displacement thickness ~0.001 m).",
        ],
    ),
    "power_plant": DatasetConfig(
        key="power_plant",
        uci_id=294,
        name="Combined Cycle Power Plant",
        description=(
            "Hourly average ambient environmental sensor readings (temperature, exhaust vacuum, "
            "ambient pressure, relative humidity) from a combined cycle power plant over 6 years."
        ),
        target="PE",
        target_units="MW",
        split_strategy="random",
        notes=[
            "Contains 41 exact duplicate rows; all valid measurements are preserved as recorded.",
            "Target PE is net hourly electrical energy output in megawatts (MW).",
        ],
    ),
    "real_estate": DatasetConfig(
        key="real_estate",
        uci_id=477,
        name="Real Estate Valuation",
        description=(
            "Historical market dataset of real estate valuation in Sindian District, New Taipei City, Taiwan."
        ),
        target="Y house price of unit area",
        target_units="10,000 NTD/Ping",
        split_strategy="random",
        exclude_columns=["No"],
        notes=[
            "Row identifier 'No' is excluded to prevent arbitrary index leakage.",
            "Target unit is 10,000 New Taiwan Dollar per Ping (1 Ping = 3.3 m^2).",
        ],
    ),
    "wine_quality_red": DatasetConfig(
        key="wine_quality_red",
        uci_id=186,
        name="Wine Quality (Red Wine)",
        description=(
            "Physicochemical laboratory measurements of red variants of Portuguese 'Vinho Verde' wine."
        ),
        target="quality",
        target_units="score (0-10 ordinal)",
        split_strategy="random",
        is_ordinal_target=True,
        notes=[
            "Uses red wine samples only (1,599 rows), excluding white wine.",
            "Target 'quality' is an ordinal integer sensory score (observed range 3 to 8), treated continuously by linear regression.",
            "Contains 240 exact duplicate feature/target rows; preserved per dataset fidelity policy.",
        ],
    ),
    "parkinsons": DatasetConfig(
        key="parkinsons",
        uci_id=189,
        name="Parkinsons Telemonitoring",
        description=(
            "Biomedical voice measurements from 42 people with early-stage Parkinson's disease "
            "recruited to a six-month trial of a telemonitoring device."
        ),
        target="total_UPDRS",
        target_units="UPDRS points",
        split_strategy="group",
        group_column="subject#",
        exclude_columns=["subject#", "motor_UPDRS"],
        notes=[
            "'subject#' is excluded from predictors and used for group-aware 80/20 splitting and 5-fold GroupKFold CV so subjects never overlap between train and test.",
            "'motor_UPDRS' is excluded because it is a closely related co-target component of 'total_UPDRS', avoiding target leakage.",
            "Many voice jitter/shimmer features are strongly collinear.",
        ],
    ),
    "superconductivity": DatasetConfig(
        key="superconductivity",
        uci_id=464,
        name="Superconductivity Data",
        description=(
            "81 elemental and weighted chemical/physical summary features extracted from superconducting materials."
        ),
        target="critical_temp",
        target_units="K",
        split_strategy="random",
        notes=[
            "High-dimensional feature space (81 predictors) with strong multicollinearity among mean, weighted mean, geometric mean, entropy, range, and std feature families.",
            "Contains 66 exact duplicate rows; all valid records are preserved.",
        ],
    ),
    "auto_mpg": DatasetConfig(
        key="auto_mpg",
        uci_id=9,
        name="Auto MPG",
        description=(
            "City-cycle fuel consumption in miles per gallon for late-1970s and early-1980s automobiles."
        ),
        target="mpg",
        target_units="mpg",
        split_strategy="random",
        exclude_columns=["car_name"],
        categorical_columns=["origin"],
        notes=[
            "'car_name' is excluded as a high-cardinality string identifier.",
            "'horsepower' contains 6 missing values, imputed using the training-split median only.",
            "'origin' (1=USA, 2=Europe, 3=Japan) is one-hot encoded (drop first) rather than treated as an ordinal continuous scale.",
        ],
    ),
    "appliances_energy": DatasetConfig(
        key="appliances_energy",
        uci_id=374,
        name="Appliances Energy Prediction",
        description=(
            "10-minute interval house temperature, humidity, weather, and appliance energy consumption "
            "measurements over ~4.5 months in a low-energy house in Belgium."
        ),
        target="Appliances",
        target_units="Wh",
        split_strategy="chronological",
        time_column="date",
        time_format="%Y-%m-%d%H:%M:%S",
        exclude_columns=["date"],
        notes=[
            "Sorted chronologically by 'date'; first 80% used for training and final 20% held out for testing.",
            "TimeSeriesSplit (5 folds) is used within training data to prevent future leakage during CV.",
            "Random probe variables 'rv1' and 'rv2' are numerically identical across all rows, making the raw design matrix rank-deficient by 1.",
        ],
    ),
    "diabetes": DatasetConfig(
        key="diabetes",
        uci_id=529,
        name="Early Stage Diabetes Risk Prediction",
        description=(
            "Symptom and demographic questionnaire records collected from patients at Sylhet Diabetes Hospital, Bangladesh."
        ),
        target="class",
        target_units="probability P(Positive)",
        split_strategy="stratified",
        binary_mappings={
            "gender": {"Male": 1, "Female": 0},
            **{col: {"Yes": 1, "No": 0} for col in DIABETES_BINARY_FEATURES},
        },
        target_mapping={"Positive": 1, "Negative": 0},
        is_binary_classification=True,
        notes=[
            "Binary target ('Positive'=1, 'Negative'=0) is modeled via linear regression as a Linear Probability Model (LPM).",
            "Both regression metrics and binary classification metrics (threshold 0.50, ROC AUC) plus out-of-[0,1] prediction rates are reported.",
            "Stratified 80/20 split (seed=42) and StratifiedKFold (5 folds) preserve class balance.",
        ],
    ),
    "weather": DatasetConfig(
        key="weather",
        uci_id=514,
        name="Bias Correction of Numerical Prediction Model Temperature Forecast",
        description=(
            "Summer next-day maximum temperature forecasts from Korea Meteorological Administration LDAPS "
            "across 25 Seoul weather stations (2013-2017)."
        ),
        target="Next_Tmax",
        target_units="deg C",
        split_strategy="chronological_date",
        time_column="Date",
        time_format="%Y-%m-%d",
        secondary_sort_column="station",
        exclude_columns=["Date", "station", "Next_Tmin"],
        drop_missing_id_time_summary_rows=True,
        notes=[
            "Two trailing summary rows with missing Date/station (min/max rows in raw UCI CSV) and 27 rows with missing target 'Next_Tmax' are dropped prior to splitting.",
            "'Next_Tmin' is excluded as a co-target not available at prediction time; 'station' is excluded because station geography is captured by lat/lon/DEM/Slope.",
            "Predictor missing values (~70-75 rows) are median-imputed using training-fold statistics only.",
            "Chronological date-level 80/20 split ensures all stations on a given date appear together with zero same-day or future leakage.",
        ],
    ),
}
