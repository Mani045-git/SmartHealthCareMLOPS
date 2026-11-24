import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input dataset
DATA_PATH = BASE_DIR / "data" / "cleaned_healthcare_dataset.csv"

# Output directory
OUTPUT_DIR = BASE_DIR / "data" / "processed"

X_TRAIN_PATH = OUTPUT_DIR / "X_train.csv"
X_TEST_PATH = OUTPUT_DIR / "X_test.csv"
Y_TRAIN_PATH = OUTPUT_DIR / "y_train.csv"
Y_TEST_PATH = OUTPUT_DIR / "y_test.csv"


def feature_engineering():

    print("Starting Feature Engineering...\n")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    target_columns = [
        "heart_risk",
        "diabetes_risk",
        "hypertension_risk",
        "kidney_risk",
        "stroke_risk",
        "cancer_screening_risk",
        "liver_risk",
        "respiratory_risk",
        "overall_health_risk"
    ]

    missing_targets = [
        col for col in target_columns if col not in df.columns
    ]

    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    feature_columns = [
        col for col in df.columns
        if col not in target_columns and col != "patient_id"
    ]

    X = df[feature_columns].copy()
    y = df[target_columns].copy()

    print("\nNumber of features:", len(feature_columns))
    print("Number of targets:", len(target_columns))

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print("\nCategorical features:", categorical_features)

    X = pd.get_dummies(
        X,
        columns=categorical_features,
        drop_first=True,
        dtype=int
    )

    risk_mapping = {
        "Low": 0,
        "Moderate": 1,
        "High": 2
    }

    for column in target_columns:
        if y[column].dtype == "object":
            y[column] = y[column].map(risk_mapping)

    if y.isnull().any().any():
        raise ValueError(
            "Target columns contain missing or unmapped values. "
            "Check target labels before training."
        )

    if X.isnull().any().any():
        raise ValueError(
            "Feature columns contain missing values. "
            "Check preprocessing."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y["overall_health_risk"]
    )

    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)

    print("\nFeature Engineering Results")
    print("--------------------------------")
    print("Original records:", len(df))
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Features after encoding:", X_train.shape[1])
    print("Target columns:", y_train.shape[1])

    print("\nTraining shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    print("\nTarget mapping:")
    print("Low      -> 0")
    print("Moderate -> 1")
    print("High     -> 2")

    print("\nFeature Engineering Completed!")

    print("\nFiles created:")
    print(X_TRAIN_PATH)
    print(X_TEST_PATH)
    print(Y_TRAIN_PATH)
    print(Y_TEST_PATH)


if __name__ == "__main__":
    feature_engineering()