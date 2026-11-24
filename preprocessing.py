import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
DATA_PATH = BASE_DIR / "data" / "smart_healthcare_monitoring_dataset.csv"
OUTPUT_PATH = BASE_DIR / "data" / "cleaned_healthcare_dataset.csv"


def preprocess_data():

    print("Starting Data Preprocessing...\n")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("Original shape:", df.shape)

    df = df.drop_duplicates()

    numerical_columns = [
        "bmi",
        "heart_rate_bpm",
        "systolic_bp",
        "diastolic_bp",
        "spo2_percent",
        "glucose_mg_dl",
        "cholesterol_mg_dl",
        "hemoglobin_g_dl",
        "creatinine_mg_dl",
        "sleep_hours"
    ]

    for column in numerical_columns:
        if column in df.columns:
            df[column] = df[column].fillna(df[column].median())

    categorical_columns = [
        "gender",
        "physical_activity",
        "smoking",
        "alcohol",
        "stress_level"
    ]

    for column in categorical_columns:
        if column in df.columns:
            if df[column].notna().any():
                df[column] = df[column].fillna(df[column].mode()[0])

    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nData Preprocessing Completed!")
    print("Cleaned dataset shape:", df.shape)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    preprocess_data()