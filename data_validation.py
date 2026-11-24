import pandas as pd
import os
DATA_PATH = r"C:\Users\manik\Downloads\smart_healthcare_monitoring_dataset.csv"
def validate_dataset():
    print("🔍 Starting Data Validation...\n")

    if not os.path.exists(DATA_PATH):
        print("❌ Dataset not found!")
        return

    df = pd.read_csv(DATA_PATH)

    print("✅ Dataset loaded successfully!\n")

    print("📊 Dataset Shape:")
    print(df.shape)

    print("\n📋 Columns:")
    print(df.columns.tolist())

    print("\n🔢 Data Types:")
    print(df.dtypes)

    print("\n❓ Missing Values:")
    print(df.isnull().sum())

    print("\n🔁 Duplicate Records:")
    print(df.duplicated().sum())

    print("\n📈 Statistical Summary:")
    print(df.describe())

    print("\n✅ Data Validation Completed!")


if __name__ == "__main__":
    validate_dataset()