from pathlib import Path

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Training data paths
X_TRAIN_PATH = BASE_DIR / "data" / "processed" / "X_train.csv"
Y_TRAIN_PATH = BASE_DIR / "data" / "processed" / "y_train.csv"

# Model output path
MODEL_PATH = BASE_DIR / "models" / "healthcare_model.pkl"


def train_model():

    print("🤖 Starting ML Model Training...\n")

    # Check input files
    if not X_TRAIN_PATH.exists() or not Y_TRAIN_PATH.exists():
        raise FileNotFoundError("Training CSV files not found in data/processed/")

    # Load training data
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)

    print("Training features shape:", X_train.shape)
    print("Training targets shape:", y_train.shape)

    # Random Forest base model
    base_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    # Multi-output model
    model = MultiOutputClassifier(base_model)

    # Train
    print("\n🌲 Training Random Forest models...")
    model.fit(X_train, y_train)

    print("✅ Model training completed!")

    # Create models folder if needed
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, MODEL_PATH)

    print("\n💾 Model saved successfully:")
    print(MODEL_PATH)


if __name__ == "__main__":
    train_model()