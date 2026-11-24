from pathlib import Path

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Test data paths
X_TEST_PATH = BASE_DIR / "data" / "processed" / "X_test.csv"
Y_TEST_PATH = BASE_DIR / "data" / "processed" / "y_test.csv"

# Model path
MODEL_PATH = BASE_DIR / "models" / "healthcare_model.pkl"

# Evaluation output path
OUTPUT_PATH = BASE_DIR / "reports" / "model_evaluation.csv"


def evaluate_model():

    print("📊 Starting Model Evaluation...\n")

    # Check required files
    for path in [X_TEST_PATH, Y_TEST_PATH, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load test data
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    # Load trained model
    model = joblib.load(MODEL_PATH)

    print("✅ Test data loaded")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Generate predictions
    predictions = model.predict(X_test)

    print("\n✅ Predictions generated successfully!")

    target_columns = y_test.columns.tolist()
    results = []

    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)

    for i, target in enumerate(target_columns):

        y_true = y_test[target]
        y_pred = predictions[:, i]

        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        )

        print(f"\n🎯 {target}")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            y_true,
            y_pred,
            zero_division=0
        ))

        results.append({
            "target": target,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    # Save evaluation results
    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("✅ MODEL EVALUATION COMPLETED!")
    print("=" * 70)

    print("\n📁 Evaluation results saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    evaluate_model()