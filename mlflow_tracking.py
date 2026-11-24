from pathlib import Path

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# MLflow tracking database
MLFLOW_DB = BASE_DIR / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")

# Test data paths
X_TEST_PATH = BASE_DIR / "data" / "processed" / "X_test.csv"
Y_TEST_PATH = BASE_DIR / "data" / "processed" / "y_test.csv"

# Model path
MODEL_PATH = BASE_DIR / "models" / "healthcare_model.pkl"


def track_model():

    print("📊 Starting MLflow Tracking...\n")

    # Check required files
    for path in [X_TEST_PATH, Y_TEST_PATH, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    # Load test data
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    # Load trained model
    model = joblib.load(MODEL_PATH)

    print("✅ Model and test data loaded")

    # Generate predictions
    predictions = model.predict(X_test)

    target_columns = y_test.columns.tolist()

    # Set experiment
    # Create a new experiment with the correct artifact location
    experiment_name = "Smart_Healthcare_Monitoring_v2"

    artifact_dir = BASE_DIR / "mlartifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_dir.resolve().as_uri()
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)

    print("MLflow Experiment:", experiment_name)
    print("Artifact Location:", artifact_dir)

    with mlflow.start_run() as run:

        print("\n🚀 MLflow Run Started")

        # Log parameters
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("max_depth", 15)
        mlflow.log_param("number_of_targets", len(target_columns))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features", X_test.shape[1])

        accuracy_values = []
        precision_values = []
        recall_values = []
        f1_values = []

        # Evaluate each target
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

            accuracy_values.append(accuracy)
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)

            # Log individual metrics
            mlflow.log_metric(f"{target}_accuracy", accuracy)
            mlflow.log_metric(f"{target}_precision", precision)
            mlflow.log_metric(f"{target}_recall", recall)
            mlflow.log_metric(f"{target}_f1", f1)

            print(f"\n🎯 {target}")
            print(f"Accuracy  : {accuracy:.4f}")
            print(f"Precision : {precision:.4f}")
            print(f"Recall    : {recall:.4f}")
            print(f"F1 Score  : {f1:.4f}")

        # Calculate average metrics
        avg_accuracy = sum(accuracy_values) / len(accuracy_values)
        avg_precision = sum(precision_values) / len(precision_values)
        avg_recall = sum(recall_values) / len(recall_values)
        avg_f1 = sum(f1_values) / len(f1_values)

        # Log average metrics
        mlflow.log_metric("average_accuracy", avg_accuracy)
        mlflow.log_metric("average_precision", avg_precision)
        mlflow.log_metric("average_recall", avg_recall)
        mlflow.log_metric("average_f1", avg_f1)

        # Log model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            name="healthcare_random_forest_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print("\n" + "=" * 60)
        print("📈 AVERAGE METRICS")
        print("=" * 60)

        print(f"Average Accuracy  : {avg_accuracy:.4f}")
        print(f"Average Precision : {avg_precision:.4f}")
        print(f"Average Recall    : {avg_recall:.4f}")
        print(f"Average F1 Score  : {avg_f1:.4f}")

        print("\n✅ MLflow tracking completed!")

        print("\n🔗 Run ID:")
        print(run.info.run_id)


if __name__ == "__main__":
    track_model()