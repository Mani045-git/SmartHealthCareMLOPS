from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "Random Forest"
    assert data["number_of_targets"] == 9
    assert data["number_of_features"] == 22


def test_prediction_endpoint():
    sample_data = {
        "age": 35,
        "height_cm": 170,
        "weight_kg": 70,
        "bmi": 24.2,
        "heart_rate_bpm": 75,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "spo2_percent": 98,
        "temperature_f": 98.6,
        "glucose_mg_dl": 95,
        "cholesterol_mg_dl": 180,
        "hemoglobin_g_dl": 14,
        "creatinine_mg_dl": 0.9,
        "sleep_hours": 7,
        "gender": "Male",
        "physical_activity": "Moderate",
        "smoking": "Never",
        "alcohol": "Occasional",
        "stress_level": "Moderate"
    }

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"

    predictions = data["predictions"]

    assert len(predictions) == 9

    for risk in predictions.values():
        assert risk in ["Low", "Moderate", "High"]


def test_prediction_with_missing_features():
    response = client.post(
        "/predict",
        json={"age": 35}
    )

    assert response.status_code == 422