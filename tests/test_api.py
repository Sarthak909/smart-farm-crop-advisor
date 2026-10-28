from fastapi.testclient import TestClient

from app.main import app, metadata


client = TestClient(app)


def sample_payload() -> dict:
    """A realistic row from the bundled Kaggle training data."""
    import pandas as pd

    return pd.read_csv("data/Crop_recommendationV2.csv").drop(columns="label").iloc[0].to_dict()


def test_health_and_model_info():
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["classes"] == metadata["classes"]
    assert info["test_accuracy"] > 0


def test_predict_returns_ranked_crops():
    response = client.post("/predict", json=sample_payload())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["crop"] in metadata["classes"]
    assert 0 <= body["confidence"] <= 1
    assert len(body["alternatives"]) == 3


def test_predict_rejects_invalid_ph():
    payload = sample_payload()
    payload["ph"] = 20
    assert client.post("/predict", json=payload).status_code == 422
