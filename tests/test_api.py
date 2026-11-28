import json

import pytest
from fastapi.testclient import TestClient

from app.main import app, logger
from src.mp083.config import COLUMNS
from src.mp083.registry import load_version


@pytest.fixture
def api_logs(caplog):
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def test_api_uses_approved_model(approved, sample, monkeypatch, api_logs):
    models, version, _ = approved
    monkeypatch.setenv("MODEL_DIR", str(models))
    with TestClient(app) as client:
        assert client.get("/").json()["docs"] == "/docs"
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/health").json()["model_ready"]
        metadata = client.get("/model-info").json()
        assert metadata["version"] == version
        assert "training_keys" not in metadata
        data = sample[COLUMNS].iloc[[0]]
        response = client.post("/predict", json=data.iloc[0].to_dict())
        assert response.status_code == 200
        assert response.json()["gb_per_day"] == pytest.approx(load_version(models)[0].predict(data)[0])
        invalid = client.post("/predict", json={"age": -1, "secret": "do-not-log-this"})
        assert invalid.status_code == 422
    logs = [json.loads(record.message) for record in api_logs.records if record.name == logger.name]
    prediction_logs = [entry for entry in logs if entry.get("route") == "/predict"]
    assert [entry["prediction_status"] for entry in prediction_logs] == ["succeeded", "failed"]
    assert prediction_logs[0]["request_id"] == response.headers["X-Request-ID"]
    assert prediction_logs[0]["timestamp"] and prediction_logs[0]["duration_ms"] >= 0
    assert "do-not-log-this" not in api_logs.text


def test_failed_model_startup_exposes_readiness_error(tmp_path, monkeypatch, api_logs, sample):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with TestClient(app) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/health").status_code == 503
        assert client.get("/model-info").status_code == 503
        assert client.post("/predict", json=sample[COLUMNS].iloc[0].to_dict()).status_code == 503
    assert "model_load_failed" in api_logs.text


def test_batch_prediction_api(approved, sample, monkeypatch):
    models, version, _ = approved
    monkeypatch.setenv("MODEL_DIR", str(models))

    with TestClient(app) as client:
        records = sample[COLUMNS].iloc[:3].to_dict(
            orient="records"
        )

        response = client.post(
            "/predict/batch",
            json={"records": records},
        )

        assert response.status_code == 200

        result = response.json()["predictions"]

        assert len(result) == 3

        for item in result:
            assert item["gb_per_day"] >= 0
            assert item["model_version"] == version


def test_batch_prediction_rejects_empty_and_large_batches(
    approved,
    monkeypatch,
):
    models, _, _ = approved
    monkeypatch.setenv("MODEL_DIR", str(models))

    with TestClient(app) as client:
        empty = client.post(
            "/predict/batch",
            json={"records": []},
        )
        assert empty.status_code == 422

        oversized = client.post(
            "/predict/batch",
            json={"records": [{}] * 101},
        )
        assert oversized.status_code == 422

