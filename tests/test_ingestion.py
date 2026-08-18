"""
test_ingestion.py — Tests for sensor data ingestion pipeline.
Run: pytest tests/test_ingestion.py -v
"""

import pytest


NESTED_PAYLOAD = {
    "id": "WS01",
    "ts": 1700000000,
    "env": {"t": 28.5, "h": 65.0, "p": 1012.0},
    "wind": {"s": 3.2, "d": "NE"},
    "rain": 0.5,
    "soil": {"t": 27.0, "m": 52.0},
    "air": {"pm25": 55.0, "pm10": 110.0},
    "rad": {"uv": 4.0, "lux": 35000.0},
    "pwr": {"bat": 3.8, "sol": 5.1},
}

FLAT_PAYLOAD = {
    "station_id": "WS02",
    "temperature": 30.0,
    "humidity": 70.0,
    "pressure": 1010.0,
    "wind_speed": 5.0,
    "wind_direction": "SW",
    "rainfall": 1.2,
    "soil_temperature": 28.0,
    "soil_moisture": 60.0,
    "pm25": 40.0,
    "pm10": 80.0,
    "uv_index": 6.0,
    "lux": 45000.0,
    "battery_voltage": 4.0,
    "solar_voltage": 5.5,
}

# Edge AI payloads (Arduino Q on-device inference)
NESTED_PAYLOAD_WITH_EDGE_AI = {
    **NESTED_PAYLOAD,
    "edge_ai": {
        "fusion_score": 72,
        "stress_index": 35,
        "rain_probability": 0.42,
        "model_version": "v1.2",
        "inference_ms": 12,
    },
    "edge_flags": {
        "anomaly_score": 0.08,
        "data_quality": "good",
    },
}

FLAT_PAYLOAD_WITH_EDGE_AI = {
    **FLAT_PAYLOAD,
    "edge_ai": {
        "fusion_score": 65,
        "stress_index": 28,
        "rain_probability": 0.15,
        "model_version": "v1.2",
        "inference_ms": 9,
    },
    "edge_flags": {
        "anomaly_score": 0.03,
        "data_quality": "good",
    },
}

BATCH_PAYLOAD = {
    "readings": [
        FLAT_PAYLOAD,
        NESTED_PAYLOAD,
        FLAT_PAYLOAD_WITH_EDGE_AI,
    ]
}


class TestIngestion:
    def test_nested_format_ingest(self, client):
        """Hardware nested JSON format (ESP32 → LoRa → Arduino Q)."""
        r = client.post("/api/sensors/data", json=NESTED_PAYLOAD)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("success", "db_error")
        assert data["station_id"] == "WS01"

    def test_flat_format_ingest(self, client):
        """Flat JSON format (direct HTTP from test scripts)."""
        r = client.post("/api/sensors/data", json=FLAT_PAYLOAD)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("success", "db_error")
        assert data["station_id"] == "WS02"

    def test_missing_fields_returns_error(self, client):
        """Partial nested payload should return error, not 500."""
        r = client.post("/api/sensors/data", json={"id": "WS01"})
        assert r.status_code == 200   # FastAPI returns 200 with error body
        assert r.json()["status"] == "error"

    def test_invalid_json_returns_error(self, client):
        """Non-JSON body should return graceful error."""
        r = client.post(
            "/api/sensors/data",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code in (200, 422)

    def test_history_returns_data_key(self, client):
        """Frontend expects `data` key in history response."""
        r = client.get("/api/sensors/history/WS01")
        assert r.status_code == 200
        body = r.json()
        assert "data" in body, "Frontend needs `data` key in history response"
        assert isinstance(body["data"], list)


class TestEdgeAIIngestion:
    """Tests for Arduino Q edge AI data ingestion."""

    def test_nested_with_edge_ai(self, client):
        """Nested format with edge_ai and edge_flags from Arduino Q."""
        r = client.post("/api/sensors/data", json=NESTED_PAYLOAD_WITH_EDGE_AI)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("success", "db_error")
        assert data.get("edge_ai") is True

    def test_flat_with_edge_ai(self, client):
        """Flat format with edge_ai fields."""
        r = client.post("/api/sensors/data", json=FLAT_PAYLOAD_WITH_EDGE_AI)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("success", "db_error")

    def test_nested_without_edge_ai_still_works(self, client):
        """Legacy payloads without edge_ai should still work."""
        r = client.post("/api/sensors/data", json=NESTED_PAYLOAD)
        assert r.status_code == 200
        assert r.json()["status"] in ("success", "db_error")

    def test_batch_ingestion(self, client):
        """Batch endpoint for Arduino Q store-and-forward."""
        r = client.post("/api/sensors/data/batch", json=BATCH_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 3
        assert body["success"] + body["failed"] == body["total"]

    def test_batch_empty_readings_error(self, client):
        """Batch with empty readings should return error."""
        r = client.post("/api/sensors/data/batch", json={"readings": []})
        assert r.status_code == 200
        assert r.json()["status"] == "error"

    def test_batch_invalid_format_partial(self, client):
        """Batch with mixed valid/invalid readings returns partial status."""
        r = client.post("/api/sensors/data/batch", json={
            "readings": [FLAT_PAYLOAD, {"bad": "data"}]
        })
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 2
        assert body["failed"] >= 1


class TestEdgeRoutes:
    """Tests for Edge AI gateway endpoints."""

    def test_edge_status(self, client):
        """Edge status endpoint should return connectivity info."""
        r = client.get("/api/edge/status")
        assert r.status_code == 200
        body = r.json()
        assert "edge_ai_enabled" in body
        assert "gateway_host" in body

    def test_edge_data_quality(self, client):
        """Data quality analytics should return distribution."""
        r = client.get("/api/edge/data-quality")
        assert r.status_code == 200
        body = r.json()
        assert "quality_distribution" in body
        assert "edge_ai_stats" in body

    def test_edge_alert_thresholds(self, client):
        """Should accept and return alert thresholds."""
        r = client.post("/api/edge/alert-thresholds", json={
            "soil_moisture_min": 20,
            "temp_max": 42,
            "pm25_max": 250,
            "humidity_min": 25,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("stored_locally", "pushed", "push_failed")
        assert body["thresholds"]["temp_max"] == 42