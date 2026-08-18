"""
Sensor Data Routes
POST /api/sensors/data            — hardware ingestion (LoRa/ESP32 nested JSON)
POST /api/sensors/ingest          — flat JSON ingestion (test_all_routes compatible)
GET  /api/sensors/latest/{station_id}
GET  /api/sensors/history/{station_id}
GET  /api/sensors/stations
POST /api/sensors/trends/store    — store a trends snapshot from the frontend
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from skyview.data.db import execute_update, get_session
from skyview.data.queries import (
    get_active_installations,
    get_latest_weather,
    get_weather_history,
)
from skyview.utils.logger import get_logger

router = APIRouter(prefix="/api/sensors", tags=["Sensors"])
logger = get_logger(__name__)

_MOCK_READING = {
    "station_id": "WS01", "temperature": 25.0, "humidity": 60.0,
    "pressure": 1013.0, "wind_speed": 5.0, "wind_direction": "N",
    "rainfall": 0.0, "soil_temperature": 26.0, "soil_moisture": 50.0,
    "pm25": 60.0, "pm10": 120.0, "uv_index": 2.0,
    "lux": 70.0, "battery_voltage": 12.5, "solar_voltage": 13.2,
}


# ── Models ────────────────────────────────────────────────────────────────────

class FlatSensorPayload(BaseModel):
    """Flat sensor payload used by test_all_routes and direct HTTP callers."""
    station_id: str = "WS01"
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[str] = None
    rainfall: Optional[float] = None
    soil_temperature: Optional[float] = None
    soil_moisture: Optional[float] = None
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    uv_index: Optional[float] = None
    lux: Optional[float] = None
    battery_voltage: Optional[float] = None
    solar_voltage: Optional[float] = None


class TrendsStoreReq(BaseModel):
    station_id: str = "WS01"
    period: str = "24h"
    data: Optional[list] = None  # raw chart data to persist if needed


# ── Helpers ───────────────────────────────────────────────────────────────────

def _insert_weather(params: dict) -> bool:
    from sqlalchemy import text
    # Ensure edge AI fields have defaults
    params.setdefault("data_quality", "unknown")
    params.setdefault("edge_fusion", None)
    params.setdefault("edge_stress", None)
    params.setdefault("edge_rain", None)
    params.setdefault("edge_anomaly", None)
    params.setdefault("edge_model", None)
    params.setdefault("edge_ms", None)
    return execute_update(
        text("""
            INSERT INTO weather_data (
                station_id, timestamp,
                temperature, humidity, pressure,
                wind_speed, wind_direction, rainfall,
                soil_temperature, soil_moisture,
                pm25, pm10, uv_index, lux,
                battery_voltage, solar_voltage,
                data_quality, edge_fusion_score, edge_stress_index,
                edge_rain_prob, edge_anomaly_score,
                edge_model_version, edge_inference_ms
            ) VALUES (
                :sid, :ts, :temp, :hum, :pres,
                :ws, :wd, :rain,
                :st, :sm, :pm25, :pm10, :uv, :lux, :bat, :sol,
                :data_quality, :edge_fusion, :edge_stress,
                :edge_rain, :edge_anomaly,
                :edge_model, :edge_ms
            )
        """),
        params,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/stations")
def list_stations():
    return {"stations": get_active_installations()}


@router.get("/latest/{station_id}")
def latest_reading(station_id: str = "WS01"):
    data = get_latest_weather(station_id)
    if not data:
        return {
            **_MOCK_READING,
            "station_id": station_id,
            "timestamp": datetime.utcnow().isoformat(),
            "_source": "mock",
        }
    return data


@router.get("/history/{station_id}")
def history(station_id: str = "WS01", hours: int = 24, limit: int = 200):
    """
    Returns historical readings.
    Response shape: { station_id, data: [...], totalRecords, returnedRecords }
    The `data` key is what the frontend (weatherData.ts) expects.
    """
    rows = get_weather_history(station_id, hours, limit)
    return {
        "station_id": station_id,
        "data": rows,
        "totalRecords": len(rows),
        "returnedRecords": len(rows),
    }


@router.post("/data")
async def ingest_sensor_data_nested(request: Request):
    """
    Hardware ingestion endpoint for ESP32/LoRa nested JSON format:
    { id, ts, env:{t,h,p}, wind:{s,d}, rain, soil:{t,m}, air:{pm25,pm10},
      rad:{uv,lux}, pwr:{bat,sol} }
    Also accepts the flat format for backwards compatibility.
    """
    try:
        data = await request.json()
    except Exception:
        return {"status": "error", "message": "Invalid JSON body"}

    # ── Extract edge AI fields (Arduino Q on-device inference) ─
    edge_ai = data.get("edge_ai", {})
    edge_flags = data.get("edge_flags", {})
    edge_params = {
        "data_quality": edge_flags.get("data_quality", "unknown"),
        "edge_fusion": edge_ai.get("fusion_score"),
        "edge_stress": edge_ai.get("stress_index"),
        "edge_rain": edge_ai.get("rain_probability"),
        "edge_anomaly": edge_flags.get("anomaly_score"),
        "edge_model": edge_ai.get("model_version"),
        "edge_ms": edge_ai.get("inference_ms"),
    }

    # ── Flat format (station_id + flat fields) ────────────────
    if "station_id" in data and "temperature" in data:
        ts = datetime.utcnow()
        success = _insert_weather({
            "sid": data.get("station_id", "UNKNOWN"), "ts": ts,
            "temp": data.get("temperature"), "hum": data.get("humidity"),
            "pres": data.get("pressure"), "ws": data.get("wind_speed"),
            "wd": data.get("wind_direction"), "rain": data.get("rainfall", 0.0),
            "st": data.get("soil_temperature"), "sm": data.get("soil_moisture"),
            "pm25": data.get("pm25"), "pm10": data.get("pm10"),
            "uv": data.get("uv_index"), "lux": data.get("lux"),
            "bat": data.get("battery_voltage"), "sol": data.get("solar_voltage"),
            **edge_params,
        })
        return {"status": "success" if success else "db_error",
                "station_id": data.get("station_id"), "timestamp": ts.isoformat()}

    # ── Nested hardware format ────────────────────────────────
    required = ["id", "ts", "env", "wind", "soil", "air", "rad", "pwr"]
    missing = [f for f in required if f not in data]
    if missing:
        return {"status": "error", "message": f"Missing fields: {missing}"}

    station_id = data.get("id", "UNKNOWN")
    ts = datetime.fromtimestamp(data.get("ts", 0))
    env  = data.get("env", {})
    wind = data.get("wind", {})
    soil = data.get("soil", {})
    air  = data.get("air", {})
    rad  = data.get("rad", {})
    pwr  = data.get("pwr", {})

    success = _insert_weather({
        "sid": station_id, "ts": ts,
        "temp": env.get("t"), "hum": env.get("h"), "pres": env.get("p"),
        "ws": wind.get("s"), "wd": wind.get("d"),
        "rain": data.get("rain", 0.0),
        "st": soil.get("t"), "sm": soil.get("m"),
        "pm25": air.get("pm25"), "pm10": air.get("pm10"),
        "uv": rad.get("uv"), "lux": rad.get("lux"),
        "bat": pwr.get("bat"), "sol": pwr.get("sol"),
        **edge_params,
    })

    logger.info("Hardware data received from %s at %s (quality=%s)",
                station_id, ts, edge_params.get("data_quality", "unknown"))
    return {
        "status": "success" if success else "db_error",
        "station_id": station_id,
        "timestamp": ts.isoformat(),
        "edge_ai": bool(edge_ai),
    }


@router.post("/trends/store")
async def store_trends(req: TrendsStoreReq):
    """
    Called by the Trends page to persist a snapshot or simply acknowledge.
    Currently logs and returns OK; extend to write to a trends table if needed.
    """
    logger.info("Trends snapshot received: station=%s period=%s points=%d",
                req.station_id, req.period, len(req.data) if req.data else 0)
    return {
        "status": "success",
        "station_id": req.station_id,
        "period": req.period,
        "message": "Trends snapshot acknowledged",
    }


@router.post("/data/batch")
async def ingest_batch(request: Request):
    """
    Batch ingestion endpoint for Arduino Q store-and-forward.
    Accepts an array of readings buffered during offline periods.
    Body: { "readings": [ {nested or flat format}, ... ] }
    """
    try:
        body = await request.json()
    except Exception:
        return {"status": "error", "message": "Invalid JSON body"}

    readings = body.get("readings", [])
    if not isinstance(readings, list) or not readings:
        return {"status": "error", "message": "'readings' must be a non-empty array"}

    results = {"total": len(readings), "success": 0, "failed": 0, "errors": []}

    for idx, reading in enumerate(readings):
        try:
            # Extract edge AI fields
            edge_ai = reading.get("edge_ai", {})
            edge_flags = reading.get("edge_flags", {})
            edge_params = {
                "data_quality": edge_flags.get("data_quality", "unknown"),
                "edge_fusion": edge_ai.get("fusion_score"),
                "edge_stress": edge_ai.get("stress_index"),
                "edge_rain": edge_ai.get("rain_probability"),
                "edge_anomaly": edge_flags.get("anomaly_score"),
                "edge_model": edge_ai.get("model_version"),
                "edge_ms": edge_ai.get("inference_ms"),
            }

            # Flat format
            if "station_id" in reading and "temperature" in reading:
                ts = datetime.utcnow()
                ok = _insert_weather({
                    "sid": reading.get("station_id", "UNKNOWN"), "ts": ts,
                    "temp": reading.get("temperature"), "hum": reading.get("humidity"),
                    "pres": reading.get("pressure"), "ws": reading.get("wind_speed"),
                    "wd": reading.get("wind_direction"), "rain": reading.get("rainfall", 0.0),
                    "st": reading.get("soil_temperature"), "sm": reading.get("soil_moisture"),
                    "pm25": reading.get("pm25"), "pm10": reading.get("pm10"),
                    "uv": reading.get("uv_index"), "lux": reading.get("lux"),
                    "bat": reading.get("battery_voltage"), "sol": reading.get("solar_voltage"),
                    **edge_params,
                })
            # Nested format
            elif "id" in reading and "env" in reading:
                ts = datetime.fromtimestamp(reading.get("ts", 0))
                env = reading.get("env", {})
                wind = reading.get("wind", {})
                soil = reading.get("soil", {})
                air = reading.get("air", {})
                rad = reading.get("rad", {})
                pwr = reading.get("pwr", {})
                ok = _insert_weather({
                    "sid": reading.get("id", "UNKNOWN"), "ts": ts,
                    "temp": env.get("t"), "hum": env.get("h"), "pres": env.get("p"),
                    "ws": wind.get("s"), "wd": wind.get("d"),
                    "rain": reading.get("rain", 0.0),
                    "st": soil.get("t"), "sm": soil.get("m"),
                    "pm25": air.get("pm25"), "pm10": air.get("pm10"),
                    "uv": rad.get("uv"), "lux": rad.get("lux"),
                    "bat": pwr.get("bat"), "sol": pwr.get("sol"),
                    **edge_params,
                })
            else:
                results["failed"] += 1
                results["errors"].append({"index": idx, "error": "Unrecognized format"})
                continue

            if ok:
                results["success"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({"index": idx, "error": "DB insert failed"})
        except Exception as exc:
            results["failed"] += 1
            results["errors"].append({"index": idx, "error": str(exc)})

    logger.info("Batch ingest: %d/%d success", results["success"], results["total"])
    return {"status": "success" if results["failed"] == 0 else "partial", **results}