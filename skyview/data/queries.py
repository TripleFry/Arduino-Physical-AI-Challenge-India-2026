"""
Common Query Helpers — reusable DB reads used by agents and endpoints.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from skyview.data.db import execute_query, get_session


def get_latest_weather(station_id: str) -> Optional[Dict[str, Any]]:
    rows = execute_query(
        """
        SELECT station_id, timestamp, temperature, humidity, pressure,
               wind_speed, wind_direction, rainfall,
               soil_temperature, soil_moisture,
               pm25, pm10, uv_index, lux,
               battery_voltage, solar_voltage,
               data_quality, edge_fusion_score, edge_stress_index,
               edge_rain_prob, edge_anomaly_score,
               edge_model_version, edge_inference_ms
        FROM weather_data
        WHERE station_id = :sid
        ORDER BY timestamp DESC LIMIT 1
        """,
        {"sid": station_id},
    )
    if not rows:
        return None
    r = rows[0]
    keys = [
        "station_id", "timestamp", "temperature", "humidity", "pressure",
        "wind_speed", "wind_direction", "rainfall",
        "soil_temperature", "soil_moisture",
        "pm25", "pm10", "uv_index", "lux",
        "battery_voltage", "solar_voltage",
        "data_quality", "edge_fusion_score", "edge_stress_index",
        "edge_rain_prob", "edge_anomaly_score",
        "edge_model_version", "edge_inference_ms",
    ]
    result = {}
    str_fields = {"station_id", "wind_direction", "timestamp", "data_quality", "edge_model_version"}
    for i, k in enumerate(keys):
        val = r[i]
        if k == "timestamp" and val:
            result[k] = val.isoformat() if hasattr(val, "isoformat") else str(val)
        elif k in str_fields:
            result[k] = val
        elif val is not None:
            result[k] = float(val) if not isinstance(val, int) else val
        else:
            result[k] = None
    return result


def get_weather_history(
    station_id: str,
    hours: int = 24,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    since = datetime.utcnow() - timedelta(hours=hours)
    rows = execute_query(
        """
        SELECT timestamp, temperature, humidity, pressure,
               wind_speed, rainfall, soil_moisture, battery_voltage
        FROM weather_data
        WHERE station_id = :sid AND timestamp >= :since
        ORDER BY timestamp DESC LIMIT :lim
        """,
        {"sid": station_id, "since": since, "lim": limit},
    )
    result = []
    for r in (rows or []):
        result.append({
            "timestamp": r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]),
            "temperature": r[1], "humidity": r[2], "pressure": r[3],
            "wind_speed": r[4], "rainfall": r[5],
            "soil_moisture": r[6], "battery_voltage": r[7],
        })
    return result


def get_active_installations() -> List[Dict[str, Any]]:
    rows = execute_query(
        "SELECT station_id, name, latitude, longitude, location_name, status, last_heartbeat FROM installations"
    )
    result = []
    for r in (rows or []):
        result.append({
            "station_id": r[0], "name": r[1],
            "latitude": r[2], "longitude": r[3],
            "location_name": r[4], "status": r[5],
            "last_heartbeat": r[6].isoformat() if r[6] and hasattr(r[6], "isoformat") else None,
        })
    return result


def get_recent_alerts(station_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
    q = "SELECT station_id, alert_type, severity, message, value, created_at FROM alerts"
    params: Dict = {"lim": limit}
    if station_id:
        q += " WHERE station_id = :sid"
        params["sid"] = station_id
    q += " ORDER BY created_at DESC LIMIT :lim"
    rows = execute_query(q, params)
    return [
        {
            "station_id": r[0], "alert_type": r[1], "severity": r[2],
            "message": r[3], "value": r[4],
            "created_at": r[5].isoformat() if r[5] and hasattr(r[5], "isoformat") else str(r[5]),
        }
        for r in (rows or [])
    ]


def get_edge_ai_latest(station_id: str) -> Optional[Dict[str, Any]]:
    """Get the latest edge AI inference results for a station."""
    rows = execute_query(
        """
        SELECT edge_fusion_score, edge_stress_index, edge_rain_prob,
               edge_anomaly_score, edge_model_version, edge_inference_ms,
               data_quality, timestamp
        FROM weather_data
        WHERE station_id = :sid AND edge_fusion_score IS NOT NULL
        ORDER BY timestamp DESC LIMIT 1
        """,
        {"sid": station_id},
    )
    if not rows:
        return None
    r = rows[0]
    return {
        "fusion_score": float(r[0]) if r[0] else None,
        "stress_index": float(r[1]) if r[1] else None,
        "rain_probability": float(r[2]) if r[2] else None,
        "anomaly_score": float(r[3]) if r[3] else None,
        "model_version": r[4],
        "inference_ms": r[5],
        "data_quality": r[6],
        "timestamp": r[7].isoformat() if r[7] and hasattr(r[7], "isoformat") else str(r[7]),
    }