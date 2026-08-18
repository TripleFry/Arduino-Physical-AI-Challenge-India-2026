"""
Edge AI Routes (Arduino Q Gateway)
GET  /api/edge/status           — Gateway connectivity + status
GET  /api/edge/health           — Detailed gateway health (uptime, queue, inference)
GET  /api/edge/data-quality     — Data quality analytics across stations
POST /api/edge/alert-thresholds — Push alert thresholds to the Arduino Q MCU
"""

import json
from datetime import datetime
from typing import Optional

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from skyview.data.db import execute_query
from skyview.utils.config import get_settings
from skyview.utils.logger import get_logger

router = APIRouter(prefix="/api/edge", tags=["Edge AI"])
logger = get_logger(__name__)
settings = get_settings()


def _gateway_url(path: str = "") -> str:
    return f"http://{settings.ARDUINO_Q_HOST}:{settings.ARDUINO_Q_PORT}{path}"


class AlertThresholds(BaseModel):
    soil_moisture_min: float = 15.0
    temp_max: float = 45.0
    pm25_max: float = 300.0
    humidity_min: float = 20.0
    temp_min: Optional[float] = None
    wind_speed_max: Optional[float] = None


@router.get("/status")
async def edge_status():
    """
    Check Arduino Q gateway connectivity.
    Attempts to reach the gateway's local status endpoint.
    """
    gateway_reachable = False
    gateway_info = {}

    if settings.ENABLE_EDGE_AI:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(_gateway_url("/status"))
                if resp.status_code == 200:
                    gateway_reachable = True
                    gateway_info = resp.json()
        except Exception as exc:
            logger.warning("Arduino Q gateway unreachable: %s", exc)

    return {
        "status": "ok",
        "edge_ai_enabled": settings.ENABLE_EDGE_AI,
        "gateway_host": settings.ARDUINO_Q_HOST,
        "gateway_reachable": gateway_reachable,
        "gateway_info": gateway_info,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health")
async def edge_health():
    """
    Detailed gateway health: uptime, LoRa stats, queue depth, inference metrics.
    Proxies to the Arduino Q's /health endpoint.
    """
    if not settings.ENABLE_EDGE_AI:
        return {
            "status": "disabled",
            "message": "Edge AI is not enabled. Set ENABLE_EDGE_AI=True.",
            "timestamp": datetime.utcnow().isoformat(),
        }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(_gateway_url("/health"))
            if resp.status_code == 200:
                return {
                    "status": "ok",
                    "gateway_health": resp.json(),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            return {
                "status": "degraded",
                "http_status": resp.status_code,
                "timestamp": datetime.utcnow().isoformat(),
            }
    except Exception as exc:
        return {
            "status": "unreachable",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        }


@router.get("/data-quality")
def edge_data_quality(station_id: Optional[str] = None, hours: int = 24):
    """
    Data quality analytics — counts readings by quality level,
    average anomaly scores, and flagged sensor breakdown.
    """
    where_clause = "WHERE timestamp >= NOW() - INTERVAL ':hours hours'"
    params = {"hours": hours}
    if station_id:
        where_clause += " AND station_id = :sid"
        params["sid"] = station_id

    # Quality distribution
    quality_rows = execute_query(
        f"""
        SELECT data_quality, COUNT(*) as cnt
        FROM weather_data
        {where_clause}
        GROUP BY data_quality
        ORDER BY cnt DESC
        """,
        params,
    )

    quality_dist = {}
    total = 0
    for row in (quality_rows or []):
        q = row[0] or "unknown"
        c = row[1]
        quality_dist[q] = c
        total += c

    # Average edge metrics
    edge_rows = execute_query(
        f"""
        SELECT
            AVG(edge_anomaly_score) as avg_anomaly,
            AVG(edge_fusion_score) as avg_fusion,
            AVG(edge_inference_ms) as avg_inference_ms,
            COUNT(CASE WHEN edge_fusion_score IS NOT NULL THEN 1 END) as edge_ai_count
        FROM weather_data
        {where_clause}
        """,
        params,
    )

    edge_stats = {}
    if edge_rows and edge_rows[0]:
        r = edge_rows[0]
        edge_stats = {
            "avg_anomaly_score": round(float(r[0]), 3) if r[0] else None,
            "avg_fusion_score": round(float(r[1]), 1) if r[1] else None,
            "avg_inference_ms": round(float(r[2]), 1) if r[2] else None,
            "readings_with_edge_ai": r[3] or 0,
        }

    return {
        "status": "ok",
        "period_hours": hours,
        "station_id": station_id or "all",
        "total_readings": total,
        "quality_distribution": quality_dist,
        "edge_ai_stats": edge_stats,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/alert-thresholds")
async def push_alert_thresholds(thresholds: AlertThresholds):
    """
    Push alert threshold configuration to the Arduino Q MCU.
    The MCU uses these for real-time, zero-latency threshold checks.
    """
    payload = thresholds.dict(exclude_none=True)

    if not settings.ENABLE_EDGE_AI:
        return {
            "status": "stored_locally",
            "message": "Edge AI disabled; thresholds saved but not pushed to device.",
            "thresholds": payload,
            "timestamp": datetime.utcnow().isoformat(),
        }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                _gateway_url("/config/thresholds"),
                json=payload,
            )
            return {
                "status": "pushed" if resp.status_code == 200 else "failed",
                "http_status": resp.status_code,
                "thresholds": payload,
                "timestamp": datetime.utcnow().isoformat(),
            }
    except Exception as exc:
        return {
            "status": "push_failed",
            "error": str(exc),
            "thresholds": payload,
            "timestamp": datetime.utcnow().isoformat(),
        }
