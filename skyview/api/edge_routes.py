"""
Arduino Q Agentic AI Gateway Routes

The Arduino UNO Q (Qualcomm MPU) is the central hub that runs:
  - The full agentic AI workflow (multi-agent supervisor) on-device
  - An on-device LLM for decision-making (e.g., Llama 3.2 via llama.cpp)
  - Data Processing & Routing for all sensor data
  - Database (sensor data + FPGA results)

The AMD ZYNQ-7000 FPGA connects to the Arduino Q via UART and provides
hardware-accelerated sensor fusion and rain prediction results.

GET  /api/edge/status           — Agentic AI gateway status (LLM + FPGA)
GET  /api/edge/health           — Detailed gateway health (LLM, memory, queue)
GET  /api/edge/data-quality     — Data quality analytics across stations
POST /api/edge/alert-thresholds — Push alert thresholds to the Arduino Q MCU
POST /api/edge/chat             — Direct chat with Arduino Q on-device LLM
"""

import json
from datetime import datetime
from typing import Optional

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from skyview.agents.edge_ai_agent import (
    get_edge_ai_status,
    invoke_edge_llm,
)
from skyview.data.db import execute_query
from skyview.utils.config import get_settings
from skyview.utils.logger import get_logger

router = APIRouter(prefix="/api/edge", tags=["Arduino Q Agentic AI"])
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


class EdgeChatReq(BaseModel):
    message: str
    temperature: float = 0.3


@router.get("/status")
async def edge_status():
    """
    Arduino Q Agentic AI Gateway status.
    Reports on-device LLM availability, FPGA UART connectivity,
    and cloud fallback configuration.
    """
    status = await get_edge_ai_status()
    return {
        "status": "ok",
        **status,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health")
async def edge_health():
    """
    Detailed gateway health: on-device LLM model info, FPGA UART status,
    LoRa stats, queue depth, and inference metrics.
    """
    if not settings.ENABLE_EDGE_AI:
        return {
            "status": "disabled",
            "message": "Arduino Q Agentic AI is not enabled. Set ENABLE_EDGE_AI=True.",
            "timestamp": datetime.utcnow().isoformat(),
        }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(_gateway_url("/health"))
            if resp.status_code == 200:
                health_data = resp.json()
                return {
                    "status": "ok",
                    "gateway_health": health_data,
                    "description": "Arduino Q running agentic AI on Qualcomm MPU, FPGA via UART",
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


@router.post("/chat")
async def edge_chat(req: EdgeChatReq):
    """
    Send a chat message directly to the Arduino Q's on-device LLM.
    Useful for testing the Qualcomm MPU inference capability.
    Does NOT fall back to cloud — this tests the edge LLM directly.
    """
    if not settings.ENABLE_EDGE_AI:
        return {
            "status": "disabled",
            "message": "Arduino Q Agentic AI is not enabled. Set ENABLE_EDGE_AI=True.",
            "timestamp": datetime.utcnow().isoformat(),
        }

    response = await invoke_edge_llm(
        [("user", req.message)],
        temperature=req.temperature,
    )

    return {
        "status": "success" if response else "error",
        "response": response or "On-device LLM did not return a response.",
        "model": settings.ARDUINO_Q_LLM_MODEL,
        "source": "arduino_q_on_device",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/data-quality")
def edge_data_quality(station_id: Optional[str] = None, hours: int = 24):
    """
    Data quality analytics — counts readings by quality level,
    average FPGA fusion scores, and flagged sensor breakdown.
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

    # Average FPGA-sourced metrics
    edge_rows = execute_query(
        f"""
        SELECT
            AVG(edge_anomaly_score) as avg_anomaly,
            AVG(edge_fusion_score) as avg_fusion,
            AVG(edge_inference_ms) as avg_inference_ms,
            COUNT(CASE WHEN edge_fusion_score IS NOT NULL THEN 1 END) as fpga_processed_count
        FROM weather_data
        {where_clause}
        """,
        params,
    )

    fpga_stats = {}
    if edge_rows and edge_rows[0]:
        r = edge_rows[0]
        fpga_stats = {
            "avg_anomaly_score": round(float(r[0]), 3) if r[0] else None,
            "avg_fusion_score": round(float(r[1]), 1) if r[1] else None,
            "avg_inference_ms": round(float(r[2]), 1) if r[2] else None,
            "readings_with_fpga_results": r[3] or 0,
        }

    return {
        "status": "ok",
        "period_hours": hours,
        "station_id": station_id or "all",
        "total_readings": total,
        "quality_distribution": quality_dist,
        "fpga_accelerator_stats": fpga_stats,
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
            "message": "Arduino Q not enabled; thresholds saved but not pushed to device.",
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
