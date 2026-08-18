"""
Farm Advisor Routes
POST /api/advisor/insights  — AI category-based advice
"""

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from skyview.data.queries import get_latest_weather
from skyview.agents.edge_ai_agent import invoke_llm_edge_first
from skyview.utils.logger import get_logger

router = APIRouter(prefix="/api/advisor", tags=["Advisor"])
logger = get_logger(__name__)


class AdvisorReq(BaseModel):
    category: str = "overview"  # alerts|crops|water|tips|soil|pests|overview
    station_id: str = "WS01"


def _build_prompt(category: str, s: dict) -> str:
    temp = s.get("temperature", 25)
    hum  = s.get("humidity", 60)
    sm   = s.get("soil_moisture", 50)
    st   = s.get("soil_temperature", "N/A")
    ws   = s.get("wind_speed", 5)
    wd   = s.get("wind_direction", "N/A")
    rain = s.get("rainfall", 0)
    lux  = s.get("lux", 70)
    uv   = s.get("uv_index", 2)
    pres = s.get("pressure", 1013)
    pm25 = s.get("pm25", "N/A")
    pm10 = s.get("pm10", "N/A")
    bat  = s.get("battery_voltage", "N/A")
    sol  = s.get("solar_voltage", "N/A")

    base = (
        f"Temp={temp}°C Humidity={hum}% Wind={ws}km/h({wd}) "
        f"Rain={rain}mm Soil Moisture={sm}% Soil Temp={st}°C "
        f"Light={lux}lux UV={uv} Pressure={pres}hPa PM2.5={pm25} "
        f"Battery={bat}V Solar={sol}V"
    )

    prompts = {
        "alerts": (
            f"You are a farm safety AI. Sensor data: {base}\n"
            'Generate 4-5 farming alerts. JSON array only: '
            '[{"severity":"warning","title":"...","message":"..."}]'
        ),
        "crops": (
            f"You are a crop advisor. Conditions: Temp={temp}°C Humidity={hum}% "
            f"Soil Moisture={sm}% Light={lux}\n"
            'Suggest 5 best crops for NOW. JSON array: '
            '[{"crop":"Rice","emoji":"🌾","suitability":"...","tips":"...","timeline":"..."}]'
        ),
        "water": (
            f"You are a farm water AI. Conditions: Temp={temp}°C Humidity={hum}% "
            f"Soil Moisture={sm}% Rain={rain}mm\n"
            'Irrigation advice. JSON: '
            '{"moisture_status":"...","schedule":"...","tips":["..."],"liters_per_acre":500,"warnings":["..."]}'
        ),
        "tips": (
            f"You are a friendly farm advisor. Conditions: Temp={temp}°C Humidity={hum}% "
            f"Wind={ws}km/h Soil Moisture={sm}%\n"
            'Give 6 practical tips for TODAY. JSON array of strings: ["🌱 Tip..."]'
        ),
        "soil": (
            f"You are a soil health AI. Data: Soil Moisture={sm}% Temp={temp}°C "
            f"Humidity={hum}% Rain={rain}mm\n"
            'Comprehensive soil analysis. JSON: '
            '{"score":75,"status":"Good","recommendations":["..."],"nutrients":"...","risk":"..."}'
        ),
        "pests": (
            f"You are a crop protection AI. Conditions: Temp={temp}°C Humidity={hum}% Rain={rain}mm\n"
            'Identify 4 likely pests/diseases. JSON array: '
            '[{"name":"...","risk":"High","symptoms":"...","remedy":"...","prevention":"..."}]'
        ),
        "overview": (
            f"You are a farm AI. Live sensor readings: {base}\n"
            'Detailed advisory. JSON: '
            '{"summary":"...","focus_points":["..."],"details":"..."}'
        ),
    }
    return prompts.get(category, prompts["overview"])


@router.post("/insights")
async def advisor_insights(req: AdvisorReq):
    sensor = get_latest_weather(req.station_id) or {}
    prompt = _build_prompt(req.category, sensor)

    ai_response = None
    raw = await invoke_llm_edge_first([("user", prompt)], temperature=0.4, timeout=15)
    if raw:
        try:
            ai_response = json.loads(raw.strip().lstrip("```json").rstrip("```"))
        except Exception:
            ai_response = raw

    return {
        "status": "success",
        "category": req.category,
        "sensor_snapshot": sensor,
        "ai_insights": ai_response,
        "timestamp": datetime.utcnow().isoformat(),
    }