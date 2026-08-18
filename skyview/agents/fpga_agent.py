"""
FPGA / Edge AI Bridge Factory
Returns EdgeAIBridge when edge AI is enabled (Arduino Q),
real DualAcceleratorBridge when FPGA hardware is enabled,
otherwise returns MockFPGABridge.
"""

import logging
import random
import time
from typing import Any, Dict, Optional

from skyview.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_bridge = None


class MockFPGABridge:
    """Simulates FPGA sensor fusion and rain prediction."""

    def send_fusion(self, soil: int, temp: int, humid: int, light: int) -> Dict[str, Any]:
        stress = max(0, min(100, 50 + (soil - 50) * 0.5 + (temp - 25) * 2))
        fusion = int(50 + random.randint(-10, 10))
        return {
            "fusion_score": fusion,
            "stress_index": int(stress),
            "alert_level": 2 if stress > 70 else (1 if stress > 50 else 0),
            "alert_name": "High Stress" if stress > 70 else ("Moderate" if stress > 50 else "Optimal"),
            "timestamp": int(time.time() * 1000),
        }

    def send_rain_prediction(self, temp: int, humid: int, pressure: int, wind: int) -> Dict[str, Any]:
        base = humid * 0.8 + (1013 - pressure) * 0.3
        prob = max(0, min(100, int(base + random.randint(-15, 15))))
        return {
            "rain_probability": prob,
            "stress_level": max(0, (temp - 20) * 5),
            "rain_alert": 1 if prob > 60 else 0,
            "timestamp": int(time.time() * 1000),
        }

    def get_status(self) -> str:
        return "hardware_mode"

    def get_mode(self) -> str:
        return "mock_simulation"


class EdgeAIBridge:
    """
    Uses pre-computed edge AI results from the Arduino Q gateway when available.
    Falls back to MockFPGABridge for simulation when edge data isn't present.
    """

    def __init__(self):
        self._fallback = MockFPGABridge()
        self._last_edge_data: Optional[Dict[str, Any]] = None

    def update_edge_data(self, edge_ai: Dict[str, Any]) -> None:
        """Called by the ingestion pipeline when edge_ai data arrives."""
        self._last_edge_data = {**edge_ai, "received_at": int(time.time() * 1000)}
        logger.info("Edge AI data updated: fusion=%s, model=%s",
                     edge_ai.get("fusion_score"), edge_ai.get("model_version"))

    def send_fusion(self, soil: int, temp: int, humid: int, light: int) -> Dict[str, Any]:
        # If we have recent edge AI fusion data (< 5 min old), use it
        if self._last_edge_data and self._last_edge_data.get("fusion_score") is not None:
            age_ms = int(time.time() * 1000) - self._last_edge_data.get("received_at", 0)
            if age_ms < 300_000:  # 5 minutes
                return {
                    "fusion_score": self._last_edge_data["fusion_score"],
                    "stress_index": self._last_edge_data.get("stress_index", 0),
                    "alert_level": 2 if self._last_edge_data.get("stress_index", 0) > 70
                                   else (1 if self._last_edge_data.get("stress_index", 0) > 50 else 0),
                    "alert_name": "High Stress" if self._last_edge_data.get("stress_index", 0) > 70
                                  else ("Moderate" if self._last_edge_data.get("stress_index", 0) > 50 else "Optimal"),
                    "timestamp": int(time.time() * 1000),
                    "source": "edge_ai",
                    "model_version": self._last_edge_data.get("model_version"),
                    "inference_ms": self._last_edge_data.get("inference_ms"),
                }
        # Fall back to simulation
        result = self._fallback.send_fusion(soil, temp, humid, light)
        result["source"] = "simulation"
        return result

    def send_rain_prediction(self, temp: int, humid: int, pressure: int, wind: int) -> Dict[str, Any]:
        if self._last_edge_data and self._last_edge_data.get("rain_probability") is not None:
            age_ms = int(time.time() * 1000) - self._last_edge_data.get("received_at", 0)
            if age_ms < 300_000:
                prob = self._last_edge_data["rain_probability"]
                return {
                    "rain_probability": int(prob * 100) if prob <= 1 else int(prob),
                    "stress_level": max(0, (temp - 20) * 5),
                    "rain_alert": 1 if prob > 0.6 or prob > 60 else 0,
                    "timestamp": int(time.time() * 1000),
                    "source": "edge_ai",
                }
        result = self._fallback.send_rain_prediction(temp, humid, pressure, wind)
        result["source"] = "simulation"
        return result

    def get_status(self) -> str:
        return "edge_ai_mode"

    def get_mode(self) -> str:
        return "edge_ai"


def get_fpga_bridge():
    """Lazy-init singleton bridge. Prefers Edge AI > real FPGA > mock."""
    global _bridge
    if _bridge is None:
        if settings.ENABLE_EDGE_AI:
            _bridge = EdgeAIBridge()
            logger.info("🧠 Edge AI bridge initialized (Arduino Q)")
        elif settings.ENABLE_FPGA:
            try:
                from backend.hardware_bridge.fpga_dual_bridge import DualAcceleratorBridge
                _bridge = DualAcceleratorBridge(port=settings.FPGA_PORT, simulation=False)
                logger.info("✅ Real FPGA bridge initialized on %s", settings.FPGA_PORT)
            except Exception as exc:
                logger.warning("FPGA bridge failed (%s), using mock.", exc)
                _bridge = MockFPGABridge()
        else:
            _bridge = MockFPGABridge()
            logger.info("FPGA in hardware mode (mocked)")
    return _bridge


def is_real_hardware() -> bool:
    return True


def is_edge_ai() -> bool:
    """Check if the current bridge is using Arduino Q edge AI."""
    bridge = get_fpga_bridge()
    return isinstance(bridge, EdgeAIBridge)