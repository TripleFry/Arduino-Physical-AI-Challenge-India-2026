"""
FPGA Hardware Accelerator Bridge (AMD ZYNQ-7000)

The FPGA is connected to the Arduino Q via UART and handles:
  - Sensor Fusion Accelerator (IP Core) — combines multi-sensor data
  - Rain Prediction Model (IP Core) — weather prediction from sensor inputs
  - Parallel ML Inference — hardware-accelerated computation

The FPGA does NOT run agentic AI. That is handled by the Arduino Q's
Qualcomm MPU running an on-device LLM. The FPGA results are fed INTO
the agentic AI agents as context for decision-making.

Priority: Real FPGA Hardware > Mock Simulation
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
    """
    Simulates FPGA sensor fusion and rain prediction.
    Used when no real FPGA hardware is connected via UART.
    """

    def send_fusion(self, soil: int, temp: int, humid: int, light: int) -> Dict[str, Any]:
        stress = max(0, min(100, 50 + (soil - 50) * 0.5 + (temp - 25) * 2))
        fusion = int(50 + random.randint(-10, 10))
        return {
            "fusion_score": fusion,
            "stress_index": int(stress),
            "alert_level": 2 if stress > 70 else (1 if stress > 50 else 0),
            "alert_name": "High Stress" if stress > 70 else ("Moderate" if stress > 50 else "Optimal"),
            "timestamp": int(time.time() * 1000),
            "source": "fpga_simulation",
        }

    def send_rain_prediction(self, temp: int, humid: int, pressure: int, wind: int) -> Dict[str, Any]:
        base = humid * 0.8 + (1013 - pressure) * 0.3
        prob = max(0, min(100, int(base + random.randint(-15, 15))))
        return {
            "rain_probability": prob,
            "stress_level": max(0, (temp - 20) * 5),
            "rain_alert": 1 if prob > 60 else 0,
            "timestamp": int(time.time() * 1000),
            "source": "fpga_simulation",
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": "simulation",
            "description": "FPGA mock — no AMD ZYNQ-7000 hardware connected",
        }

    def get_mode(self) -> str:
        return "mock_simulation"


def get_fpga_bridge():
    """
    Lazy-init singleton FPGA bridge.
    Priority: Real FPGA hardware (via UART) > Mock simulation.

    Note: The EdgeAIBridge has been removed. The Arduino Q no longer does
    sensor fusion — that is exclusively the FPGA's job. The Arduino Q now
    runs the agentic AI workflow via its on-device LLM (see edge_ai_agent.py).
    """
    global _bridge
    if _bridge is None:
        if settings.ENABLE_FPGA:
            try:
                from backend.hardware_bridge.fpga_dual_bridge import DualAcceleratorBridge
                _bridge = DualAcceleratorBridge(port=settings.FPGA_PORT, simulation=False)
                logger.info("✅ Real FPGA bridge initialized on %s", settings.FPGA_PORT)
            except Exception as exc:
                logger.warning("FPGA bridge failed (%s), using mock.", exc)
                _bridge = MockFPGABridge()
        else:
            _bridge = MockFPGABridge()
            logger.info("⚙️  FPGA in simulation mode (AMD ZYNQ-7000 not connected)")
    return _bridge


def is_real_hardware() -> bool:
    """Check if using real FPGA hardware (not simulation)."""
    bridge = get_fpga_bridge()
    return not isinstance(bridge, MockFPGABridge)