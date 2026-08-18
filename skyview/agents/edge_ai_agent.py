"""
Arduino Q — On-Device Agentic AI Gateway

The Arduino UNO Q (Qualcomm MPU) is the central hub that:
  1. Receives sensor data from ESP32 via LoRa
  2. Communicates with the AMD ZYNQ-7000 FPGA via UART for
     hardware-accelerated sensor fusion and rain prediction
  3. Runs the entire agentic AI workflow on-device using a local LLM
     (OpenAI-compatible endpoint, e.g., llama.cpp / Ollama)
  4. Falls back to cloud Groq API only when the on-device LLM is unavailable

Architecture:
  ESP32 --LoRa--> Arduino Q <--UART--> FPGA (sensor fusion + rain prediction)
                      |
                  On-Device LLM (Qualcomm MPU)
                      |
                  Agentic AI Agents (Weather, Farm, Alert, Mandi, Trend)
                      |
                  React Dashboard / Flutter App / WhatsApp
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from skyview.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── On-Device LLM State ──────────────────────────────────────────────────────

_edge_llm_available: Optional[bool] = None
_edge_llm_last_check: float = 0
_HEALTH_CHECK_INTERVAL = 30  # seconds between health checks


async def _check_edge_llm_health() -> bool:
    """Ping the Arduino Q's on-device LLM endpoint to see if it's ready."""
    global _edge_llm_available, _edge_llm_last_check

    now = time.time()
    if _edge_llm_available is not None and (now - _edge_llm_last_check) < _HEALTH_CHECK_INTERVAL:
        return _edge_llm_available

    if not settings.ENABLE_EDGE_AI:
        _edge_llm_available = False
        _edge_llm_last_check = now
        return False

    try:
        # Try the models endpoint (standard OpenAI-compatible)
        base_url = settings.ARDUINO_Q_LLM_ENDPOINT.rsplit("/chat/completions", 1)[0]
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/models")
            _edge_llm_available = resp.status_code == 200
    except Exception:
        # Fall back to a simple connectivity check
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(
                    f"http://{settings.ARDUINO_Q_HOST}:{settings.ARDUINO_Q_PORT}/status"
                )
                _edge_llm_available = resp.status_code == 200
        except Exception:
            _edge_llm_available = False

    _edge_llm_last_check = now
    if _edge_llm_available:
        logger.info("🧠 Arduino Q on-device LLM is available")
    else:
        logger.debug("Arduino Q on-device LLM is not reachable")
    return _edge_llm_available


# ── On-Device LLM Invocation ─────────────────────────────────────────────────

async def invoke_edge_llm(
    messages: Any,
    model: Optional[str] = None,
    temperature: float = 0.25,
    timeout: Optional[int] = None,
) -> Optional[str]:
    """
    Send an LLM inference request to the Arduino Q's on-device model.
    Uses the OpenAI-compatible chat/completions API format.

    Returns the response content string, or None on failure.
    """
    if not settings.ENABLE_EDGE_AI:
        return None

    # Convert LangChain-style (role, content) tuples to OpenAI messages format
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, tuple) and len(msg) == 2:
            formatted_messages.append({"role": msg[0], "content": msg[1]})
        elif isinstance(msg, dict):
            formatted_messages.append(msg)
        else:
            # LangChain message objects
            formatted_messages.append({
                "role": getattr(msg, "type", "user"),
                "content": getattr(msg, "content", str(msg)),
            })

    payload = {
        "model": model or settings.ARDUINO_Q_LLM_MODEL,
        "messages": formatted_messages,
        "temperature": temperature,
        "max_tokens": 1024,
    }

    req_timeout = timeout or settings.ARDUINO_Q_LLM_TIMEOUT

    try:
        async with httpx.AsyncClient(timeout=float(req_timeout)) as client:
            resp = await client.post(
                settings.ARDUINO_Q_LLM_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                if content:
                    logger.debug(
                        "Edge LLM response received (model=%s, tokens=%s)",
                        data.get("model", "unknown"),
                        data.get("usage", {}).get("total_tokens", "?"),
                    )
                    return content
            else:
                logger.warning(
                    "Edge LLM returned HTTP %d: %s",
                    resp.status_code, resp.text[:200],
                )
    except httpx.TimeoutException:
        logger.warning("Edge LLM request timed out after %ds", req_timeout)
    except Exception as exc:
        logger.warning("Edge LLM request failed: %s", exc)

    return None


# ── FPGA Results via UART ─────────────────────────────────────────────────────

async def get_fpga_results_from_gateway() -> Optional[Dict[str, Any]]:
    """
    Fetch the latest FPGA sensor fusion + rain prediction results
    from the Arduino Q gateway. The FPGA (AMD ZYNQ-7000) is connected
    to the Arduino Q via UART; the gateway exposes these results via HTTP.

    Returns a dict with keys like:
      - fusion_score, stress_index, alert_level
      - rain_probability, rain_alert
      - fpga_inference_ms
    or None if unavailable.
    """
    if not settings.ENABLE_EDGE_AI:
        return None

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"http://{settings.ARDUINO_Q_HOST}:{settings.ARDUINO_Q_PORT}/fpga/latest"
            )
            if resp.status_code == 200:
                data = resp.json()
                logger.debug("FPGA results received via gateway: %s", data)
                return data
    except Exception as exc:
        logger.debug("Could not fetch FPGA results from gateway: %s", exc)

    return None


def build_fpga_context(fpga_data: Optional[Dict[str, Any]]) -> str:
    """
    Format FPGA hardware-accelerated results into a context string
    that can be injected into agent prompts.
    """
    if not fpga_data:
        return ""

    lines = ["FPGA Hardware Accelerator Results (AMD ZYNQ-7000 via UART):"]

    fusion = fpga_data.get("fusion_score")
    if fusion is not None:
        stress = fpga_data.get("stress_index", "N/A")
        alert = fpga_data.get("alert_level", "N/A")
        lines.append(f"  - Sensor Fusion Score: {fusion}/100")
        lines.append(f"  - Stress Index: {stress}")
        lines.append(f"  - Alert Level: {alert}")

    rain_prob = fpga_data.get("rain_probability")
    if rain_prob is not None:
        rain_alert = fpga_data.get("rain_alert", 0)
        lines.append(f"  - Rain Probability: {rain_prob}%")
        lines.append(f"  - Rain Alert: {'YES' if rain_alert else 'No'}")

    inference_ms = fpga_data.get("fpga_inference_ms")
    if inference_ms is not None:
        lines.append(f"  - FPGA Inference Latency: {inference_ms}ms")

    return "\n".join(lines)


# ── Edge-First LLM with Cloud Fallback ────────────────────────────────────────

async def invoke_llm_edge_first(
    messages: Any,
    model: Optional[str] = None,
    temperature: float = 0.25,
    timeout: int = 30,
    retries: int = 3,
    include_fpga_context: bool = False,
) -> Optional[str]:
    """
    Primary LLM invocation function for the entire agentic AI workflow.

    Routing priority:
      1. Arduino Q on-device LLM (Qualcomm MPU) — if ENABLE_EDGE_AI is True
      2. Cloud Groq API — fallback when edge is unavailable or fails

    If include_fpga_context is True, fetches the latest FPGA results
    and prepends them as system context to the prompt.
    """
    # Optionally enrich prompts with FPGA hardware results
    if include_fpga_context:
        fpga_data = await get_fpga_results_from_gateway()
        fpga_ctx = build_fpga_context(fpga_data)
        if fpga_ctx:
            messages = _prepend_fpga_context(messages, fpga_ctx)

    # ── Try Arduino Q on-device LLM first ──
    if settings.ENABLE_EDGE_AI:
        is_healthy = await _check_edge_llm_health()
        if is_healthy:
            logger.info("🧠 Routing to Arduino Q on-device LLM (Qualcomm MPU)")
            result = await invoke_edge_llm(
                messages,
                model=model,
                temperature=temperature,
                timeout=timeout,
            )
            if result:
                return result
            logger.warning("Edge LLM returned empty — falling back to cloud")

        if not settings.EDGE_AI_FALLBACK_TO_CLOUD:
            logger.error("Edge LLM unavailable and cloud fallback is disabled")
            return None

    # ── Fallback to cloud Groq API ──
    logger.debug("Falling back to cloud Groq API")
    from skyview.utils.llm_pool import invoke_llm
    return await invoke_llm(
        messages,
        model=model,
        temperature=temperature,
        timeout=timeout,
        retries=retries,
    )


def _prepend_fpga_context(messages: Any, fpga_ctx: str) -> list:
    """Prepend FPGA context as a system message to the message list."""
    fpga_system_msg = (
        "system",
        f"The following hardware-accelerated analysis has been computed by "
        f"the AMD ZYNQ-7000 FPGA connected via UART. Use these results to "
        f"inform your response:\n\n{fpga_ctx}"
    )

    if isinstance(messages, list) and messages:
        return [fpga_system_msg] + list(messages)
    return [fpga_system_msg, messages]


# ── Status & Diagnostics ─────────────────────────────────────────────────────

async def get_edge_ai_status() -> Dict[str, Any]:
    """
    Comprehensive status of the Arduino Q Agentic AI Gateway.
    Reports on-device LLM health, FPGA connectivity, and system info.
    """
    status = {
        "edge_ai_enabled": settings.ENABLE_EDGE_AI,
        "arduino_q_host": settings.ARDUINO_Q_HOST,
        "arduino_q_port": settings.ARDUINO_Q_PORT,
        "on_device_llm": {
            "endpoint": settings.ARDUINO_Q_LLM_ENDPOINT,
            "model": settings.ARDUINO_Q_LLM_MODEL,
            "timeout_s": settings.ARDUINO_Q_LLM_TIMEOUT,
            "available": False,
        },
        "fpga": {
            "uart_port": settings.FPGA_UART_PORT,
            "baud_rate": settings.FPGA_UART_BAUD,
            "latest_results": None,
        },
        "cloud_fallback_enabled": settings.EDGE_AI_FALLBACK_TO_CLOUD,
    }

    if settings.ENABLE_EDGE_AI:
        # Check LLM health
        status["on_device_llm"]["available"] = await _check_edge_llm_health()

        # Check FPGA results
        fpga_data = await get_fpga_results_from_gateway()
        if fpga_data:
            status["fpga"]["latest_results"] = fpga_data

        # Try to get detailed system info from the gateway
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"http://{settings.ARDUINO_Q_HOST}:{settings.ARDUINO_Q_PORT}/status"
                )
                if resp.status_code == 200:
                    status["gateway_info"] = resp.json()
        except Exception:
            pass

    return status
