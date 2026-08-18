"""
Groq API Key Load Balancer
Round-robin distribution across multiple API keys with automatic failover.

Usage:
    from skyview.utils.llm_pool import get_llm, invoke_llm

    result = await invoke_llm(messages, timeout=15)
"""

import asyncio
import logging
import time
from itertools import cycle
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    from langchain_groq import ChatGroq

    logger_import_ok = True
except Exception as exc:  # pragma: no cover
    ChatGroq = None
    logger_import_ok = False
    _import_error = exc

if TYPE_CHECKING:
    from langchain_groq import ChatGroq as ChatGroqType

from skyview.utils.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

# ── Internal state ────────────────────────────────────────────────────────────

_keys: List[str] = _settings.GROQ_API_KEYS
_key_cycle = cycle(_keys) if _keys else iter([])

_key_errors: Dict[str, int] = {}
_key_last_error: Dict[str, float] = {}

_ERROR_COOLDOWN_SEC = 60
_MAX_ERRORS = 3


def _next_healthy_key() -> Optional[str]:
    """Pick next key that is not in cooldown."""
    now = time.time()

    for _ in range(max(len(_keys), 1)):
        try:
            key = next(_key_cycle)
        except StopIteration:
            return None

        errors = _key_errors.get(key, 0)
        last_err = _key_last_error.get(key, 0)

        if errors >= _MAX_ERRORS and (
            now - last_err
        ) < _ERROR_COOLDOWN_SEC:
            continue

        _key_errors[key] = 0
        return key

    return _keys[0] if _keys else None


def _mark_key_error(key: str) -> None:
    _key_errors[key] = _key_errors.get(key, 0) + 1
    _key_last_error[key] = time.time()


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.25,
    timeout: int = 30,
    key: Optional[str] = None,
) -> Optional[Any]:
    """
    Return a ChatGroq instance using the next healthy API key.
    """

    if ChatGroq is None:
        logger.error(
            "Failed to import langchain_groq. Import error: %s",
            globals().get("_import_error"),
        )
        return None

    chosen_key = key or _next_healthy_key()

    if not chosen_key:
        logger.warning("No Groq API keys configured")
        return None

    return ChatGroq(
        model=model or _settings.LLM_MODEL,
        api_key=chosen_key,
        temperature=temperature,
        timeout=timeout,
    )


async def invoke_llm(
    messages: Any,
    model: Optional[str] = None,
    temperature: float = 0.25,
    timeout: int = 30,
    retries: int = 3,
) -> Optional[str]:
    """
    Invoke the LLM with automatic key rotation and retry on failure.

    Args:
        messages: LangChain-style messages or list of (role, text) tuples.
        model: Override model name.
        temperature: Sampling temperature.
        timeout: Per-request timeout.
        retries: Number of attempts across different keys.

    Returns:
        Response content string, or None on complete failure.
    """

    if not _keys:
        logger.warning("invoke_llm: No API keys available")
        return None

    if ChatGroq is None:
        logger.error(
            "Failed to import langchain_groq. Import error: %s",
            globals().get("_import_error"),
        )
        return None

    last_exc = None

    for attempt in range(retries):
        key = _next_healthy_key()

        if not key:
            break

        try:
            llm = ChatGroq(
                model=model or _settings.LLM_MODEL,
                api_key=key,
                temperature=temperature,
                timeout=timeout,
            )

            response = await asyncio.to_thread(
                llm.invoke,
                messages,
            )

            return response.content

        except Exception as exc:
            last_exc = exc

            _mark_key_error(key)

            logger.warning(
                "LLM key attempt %d/%d failed (key=...%s): %s",
                attempt + 1,
                retries,
                key[-6:],
                exc,
            )

            await asyncio.sleep(0.3 * (attempt + 1))

    logger.error(
        "All LLM retries exhausted. Last error: %s",
        last_exc,
    )

    return None


def pool_status() -> Dict[str, Any]:
    """Return current load-balancer health info."""

    now = time.time()
    statuses = []

    for key in _keys:
        errors = _key_errors.get(key, 0)
        last_err = _key_last_error.get(key, 0)

        cooling = (
            errors >= _MAX_ERRORS
            and (now - last_err) < _ERROR_COOLDOWN_SEC
        )

        statuses.append(
            {
                "key_suffix": key[-6:],
                "consecutive_errors": errors,
                "cooling_down": cooling,
                "cooldown_remaining_s": (
                    max(
                        0,
                        int(
                            _ERROR_COOLDOWN_SEC
                            - (now - last_err)
                        ),
                    )
                    if cooling
                    else 0
                ),
            }
        )

    return {
        "total_keys": len(_keys),
        "healthy_keys": sum(
            1 for s in statuses if not s["cooling_down"]
        ),
        "keys": statuses,
    }


# ── Edge-First Convenience Export ─────────────────────────────────────────────
# All agent code should call invoke_llm_edge_first() instead of invoke_llm().
# This tries the Arduino Q on-device LLM first, then falls back to cloud Groq.

def get_invoke_llm_edge_first():
    """
    Lazy import to avoid circular dependencies.
    Returns the invoke_llm_edge_first function from edge_ai_agent.
    """
    from skyview.agents.edge_ai_agent import invoke_llm_edge_first
    return invoke_llm_edge_first