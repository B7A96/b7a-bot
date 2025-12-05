import os
import time
from collections import deque
from typing import Dict, Any, Optional

import requests

COINGLASS_BASE_URL = "https://open-api-v4.coinglass.com"
COINGLASS_API_KEY = os.getenv("COINGLASS_API_KEY")


class CoinglassError(Exception):
    """Generic Coinglass client error."""
    pass


# =========================
# Simple rate limiter + per-symbol cache
# =========================

# نخزن أزمنة آخر الريكوستات (لكل الإندبوينتات)
_REQUEST_TIMES = deque()
_MAX_REQUESTS_PER_MINUTE = 20  # أقل من حد خطة HOBBYIST علشان نكون بأمان

# كاش لكل عملة
_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TS: Dict[str, float] = {}
_CACHE_TTL_SECONDS = 60.0  # نخزن الإنتل لمدة 60 ثانية


def _normalize_symbol(symbol: str) -> str:
    s = (symbol or "").upper()
    # BTCUSDT -> BTC
    for suffix in ("USDT", "PERP", "USD"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def _rate_limit_allow() -> bool:
    """
    Local rate limiter to stay well under Hobbyist plan limits.
    """
    now = time.time()
    # نشيل أي ريكوست أقدم من 60 ثانية
    while _REQUEST_TIMES and now - _REQUEST_TIMES[0] > 60.0:
        _REQUEST_TIMES.popleft()
    if len(_REQUEST_TIMES) >= _MAX_REQUESTS_PER_MINUTE:
        return False
    _REQUEST_TIMES.append(now)
    return True


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    ts = _CACHE_TS.get(key)
    if ts is None:
        return None
    if time.time() - ts > _CACHE_TTL_SECONDS:
        return None
    return _CACHE.get(key)


def _cache_set(key: str, value: Dict[str, Any]) -> None:
    _CACHE[key] = value
    _CACHE_TS[key] = time.time()


def _api_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Thin wrapper around requests.get with auth and basic error handling.
    Returns the Coinglass top-level JSON (dict).
    """
    if not COINGLASS_API_KEY:
        raise CoinglassError("COINGLASS_API_KEY not set")

    if not _rate_limit_allow():
        raise CoinglassError("Local Coinglass rate limit reached; skipping request for now")

    url = f"{COINGLASS_BASE_URL}{path}"
    headers = {
        "CG-API-KEY": COINGLASS_API_KEY,
        "Accept": "application/json",
    }
    resp = requests.get(url, headers=headers, params=params or {}, timeout=timeout)
    try:
        data = resp.json()
    except Exception as e:
        raise CoinglassError(f"Invalid JSON from Coinglass: {e}") from e

    if resp.status_code != 200 or not isinstance(data, dict):
        raise CoinglassError(f"Bad response from Coinglass ({resp.status_code}): {data!r}")

    # v4 غالباً ترجع بهذا الشكل: {"code":"0","msg":"success","data":...}
    code = str(data.get("code", "0"))
    if code != "0":
        raise CoinglassError(f"Coinglass error code={code}, msg={data.get('msg')}")
    return data


def _build_oi_intel(symbol_base: str) -> Dict[str, Any]:
    """
    نستخدم /api/futures/open-interest/exchange-list لجلب Intel عن الـ Open Interest.
    """
    try:
        data = _api_get(
            "/api/futures/open-interest/exchange-list",
            params={"symbol": symbol_base},
            timeout=5.0,
        )
        items = data.get("data") or []
        if not isinstance(items, list) or not items:
            raise CoinglassError("Empty OI exchange-list data")

        # نفضّل الصف المجمع (exchange == "All") لو موجود
        agg = None
        for row in items:
            if str(row.get("exchange")).lower() == "all":
                agg = row
                break
        if agg is None:
            agg = items[0]

        oi_usd = float(agg.get("open_interest_usd") or 0.0)
        oi_chg_24h = agg.get("open_interest_change_percent_24h")
        try:
            oi_chg_24h = float(oi_chg_24h) if oi_chg_24h is not None else None
        except Exception:
            oi_chg_24h = None

        # Bias بسيط بناءً على تغير الـ OI خلال 24 ساعة
        if oi_chg_24h is None:
            bias = "NEUTRAL"
        elif oi_chg_24h >= 5.0:
            bias = "LEVERAGE_UP"
        elif oi_chg_24h <= -5.0:
            bias = "LEVERAGE_DOWN"
        else:
            bias = "NEUTRAL"

        return {
            "available": True,
            "oi_usd": oi_usd,
            "oi_change_24h": oi_chg_24h,
            "oi_bias": bias,
        }
    except CoinglassError as e:
        return {
            "available": False,
            "oi_usd": None,
            "oi_change_24h": None,
            "oi_bias": "NEUTRAL",
            "error": str(e),
        }


def get_coinglass_intel(symbol: str) -> Dict[str, Any]:
    """
    High-level intel used by the engine.

    الشكل المرجع:
      {
        "available": bool,
        "open_interest": { ... },
        "futures_status": { ... },
        "spot_status": { ... },
        "btc_etf": { ... },
        "error": Optional[str],
      }
    """
    base = _normalize_symbol(symbol)
    cache_key = f"intel:{base}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # هيكل افتراضي
    result: Dict[str, Any] = {
        "available": False,
        "open_interest": {
            "available": False,
            "oi_usd": None,
            "oi_change_24h": None,
            "oi_bias": "NEUTRAL",
        },
        "futures_status": {
            "available": False,
            "listed": None,
            "raw": None,
        },
        "spot_status": {
            "available": False,
            "listed": None,
            "raw": None,
        },
        "btc_etf": {
            "available": False,
            "funds": 0.0,
            "trading_count": 0,
            "halted_count": 0,
        },
        "error": None,
    }

    try:
        oi = _build_oi_intel(base)
        result["open_interest"] = oi
        result["available"] = bool(oi.get("available"))
    except Exception as e:
        # لو صار أي شيء غير متوقع نخليها غير متاحة
        result["error"] = str(e)

    _cache_set(cache_key, result)
    return result
