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

def _build_funding_intel(symbol_base: str) -> Dict[str, Any]:
    """
    Funding Intel بسيط من:
    /api/futures/funding-rate/exchange-list
    ناخذ BINANCE / stablecoin margin لو متوفر.
    """
    try:
        data = _api_get(
            "/api/futures/funding-rate/exchange-list",
            params={"symbol": symbol_base},
            timeout=5.0,
        )
        items = data.get("data") or []
        if not isinstance(items, list) or not items:
            raise CoinglassError("Empty funding exchange-list data")

        # نحاول نلقط نفس العملة من الليست
        row = None
        for it in items:
            if str(it.get("symbol", "")).upper() == symbol_base.upper():
                row = it
                break
        if row is None:
            row = items[0]

        stable_list = row.get("stablecoin_margin_list") or []
        chosen = None
        for x in stable_list:
            ex = str(x.get("exchange", "")).upper()
            if "BINANCE" in ex:
                chosen = x
                break
        if chosen is None and stable_list:
            chosen = stable_list[0]

        if chosen is None:
            raise CoinglassError("No stablecoin funding data")

        rate = float(chosen.get("funding_rate") or 0.0)
        interval = chosen.get("funding_rate_interval")
        next_ts = chosen.get("next_funding_time")

        abs_rate = abs(rate)
        if abs_rate >= 0.08:
            severity = "EXTREME"
        elif abs_rate >= 0.03:
            severity = "HIGH"
        elif abs_rate >= 0.01:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        if rate > 0:
            side_bias = "LONG"
        elif rate < 0:
            side_bias = "SHORT"
        else:
            side_bias = "NEUTRAL"

        return {
            "available": True,
            "rate": rate,
            "abs_rate": abs_rate,
            "interval_hours": interval,
            "next_funding_time": next_ts,
            "severity": severity,      # LOW / MEDIUM / HIGH / EXTREME
            "side_bias": side_bias,    # LONG / SHORT / NEUTRAL
        }
    except CoinglassError as e:
        return {
            "available": False,
            "rate": None,
            "abs_rate": None,
            "interval_hours": None,
            "next_funding_time": None,
            "severity": "NONE",
            "side_bias": "NEUTRAL",
            "error": str(e),
        }


def _build_liquidation_intel(symbol_base: str) -> Dict[str, Any]:
    """
    Liquidation Intel من:
    /api/futures/liquidation/exchange-list

    نستخدم صف "All" كإجمالي:
      - long_liquidation_usd
      - short_liquidation_usd
    ونستنتج منها bias + intensity (0..1).
    """
    try:
        data = _api_get(
            "/api/futures/liquidation/exchange-list",
            params={"symbol": symbol_base},
            timeout=5.0,
        )
        items = data.get("data") or []
        if not isinstance(items, list) or not items:
            raise CoinglassError("Empty liquidation exchange-list data")

        agg = None
        for it in items:
            if str(it.get("exchange", "")).upper() == "ALL":
                agg = it
                break
        if agg is None:
            agg = items[0]

        total = float(agg.get("liquidation_usd") or 0.0)
        long_usd = float(agg.get("long_liquidation_usd") or 0.0)
        short_usd = float(agg.get("short_liquidation_usd") or 0.0)

        bias = "BALANCED"
        intensity = 0.0

        if total > 0:
            long_ratio = long_usd / total
            short_ratio = short_usd / total

            # Long washout قوي
            if long_ratio >= 0.7:
                bias = "LONG_WASHOUT"
                intensity = long_ratio
            # Short washout قوي
            elif short_ratio >= 0.7:
                bias = "SHORT_WASHOUT"
                intensity = short_ratio
            # انحياز واضح بدون ما يكون washout كامل
            elif abs(short_ratio - long_ratio) >= 0.3:
                bias = "SHORT_DOMINANT" if short_ratio > long_ratio else "LONG_DOMINANT"
                intensity = abs(short_ratio - long_ratio)

        return {
            "available": True,
            "liquidation_usd": total,
            "long_liquidation_usd": long_usd,
            "short_liquidation_usd": short_usd,
            "bias": bias,                  # LONG_WASHOUT / SHORT_WASHOUT / BALANCED / ...
            "intensity": float(intensity), # 0..1
        }
    except CoinglassError as e:
        return {
            "available": False,
            "liquidation_usd": None,
            "long_liquidation_usd": None,
            "short_liquidation_usd": None,
            "bias": "BALANCED",
            "intensity": 0.0,
            "error": str(e),
        }


def get_coinglass_intel(symbol: str) -> Dict[str, Any]:
    """
    High-level intel used by الـ Engine.

    الشكل المرجع:
      {
        "available": bool,
        "symbol": "BTC",
        "open_interest": {...},
        "funding": {...},
        "liquidation": {...},
        "futures_status": {...},
        "spot_status": {...},
        "btc_etf": {...},
        "error": Optional[str],
      }
    """
    base = _normalize_symbol(symbol)
    cache_key = f"intel:{base}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    result: Dict[str, Any] = {
        "available": False,
        "symbol": base,
        "open_interest": {
            "available": False,
            "oi_usd": None,
            "oi_change_24h": None,
            "oi_bias": "NEUTRAL",
        },
        "funding": {
            "available": False,
            "rate": None,
            "abs_rate": None,
            "interval_hours": None,
            "next_funding_time": None,
            "severity": "NONE",
            "side_bias": "NEUTRAL",
        },
        "liquidation": {
            "available": False,
            "liquidation_usd": None,
            "long_liquidation_usd": None,
            "short_liquidation_usd": None,
            "bias": "BALANCED",
            "intensity": 0.0,
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

    # OI Intel
    try:
        oi = _build_oi_intel(base)
        result["open_interest"] = oi
        if oi.get("available"):
            result["available"] = True
    except Exception as e:
        result["error"] = str(e)

    # Funding Intel
    try:
        funding = _build_funding_intel(base)
        result["funding"] = funding
        if funding.get("available"):
            result["available"] = True
    except Exception as e:
        if not result["error"]:
            result["error"] = str(e)

    # Liquidation Intel
    try:
        liq = _build_liquidation_intel(base)
        result["liquidation"] = liq
        if liq.get("available"):
            result["available"] = True
    except Exception as e:
        if not result["error"]:
            result["error"] = str(e)

    _cache_set(cache_key, result)
    return result

