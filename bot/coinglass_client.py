import os
import time
import requests
from typing import Dict, Any, Optional, List


COINGLASS_BASE_URL = "https://open-api-v4.coinglass.com"
COINGLASS_API_KEY = os.getenv("COINGLASS_API_KEY")


class CoinglassError(Exception):
    pass


def _headers() -> Dict[str, str]:
    if not COINGLASS_API_KEY:
        raise CoinglassError("COINGLASS_API_KEY is not set in environment variables")
    return {
        "accept": "application/json",
        "CG-API-KEY": COINGLASS_API_KEY,
    }


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = COINGLASS_BASE_URL + path
    resp = requests.get(url, headers=_headers(), params=params or {}, timeout=10)
    if resp.status_code != 200:
        raise CoinglassError(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    # حسب docs: code = "0" يعني success
    if str(data.get("code")) != "0":
        raise CoinglassError(f"Coinglass error: {data}")
    return data


# ======================
# Simple Cache + Flags
# ======================
_TOP_CACHE: Dict[tuple, tuple] = {}       # key -> (ts, result)
_TOP_CACHE_TTL = 300                      # 5 دقائق

_LIQ_CACHE: Dict[str, tuple] = {}         # symbol -> (ts, result)
_LIQ_CACHE_TTL = 180                      # 3 دقائق

_LIQ_SUPPORTED = True                     # لو طلع "Not Supported" نعطله نهائياً


# ================================
# 1) Top Long/Short Position Ratio
# ================================
# endpoint: /api/futures/top-long-short-position-ratio/history :contentReference[oaicite:1]{index=1}

def get_top_long_short_ratio(
    symbol: str,
    exchange: str = "Binance",
    interval: str = "4h",
    limit: int = 1,
) -> Dict[str, Any]:
    """
    يرجّع آخر قيمة لـ Long/Short Ratio للـ top traders.
    الآن فيه:
      - Caching 5 دقائق على مستوى (symbol, exchange, interval)
      - حماية بسيطة من Too Many Requests
    """
    key = (symbol.upper(), exchange, interval)
    now = time.time()

    # 1) إذا عندنا كاش جديد → رجّعه مباشرة
    if key in _TOP_CACHE:
        ts, cached = _TOP_CACHE[key]
        if now - ts < _TOP_CACHE_TTL:
            return cached

    path = "/api/futures/top-long-short-position-ratio/history"
    params = {
        "symbol": symbol.upper(),
        "exchange": exchange,
        "interval": interval,
        "limit": limit,
    }

    try:
        data = _get(path, params=params)
        items: List[Dict[str, Any]] = data.get("data", []) or []
        if not items:
            result = {
                "available": False,
                "top_long_pct": None,
                "top_short_pct": None,
                "top_long_short_ratio": None,
            }
        else:
            last = items[-1]
            result = {
                "available": True,
                "top_long_pct": float(last.get("top_position_long_percent", 0.0)),
                "top_short_pct": float(last.get("top_position_short_percent", 0.0)),
                "top_long_short_ratio": float(
                    last.get("top_position_long_short_ratio", 0.0)
                ),
            }

    except CoinglassError as e:
        # لو Too Many Requests أو أي خطأ → نرجّع نتيجة محايدة
        print("Coinglass top ratio error:", e)
        result = {
            "available": False,
            "top_long_pct": None,
            "top_short_pct": None,
            "top_long_short_ratio": None,
        }

    _TOP_CACHE[key] = (now, result)
    return result



# ================================
# 2) Liquidation Coin List (حجم التصفيات)
# ================================
# endpoint: /api/futures/liquidation/coin-list :contentReference[oaicite:2]{index=2}

def get_liquidation_intel(
    symbol: str,
    interval: str = "4h",
    window: int = 24,
) -> Dict[str, Any]:
    """
    Liquidation Intel مبسّط مع مراعاة:
      - بعض الخطط (مثل Hobbyist) ما تدعم هالـ endpoint → "Not Supported"
      - نستخدم Cache عشان ما نكرر الطلب.
    لو الخطة ما تدعم → يرجّع available=False دائماً بدون ما يزعج Coinglass.
    """
    global _LIQ_SUPPORTED

    # لو عرفنا من قبل إنه غير مدعوم → رجّع محايد على طول
    if not _LIQ_SUPPORTED:
        return {
            "available": False,
            "error": "Not supported on current Coinglass plan",
            "long_liq": None,
            "short_liq": None,
            "liq_bias": "NEUTRAL",
        }

    symbol_u = symbol.upper()
    now = time.time()

    # كاش لكل رمز
    if symbol_u in _LIQ_CACHE:
        ts, cached = _LIQ_CACHE[symbol_u]
        if now - ts < _LIQ_CACHE_TTL:
            return cached

    path = "/api/futures/liquidation/aggregated-history"
    params = {
        "symbol": symbol_u,
        "interval": interval,   # مثل "4h"
        "window": window,       # مثلاً 24 ساعة
    }

    try:
        data = _get(path, params=params)
    except CoinglassError as e:
        msg = str(e)
        print("Coinglass liquidation error:", msg)

        # لو الخطة ما تدعم أو رجّع Not Supported → عطّل الـ endpoint نهائياً
        if "Not Supported" in msg or "Account level" in msg:
            _LIQ_SUPPORTED = False

        result = {
            "available": False,
            "error": msg,
            "long_liq": None,
            "short_liq": None,
            "liq_bias": "NEUTRAL",
        }
        _LIQ_CACHE[symbol_u] = (now, result)
        return result

    items: List[Dict[str, Any]] = data.get("data", []) or []
    if not items:
        result = {
            "available": False,
            "error": None,
            "long_liq": None,
            "short_liq": None,
            "liq_bias": "NEUTRAL",
        }
        _LIQ_CACHE[symbol_u] = (now, result)
        return result

    long_liq_total = 0.0
    short_liq_total = 0.0

    for it in items:
        long_liq_total += float(it.get("longVolUsd", 0.0))
        short_liq_total += float(it.get("shortVolUsd", 0.0))

    if long_liq_total + short_liq_total <= 0:
        liq_bias = "NEUTRAL"
    elif long_liq_total > short_liq_total * 1.2:
        liq_bias = "LONG_FLUSH_SOON"
    elif short_liq_total > long_liq_total * 1.2:
        liq_bias = "SHORT_SQUEEZE_SOON"
    else:
        liq_bias = "BALANCED"

    result = {
        "available": True,
        "error": None,
        "long_liq": long_liq_total,
        "short_liq": short_liq_total,
        "liq_bias": liq_bias,
    }

    _LIQ_CACHE[symbol_u] = (now, result)
    return result
