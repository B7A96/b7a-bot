import os
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
    يستخدمها البوت ليفهم منو متغلب: اللونغ ولا الشورت.
    """
    path = "/api/futures/top-long-short-position-ratio/history"
    params = {
        "symbol": symbol.upper(),
        "exchange": exchange,
        "interval": interval,
        "limit": limit,
    }
    data = _get(path, params=params)
    items: List[Dict[str, Any]] = data.get("data", []) or []
    if not items:
        return {
            "available": False,
            "top_long_pct": None,
            "top_short_pct": None,
            "top_long_short_ratio": None,
        }

    last = items[-1]
    return {
        "available": True,
        "top_long_pct": float(last.get("top_position_long_percent", 0.0)),
        "top_short_pct": float(last.get("top_position_short_percent", 0.0)),
        "top_long_short_ratio": float(last.get("top_position_long_short_ratio", 0.0)),
    }


# ================================
# 2) Liquidation Coin List (حجم التصفيات)
# ================================
# endpoint: /api/futures/liquidation/coin-list :contentReference[oaicite:2]{index=2}

def get_liquidation_intel(
    symbol: str,
    exchange: str = "Binance",
    window: str = "4h",
) -> Dict[str, Any]:
    """
    يرجّع صورة عن تصفيات LONG و SHORT على العملة من endpoint:
      /api/futures/liquidation/coin-list

    نختار نافذة زمنية (1h / 4h / 12h / 24h) ونطلع منها حجم التصفيات بالدولار.
    """

    path = "/api/futures/liquidation/coin-list"

    # Coinglass يستخدم exName لاسم المنصة (Binance, OKX, Bybit, ...)
    params = {
        "exName": exchange,
    }

    try:
        data = _get(path, params=params)
    except CoinglassError as e:
        return {
            "available": False,
            "error": str(e),
            "long_liq": None,
            "short_liq": None,
            "liq_bias": "NEUTRAL",
            "window": window,
        }

    items: List[Dict[str, Any]] = data.get("data", []) or []
    if not items:
        return {
            "available": False,
            "error": None,
            "long_liq": None,
            "short_liq": None,
            "liq_bias": "NEUTRAL",
            "window": window,
        }

    # نبحث عن العملة المطلوبة داخل الليست
    base = symbol.upper().replace("USDT", "")
    coin = None
    for c in items:
        code = str(c.get("symbol") or c.get("baseAsset") or "").upper()
        if code == base:
            coin = c
            break

    # لو ما لقيناها نستخدم أول عنصر كـ fallback (بس غالباً بنلاقيها)
    if coin is None:
        coin = items[0]

    # نختار الحقول حسب النافذة الزمنية
    field_map = {
        "1h": ("long_liquidation_usd_1h", "short_liquidation_usd_1h"),
        "4h": ("long_liquidation_usd_4h", "short_liquidation_usd_4h"),
        "12h": ("long_liquidation_usd_12h", "short_liquidation_usd_12h"),
        "24h": ("long_liquidation_usd_24h", "short_liquidation_usd_24h"),
    }
    long_key, short_key = field_map.get(window, field_map["4h"])

    long_liq = float(coin.get(long_key, 0.0) or 0.0)
    short_liq = float(coin.get(short_key, 0.0) or 0.0)

    if long_liq + short_liq <= 0:
        liq_bias = "NEUTRAL"
    elif long_liq > short_liq * 1.2:
        liq_bias = "LONG_FLUSH_SOON"      # تصفية لونغات محتملة لو نزل
    elif short_liq > long_liq * 1.2:
        liq_bias = "SHORT_SQUEEZE_SOON"   # شورت سكويز محتمل لو صعد
    else:
        liq_bias = "BALANCED"

    return {
        "available": True,
        "error": None,
        "long_liq": long_liq,
        "short_liq": short_liq,
        "liq_bias": liq_bias,
        "window": window,
    }

