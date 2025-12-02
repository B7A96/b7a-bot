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
    """
    Helper بسيط للنداء على Coinglass V4.
    يرفع CoinglassError لو صار HTTP error أو code != "0".
    """
    url = COINGLASS_BASE_URL + path
    resp = requests.get(url, headers=_headers(), params=params or {}, timeout=8)
    if resp.status_code != 200:
        raise CoinglassError(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    if str(data.get("code")) != "0":
        raise CoinglassError(f"Coinglass error: {data}")
    return data


def _symbol_base(symbol: str) -> str:
    """
    يحول BTCUSDT → BTC
    """
    s = symbol.strip().upper()
    if s.endswith("USDT"):
        s = s[:-4]
    return s


# ============================
# 1) Futures Open Interest
#    /api/futures/open-interest/exchange-list
# ============================

def get_open_interest_intel(symbol: str) -> Dict[str, Any]:
    """
    إنتل عن الـ Open Interest للعقود الدائمة من Coinglass (endpoint مسموح لـ Hobbyist).

    نرجع:
      - oi_usd        : إجمالي عقود مفتوحة بالدولار (كل المنصات)
      - oi_change_24h : نسبة التغير خلال 24 ساعة
      - oi_bias       : LEVERAGE_UP / LEVERAGE_DOWN / NEUTRAL
    """
    base = _symbol_base(symbol)
    try:
        data = _get("/api/futures/open-interest/exchange-list", params={"symbol": base})
    except CoinglassError as e:
        return {
            "available": False,
            "error": str(e),
            "oi_usd": None,
            "oi_change_24h": None,
            "oi_bias": "NEUTRAL",
        }

    items: List[Dict[str, Any]] = data.get("data") or []
    if not isinstance(items, list) or not items:
        return {
            "available": False,
            "error": None,
            "oi_usd": None,
            "oi_change_24h": None,
            "oi_bias": "NEUTRAL",
        }

    # نفضّل صف "All" لو موجود (كل المنصات)
    row: Dict[str, Any] = items[0]
    for it in items:
        ex_name = str(it.get("exName") or it.get("exchange") or "").lower()
        if ex_name == "all":
            row = it
            break

    def _pick(keys, default=0.0):
        for k in keys:
            if k in row and row[k] is not None:
                try:
                    return float(row[k])
                except Exception:
                    pass
        return float(default)

    oi_usd = _pick(
        ["openInterestUsd", "open_interest_usd", "openInterestValue", "open_interest_value"],
        0.0,
    )
    oi_change_24h = _pick(
        ["openInterestChangePercent24h", "open_interest_change_percent_24h", "openInterest24hChangePercent"],
        0.0,
    )

    if oi_change_24h > 5:
        oi_bias = "LEVERAGE_UP"
    elif oi_change_24h < -5:
        oi_bias = "LEVERAGE_DOWN"
    else:
        oi_bias = "NEUTRAL"

    return {
        "available": True,
        "error": None,
        "oi_usd": oi_usd,
        "oi_change_24h": oi_change_24h,
        "oi_bias": oi_bias,
    }


# ============================
# 2) Supported Coins (Futures / Spot)
#    /api/futures/supported-coins
#    /api/spot/supported-coins
# ============================

def _lookup_supported(symbol: str, market: str) -> Dict[str, Any]:
    """
    Helper عام للـ futures / spot.
    نستخدمه مرة لكل تحليل (ما نحرق الـ rate-limit).
    """
    base = _symbol_base(symbol)
    if market == "futures":
        path = "/api/futures/supported-coins"
    elif market == "spot":
        path = "/api/spot/supported-coins"
    else:
        raise ValueError("market must be 'futures' or 'spot'")

    try:
        data = _get(path)
    except CoinglassError as e:
        return {"available": False, "error": str(e), "listed": None, "raw": None}

    items: List[Dict[str, Any]] = data.get("data") or []
    if not isinstance(items, list):
        items = []

    found: Optional[Dict[str, Any]] = None
    for it in items:
        sym = str(it.get("symbol") or it.get("coin") or it.get("baseAsset") or "").upper()
        if sym == base:
            found = it
            break

    return {
        "available": True,
        "error": None,
        "listed": found is not None,
        "raw": found,
    }


def get_futures_supported(symbol: str) -> Dict[str, Any]:
    return _lookup_supported(symbol, "futures")


def get_spot_supported(symbol: str) -> Dict[str, Any]:
    return _lookup_supported(symbol, "spot")


# ============================
# 3) Bitcoin ETF List
#    /api/etf/bitcoin/list
# ============================

def get_bitcoin_etf_intel() -> Dict[str, Any]:
    """
    Bitcoin ETF intel (متاح حتى لـ Hobbyist).
    نستخدمه فقط لما يكون الزوج BTCUSDT كـ Sentiment مؤسسات.
    """
    try:
        data = _get("/api/etf/bitcoin/list")
    except CoinglassError as e:
        return {
            "available": False,
            "error": str(e),
            "funds": 0,
            "trading_count": 0,
            "halted_count": 0,
        }

    items: List[Dict[str, Any]] = data.get("data") or []
    if not isinstance(items, list):
        items = []

    trading = 0
    halted = 0
    for it in items:
        status = str(it.get("market_status") or it.get("marketStatus") or "").lower()
        if "trading" in status or status == "open":
            trading += 1
        elif "halt" in status or "closed" in status or "suspend" in status:
            halted += 1

    return {
        "available": True,
        "error": None,
        "funds": len(items),
        "trading_count": trading,
        "halted_count": halted,
    }


# ============================
# 4) High-level aggregator used by engine.py
# ============================

def get_coinglass_intel(symbol: str) -> Dict[str, Any]:
    """
    واجهة موحدة يستدعيها الـ B7A Ultra Engine.
    ترجع dict فيه:
      - open_interest
      - futures_status
      - spot_status
      - btc_etf (لو BTC فقط)
    """
    if not COINGLASS_API_KEY:
        # ما في مفتاح → نرجع Neutral
        return {
            "available": False,
            "open_interest": {"available": False, "oi_usd": None, "oi_change_24h": None, "oi_bias": "NEUTRAL"},
            "futures_status": {"available": False, "listed": None, "raw": None},
            "spot_status": {"available": False, "listed": None, "raw": None},
            "btc_etf": {"available": False, "funds": 0, "trading_count": 0, "halted_count": 0},
        }

    base = _symbol_base(symbol)

    oi = get_open_interest_intel(base)
    fut = get_futures_supported(base)
    spot = get_spot_supported(base)

    btc_etf = {"available": False, "funds": 0, "trading_count": 0, "halted_count": 0}
    if base == "BTC":
        btc_etf = get_bitcoin_etf_intel()

    return {
        "available": True,
        "open_interest": oi,
        "futures_status": fut,
        "spot_status": spot,
        "btc_etf": btc_etf,
    }
