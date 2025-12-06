import requests
from typing import List, Dict, Any, Optional


# ==========================
# 1) سعر Binance المباشر
# ==========================
def get_binance_price(symbol: str) -> Optional[float]:
    """
    إرجاع آخر سعر من Binance.
    """
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}USDT"
        r = requests.get(url, timeout=5)
        data = r.json()
        return float(data.get("price"))
    except Exception:
        return None


# ==========================
# 2) أعلى العملات فوليوم
# ==========================
def get_top_volume_symbols(limit: int = 40) -> List[str]:
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        r = requests.get(url, timeout=5)
        data = r.json()
    except Exception:
        return []

    rows = []
    for item in data:
        sym = item.get("symbol", "")
        if sym.endswith("USDT"):
            vol = float(item.get("quoteVolume", 0.0))
            base = sym.replace("USDT", "")
            rows.append((base, vol))

    rows.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in rows[:limit]]


# ==========================
# 3) أكبر الرابحين
# ==========================
def get_top_gainers() -> List[tuple]:
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=5)
        data = r.json()
    except Exception:
        return []

    gains = []
    for item in data:
        sym = item.get("symbol")
        if sym.endswith("USDT"):
            pct = float(item.get("priceChangePercent") or 0.0)
            if pct > 0:
                gains.append((sym.replace("USDT", ""), pct))

    gains.sort(key=lambda x: x[1], reverse=True)
    return gains


# ==========================
# 4) أكبر الخاسرين
# ==========================
def get_top_losers() -> List[tuple]:
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=5)
        data = r.json()
    except Exception:
        return []

    losses = []
    for item in data:
        sym = item.get("symbol")
        if sym.endswith("USDT"):
            pct = float(item.get("priceChangePercent") or 0.0)
            if pct < 0:
                losses.append((sym.replace("USDT", ""), pct))

    losses.sort(key=lambda x: x[1])
    return losses
