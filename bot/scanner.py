import requests
from typing import List

BINANCE_BASE_URL = "https://api.binance.com"


def get_top_usdt_symbols(limit: int = 40) -> List[str]:
    """
    يرجع قائمة بأعلى عملات USDT من حيث حجم التداول (quoteVolume)
    مثال: ["BTC", "ETH", "SOL", ...]
    """
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/24hr"

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for item in data:
        symbol = item.get("symbol", "")
        # نبي أزواج USDT فقط
        if not symbol.endswith("USDT"):
            continue

        # نستبعد التوكنات الغريبة (رافعة / UP / DOWN ...)
        if any(bad in symbol for bad in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"]):
            continue

        try:
            quote_vol = float(item.get("quoteVolume", 0.0))
        except ValueError:
            quote_vol = 0.0

        rows.append((symbol, quote_vol))

    # ترتيب حسب الفوليوم من الأكبر للأصغر
    rows.sort(key=lambda x: x[1], reverse=True)

    # نأخذ أول limit عملة
    top = [s for s, _ in rows[:limit]]

    # نرجع اسم العملة بدون USDT
    return [s.replace("USDT", "") for s in top]
