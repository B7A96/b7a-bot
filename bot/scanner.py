import time
from typing import List, Tuple, Optional

import requests

# =========================
# إعدادات عامة
# =========================

SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"

# Cache داخلي لتقليل الضغط على Binance
_CACHE_TTL_SECONDS = 30.0  # نحدّث القائمة كل 30 ثانية فقط
_last_cache_ts: float = 0.0
_last_cached_symbols: List[str] = []


class BinanceScannerError(Exception):
    """خطأ عام في جلب بيانات السكانر من بايننس."""
    pass


# =========================
# دوال مساعدة لطلبات HTTP
# =========================

def _safe_get(url: str, timeout: float = 10.0, max_retries: int = 3) -> list:
    """
    يرسل GET مع إعادة المحاولة عند الأخطاء المؤقتة.
    يرجّع list (JSON).
    """
    backoff = 1.0
    last_exc: Optional[Exception] = None

    for _ in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            # أحياناً ترجع dict فيها "code"/"msg"
            raise BinanceScannerError(f"Unexpected JSON type from {url}: {type(data)}")
        except Exception as e:
            last_exc = e
            time.sleep(backoff)
            backoff *= 1.5  # backoff بسيط

    # لو وصلنا هنا، فشلنا بعد كل المحاولات
    raise BinanceScannerError(f"Failed to GET {url}: {last_exc}")


def _fetch_24hr_spot() -> list:
    url = f"{SPOT_BASE_URL}/api/v3/ticker/24hr"
    return _safe_get(url, timeout=10.0, max_retries=3)


def _fetch_24hr_futures() -> list:
    url = f"{FUTURES_BASE_URL}/fapi/v1/ticker/24hr"
    return _safe_get(url, timeout=10.0, max_retries=3)


# =========================
# المنطق الرئيسي
# =========================

def _extract_top_usdt_from_tickers(tickers: list, limit: int) -> List[str]:
    """
    يستخرج أعلى عملات USDT من list جاي من /ticker/24hr
    ويرتبها حسب quoteVolume تنازلياً.
    """
    rows: List[Tuple[str, float]] = []

    for item in tickers:
        symbol = item.get("symbol", "")
        if not symbol.endswith("USDT"):
            continue

        # نتجنب الأزواج الغريبة مثل BUSDUSDT أو ما شابه لو حاب تستثنيها لاحقاً
        if symbol in ("BUSDUSDT",):
            continue

        try:
            quote_vol = float(item.get("quoteVolume", 0.0))
        except Exception:
            quote_vol = 0.0

        if quote_vol <= 0:
            continue

        rows.append((symbol, quote_vol))

    if not rows:
        raise BinanceScannerError("No valid USDT symbols found in tickers.")

    # ترتيب من الأكبر إلى الأصغر
    rows.sort(key=lambda x: x[1], reverse=True)

    # أخذ أول limit رمز
    top = [s for s, _ in rows[:limit]]

    # نرجّع الأسماء بدون USDT عشان /signal يشتغل مثل قبل: BTC, ETH, ...
    return [s.replace("USDT", "") for s in top]


def get_top_usdt_symbols(limit: int = 40, use_cache: bool = True) -> List[str]:
    """
    يرجع قائمة بأعلى عملات USDT من حيث حجم التداول (quoteVolume).
    مثال: ["BTC", "ETH", "SOL", ...].

    - فيه Cache داخلي لمدة 30 ثانية لحماية البوت لو صار عليه ضغط كبير.
    - يحاول أولاً Spot API، لو فشل → Futures API.
    - لو كلهم فشلوا لكن عندنا Cache سابق → نرجّع الكاش.
    """
    global _last_cache_ts, _last_cached_symbols

    now = time.time()

    # 1) استخدام الكاش لو صالح ومفعّل
    if use_cache and _last_cached_symbols and (now - _last_cache_ts) < _CACHE_TTL_SECONDS:
        # نرجع فقط limit عناصر من الكاش
        return _last_cached_symbols[:limit]

    # 2) نحاول جلب البيانات من Spot أولاً
    last_error: Optional[Exception] = None
    tickers: Optional[list] = None

    try:
        tickers = _fetch_24hr_spot()
    except Exception as e:
        last_error = e

    # 3) لو Spot فشل، نحاول Futures
    if tickers is None:
        try:
            tickers = _fetch_24hr_futures()
        except Exception as e:
            last_error = e

    # 4) لو ما قدرنا نجيب ولا من Spot ولا Futures
    if tickers is None:
        # لو عندنا كاش قديم نرجّعه بدل ما نخرب تجربة المستخدم
        if _last_cached_symbols:
            return _last_cached_symbols[:limit]
        # ما في كاش أيضاً → نرفع خطأ واضح
        raise BinanceScannerError(f"Failed to fetch 24hr tickers from Binance. Last error: {last_error}")

    # 5) نفلتر ونرتّب
    symbols = _extract_top_usdt_from_tickers(tickers, limit=limit)

    # 6) نحفظ في الكاش للتالي
    _last_cache_ts = now
    _last_cached_symbols = symbols[:]

    return symbols
