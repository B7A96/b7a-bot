# bot/intel_hub.py

import time
from typing import Dict, Any, Optional

import requests
import numpy as np

# كاش بسيط – عشان ما نجلد الـ APIs كل ثانية
_GLOBAL_INTEL_CACHE: Dict[str, Any] = {
    "last_update": 0.0,
    "data": None,
}


def _fetch_fear_greed() -> Optional[int]:
    """
    يجلب آخر قيمة لـ Crypto Fear & Greed Index من alternative.me
    API: https://api.alternative.me/fng/  (endpoint: /fng/)
    يرجع رقم من 0 (خوف شديد) إلى 100 (طمع شديد).
    """
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        data = r.json()
        v = data["data"][0]["value"]
        return int(v)
    except Exception:
        return None


def _fetch_btc_klines(interval: str = "1h", limit: int = 100) -> Optional[Dict[str, np.ndarray]]:
    """
    يجلب شموع BTCUSDT من Binance – نستخدمها كتمثيل للسوق كله.
    """
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=5)
        raw = r.json()
        if not isinstance(raw, list) or not raw:
            return None

        close = np.array([float(x[4]) for x in raw], dtype=float)
        high = np.array([float(x[2]) for x in raw], dtype=float)
        low = np.array([float(x[3]) for x in raw], dtype=float)
        volume = np.array([float(x[5]) for x in raw], dtype=float)

        return {
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
        }
    except Exception:
        return None


def _compute_trend_and_regime(ohlcv: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    يحسب ترند BTC + حالة السوق (Regime) من الشموع.
      - UP/DOWN/FLAT
      - TRENDING / CHOP / CRASH
    """
    close = ohlcv["close"]
    volume = ohlcv["volume"]

    if len(close) < 20:
        return {"trend": "FLAT", "regime": "CHOP", "shock": False}

    ma_short = close[-5:].mean()
    ma_long = close[-20:].mean()

    rng = ohlcv["high"] - ohlcv["low"]
    atr = float(rng[-20:].mean())

    last = float(close[-1])
    prev = float(close[-2])
    change_1 = (last - prev) / prev * 100.0

    vol_recent = float(volume[-1])
    vol_avg = float(volume[-20:].mean())
    vol_surge = vol_recent > vol_avg * 2.0

    crash = change_1 <= -3.5 and vol_surge
    shock = abs(change_1) >= 3.0 and vol_surge

    if crash:
        regime = "CRASH"
    else:
        recent_max = float(close[-20:].max())
        recent_min = float(close[-20:].min()) or 1e-9
        width_pct = (recent_max - recent_min) / recent_min * 100.0

        if width_pct < 3.0:
            regime = "CHOP"
        else:
            regime = "TRENDING"

    if ma_short > ma_long * 1.003:
        trend = "UP"
    elif ma_short < ma_long * 0.997:
        trend = "DOWN"
        # وإلا نعتبره FLAT
    else:
        trend = "FLAT"

    return {
        "trend": trend,
        "regime": regime,
        "shock": shock,
        "change_1": change_1,
        "atr": atr,
        "vol_surge": vol_surge,
    }


def get_global_intel() -> Dict[str, Any]:
    """
    B7A Ultra – Global Intel Hub V1

    يرجّع نظرة عامة عن السوق:
      - btc_trend / btc_regime
      - fear_greed_index
      - global_mood_score (0 - 100)
      - shock_mode (حركة عنيفة على BTC؟)
    """
    now = time.time()
    if (
        _GLOBAL_INTEL_CACHE["data"] is not None
        and now - _GLOBAL_INTEL_CACHE["last_update"] < 60
    ):
        return _GLOBAL_INTEL_CACHE["data"]

    btc_klines = _fetch_btc_klines(interval="1h", limit=100)
    if btc_klines is None:
        btc_info = {
            "trend": "FLAT",
            "regime": "CHOP",
            "shock": False,
            "change_1": 0.0,
            "atr": 0.0,
            "vol_surge": False,
        }
    else:
        btc_info = _compute_trend_and_regime(btc_klines)

    fg = _fetch_fear_greed()
    if fg is None:
        mood = 50.0
    else:
        mood = float(fg)

    shock_mode = bool(btc_info.get("shock"))

    data = {
        "btc_trend": btc_info.get("trend", "FLAT"),
        "btc_regime": btc_info.get("regime", "CHOP"),
        "btc_change_1": float(btc_info.get("change_1", 0.0) or 0.0),
        "fear_greed_index": fg,
        "global_mood_score": mood,
        "shock_mode": shock_mode,
    }

    _GLOBAL_INTEL_CACHE["data"] = data
    _GLOBAL_INTEL_CACHE["last_update"] = now
    return data
