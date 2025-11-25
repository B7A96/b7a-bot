# bot/indicators.py
"""
SniperFlow Ultra - Indicator Engine
هنا نحسب المؤشرات الأساسية:
RSI, EMA, MACD, Bollinger Bands, VWAP
"""

from typing import List, Optional, Dict
import numpy as np


def _to_np(values: List[float]) -> np.ndarray:
    """تحويل الليست إلى numpy array مع فلترة None."""
    return np.array([v for v in values if v is not None], dtype=float)


# =======================
# EMA & SMA
# =======================

def sma(values: List[float], period: int) -> Optional[float]:
    arr = _to_np(values)
    if arr.size < period:
        return None
    return float(arr[-period:].mean())


def ema(values: List[float], period: int) -> Optional[float]:
    arr = _to_np(values)
    if arr.size < period:
        return None

    k = 2 / (period + 1)
    ema_val = arr[0]
    for price in arr[1:]:
        ema_val = price * k + ema_val * (1 - k)
    return float(ema_val)


# =======================
# RSI
# =======================

def rsi(values: List[float], period: int = 14) -> Optional[float]:
    """
    RSI كلاسيكي 14 فترة.
    """
    arr = _to_np(values)
    if arr.size <= period:
        return None

    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    if avg_loss == 0:
        return 100.0

    # نستخدم طريقة Wilder
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return float(rsi_val)


# =======================
# MACD
# =======================

def macd(
    values: List[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Optional[Dict[str, float]]:
    """
    يرجّع dict:
    {
        "macd": آخر قيمة MACD,
        "signal": خط الإشارة,
        "hist": الهيستوجرام
    }
    """
    arr = _to_np(values)
    if arr.size < slow + signal_period:
        return None

    ema_fast_vals = []
    ema_slow_vals = []
    ema_fast_val = arr[0]
    ema_slow_val = arr[0]

    k_fast = 2 / (fast + 1)
    k_slow = 2 / (slow + 1)

    for price in arr:
        ema_fast_val = price * k_fast + ema_fast_val * (1 - k_fast)
        ema_slow_val = price * k_slow + ema_slow_val * (1 - k_slow)
        ema_fast_vals.append(ema_fast_val)
        ema_slow_vals.append(ema_slow_val)

    ema_fast_arr = np.array(ema_fast_vals)
    ema_slow_arr = np.array(ema_slow_vals)
    macd_line = ema_fast_arr - ema_slow_arr

    # نحسب EMA لخط الماكد كـ signal
    signal_vals = []
    signal_val = macd_line[0]
    k_sig = 2 / (signal_period + 1)
    for v in macd_line:
        signal_val = v * k_sig + signal_val * (1 - k_sig)
        signal_vals.append(signal_val)

    macd_last = float(macd_line[-1])
    signal_last = float(signal_vals[-1])
    hist_last = float(macd_last - signal_last)

    return {
        "macd": macd_last,
        "signal": signal_last,
        "hist": hist_last,
    }


# =======================
# Bollinger Bands
# =======================

def bollinger_bands(
    values: List[float],
    period: int = 20,
    std_factor: float = 2.0,
) -> Optional[Dict[str, float]]:
    """
    يرجّع dict:
    {
        "mid": خط المتوسط,
        "upper": الباند العلوي,
        "lower": الباند السفلي
    }
    """
    arr = _to_np(values)
    if arr.size < period:
        return None

    window = arr[-period:]
    mid = window.mean()
    std = window.std(ddof=0)

    upper = mid + std_factor * std
    lower = mid - std_factor * std

    return {
        "mid": float(mid),
        "upper": float(upper),
        "lower": float(lower),
    }


# =======================
# VWAP
# =======================

def vwap(
    high: List[float],
    low: List[float],
    close: List[float],
    volume: List[float],
) -> Optional[float]:
    """
    VWAP بسيط باستخدام (H+L+C)/3 * Volume.
    """
    if not (len(high) == len(low) == len(close) == len(volume)):
        return None

    h = _to_np(high)
    l = _to_np(low)
    c = _to_np(close)
    v = _to_np(volume)

    if h.size == 0:
        return None

    typical_price = (h + l + c) / 3.0
    cum_vp = np.sum(typical_price * v)
    cum_vol = np.sum(v)

    if cum_vol == 0:
        return None

    return float(cum_vp / cum_vol)


# =======================
# حزمة مؤشرات كاملة لإطار زمني واحد
# =======================

def compute_all_indicators(
    closes: List[float],
    highs: Optional[List[float]] = None,
    lows: Optional[List[float]] = None,
    volumes: Optional[List[float]] = None,
) -> Dict[str, Optional[float]]:
    """
    تحسب أهم المؤشرات لإطار زمني واحد (مثلاً 15m أو 1h).
    """
    result: Dict[str, Optional[float]] = {}

    result["rsi"] = rsi(closes)
    result["ema_50"] = ema(closes, 50)
    result["ema_200"] = ema(closes, 200)

    macd_result = macd(closes)
    if macd_result:
        result["macd"] = macd_result["macd"]
        result["macd_signal"] = macd_result["signal"]
        result["macd_hist"] = macd_result["hist"]
    else:
        result["macd"] = None
        result["macd_signal"] = None
        result["macd_hist"] = None

    bb = bollinger_bands(closes)
    if bb:
        result["bb_mid"] = bb["mid"]
        result["bb_upper"] = bb["upper"]
        result["bb_lower"] = bb["lower"]
    else:
        result["bb_mid"] = None
        result["bb_upper"] = None
        result["bb_lower"] = None

    if highs and lows and volumes:
        result["vwap"] = vwap(highs, lows, closes, volumes)
    else:
        result["vwap"] = None

    return result
