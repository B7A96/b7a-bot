import time
from typing import Dict, Any, List, Tuple

import numpy as np
import requests

BINANCE_BASE_URL = "https://api.binance.com"

# الفريمات اللي نستخدمها في التحليل
TIMEFRAMES = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


class MarketDataError(Exception):
    """Raised when we cannot fetch data from Binance."""


def _normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"
    return symbol


def fetch_klines(symbol: str, interval: str, limit: int = 200) -> Dict[str, np.ndarray]:
    """
    يجلب بيانات OHLCV من Binance ويحولها إلى numpy arrays.
    """
    symbol = _normalize_symbol(symbol)
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        raise MarketDataError(f"Binance error {resp.status_code}: {resp.text}")

    raw = resp.json()
    if not raw:
        raise MarketDataError("No kline data returned from Binance")

    opens, highs, lows, closes, volumes = [], [], [], [], []
    for k in raw:
        opens.append(float(k[1]))
        highs.append(float(k[2]))
        lows.append(float(k[3]))
        closes.append(float(k[4]))
        volumes.append(float(k[5]))

    return {
        "open": np.array(opens, dtype=float),
        "high": np.array(highs, dtype=float),
        "low": np.array(lows, dtype=float),
        "close": np.array(closes, dtype=float),
        "volume": np.array(volumes, dtype=float),
    }


# =========================
# Indicators
# =========================

def ema(series: np.ndarray, period: int) -> np.ndarray:
    if series.size < period:
        raise ValueError("Not enough data for EMA")
    alpha = 2 / (period + 1)
    result = np.zeros_like(series)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    if series.size < period + 1:
        raise ValueError("Not enough data for RSI")
    deltas = np.diff(series)
    seed = deltas[:period]
    up = seed[seed > 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi_values = np.zeros_like(series)
    rsi_values[:period] = 100.0 - (100.0 / (1.0 + rs))

    up_vals = np.where(deltas > 0, deltas, 0.0)
    down_vals = np.where(deltas < 0, -deltas, 0.0)

    up_ema = up
    down_ema = down
    for i in range(period, len(series) - 1):
        up_ema = (up_ema * (period - 1) + up_vals[i]) / period
        down_ema = (down_ema * (period - 1) + down_vals[i]) / period
        rs = up_ema / down_ema if down_ema != 0 else 0
        rsi_values[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return rsi_values


def macd(series: np.ndarray, fast: int = 12, slow: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    if series.size < slow + signal_period:
        raise ValueError("Not enough data for MACD")
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    return macd_line, signal_line


def bollinger_bands(series: np.ndarray, period: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if series.size < period:
        raise ValueError("Not enough data for Bollinger Bands")
    sma = np.convolve(series, np.ones(period) / period, mode="valid")
    padded_sma = np.concatenate([np.full(series.size - sma.size, np.nan), sma])
    rolling_std = []
    for i in range(series.size):
        if i < period - 1:
            rolling_std.append(np.nan)
        else:
            window = series[i - period + 1: i + 1]
            rolling_std.append(np.std(window))
    rolling_std = np.array(rolling_std)
    upper = padded_sma + num_std * rolling_std
    lower = padded_sma - num_std * rolling_std
    return lower, padded_sma, upper


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    typical_price = (high + low + close) / 3.0
    cumulative_vp = np.cumsum(typical_price * volume)
    cumulative_volume = np.cumsum(volume)
    return cumulative_vp / np.maximum(cumulative_volume, 1e-9)


def volume_surge(volume: np.ndarray, lookback: int = 20, threshold: float = 2.0) -> bool:
    if volume.size < lookback + 1:
        return False
    recent = volume[-1]
    avg_prev = volume[-(lookback + 1): -1].mean()
    return recent > threshold * avg_prev


def price_change(series: np.ndarray, period: int = 1) -> float:
    if series.size < period + 1:
        return 0.0
    return (series[-1] - series[-period - 1]) / series[-period - 1] * 100.0


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average True Range لقياس تذبذب السعر.
    نستخدمه لحساب وقف الخسارة والأهداف.
    """
    if len(close) < period + 1:
        raise ValueError("Not enough data for ATR")

    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        high_low = high[i] - low[i]
        high_close_prev = abs(high[i] - close[i - 1])
        low_close_prev = abs(low[i] - close[i - 1])
        tr[i] = max(high_low, high_close_prev, low_close_prev)

    atr_values = np.zeros_like(close)
    atr_values[period - 1] = tr[:period].mean()

    for i in range(period, len(close)):
        atr_values[i] = (atr_values[i - 1] * (period - 1) + tr[i]) / period

    return atr_values


# =========================
# Liquidity Map Engine
# =========================

def _detect_swings(high: np.ndarray, low: np.ndarray, left: int = 2, right: int = 2) -> Tuple[List[int], List[int]]:
    """
    يحدد swing highs و swing lows بسيطة (قمة أعلى من الجيران / قاع أقل من الجيران).
    """
    n = len(high)
    swing_highs: List[int] = []
    swing_lows: List[int] = []
    for i in range(left, n - right):
        window_high = high[i - left: i + right + 1]
        if high[i] >= window_high.max():
            swing_highs.append(i)
        window_low = low[i - left: i + right + 1]
        if low[i] <= window_low.min():
            swing_lows.append(i)
    return swing_highs, swing_lows


def _cluster_levels(prices: List[float], tolerance: float = 0.001) -> List[Dict[str, Any]]:
    """
    يجمع القمم / القيعان المتقاربة في مستوى واحد (zone).
    tolerance = 0.001 يعني تقريباً 0.1% فرق.
    """
    if not prices:
        return []
    prices_sorted = sorted(prices)
    levels: List[Dict[str, Any]] = []
    for p in prices_sorted:
        if not levels:
            levels.append({"price": float(p), "count": 1})
            continue
        last = levels[-1]
        if abs(p - last["price"]) / max(last["price"], 1e-9) <= tolerance:
            # ندمج في نفس المستوى
            new_count = last["count"] + 1
            last["price"] = (last["price"] * last["count"] + p) / new_count
            last["count"] = new_count
        else:
            levels.append({"price": float(p), "count": 1})
    return levels


def build_liquidity_map(ohlcv: Dict[str, np.ndarray], name: str) -> Dict[str, Any]:
    """
    يبني خريطة سيولة بسيطة:
    - يبحث عن swing highs / lows
    - يكوّن مستويات سيولة فوق السعر (Buy-side liquidity) وتحت السعر (Sell-side liquidity)
    - يحسب قوة كل منطقة + انحياز السيولة (أعلى / أسفل)
    """
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]

    last_close = float(close[-1])

    swing_highs, swing_lows = _detect_swings(high, low, left=2, right=2)

    high_prices = [high[i] for i in swing_highs]
    low_prices = [low[i] for i in swing_lows]

    # نكوّن مستويات من القمم والقيعان
    high_levels = _cluster_levels(high_prices, tolerance=0.0015)
    low_levels = _cluster_levels(low_prices, tolerance=0.0015)

    zones: List[Dict[str, Any]] = []

    above_strength = 0.0
    below_strength = 0.0

    # مستويات فوق السعر (Buy-side liquidity)
    for lvl in high_levels:
        price = float(lvl["price"])
        count = int(lvl["count"])
        distance_pct = (price - last_close) / last_close * 100.0
        if distance_pct <= 0:
            # مستوى أصبح تحت السعر الحالي -> سيولة قديمة
            continue
        # كل ما كان أقرب للسعر وكل ما تكرر أكثر، تزيد قوة السيولة
        base = min(count * 8.0, 40.0)
        if distance_pct < 1:
            dist_score = 40.0
        elif distance_pct < 3:
            dist_score = 30.0
        elif distance_pct < 5:
            dist_score = 20.0
        else:
            dist_score = 10.0
        strength = max(5.0, min(base + dist_score, 100.0))
        above_strength += strength
        zones.append(
            {
                "price": price,
                "side": "BUY",  # سيولة فوق السعر، عادة يجذب السعر لفوق
                "count": count,
                "distance_pct": distance_pct,
                "strength": strength,
            }
        )

    # مستويات تحت السعر (Sell-side liquidity)
    for lvl in low_levels:
        price = float(lvl["price"])
        count = int(lvl["count"])
        distance_pct = (last_close - price) / last_close * 100.0
        if distance_pct <= 0:
            # مستوى أصبح فوق السعر -> سيولة قديمة
            continue
        base = min(count * 8.0, 40.0)
        if distance_pct < 1:
            dist_score = 40.0
        elif distance_pct < 3:
            dist_score = 30.0
        elif distance_pct < 5:
            dist_score = 20.0
        else:
            dist_score = 10.0
        strength = max(5.0, min(base + dist_score, 100.0))
        below_strength += strength
        zones.append(
            {
                "price": price,
                "side": "SELL",  # سيولة تحت السعر، يجذب السعر لتحت
                "count": count,
                "distance_pct": distance_pct,
                "strength": strength,
            }
        )

    total_strength = above_strength + below_strength
    if total_strength <= 0:
        imbalance = 0.0
        bias = "FLAT"
        liq_score = 0.0
    else:
        imbalance = (above_strength - below_strength) / total_strength  # من -1 إلى +1
        if imbalance > 0.2:
            bias = "UP"   # سيولة أقوى فوق السعر -> السوق يميل يجمع سيولة لفوق
        elif imbalance < -0.2:
            bias = "DOWN"  # سيولة أقوى تحت السعر -> يميل يجمع سيولة لتحت
        else:
            bias = "FLAT"
        liq_score = abs(imbalance) * 100.0

    return {
        "timeframe": name,
        "zones": zones,
        "above_strength": float(above_strength),
        "below_strength": float(below_strength),
        "imbalance": float(imbalance),
        "bias": bias,
        "score": float(liq_score),
        "last_price": last_close,
    }


# =========================
# تحليل كل فريم
# =========================

def analyse_timeframe(ohlcv: Dict[str, np.ndarray], name: str) -> Dict[str, Any]:
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]

    info: Dict[str, Any] = {"timeframe": name}

    try:
        ema200 = ema(close, 200)[-1]
    except ValueError:
        ema200 = float("nan")

    try:
        rsi_arr = rsi(close, 14)
        rsi_last = float(rsi_arr[-1])
    except ValueError:
        rsi_last = float("nan")

    try:
        macd_line, sig_line = macd(close)
        macd_last = float(macd_line[-1])
        macd_signal_last = float(sig_line[-1])
    except ValueError:
        macd_last = float("nan")
        macd_signal_last = float("nan")

    try:
        lower_bb, mid_bb, upper_bb = bollinger_bands(close)
        lower_last = float(lower_bb[-1])
        upper_last = float(upper_bb[-1])
    except ValueError:
        lower_last = float("nan")
        upper_last = float("nan")

    vwap_arr = vwap(high, low, close, volume)
    vwap_last = float(vwap_arr[-1])

    vol_surge = volume_surge(volume)
    change_1 = price_change(close, 1)
    change_4 = price_change(close, 4)

    bullish_points = 0
    bearish_points = 0

    last_close = float(close[-1])

    if not np.isnan(ema200):
        if last_close > ema200:
            bullish_points += 1
        else:
            bearish_points += 1

    if not np.isnan(rsi_last):
        if 50 <= rsi_last <= 70:
            bullish_points += 1
        elif rsi_last > 70:
            bearish_points += 1  # overbought
        elif rsi_last < 30:
            bullish_points += 1  # oversold

    if not np.isnan(macd_last) and not np.isnan(macd_signal_last):
        if macd_last > macd_signal_last:
            bullish_points += 1
        else:
            bearish_points += 1

    if not np.isnan(lower_last) and not np.isnan(upper_last):
        if last_close <= lower_last:
            bullish_points += 1
        elif last_close >= upper_last:
            bearish_points += 1

    trend_score = (bullish_points - bearish_points) * 10 + 50
    trend_score = max(0, min(100, trend_score))

    pump_dump_risk = "LOW"
    if abs(change_1) > 3 and vol_surge:
        pump_dump_risk = "MEDIUM"
    if abs(change_1) > 6 and vol_surge:
        pump_dump_risk = "HIGH"

    # 🔥 خريطة السيولة لهذا الفريم
    liq_map = build_liquidity_map(ohlcv, name)
    liq_bias = liq_map.get("bias", "FLAT")
    liq_score = liq_map.get("score", 0.0)
    liq_above = liq_map.get("above_strength", 0.0)
    liq_below = liq_map.get("below_strength", 0.0)

    info.update(
        {
            "close": last_close,
            "ema200": ema200,
            "rsi": rsi_last,
            "macd": macd_last,
            "macd_signal": macd_signal_last,
            "bb_lower": lower_last,
            "bb_upper": upper_last,
            "vwap": vwap_last,
            "volume_surge": vol_surge,
            "change_1": change_1,
            "change_4": change_4,
            "trend_score": trend_score,
            "pump_dump_risk": pump_dump_risk,
            # Liquidity info
            "liquidity": liq_map,
            "liq_bias": liq_bias,
            "liq_score": liq_score,
            "liq_above": liq_above,
            "liq_below": liq_below,
        }
    )

    if bullish_points > bearish_points:
        info["trend"] = "BULLISH"
    elif bearish_points > bullish_points:
        info["trend"] = "BEARISH"
    else:
        info["trend"] = "RANGING"

    return info


# =========================
# دمج الفريمات واتخاذ القرار
# =========================

def combine_timeframes(tf_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    weights = {
        "15m": 0.2,
        "1h": 0.3,
        "4h": 0.3,
        "1d": 0.2,
    }

    score = 0.0
    total_weight = 0.0
    bullish_votes = 0.0
    bearish_votes = 0.0

    max_pump_risk = "LOW"

    liq_above_total = 0.0
    liq_below_total = 0.0

    for tf, data in tf_data.items():
        w = weights.get(tf, 0.0)
        tf_score = data.get("trend_score", 50)
        score += tf_score * w
        total_weight += w

        if data.get("trend") == "BULLISH":
            bullish_votes += w
        elif data.get("trend") == "BEARISH":
            bearish_votes += w

        risk = data.get("pump_dump_risk", "LOW")
        if risk == "HIGH":
            max_pump_risk = "HIGH"
        elif risk == "MEDIUM" and max_pump_risk != "HIGH":
            max_pump_risk = "MEDIUM"

        liq_above_total += data.get("liq_above", 0.0) * w
        liq_below_total += data.get("liq_below", 0.0) * w

    if total_weight > 0:
        score /= total_weight

    if bullish_votes > bearish_votes:
        global_trend = "BULLISH"
    elif bearish_votes > bullish_votes:
        global_trend = "BEARISH"
    else:
        global_trend = "RANGING"

    # 🔥 تجميع انحياز السيولة الكلي
    if liq_above_total + liq_below_total > 0:
        liq_imbalance = (liq_above_total - liq_below_total) / (liq_above_total + liq_below_total)
        if liq_imbalance > 0.2:
            liquidity_bias = "UP"
        elif liq_imbalance < -0.2:
            liquidity_bias = "DOWN"
        else:
            liquidity_bias = "FLAT"
        liquidity_score = abs(liq_imbalance) * 100.0
    else:
        liq_imbalance = 0.0
        liquidity_bias = "FLAT"
        liquidity_score = 0.0

    rsi_1h = tf_data.get("1h", {}).get("rsi")
    rsi_4h = tf_data.get("4h", {}).get("rsi")

    overbought = any(
        r is not None and not np.isnan(r) and r > 70 for r in [rsi_1h, rsi_4h]
    )
    oversold = any(
        r is not None and not np.isnan(r) and r < 30 for r in [rsi_1h, rsi_4h]
    )

    # منطقة عدم التداول
    action = "WAIT"

    if score >= 70 and bullish_votes > bearish_votes and not overbought and max_pump_risk != "HIGH":
        action = "BUY"
    elif score <= 30 and bearish_votes > bullish_votes and not oversold:
        action = "SELL"

    # لو الإشارة عندنا مترددة (بين 60 و 70) نستخدم انحياز السيولة لدفع القرار
    if action == "WAIT" and 60 <= score < 70 and max_pump_risk != "HIGH":
        if liquidity_bias == "UP" and bullish_votes >= bearish_votes:
            action = "BUY"
        elif liquidity_bias == "DOWN" and bearish_votes >= bullish_votes:
            action = "SELL"

    distance = abs(score - 50)
    if distance > 25:
        confidence = "HIGH"
    elif distance > 15:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    if max_pump_risk == "HIGH" and action == "BUY":
        action = "WAIT"

    return {
        "score": round(float(score), 2),
        "trend": global_trend,
        "action": action,
        "confidence": confidence,
        "pump_dump_risk": max_pump_risk,
        "liquidity_bias": liquidity_bias,
        "liquidity_score": round(float(liquidity_score), 2),
    }


# =========================
# اختيار نسب المخاطرة / الربح (احتياطي لو حبّينا نستخدمه لاحقاً)
# =========================

def choose_risk_reward(decision: Dict[str, Any], tf_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    يحدد نسب المخاطرة والربح تلقائياً حسب:
    - قوة الاتجاه (score + trend)
    - درجة الثقة
    - التقلب (من تغيّر السعر في 15m و 1h)
    - مخاطر Pump/Dump
    """
    score = decision.get("score", 50)
    confidence = decision.get("confidence", "LOW")
    pump_risk = decision.get("pump_dump_risk", "LOW")
    trend = decision.get("trend", "RANGING")

    change_15 = abs(tf_results.get("15m", {}).get("change_1", 0.0) or 0.0)
    change_1h = abs(tf_results.get("1h", {}).get("change_1", 0.0) or 0.0)
    volatility = max(change_15, change_1h)

    if volatility < 0.5:
        vol_level = "LOW"
    elif volatility < 1.5:
        vol_level = "MEDIUM"
    else:
        vol_level = "HIGH"

    risk_pct = 0.015   # 1.5%
    reward_pct = 0.03  # 3.0%

    strong_trend = (score >= 75 and confidence == "HIGH" and trend in ("BULLISH", "BEARISH"))
    medium_trend = (60 <= score < 75)

    if strong_trend and pump_risk == "LOW":
        risk_pct = 0.02
        reward_pct = 0.06
    elif medium_trend and pump_risk != "HIGH":
        risk_pct = 0.018
        reward_pct = 0.04
    else:
        risk_pct = 0.015
        reward_pct = 0.025

    if vol_level == "HIGH":
        risk_pct *= 1.3
        reward_pct *= 1.3
    elif vol_level == "LOW":
        risk_pct *= 0.8
        reward_pct *= 0.8

    if pump_risk == "MEDIUM":
        reward_pct *= 0.8

    return {
        "risk_pct": float(risk_pct),
        "reward_pct": float(reward_pct),
    }


def build_trade_levels(last_close: float,
                       ohlcv: Dict[str, np.ndarray],
                       action: str) -> Dict[str, float]:
    """
    يحسب مستويات SL / TP1 / TP2 بناءً على ATR من فريم أساسي (مثلاً 1h).
    """
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]

    try:
        atr_vals = atr(high, low, close, period=14)
        atr_last = float(atr_vals[-1])
    except ValueError:
        return {}

    levels: Dict[str, float] = {}

    if action == "BUY":
        sl = last_close - 1.5 * atr_last
        tp1 = last_close + 2.0 * atr_last
        tp2 = last_close + 3.0 * atr_last
    elif action == "SELL":
        sl = last_close + 1.5 * atr_last
        tp1 = last_close - 2.0 * atr_last
        tp2 = last_close - 3.0 * atr_last
    else:
        return {}

    levels["sl"] = sl
    levels["tp1"] = tp1
    levels["tp2"] = tp2
    return levels


# =========================
# نقطة الدخول الرئيسية
# =========================

def generate_signal(symbol: str) -> Dict[str, Any]:
    """
    Main Ultra Engine entrypoint.

    Returns a dict ready to be formatted by the Telegram layer.
    """
    symbol_norm = _normalize_symbol(symbol)
    tf_results: Dict[str, Dict[str, Any]] = {}

    # 1) نجيب بيانات كل الفريمات
    for name, interval in TIMEFRAMES.items():
        try:
            ohlcv = fetch_klines(symbol_norm, interval)
            tf_info = analyse_timeframe(ohlcv, name)
            tf_results[name] = tf_info
            time.sleep(0.1)  # نرفق شوي على Binance
        except Exception as e:
            tf_results[name] = {
                "timeframe": name,
                "error": str(e),
                "trend": "UNKNOWN",
                "trend_score": 50,
                "pump_dump_risk": "LOW",
            }

    # 2) ندمج الفريمات في قرار واحد
    combined = combine_timeframes(tf_results)

    # نحاول نستخدم إغلاق فريم 1h كسعر مرجعي، وإذا مو موجود نرجع لفريم 15m
    last_close = tf_results.get("1h", tf_results.get("15m", {})).get("close")

    # 3) توليد TP / SL بناءً على السعر والسكور والثقة
    tp = None
    sl = None
    rr = None  # Risk/Reward
    risk_pct = None
    reward_pct = None

    if last_close is not None:
        price = float(last_close)

        if combined["confidence"] == "HIGH":
            risk_pct = 2.0
        elif combined["confidence"] == "MEDIUM":
            risk_pct = 1.5
        else:
            risk_pct = 1.0

        if combined["score"] >= 75:
            reward_mult = 2.5
        elif combined["score"] >= 65:
            reward_mult = 2.0
        else:
            reward_mult = 1.5

        reward_pct = risk_pct * reward_mult

        action = combined["action"]

        if action == "BUY":
            sl = round(price * (1 - risk_pct / 100), 4)
            tp = round(price * (1 + reward_pct / 100), 4)
        elif action == "SELL":
            sl = round(price * (1 + risk_pct / 100), 4)
            tp = round(price * (1 - reward_pct / 100), 4)

        if tp is not None and sl is not None and price != sl:
            rr = round(abs((tp - price) / (price - sl)), 2)

    # 4) نص توضيحي ذكي مختصر
    reason_lines: List[str] = []
    reason_lines.append(f"الاتجاه العام: {combined['trend']}")
    reason_lines.append(
        "أقوى الفريمات: "
        + ", ".join(
            tf for tf, d in tf_results.items()
            if d.get("trend_score", 50) >= combined["score"]
        )
    )

    # انحياز السيولة الكلي
    liq_bias = combined.get("liquidity_bias")
    liq_score = combined.get("liquidity_score", 0.0)
    if liq_bias == "UP":
        reason_lines.append(
            f"السيولة المتراكمة أقوى أعلى السعر (Liquidity Score ≈ {liq_score:.0f}) → السوق يميل يجمع السيولة من فوق."
        )
    elif liq_bias == "DOWN":
        reason_lines.append(
            f"السيولة المتراكمة أقوى أسفل السعر (Liquidity Score ≈ {liq_score:.0f}) → السوق يميل يجمع السيولة من تحت."
        )

    if combined["pump_dump_risk"] != "LOW":
        reason_lines.append(
            f"تنبيه: احتمالية حركة حادة (Pump/Dump) = {combined['pump_dump_risk']} – انتبه مع الدخول."
        )

    explanation = " | ".join(reason_lines)

    return {
        "symbol": symbol_norm,
        "last_price": float(last_close) if last_close is not None else None,
        "timeframes": tf_results,
        "decision": combined,
        "reason": explanation,
        # خطة الصفقة
        "tp": tp,
        "sl": sl,
        "rr": rr,
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
    }
