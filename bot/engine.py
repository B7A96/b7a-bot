import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import requests
import csv
import os
from datetime import datetime

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
# Trade Logger
# =========================

def log_trade(data: Dict[str, Any]):
    """
    يسجل الصفقات الحقيقية في ملف CSV اسمه trades_log.csv
    """
    log_file = "trades_log.csv"

    file_exists = os.path.isfile(log_file)

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # لو أول مرة ننشئ الملف، نكتب الهيدر
        if not file_exists:
            writer.writerow([
                "datetime", "symbol", "action", "price",
                "tp", "sl", "rr",
                "grade", "score", "confidence",
                "pump_risk", "market_regime", "liquidity_bias",
                "no_trade"
            ])

        decision = data.get("decision", {})

        writer.writerow([
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            data.get("symbol"),
            decision.get("action"),
            data.get("last_price"),
            data.get("tp"),
            data.get("sl"),
            data.get("rr"),
            decision.get("grade"),
            decision.get("score"),
            decision.get("confidence"),
            decision.get("pump_dump_risk"),
            decision.get("market_regime"),
            decision.get("liquidity_bias"),
            decision.get("no_trade"),
        ])


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


def macd(series: np.ndarray, fast: int = 12, slow: int = 26,
         signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    if series.size < slow + signal_period:
        raise ValueError("Not enough data for MACD")
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    return macd_line, signal_line


def bollinger_bands(series: np.ndarray, period: int = 20,
                    num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         volume: np.ndarray) -> np.ndarray:
    typical_price = (high + low + close) / 3.0
    cumulative_vp = np.cumsum(typical_price * volume)
    cumulative_volume = np.cumsum(volume)
    return cumulative_vp / np.maximum(cumulative_volume, 1e-9)


def volume_surge(volume: np.ndarray, lookback: int = 20,
                 threshold: float = 2.0) -> bool:
    if volume.size < lookback + 1:
        return False
    recent = volume[-1]
    avg_prev = volume[-(lookback + 1): -1].mean()
    return recent > threshold * avg_prev


def price_change(series: np.ndarray, period: int = 1) -> float:
    if series.size < period + 1:
        return 0.0
    return (series[-1] - series[-period - 1]) / series[-period - 1] * 100.0


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> np.ndarray:
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

def _detect_swings(high: np.ndarray, low: np.ndarray,
                   left: int = 2, right: int = 2) -> Tuple[List[int], List[int]]:
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
            new_count = last["count"] + 1
            last["price"] = (last["price"] * last["count"] + p) / new_count
            last["count"] = new_count
        else:
            levels.append({"price": float(p), "count": 1})
    return levels


def build_liquidity_map(ohlcv: Dict[str, np.ndarray], name: str) -> Dict[str, Any]:
    """
    يبني خريطة سيولة بسيطة.
    """
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]

    last_close = float(close[-1])

    swing_highs, swing_lows = _detect_swings(high, low, left=2, right=2)

    high_prices = [high[i] for i in swing_highs]
    low_prices = [low[i] for i in swing_lows]

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
        above_strength += strength
        zones.append(
            {
                "price": price,
                "side": "BUY",
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
                "side": "SELL",
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
        imbalance = (above_strength - below_strength) / total_strength
        if imbalance > 0.2:
            bias = "UP"
        elif imbalance < -0.2:
            bias = "DOWN"
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

    # EMA 200
    try:
        ema200 = ema(close, 200)[-1]
    except ValueError:
        ema200 = float("nan")

    # RSI
    rsi_arr = None
    try:
        rsi_arr = rsi(close, 14)
        rsi_last = float(rsi_arr[-1])
    except ValueError:
        rsi_last = float("nan")
        rsi_arr = None

    # MACD
    try:
        macd_line, sig_line = macd(close)
        macd_last = float(macd_line[-1])
        macd_signal_last = float(sig_line[-1])
    except ValueError:
        macd_last = float("nan")
        macd_signal_last = float("nan")

    # Bollinger Bands
    try:
        lower_bb, mid_bb, upper_bb = bollinger_bands(close)
        lower_last = float(lower_bb[-1])
        upper_last = float(upper_bb[-1])
    except ValueError:
        lower_last = float("nan")
        upper_last = float("nan")

    # VWAP
    vwap_arr = vwap(high, low, close, volume)
    vwap_last = float(vwap_arr[-1])

    vol_surge = volume_surge(volume)
    change_1 = price_change(close, 1)
    change_4 = price_change(close, 4)

    bullish_points = 0
    bearish_points = 0

    last_close = float(close[-1])

    # اتجاه بالنسبة للـ EMA200
    if not np.isnan(ema200):
        if last_close > ema200:
            bullish_points += 1
        else:
            bearish_points += 1

    # سلوك RSI
    if not np.isnan(rsi_last):
        if 50 <= rsi_last <= 70:
            bullish_points += 1
        elif rsi_last > 70:
            bearish_points += 1
        elif rsi_last < 30:
            bullish_points += 1

    # MACD Cross
    if not np.isnan(macd_last) and not np.isnan(macd_signal_last):
        if macd_last > macd_signal_last:
            bullish_points += 1
        else:
            bearish_points += 1

    # Bollinger Touch
    if not np.isnan(lower_last) and not np.isnan(upper_last):
        if last_close <= lower_last:
            bullish_points += 1
        elif last_close >= upper_last:
            bearish_points += 1

    # =========================
    # Market Regime Detector
    # =========================
    try:
        atr_vals = atr(high, low, close, period=14)
        atr_last = float(atr_vals[-1])
    except Exception:
        atr_last = float("nan")

    distance_from_ema200 = (
        abs(last_close - ema200) if not np.isnan(ema200) else 0.0
    )

    if (
        not np.isnan(ema200)
        and not np.isnan(atr_last)
        and distance_from_ema200 > atr_last * 1.2
        and atr_last > (0.002 * last_close)  # ATR > 0.2% من السعر
    ):
        market_regime = "TRENDING"
    else:
        market_regime = "RANGING"

    # =========================
    # Breakout Detector
    # =========================
    lookback = min(20, len(close))
    if lookback < 5:
        recent_high = last_close
        recent_low = last_close
    else:
        recent_high = float(np.max(high[-lookback:]))
        recent_low = float(np.min(low[-lookback:]))

    is_breakout_up = False
    is_breakout_down = False

    if last_close > recent_high and change_1 > 0:
        is_breakout_up = True

    if last_close < recent_low and change_1 < 0:
        is_breakout_down = True

    # =========================
    # RSI Divergence
    # =========================
    has_bull_div = False
    has_bear_div = False

    if rsi_arr is not None and len(close) >= 20:
        # نستخدم نقطة قبل 10 شموع كنقطة مقارنة بسيطة
        prev_idx = -10

        prev_low = close[prev_idx]
        curr_low = close[-1]
        prev_rsi = rsi_arr[prev_idx]
        curr_rsi = rsi_arr[-1]

        if not np.isnan(prev_rsi) and not np.isnan(curr_rsi):
            # Bullish Divergence: السعر ينزل، RSI يطلع
            if curr_low < prev_low and curr_rsi > prev_rsi:
                has_bull_div = True

            # Bearish Divergence: السعر يطلع، RSI ينزل
            if curr_low > prev_low and curr_rsi < prev_rsi:
                has_bear_div = True

    # نكافئ/نعاقب حسب الدايفرجنس
    if has_bull_div:
        bullish_points += 1
    if has_bear_div:
        bearish_points += 1

    # =========================
    # Whale / Smart Money Detector
    # =========================
    whale_score = 0.0

    # جزء 1: انهيارات وفوليوم غريب
    if vol_surge:
        whale_score += 15

    # جزء 2: شموع سريعة جداً (Volatility)
    if abs(change_1) > 1.8:
        whale_score += 10
    if abs(change_1) > 3.5:
        whale_score += 15

    # جزء 3: ATR مرتفع = في تلاعب أو تجميع
    if not np.isnan(atr_last):
        if atr_last > (0.004 * last_close):  # ATR > 0.4% من السعر
            whale_score += 10
        if atr_last > (0.007 * last_close):  # ATR > 0.7%
            whale_score += 15

    # جزء 4: شموع بها Wicks قوية (جمع سيولة)
    if abs(high[-1] - close[-1]) > (0.004 * last_close):
        whale_score += 5
    if abs(close[-1] - low[-1]) > (0.004 * last_close):
        whale_score += 5

    whale_score = min(100.0, whale_score)

    # =========================
    # Trend Score + Pump/Dump
    # =========================
    trend_score = (bullish_points - bearish_points) * 10 + 50

    # تعديل بسيط للترند حسب وضع السوق والاختراق
    if market_regime == "TRENDING" and (is_breakout_up or is_breakout_down):
        trend_score += 5
    if market_regime == "RANGING" and (is_breakout_up or is_breakout_down):
        trend_score -= 5

    trend_score = max(0, min(100, trend_score))

    pump_dump_risk = "LOW"
    if abs(change_1) > 3 and vol_surge:
        pump_dump_risk = "MEDIUM"
    if abs(change_1) > 6 and vol_surge:
        pump_dump_risk = "HIGH"

    # خريطة السيولة لهذا الفريم
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
            "liquidity": liq_map,
            "liq_bias": liq_bias,
            "liq_score": liq_score,
            "liq_above": liq_above,
            "liq_below": liq_below,
            # إضافات الذكاء الجديد
            "market_regime": market_regime,
            "is_breakout_up": is_breakout_up,
            "is_breakout_down": is_breakout_down,
            "has_bull_div": has_bull_div,
            "has_bear_div": has_bear_div,
            "whale_score": whale_score,
        }
    )

    if bullish_points > bearish_points:
        info["trend"] = "BULLISH"
    elif bearish_points > bullish_points:
        info["trend"] = "BEARISH"
    else:
        info["trend"] = "RANGING"

    # =========================
    # Pre-Pump / Pre-Dump Detector
    # =========================
    pre_pump = False
    pre_dump = False

    # شروط قبل الانفجار:
    # - فوليوم أعلى من الطبيعي
    # - RSI يرتفع من مناطق منخفضة
    # - شمعة خضراء قوية + ATR عالي
    if (
        vol_surge
        and change_1 > 0.8
        and rsi_last > 35
        and not np.isnan(atr_last)
        and atr_last > (0.003 * last_close)
    ):
        pre_pump = True

    # شروط قبل الانهيار:
    if (
        vol_surge
        and change_1 < -0.8
        and rsi_last < 65
        and not np.isnan(atr_last)
        and atr_last > (0.003 * last_close)
    ):
        pre_dump = True

    info["pre_pump"] = pre_pump
    info["pre_dump"] = pre_dump

    return info


# =========================
# دمج الفريمات واتخاذ القرار
# =========================

def combine_timeframes(tf_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    دمج الفريمات في قرار واحد مع Ultra Filter V2:
    - يشدد على توافق الفريمات
    - يعتمد على فريم مرجعي (4h / 1d)
    - يرفع جودة Grade ويقلل الإشارات العشوائية
    """
    weights = {
        "15m": 0.2,
        "1h": 0.3,
        "4h": 0.3,
        "1d": 0.2,
    }

    score_sum = 0.0
    total_weight = 0.0
    bullish_votes = 0.0
    bearish_votes = 0.0

    max_pump_risk = "LOW"

    liq_above_total = 0.0
    liq_below_total = 0.0

    trending_weight = 0.0
    ranging_weight = 0.0
    breakout_up_weight = 0.0
    breakout_down_weight = 0.0
    bull_div_weight = 0.0
    bear_div_weight = 0.0

    whale_sum = 0.0
    pump_flags = 0.0
    dump_flags = 0.0

    for tf, data in tf_data.items():
        w = weights.get(tf, 0.0)
        if w <= 0:
            continue

        tf_score = data.get("trend_score", 50)
        score_sum += tf_score * w
        total_weight += w

        tf_trend = data.get("trend")
        if tf_trend == "BULLISH":
            bullish_votes += w
        elif tf_trend == "BEARISH":
            bearish_votes += w

        # pump/dump
        risk = data.get("pump_dump_risk", "LOW")
        if risk == "HIGH":
            max_pump_risk = "HIGH"
        elif risk == "MEDIUM" and max_pump_risk != "HIGH":
            max_pump_risk = "MEDIUM"

        # السيولة
        liq_above_total += data.get("liq_above", 0.0) * w
        liq_below_total += data.get("liq_below", 0.0) * w

        # وضع السوق (ترند / رينج)
        regime = data.get("market_regime")
        if regime == "TRENDING":
            trending_weight += w
        elif regime == "RANGING":
            ranging_weight += w

        # اختراقات
        if data.get("is_breakout_up"):
            breakout_up_weight += w
        if data.get("is_breakout_down"):
            breakout_down_weight += w

        # دايفرجنس
        if data.get("has_bull_div"):
            bull_div_weight += w
        if data.get("has_bear_div"):
            bear_div_weight += w

        # Whale & Pre Pump Weighted Aggregation
        whale_sum += data.get("whale_score", 0.0) * w
        pump_flags += (1 if data.get("pre_pump") else 0) * w
        dump_flags += (1 if data.get("pre_dump") else 0) * w

    if total_weight > 0:
        base_score = score_sum / total_weight
    else:
        base_score = 50.0

    # نسب توافق الاتجاه بين الفريمات
    bull_align = bullish_votes / total_weight if total_weight > 0 else 0.0
    bear_align = bearish_votes / total_weight if total_weight > 0 else 0.0

    # ترند عام
    if bullish_votes > bearish_votes:
        global_trend = "BULLISH"
    elif bearish_votes > bullish_votes:
        global_trend = "BEARISH"
    else:
        global_trend = "RANGING"

    # وضع السوق العام
    if trending_weight > ranging_weight * 1.1:
        global_regime = "TRENDING"
    elif ranging_weight > trending_weight * 1.1:
        global_regime = "RANGING"
    else:
        global_regime = "MIXED"

    # تجميع انحياز السيولة الكلي
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

    # RSI العام (1h + 4h)
    rsi_1h = tf_data.get("1h", {}).get("rsi")
    rsi_4h = tf_data.get("4h", {}).get("rsi")

    overbought = any(
        r is not None and not np.isnan(r) and r > 70 for r in [rsi_1h, rsi_4h]
    )
    oversold = any(
        r is not None and not np.isnan(r) and r < 30 for r in [rsi_1h, rsi_4h]
    )

    # =========================
    # تعديل السكور بالذكاء الجديد
    # =========================
    combined_score = base_score

    # مكافأة السوق الترندي
    if global_regime == "TRENDING":
        combined_score += 3

    # مكافأة/عقاب الاختراقات
    if global_trend == "BULLISH":
        if breakout_up_weight > 0.15:
            combined_score += 4
        if breakout_down_weight > 0.15:
            combined_score -= 4
    elif global_trend == "BEARISH":
        if breakout_down_weight > 0.15:
            combined_score += 4
        if breakout_up_weight > 0.15:
            combined_score -= 4

    # دايفرجنس ضد الترند
    if global_trend == "BULLISH" and bear_div_weight > 0.15:
        combined_score -= 5
    if global_trend == "BEARISH" and bull_div_weight > 0.15:
        combined_score += 5

    # السيولة
    if liquidity_bias == "UP" and global_trend == "BULLISH":
        combined_score += 3
    elif liquidity_bias == "DOWN" and global_trend == "BULLISH":
        combined_score -= 3
    elif liquidity_bias == "DOWN" and global_trend == "BEARISH":
        combined_score += 3
    elif liquidity_bias == "UP" and global_trend == "BEARISH":
        combined_score -= 3

    combined_score = max(0.0, min(100.0, combined_score))

    # =========================
    # فريم مرجعي قوي (Anchor)
    # =========================
    strong_bull_anchor = (
        tf_data.get("4h", {}).get("trend") == "BULLISH"
        and tf_data.get("4h", {}).get("trend_score", 50) >= 60
    ) or (
        tf_data.get("1d", {}).get("trend") == "BULLISH"
        and tf_data.get("1d", {}).get("trend_score", 50) >= 55
    )

    strong_bear_anchor = (
        tf_data.get("4h", {}).get("trend") == "BEARISH"
        and tf_data.get("4h", {}).get("trend_score", 50) >= 60
    ) or (
        tf_data.get("1d", {}).get("trend") == "BEARISH"
        and tf_data.get("1d", {}).get("trend_score", 50) >= 55
    )

    # =========================
    # Whale Intelligence Merge
    # =========================
    if total_weight > 0:
        whale_score_final = whale_sum / total_weight
        pre_pump_score = pump_flags / total_weight
        pre_dump_score = dump_flags / total_weight
    else:
        whale_score_final = 0.0
        pre_pump_score = 0.0
        pre_dump_score = 0.0

    # =========================
    # اتخاذ قرار BUY / SELL / WAIT
    # =========================
    action = "WAIT"

    # عشان الإشارة تكون BUY لازم:
    # - سكور عالي
    # - توافق فريمات قوي مع الاتجاه
    # - فريم مرجعي داعم
    if (
        combined_score >= 72
        and bull_align >= 0.6
        and not overbought
        and max_pump_risk != "HIGH"
        and strong_bull_anchor
    ):
        action = "BUY"

    # شروط SELL المعاكسة
    if (
        combined_score <= 28
        and bear_align >= 0.6
        and not oversold
        and strong_bear_anchor
    ):
        action = "SELL"

    # المنطقة الرمادية → نحتكم للسيولة إذا في أفضلية واضحة
    if action == "WAIT" and 60 <= combined_score < 72 and max_pump_risk != "HIGH":
        if liquidity_bias == "UP" and bull_align >= 0.6 and strong_bull_anchor:
            action = "BUY"
        elif liquidity_bias == "DOWN" and bear_align >= 0.6 and strong_bear_anchor:
            action = "SELL"

    # الثقة
    distance = abs(combined_score - 50.0)
    if distance > 25:
        confidence = "HIGH"
    elif distance > 15:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # حماية من Pump/Dump
    if max_pump_risk == "HIGH" and action == "BUY":
        action = "WAIT"

    # =========================
    # Grade + No-Trade
    # =========================
    if (
        combined_score >= 80
        and confidence == "HIGH"
        and max_pump_risk == "LOW"
        and ((action == "BUY" and bull_align >= 0.7) or (action == "SELL" and bear_align >= 0.7))
    ):
        grade = "A+"
    elif (
        combined_score >= 70
        and max_pump_risk != "HIGH"
        and confidence in ("HIGH", "MEDIUM")
        and (bull_align >= 0.55 or bear_align >= 0.55)
    ):
        grade = "A"
    elif combined_score >= 58:
        grade = "B"
    else:
        grade = "C"

    no_trade = False

    # شروط المنطقة المحظورة No-Trade
    if grade == "C" or confidence == "LOW" or max_pump_risk == "HIGH":
        no_trade = True

    if action == "WAIT":
        no_trade = True

    # لو السيولة تقريباً متعادلة تماماً → نعتبرها No-Trade
    if liquidity_score < 5:
        no_trade = True

    return {
        "score": round(float(combined_score), 2),
        "trend": global_trend,
        "action": action,
        "confidence": confidence,
        "pump_dump_risk": max_pump_risk,
        "liquidity_bias": liquidity_bias,
        "liquidity_score": round(float(liquidity_score), 2),
        "market_regime": global_regime,
        "grade": grade,
        "no_trade": no_trade,
        # معلومات إضافية مفيدة
        "bull_align": round(float(bull_align), 2),
        "bear_align": round(float(bear_align), 2),
        "whale_score": round(float(whale_score_final), 2),
        "pre_pump_score": round(float(pre_pump_score), 2),
        "pre_dump_score": round(float(pre_dump_score), 2),
    }


# =========================
# اختيار نسب المخاطرة / الربح (اختياري)
# =========================

def choose_risk_reward(decision: Dict[str, Any],
                       tf_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    يحدد نسب المخاطرة والربح تلقائياً (غير مستخدمة حالياً في generate_signal
    لكن نقدر نستفيد منها لاحقاً).
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

    risk_pct = 0.015
    reward_pct = 0.03

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


# =========================
# حساب TP/SL + R:R باستخدام ATR + Fallback
# =========================

def compute_trade_levels(
    symbol_norm: str,
    price: float,
    action: str,
    risk_pct: float,
    reward_pct: float,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    يحسب SL و TP و R:R باستخدام:
    - ATR من فريم 1h، ولو ما يكفي → نجرب 15m
    - لو ATR ما اشتغل نهائياً → نرجع لطريقة النِسَب المئوية فقط
    """
    # 1) نحاول نجيب ATR من 1h ثم من 15m
    atr_value: Optional[float] = None
    for tf_key in ["1h", "15m"]:
        interval = TIMEFRAMES.get(tf_key)
        if not interval:
            continue

        try:
            ohlcv = fetch_klines(symbol_norm, interval, limit=200)
            atr_vals = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
            atr_value = float(atr_vals[-1])
            break
        except Exception:
            continue

    # 2) نحول النِسَب إلى مسافات سعرية
    sl_dist_pct = price * (risk_pct / 100.0)
    tp_dist_pct = price * (reward_pct / 100.0)

    if atr_value is not None:
        # نستخدم الأكبر بين ATR والنسبة المئوية
        sl_dist = max(sl_dist_pct, 1.2 * atr_value)
        tp_dist = max(tp_dist_pct, 1.5 * atr_value)
    else:
        # Fallback: ما عندنا ATR → نشتغل بالـ % فقط
        sl_dist = sl_dist_pct
        tp_dist = tp_dist_pct

    if sl_dist <= 0 or tp_dist <= 0 or action not in ("BUY", "SELL"):
        return None, None, None

    if action == "BUY":
        sl = round(price - sl_dist, 4)
        tp = round(price + tp_dist, 4)
    else:  # SELL
        sl = round(price + sl_dist, 4)
        tp = round(price - tp_dist, 4)

    rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else None
    return sl, tp, rr


# =========================
# نقطة الدخول الرئيسية
# =========================

def generate_signal(symbol: str) -> Dict[str, Any]:
    """
    Main Ultra Engine entrypoint.
    """
    symbol_norm = _normalize_symbol(symbol)
    tf_results: Dict[str, Dict[str, Any]] = {}

    # نحتفظ بآخر سعر واضح (نفضّل 1h ثم 15m)
    last_close: Optional[float] = None

    # 1) نجيب بيانات كل الفريمات
    for name, interval in TIMEFRAMES.items():
        try:
            ohlcv = fetch_klines(symbol_norm, interval)
            tf_info = analyse_timeframe(ohlcv, name)
            tf_results[name] = tf_info

            # نخزن آخر سعر للفريمات المهمة
            if name == "1h":
                last_close = tf_info.get("close", last_close)
            elif name == "15m" and last_close is None:
                last_close = tf_info.get("close", last_close)

            time.sleep(0.1)
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

    # منطق خاص بالإشارات قبل الانفجار + الحيتان
    pre_pump_flag = combined.get("pre_pump_score", 0.0) >= 0.25
    whale_alert = combined.get("whale_score", 0.0) >= 40.0

    # لو عندنا Pre-Pump وقرار BUY ومش No-Trade → نرفع Grade إلى A+
    if (
        pre_pump_flag
        and combined.get("action") == "BUY"
        and combined.get("no_trade") is False
        and combined.get("grade") in ("A", "B")
    ):
        combined["grade"] = "A+"

    tp: Optional[float] = None
    sl: Optional[float] = None
    rr: Optional[float] = None
    risk_pct: Optional[float] = None
    reward_pct: Optional[float] = None

    if last_close is not None:
        price = float(last_close)

        # نسب المخاطرة حسب الثقة
        if combined["confidence"] == "HIGH":
            risk_pct = 2.0
        elif combined["confidence"] == "MEDIUM":
            risk_pct = 1.5
        else:
            risk_pct = 1.0

        # مضاعف هدف الربح حسب السكور
        if combined["score"] >= 75:
            reward_mult = 2.5
        elif combined["score"] >= 65:
            reward_mult = 2.0
        else:
            reward_mult = 1.5

        reward_pct = risk_pct * reward_mult
        action = combined["action"]

        if action in ("BUY", "SELL"):
            sl, tp, rr = compute_trade_levels(
                symbol_norm=symbol_norm,
                price=price,
                action=action,
                risk_pct=risk_pct,
                reward_pct=reward_pct,
            )

    # 4) نص توضيحي ذكي مختصر
    reason_lines: List[str] = []

    grade = combined.get("grade")
    no_trade = combined.get("no_trade", False)
    market_regime = combined.get("market_regime", "UNKNOWN")

    if grade:
        reason_lines.append(f"تصنيف الإشارة (Grade): {grade}")
    reason_lines.append(f"وضع السوق العام: {market_regime}")
    if no_trade:
        reason_lines.append("⚠️ هذه المنطقة مصنّفة حالياً كـ No-Trade Zone حسب فلتر B7A Ultra.")

    if whale_alert:
        reason_lines.append("🐋 تنبيه: حركة سيولة قوية (Whale Activity Detected).")

    if pre_pump_flag and not no_trade:
        reason_lines.append("🚀 إشارة ما قبل الانفجار (Pre-Pump Pattern) مرصودة على الفريمات.")

    reason_lines.append(f"الاتجاه العام: {combined['trend']}")
    reason_lines.append(
        "أقوى الفريمات: "
        + ", ".join(
            tf for tf, d in tf_results.items()
            if d.get("trend_score", 50) >= combined["score"]
        )
    )

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

    result: Dict[str, Any] = {
        "symbol": symbol_norm,
        "last_price": last_close,
        "timeframes": tf_results,
        "decision": combined,
        "reason": explanation,
        "tp": tp,
        "sl": sl,
        "rr": rr,
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
        "pre_pump": pre_pump_flag,
        "whale_alert": whale_alert,
    }

    # تسجيل الصفقات الفعلية فقط
    if (
        combined.get("action") in ("BUY", "SELL")
        and combined.get("no_trade") is False
        and last_close is not None
    ):
        try:
            log_trade(result)
        except Exception as e:
            print("log_trade error:", e)

    return result
