import time
from typing import Dict, Any, List, Tuple

import numpy as np
import requests


BINANCE_BASE_URL = "https://api.binance.com"
# Timeframes we will use in the Ultra Engine
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
    Fetch OHLCV candles from Binance and return numpy arrays.
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
# Indicator calculations
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
            window = series[i - period + 1 : i + 1]
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
    avg_prev = volume[-(lookback + 1) : -1].mean()
    return recent > threshold * avg_prev


def price_change(series: np.ndarray, period: int = 1) -> float:
    if series.size < period + 1:
        return 0.0
    return (series[-1] - series[-period - 1]) / series[-period - 1] * 100.0


# =========================
# Scoring & decision engine
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

    # Trend detection
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
            bullish_points += 1  # oversold bounce potential

    if not np.isnan(macd_last) and not np.isnan(macd_signal_last):
        if macd_last > macd_signal_last:
            bullish_points += 1
        else:
            bearish_points += 1

    if not np.isnan(lower_last) and not np.isnan(upper_last):
        if last_close <= lower_last:
            bullish_points += 1  # near lower band -> possible bounce
        elif last_close >= upper_last:
            bearish_points += 1  # extended move

    trend_score = (bullish_points - bearish_points) * 10 + 50
    trend_score = max(0, min(100, trend_score))

    # Pump & dump style risk (very simple heuristic)
    pump_dump_risk = "LOW"
    if abs(change_1) > 3 and vol_surge:
        pump_dump_risk = "MEDIUM"
    if abs(change_1) > 6 and vol_surge:
        pump_dump_risk = "HIGH"

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
        }
    )

    if bullish_points > bearish_points:
        info["trend"] = "BULLISH"
    elif bearish_points > bullish_points:
        info["trend"] = "BEARISH"
    else:
        info["trend"] = "RANGING"

    return info


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

    for tf, data in tf_data.items():
        w = weights.get(tf, 0.0)
        score += data["trend_score"] * w
        total_weight += w

        if data["trend"] == "BULLISH":
            bullish_votes += w
        elif data["trend"] == "BEARISH":
            bearish_votes += w

        # Pump/dump aggregation
        risk = data.get("pump_dump_risk", "LOW")
        if risk == "HIGH":
            max_pump_risk = "HIGH"
        elif risk == "MEDIUM" and max_pump_risk != "HIGH":
            max_pump_risk = "MEDIUM"

    if total_weight > 0:
        score /= total_weight

    if bullish_votes > bearish_votes:
        global_trend = "BULLISH"
    elif bearish_votes > bullish_votes:
        global_trend = "BEARISH"
    else:
        global_trend = "RANGING"

    if score >= 70 and max_pump_risk != "HIGH":
        action = "BUY"
    elif score <= 30:
        action = "SELL"
    else:
        action = "WAIT"

    distance = abs(score - 50)
    if distance > 25:
        confidence = "HIGH"
    elif distance > 15:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    if max_pump_risk == "HIGH" and action == "BUY":
        # Override: strong warning against buying into a potential pump
        action = "WAIT"

    return {
        "score": round(float(score), 2),
        "trend": global_trend,
        "action": action,
        "confidence": confidence,
        "pump_dump_risk": max_pump_risk,
    }


def generate_signal(symbol: str) -> Dict[str, Any]:
    """
    Main Ultra Engine entrypoint.

    Returns a dict ready to be formatted by the Telegram layer.
    """
    symbol_norm = _normalize_symbol(symbol)
    tf_results: Dict[str, Dict[str, Any]] = {}

    for name, interval in TIMEFRAMES.items():
        try:
            ohlcv = fetch_klines(symbol_norm, interval)
            tf_info = analyse_timeframe(ohlcv, name)
            tf_results[name] = tf_info
            time.sleep(0.1)  # be gentle with Binance
        except Exception as e:
            tf_results[name] = {
                "timeframe": name,
                "error": str(e),
                "trend": "UNKNOWN",
                "trend_score": 50,
                "pump_dump_risk": "LOW",
            }

    combined = combine_timeframes(tf_results)

    last_close = tf_results.get("1h", tf_results.get("15m", {})).get("close")

    reason_lines: List[str] = []
    reason_lines.append(f"الاتجاه العام: {combined['trend']}")
    reason_lines.append(f"أقوى الفريمات: " + ", ".join(
        tf for tf, d in tf_results.items() if d.get("trend_score", 50) >= combined["score"]
    ))
    if combined["pump_dump_risk"] != "LOW":
        reason_lines.append(f"تنبيه: احتمالية Pump/Dump = {combined['pump_dump_risk']} (حذر مع الدخول).")

    explanation = " | ".join(reason_lines)

    return {
        "symbol": symbol_norm,
        "last_price": float(last_close) if last_close is not None else None,
        "timeframes": tf_results,
        "decision": combined,
        "reason": explanation,
    }
