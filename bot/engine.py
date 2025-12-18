import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import requests
import csv
import os
from datetime import datetime

from .coinglass_client import get_coinglass_intel
from .analytics import performance_intel
from .intel_hub import get_global_intel
from .onchain_intel import get_onchain_intel

# ✅ Flow Engine خارجي فقط
from bot.flow_engine import compute_flow_engine


# =========================
# إعدادات عامة
# =========================

BINANCE_SPOT_BASE_URL = "https://api.binance.com"
BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"

TIMEFRAMES = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


class MarketDataError(Exception):
    """Raised when we cannot fetch data from Binance."""
    pass


def _normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"
    return symbol


# =========================
# Arkham Smart Money Intel (Placeholder)
# =========================

def get_arkham_intel(symbol: str) -> Dict[str, Any]:
    """
    Placeholder لـ Arkham Intelligence.
    حالياً ترجع قيم محايدة، لاحقاً نستبدلها بنداءات API الحقيقية.
    """
    return {
        "whale_inflow_score": 0.0,
        "whale_outflow_score": 0.0,
        "smart_money_bias": "NEUTRAL",  # UP / DOWN / NEUTRAL
        "cex_inflow_score": 0.0,
        "cex_outflow_score": 0.0,
        "intel_confidence": "LOW",      # LOW / MEDIUM / HIGH
    }


# =========================
# Binance Sentiment (بديل مجاني لـ Coinglass)
# =========================

def fetch_binance_sentiment(symbol: str) -> Dict[str, Any]:
    """
    يقرأ Top Long/Short Accounts Ratio من Binance Futures.
    يستخدم كـ Smart Sentiment خفيف:
      - bias: LONG / SHORT / NEUTRAL
      - strength: فرق القوة بين الطرفين (0 - 100 تقريباً)
    """
    symbol_norm = _normalize_symbol(symbol)
    url = f"{BINANCE_FUTURES_BASE_URL}/futures/data/topLongShortAccountRatio"
    params = {
        "symbol": symbol_norm,
        "period": "5m",
        "limit": 50,
    }

    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code != 200:
            return {
                "available": False,
                "long_pct": None,
                "short_pct": None,
                "bias": "NEUTRAL",
                "strength": 0.0,
            }

        data = resp.json()
        if not isinstance(data, list) or not data:
            return {
                "available": False,
                "long_pct": None,
                "short_pct": None,
                "bias": "NEUTRAL",
                "strength": 0.0,
            }

        last = data[-1]
        long_ratio = float(last.get("longAccount", 0.0))
        short_ratio = float(last.get("shortAccount", 0.0))

        if long_ratio + short_ratio > 0:
            total = long_ratio + short_ratio
            long_pct = long_ratio / total * 100.0
            short_pct = short_ratio / total * 100.0
        else:
            long_pct = 0.0
            short_pct = 0.0

        bias = "NEUTRAL"
        strength = 0.0

        if long_pct > short_pct * 1.1:
            bias = "LONG"
            strength = long_pct - short_pct
        elif short_pct > long_pct * 1.1:
            bias = "SHORT"
            strength = short_pct - long_pct

        return {
            "available": True,
            "long_pct": long_pct,
            "short_pct": short_pct,
            "bias": bias,
            "strength": strength,
            "raw": last,
        }
    except Exception:
        return {
            "available": False,
            "long_pct": None,
            "short_pct": None,
            "bias": "NEUTRAL",
            "strength": 0.0,
        }


# =========================
# جلب البيانات من Binance
# =========================

def fetch_klines(symbol: str, interval: str, limit: int = 200) -> Dict[str, np.ndarray]:
    """
    يجلب بيانات الشموع من بايننس (Spot).
    ملاحظة: نفترض أن الـ symbol جاي جاهز (USDT مضاف لو تحتاجه).
    """
    url = f"{BINANCE_SPOT_BASE_URL}/api/v3/klines"
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
# Orderbook Pressure Engine
# =========================

def fetch_orderbook(symbol: str, limit: int = 100) -> Dict[str, Any]:
    """
    يجلب دفتر الأوامر (Orderbook) من Binance.
    نستخدمه لقياس ضغط الشراء/البيع (BID/ASK Pressure).
    """
    symbol = _normalize_symbol(symbol)
    url = f"{BINANCE_SPOT_BASE_URL}/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}

    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        raise MarketDataError(f"Binance orderbook error {resp.status_code}: {resp.text}")

    data = resp.json()
    bids_raw = data.get("bids", [])
    asks_raw = data.get("asks", [])

    bids = [(float(p), float(q)) for p, q in bids_raw]
    asks = [(float(p), float(q)) for p, q in asks_raw]

    return {"bids": bids, "asks": asks}


def analyse_orderbook(symbol_norm: str, limit: int = 100) -> Dict[str, Any]:
    """
    يحلل دفتر الأوامر ويحسب:
    - bias: BID / ASK / FLAT
    - score: قوة الانحياز (0 - 100)
    - bid/ask walls: مستويات سيولة قوية قريبة من السعر
    """
    ob = fetch_orderbook(symbol_norm, limit=limit)
    bids = ob["bids"]
    asks = ob["asks"]

    if not bids or not asks:
        return {
            "bias": "FLAT",
            "score": 0.0,
            "total_bid": 0.0,
            "total_ask": 0.0,
            "bid_walls": [],
            "ask_walls": [],
        }

    total_bid = sum(q for _, q in bids)
    total_ask = sum(q for _, q in asks)

    if total_bid + total_ask <= 0:
        return {
            "bias": "FLAT",
            "score": 0.0,
            "total_bid": float(total_bid),
            "total_ask": float(total_ask),
            "bid_walls": [],
            "ask_walls": [],
        }

    imbalance = (total_bid - total_ask) / (total_bid + total_ask)

    if imbalance > 0.15:
        bias = "BID"
    elif imbalance < -0.15:
        bias = "ASK"
    else:
        bias = "FLAT"

    score = max(0.0, min(100.0, abs(imbalance) * 100.0))

    bid_walls = []
    ask_walls = []
    if total_bid > 0:
        bid_threshold = 0.03 * total_bid
        for price, qty in bids:
            if qty >= bid_threshold:
                bid_walls.append({"price": price, "qty": qty})

    if total_ask > 0:
        ask_threshold = 0.03 * total_ask
        for price, qty in asks:
            if qty >= ask_threshold:
                ask_walls.append({"price": price, "qty": qty})

    bid_walls = sorted(bid_walls, key=lambda x: x["qty"], reverse=True)[:5]
    ask_walls = sorted(ask_walls, key=lambda x: x["qty"], reverse=True)[:5]

    return {
        "bias": bias,
        "score": float(score),
        "total_bid": float(total_bid),
        "total_ask": float(total_ask),
        "bid_walls": bid_walls,
        "ask_walls": ask_walls,
    }


# =========================
# Trade Logger
# =========================

def log_trade(data: Dict[str, Any]):
    """
    يسجل الصفقات الفعلية في ملف CSV اسمه trades_log.csv
    ويمكن لاحقاً نضيف نتيجة الصفقة (WIN/LOSS) يدويًا في ملف CSV.
    """
    log_file = "trades_log.csv"
    file_exists = os.path.isfile(log_file)

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "datetime", "symbol", "action", "price",
                    "tp", "sl", "rr",
                    "grade", "score", "confidence",
                    "pump_risk", "market_regime", "liquidity_bias",
                    "no_trade",
                    "result",
                ]
            )

        decision = data.get("decision", {})

        writer.writerow(
            [
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
                data.get("result", ""),
            ]
        )


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


def macd(
    series: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    if series.size < slow + signal_period:
        raise ValueError("Not enough data for MACD")
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    return macd_line, signal_line


def bollinger_bands(
    series: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
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


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
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

def _detect_swings(
    high: np.ndarray,
    low: np.ndarray,
    left: int = 2,
    right: int = 2,
) -> Tuple[List[int], List[int]]:
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


def _cluster_levels(prices: List[float], tolerance: float = 0.0015) -> List[Dict[str, Any]]:
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

    try:
        ema200 = ema(close, 200)[-1]
    except ValueError:
        ema200 = float("nan")

    rsi_arr = None
    try:
        rsi_arr = rsi(close, 14)
        rsi_last = float(rsi_arr[-1])
    except ValueError:
        rsi_last = float("nan")
        rsi_arr = None

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
            bearish_points += 1
        elif rsi_last < 30:
            bullish_points += 1

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

    # =========================
    # Market Regime Detector
    # =========================
    try:
        atr_vals = atr(high, low, close, period=14)
        atr_last = float(atr_vals[-1])
    except Exception:
        atr_last = float("nan")

    distance_from_ema200 = abs(last_close - ema200) if not np.isnan(ema200) else 0.0

    if (
        not np.isnan(ema200)
        and not np.isnan(atr_last)
        and distance_from_ema200 > atr_last * 1.2
        and atr_last > (0.002 * last_close)
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
        prev_idx = -10
        prev_low = close[prev_idx]
        curr_low = close[-1]
        prev_rsi = rsi_arr[prev_idx]
        curr_rsi = rsi_arr[-1]

        if not np.isnan(prev_rsi) and not np.isnan(curr_rsi):
            if curr_low < prev_low and curr_rsi > prev_rsi:
                has_bull_div = True
            if curr_low > prev_low and curr_rsi < prev_rsi:
                has_bear_div = True

    if has_bull_div:
        bullish_points += 1
    if has_bear_div:
        bearish_points += 1

    trend_score = (bullish_points - bearish_points) * 10 + 50

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
            "score": trend_score,
            "pump_dump_risk": pump_dump_risk,
            "liquidity": liq_map,
            "liq_bias": liq_bias,
            "liq_score": liq_score,
            "liq_above": liq_above,
            "liq_below": liq_below,
            "market_regime": market_regime,
            "regime": market_regime,
            "is_breakout_up": is_breakout_up,
            "is_breakout_down": is_breakout_down,
            "has_bull_div": has_bull_div,
            "has_bear_div": has_bear_div,
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

def combine_timeframes(
    tf_data: Dict[str, Dict[str, Any]],
    arkham_intel: Optional[Dict[str, Any]] = None,
    orderbook_intel: Optional[Dict[str, Any]] = None,
    binance_sentiment: Optional[Dict[str, Any]] = None,
    mode: str = "balanced",
) -> Dict[str, Any]:
    """
    دمج الفريمات في قرار واحد.
    يدعم 3 أوضاع:
      - safe
      - balanced
      - momentum
    """
    mode = (mode or "balanced").lower()
    if mode not in ("safe", "balanced", "momentum"):
        mode = "balanced"

    weights = {"15m": 0.2, "1h": 0.3, "4h": 0.3, "1d": 0.2}

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

        risk = data.get("pump_dump_risk", "LOW")
        if risk == "HIGH":
            max_pump_risk = "HIGH"
        elif risk == "MEDIUM" and max_pump_risk != "HIGH":
            max_pump_risk = "MEDIUM"

        liq_above_total += data.get("liq_above", 0.0) * w
        liq_below_total += data.get("liq_below", 0.0) * w

        regime = data.get("market_regime")
        if regime == "TRENDING":
            trending_weight += w
        elif regime == "RANGING":
            ranging_weight += w

        if data.get("is_breakout_up"):
            breakout_up_weight += w
        if data.get("is_breakout_down"):
            breakout_down_weight += w

        if data.get("has_bull_div"):
            bull_div_weight += w
        if data.get("has_bear_div"):
            bear_div_weight += w

    base_score = (score_sum / total_weight) if total_weight > 0 else 50.0

    bull_align = bullish_votes / total_weight if total_weight > 0 else 0.0
    bear_align = bearish_votes / total_weight if total_weight > 0 else 0.0

    if bullish_votes > bearish_votes:
        global_trend = "BULLISH"
    elif bearish_votes > bullish_votes:
        global_trend = "BEARISH"
    else:
        global_trend = "RANGING"

    if trending_weight > ranging_weight * 1.1:
        global_regime = "TRENDING"
    elif ranging_weight > trending_weight * 1.1:
        global_regime = "RANGING"
    else:
        global_regime = "MIXED"

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
        liquidity_bias = "FLAT"
        liquidity_score = 0.0

    rsi_1h = tf_data.get("1h", {}).get("rsi")
    rsi_4h = tf_data.get("4h", {}).get("rsi")
    rsi_1d = tf_data.get("1d", {}).get("rsi")

    def _is_overbought(x):
        return x is not None and not np.isnan(x) and x > 70.0

    def _is_oversold(x):
        return x is not None and not np.isnan(x) and x < 30.0

    overbought = any(_is_overbought(r) for r in [rsi_1h, rsi_4h, rsi_1d])
    oversold = any(_is_oversold(r) for r in [rsi_1h, rsi_4h, rsi_1d])

    combined_score = base_score

    if global_regime == "TRENDING":
        combined_score += 3

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

    if global_trend == "BULLISH" and bear_div_weight > 0.15:
        combined_score -= 5
    if global_trend == "BEARISH" and bull_div_weight > 0.15:
        combined_score += 5

    if liquidity_bias == "UP" and global_trend == "BULLISH":
        combined_score += 3
    elif liquidity_bias == "DOWN" and global_trend == "BULLISH":
        combined_score -= 3
    elif liquidity_bias == "DOWN" and global_trend == "BEARISH":
        combined_score += 3
    elif liquidity_bias == "UP" and global_trend == "BEARISH":
        combined_score -= 3

    orderbook_bias = "FLAT"
    orderbook_score = 0.0
    sentiment_bias = "NEUTRAL"
    sentiment_strength = 0.0

    if arkham_intel:
        try:
            whale_in = float(arkham_intel.get("whale_inflow_score", 0.0) or 0.0)
            whale_out = float(arkham_intel.get("whale_outflow_score", 0.0) or 0.0)
            intel_bias = arkham_intel.get("smart_money_bias", "NEUTRAL")
            intel_conf = arkham_intel.get("intel_confidence", "LOW")
            intel_weight_map = {"LOW": 0.3, "MEDIUM": 0.7, "HIGH": 1.0}
            intel_weight = intel_weight_map.get(intel_conf, 0.3)

            raw_delta = (whale_in - whale_out) / 100.0
            delta = raw_delta * 3.0 * intel_weight

            if intel_bias == "UP" and global_trend == "BEARISH":
                delta *= 0.5
            if intel_bias == "DOWN" and global_trend == "BULLISH":
                delta *= 0.5

            combined_score += delta
        except Exception:
            pass

    if orderbook_intel:
        try:
            orderbook_bias = orderbook_intel.get("bias", "FLAT")
            orderbook_score = float(orderbook_intel.get("score", 0.0) or 0.0)
            ob_intensity = min(orderbook_score / 100.0, 1.0)
            delta = 0.0
            if orderbook_bias == "BID":
                if global_trend == "BULLISH":
                    delta += 4.0 * ob_intensity
                elif global_trend == "BEARISH":
                    delta += 2.0 * ob_intensity
            elif orderbook_bias == "ASK":
                if global_trend == "BEARISH":
                    delta -= 4.0 * ob_intensity
                elif global_trend == "BULLISH":
                    delta -= 2.0 * ob_intensity
            combined_score += delta
        except Exception:
            pass

    if binance_sentiment:
        try:
            sentiment_bias = binance_sentiment.get("bias", "NEUTRAL")
            sentiment_strength = float(binance_sentiment.get("strength", 0.0) or 0.0)
            if sentiment_bias != "NEUTRAL" and sentiment_strength > 5:
                s_intensity = min(sentiment_strength / 50.0, 1.0)
                delta = 0.0
                if sentiment_bias == "LONG":
                    if global_trend == "BULLISH":
                        delta += 3.0 * s_intensity
                    elif global_trend == "BEARISH":
                        delta += 1.5 * s_intensity
                elif sentiment_bias == "SHORT":
                    if global_trend == "BEARISH":
                        delta -= 3.0 * s_intensity
                    elif global_trend == "BULLISH":
                        delta -= 1.5 * s_intensity
                combined_score += delta
        except Exception:
            pass

    combined_score = max(0.0, min(100.0, combined_score))

    def _extended_side(tf_name: str) -> str:
        data = tf_data.get(tf_name, {})
        c = data.get("close")
        ema200_v = data.get("ema200")
        if c is None or ema200_v is None or np.isnan(ema200_v) or ema200_v == 0:
            return "NONE"
        dist_pct = abs(c - ema200_v) / abs(ema200_v) * 100.0
        if dist_pct < 8.0:
            return "NONE"
        return "UP" if c > ema200_v else "DOWN"

    ext_4h = _extended_side("4h")
    ext_1d = _extended_side("1d")
    extended_up = (ext_4h == "UP") or (ext_1d == "UP")
    extended_down = (ext_4h == "DOWN") or (ext_1d == "DOWN")

    strong_bull_anchor = (
        (tf_data.get("4h", {}).get("trend") == "BULLISH" and tf_data.get("4h", {}).get("trend_score", 50) >= 60)
        or (tf_data.get("1d", {}).get("trend") == "BULLISH" and tf_data.get("1d", {}).get("trend_score", 50) >= 55)
    )

    strong_bear_anchor = (
        (tf_data.get("4h", {}).get("trend") == "BEARISH" and tf_data.get("4h", {}).get("trend_score", 50) >= 60)
        or (tf_data.get("1d", {}).get("trend") == "BEARISH" and tf_data.get("1d", {}).get("trend_score", 50) >= 55)
    )

    safety_block_buy = False
    safety_block_sell = False
    if mode == "safe":
        if global_trend == "BULLISH" and extended_up and overbought:
            safety_block_buy = True
        if global_trend == "BEARISH" and extended_down and oversold:
            safety_block_sell = True

    long_score = combined_score
    short_score = 100.0 - combined_score

    long_score += bull_align * 20.0
    long_score -= bear_align * 10.0
    short_score += bear_align * 20.0
    short_score -= bull_align * 10.0

    if global_trend == "BULLISH":
        long_score += 5.0
        short_score -= 3.0
    elif global_trend == "BEARISH":
        short_score += 5.0
        long_score -= 3.0

    if liquidity_bias == "UP":
        long_score += 3.0
    elif liquidity_bias == "DOWN":
        short_score += 3.0

    if extended_up and overbought:
        long_score -= 8.0
    if extended_down and oversold:
        short_score -= 8.0

    long_score = max(0.0, min(100.0, long_score))
    short_score = max(0.0, min(100.0, short_score))

    if mode == "safe":
        long_min = 70.0
        short_min = 80.0
        gray_low = 52.0
        gray_high = 70.0
    elif mode == "momentum":
        long_min = 58.0
        short_min = 58.0
        gray_low = 48.0
        gray_high = 70.0
    else:
        long_min = 65.0
        short_min = 60.0
        gray_low = 50.0
        gray_high = 70.0

    sell_oversold_block = oversold and global_trend != "BULLISH"
    buy_overbought_block = overbought and global_trend != "BULLISH"

    action = "WAIT"

    if (
        long_score >= long_min
        and long_score >= short_score
        and bull_align >= 0.40
        and not buy_overbought_block
        and max_pump_risk != "HIGH"
        and (strong_bull_anchor or (global_regime in ("TRENDING", "RANGING") and liquidity_bias in ("UP", "FLAT")))
    ):
        action = "BUY"

    if (
        short_score >= short_min
        and short_score > long_score
        and bear_align >= 0.30
        and not sell_oversold_block
        and (strong_bear_anchor or (global_trend == "BEARISH" and liquidity_bias in ("DOWN", "FLAT")))
    ):
        action = "SELL"

    if action == "WAIT" and gray_low <= combined_score < gray_high and max_pump_risk != "HIGH":
        if (
            liquidity_bias == "UP"
            and bull_align >= 0.45
            and not buy_overbought_block
            and (strong_bull_anchor or breakout_up_weight > 0.20)
        ):
            action = "BUY"
        elif (
            liquidity_bias == "DOWN"
            and bear_align >= 0.40
            and not sell_oversold_block
            and (strong_bear_anchor or breakout_down_weight > 0.20)
        ):
            action = "SELL"

    # ✅ Pump Sniper للمومنتوم (مرة واحدة فقط)
    pump_momentum = False
    if mode == "momentum" and max_pump_risk != "HIGH":
        tf15 = tf_data.get("15m", {})
        if (
            tf15.get("market_regime") == "TRENDING"
            and tf15.get("is_breakout_up")
            and tf15.get("volume_surge")
            and bull_align >= 0.35
        ):
            action = "BUY"
            pump_momentum = True

    distance = abs(combined_score - 50.0)
    if distance >= 22:
        confidence = "HIGH"
    elif distance >= 12:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    if max_pump_risk == "HIGH" and action in ("BUY", "SELL"):
        action = "WAIT"

    if safety_block_buy and action == "BUY":
        action = "WAIT"
    if safety_block_sell and action == "SELL":
        action = "WAIT"

    if (
        combined_score >= 78
        and confidence == "HIGH"
        and max_pump_risk == "LOW"
        and ((action == "BUY" and bull_align >= 0.65) or (action == "SELL" and bear_align >= 0.65))
    ):
        grade = "A+"
    elif (
        combined_score >= 68
        and max_pump_risk != "HIGH"
        and confidence in ("HIGH", "MEDIUM")
        and (bull_align >= 0.50 or bear_align >= 0.50)
    ):
        grade = "A"
    elif combined_score >= 55:
        grade = "B"
    else:
        grade = "C"

    no_trade = False
    if max_pump_risk == "HIGH":
        no_trade = True
    elif action == "WAIT":
        no_trade = True
    elif mode == "safe":
        if grade in ("C",) or confidence == "LOW" or liquidity_score < 8:
            no_trade = True
    elif mode == "balanced":
        if grade == "C" or confidence == "LOW" or liquidity_score < 5:
            no_trade = True
    else:
        if confidence == "LOW" or liquidity_score < 3:
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
        "mode": mode,
        "bull_align": round(float(bull_align), 2),
        "bear_align": round(float(bear_align), 2),
        "safety_block_buy": bool(safety_block_buy),
        "safety_block_sell": bool(safety_block_sell),
        "orderbook_bias": orderbook_bias,
        "orderbook_score": round(float(orderbook_score), 2),
        "binance_sentiment_bias": sentiment_bias,
        "binance_sentiment_strength": round(float(sentiment_strength), 2),
        "pump_momentum": pump_momentum,
        "bear_score": round(float(short_score), 2),
        "long_score": round(float(long_score), 2),
        "short_score": round(float(short_score), 2),
    }


# =========================
# Dynamic ATR Multi-TP
# =========================

def compute_trade_levels_multi(
    decision: Dict[str, Any],
    symbol_norm: str,
    price: float,
    risk_pct: float,
    mode: str = "balanced",
) -> Dict[str, Optional[float]]:
    mode = (mode or "balanced").lower()
    if mode not in ("safe", "balanced", "momentum"):
        mode = "balanced"

    atr_value: Optional[float] = None
    try:
        ohlcv = fetch_klines(symbol_norm, TIMEFRAMES["1h"], limit=200)
        atr_vals = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
        atr_value = float(atr_vals[-1])
    except Exception:
        try:
            ohlcv = fetch_klines(symbol_norm, TIMEFRAMES["15m"], limit=200)
            atr_vals = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
            atr_value = float(atr_vals[-1])
        except Exception:
            atr_value = None

    if atr_value is None or atr_value <= 0:
        return {"sl": None, "tp1": None, "tp2": None, "tp3": None, "rr1": None, "rr2": None, "rr3": None}

    action = decision.get("action")
    grade = decision.get("grade", "C")
    score = float(decision.get("score", 50.0) or 50.0)

    if grade == "A+":
        tp_mults = (1.2, 2.0, 3.0)
        base_sl_mult = 1.3
    elif grade == "A":
        tp_mults = (1.0, 1.8, 2.5)
        base_sl_mult = 1.25
    else:
        tp_mults = (0.8, 1.5, 2.0)
        base_sl_mult = 1.1

    if score < 70:
        tp_mults = tuple(m * 0.9 for m in tp_mults)

    if mode == "safe":
        sl_mode_factor = 1.3
        tp_mode_factor = 0.85
    elif mode == "momentum":
        sl_mode_factor = 0.8
        tp_mode_factor = 1.2
    else:
        sl_mode_factor = 1.0
        tp_mode_factor = 1.0

    sl_dist_atr = atr_value * base_sl_mult * sl_mode_factor
    sl_dist_pct = price * (risk_pct / 100.0)
    sl_dist = max(sl_dist_atr * 0.7 + sl_dist_pct * 0.3, atr_value * 0.5)

    tp_dists = [atr_value * m * tp_mode_factor for m in tp_mults]

    if action not in ("BUY", "SELL"):
        return {"sl": None, "tp1": None, "tp2": None, "tp3": None, "rr1": None, "rr2": None, "rr3": None}

    if action == "BUY":
        sl = round(price - sl_dist, 6)
        tp1 = round(price + tp_dists[0], 6)
        tp2 = round(price + tp_dists[1], 6)
        tp3 = round(price + tp_dists[2], 6)
    else:
        sl = round(price + sl_dist, 6)
        tp1 = round(price - tp_dists[0], 6)
        tp2 = round(price - tp_dists[1], 6)
        tp3 = round(price - tp_dists[2], 6)

    rr1 = round(tp_dists[0] / sl_dist, 2) if sl_dist > 0 else None
    rr2 = round(tp_dists[1] / sl_dist, 2) if sl_dist > 0 else None
    rr3 = round(tp_dists[2] / sl_dist, 2) if sl_dist > 0 else None

    return {"sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "rr1": rr1, "rr2": rr2, "rr3": rr3}


# =========================
# ULTRA Hacker Filter
# =========================

def _is_ultra_hacker_signal(
    combined: Dict[str, Any],
    tf_data: Dict[str, Dict[str, Any]],
    global_intel: Dict[str, Any],
    coinglass: Optional[Dict[str, Any]] = None,
    onchain_intel: Optional[Dict[str, Any]] = None,
) -> bool:
    action = (combined.get("action") or "").upper()
    if action not in ("BUY", "SELL"):
        return False

    score = float(combined.get("score") or 0.0)
    confidence = (combined.get("confidence") or "").upper()
    pump_risk = (combined.get("pump_dump_risk") or "MEDIUM").upper()

    if score < 70:
        return False
    if confidence == "LOW":
        return False
    if pump_risk == "HIGH":
        return False

    trends = {tf: (d or {}).get("trend") for tf, d in tf_data.items()}
    regimes = {tf: (d or {}).get("regime") for tf, d in tf_data.items()}

    bullish_cnt = sum(1 for t in trends.values() if t == "BULLISH")
    bearish_cnt = sum(1 for t in trends.values() if t == "BEARISH")
    trending_cnt = sum(1 for r in regimes.values() if r == "TRENDING")

    if action == "BUY":
        if bullish_cnt < 3:
            return False
    else:
        if bearish_cnt < 3:
            return False

    if trending_cnt < 2:
        return False

    btc_trend = (global_intel or {}).get("btc_trend", "FLAT")
    shock_mode = bool((global_intel or {}).get("shock_mode"))
    global_mood = float((global_intel or {}).get("global_mood_score") or 50.0)

    if shock_mode:
        return False

    if action == "BUY" and btc_trend == "BEARISH" and global_mood < 45:
        return False

    if action == "SELL" and btc_trend == "BULLISH" and global_mood > 55:
        return False

    if onchain_intel and onchain_intel.get("available"):
        if onchain_intel.get("dump_risk") == "HIGH":
            return False

        btc_chain = (onchain_intel.get("btc") or {})
        activity_score = float(btc_chain.get("activity_score") or 50.0)

        if action == "BUY" and activity_score < 40:
            return False

    if coinglass and coinglass.get("available"):
        funding = (coinglass.get("funding") or {}).get("funding_bias", "NEUTRAL").upper()
        liq_side = (coinglass.get("liquidations") or {}).get("side", "NONE").upper()
        oi_bias = (coinglass.get("open_interest") or {}).get("oi_bias", "NEUTRAL").upper()

        if action == "BUY" and funding == "LONG_CROWDED":
            return False
        if action == "SELL" and funding == "SHORT_CROWDED":
            return False

        if action == "BUY" and liq_side == "LONG":
            return False
        if action == "SELL" and liq_side == "SHORT":
            return False

        if action == "BUY" and oi_bias == "LEVERAGE_DOWN":
            return False
        if action == "SELL" and oi_bias == "LEVERAGE_UP":
            return False

    return True


# =========================
# B7A SHIELD (تحذير فقط)
# =========================

def _apply_shield(
    combined: Dict[str, Any],
    global_intel: Dict[str, Any],
    coinglass: Optional[Dict[str, Any]] = None,
    onchain_intel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    action = (combined.get("action") or "").upper()
    grade = str(combined.get("grade") or "").upper()
    mode = (combined.get("mode") or "balanced").lower()

    reasons: List[str] = []
    suggest_no_trade = False

    gi = global_intel or {}

    shock_mode = bool(gi.get("shock_mode"))
    btc_regime = gi.get("btc_regime", "CHOP")

    try:
        btc_change_1 = float(gi.get("btc_change_1") or 0.0)
    except Exception:
        btc_change_1 = 0.0

    fg_raw = gi.get("fear_greed_index")
    try:
        fg_val = int(fg_raw) if fg_raw is not None else None
    except Exception:
        fg_val = None

    if shock_mode:
        reasons.append("BTC Shock Mode مُفعّل – حركة عنيفة / أخبار قوية.")
        suggest_no_trade = True

    if btc_regime in ("CRASH", "PANIC") or btc_change_1 <= -7.0:
        reasons.append(f"BTC في حالة هبوط حاد ({btc_change_1:.1f}%) – تجنّب صفقات جديدة.")
        suggest_no_trade = True

    if fg_val is not None:
        if action == "BUY" and fg_val >= 85:
            reasons.append(f"Extreme Greed (Fear & Greed = {fg_val}) – لونغ جديد خطير.")
            if mode != "momentum":
                suggest_no_trade = True

        if action == "SELL" and fg_val <= 10:
            reasons.append(f"Extreme Fear (Fear & Greed = {fg_val}) – شورت إضافي خطير.")
            if mode == "safe":
                suggest_no_trade = True

    if onchain_intel and onchain_intel.get("available"):
        dump_risk = onchain_intel.get("dump_risk", "MEDIUM")
        if dump_risk == "HIGH":
            reasons.append("On-Chain: Dump Risk HIGH من شبكة BTC.")
            suggest_no_trade = True

        btc_chain = (onchain_intel.get("btc") or {})
        try:
            activity_score = float(btc_chain.get("activity_score") or 50.0)
        except Exception:
            activity_score = 50.0

        if action == "BUY" and activity_score < 35.0 and mode in ("safe", "balanced"):
            reasons.append(f"On-Chain: نشاط BTC ضعيف ({activity_score:.1f}/100) – لونغ غير مفضّل.")
            suggest_no_trade = True

    if coinglass and coinglass.get("available"):
        funding = (coinglass.get("funding") or {})
        funding_bias = str(funding.get("funding_bias") or "NEUTRAL").upper()
        try:
            funding_score = float(funding.get("funding_score") or 0.0)
        except Exception:
            funding_score = 0.0

        if funding_bias in ("LONG_CROWDED", "SHORT_CROWDED") and funding_score >= 80:
            reasons.append(f"Coinglass: Funding {funding_bias} (score={funding_score:.0f}) – احتمال سحبة/سكويز.")
            if mode != "momentum":
                suggest_no_trade = True

        liq = (coinglass.get("liquidation") or coinglass.get("liquidations") or {})
        liq_bias = str(liq.get("bias") or liq.get("side") or "NONE").upper()
        try:
            liq_intensity = float(liq.get("intensity") or 0.0)
        except Exception:
            liq_intensity = 0.0

        if liq_bias in ("LONG_WASHOUT", "SHORT_WASHOUT") and liq_intensity >= 0.7:
            reasons.append(f"Coinglass: {liq_bias} قوي (intensity={liq_intensity:.2f}) – السوق تحت تصفية.")
            if mode == "safe":
                suggest_no_trade = True

        oi = (coinglass.get("open_interest") or {})
        try:
            oi_chg_24h = float(oi.get("oi_change_24h") or 0.0)
        except Exception:
            oi_chg_24h = 0.0

        if abs(oi_chg_24h) >= 35.0 and mode == "safe":
            reasons.append(f"Coinglass: تغير OI {oi_chg_24h:.1f}% خلال 24h – رافعة خطرة على الزوج.")
            suggest_no_trade = True

    try:
        liq_score = float(combined.get("liquidity_score") or 0.0)
    except Exception:
        liq_score = 0.0

    if liq_score < 3.0:
        reasons.append("سيولة ضعيفة جدًا (Liquidity Score < 3) – سبريد/عمق سيء.")
        suggest_no_trade = True

    market_regime = str(combined.get("market_regime") or "").upper()
    if market_regime in ("NOISE", "CHOP") and grade in ("B", "C") and mode == "safe":
        reasons.append("سوق متذبذب + Grade ضعيف في SAFE Mode – الانتظار أفضل.")
        suggest_no_trade = True

    combined["shield_active"] = bool(reasons)
    combined["shield_suggest_no_trade"] = bool(suggest_no_trade)
    combined["shield_reasons"] = reasons

    return combined


# =========================
# نقطة الدخول الرئيسية
# =========================

def generate_signal(
    symbol: str,
    mode: str = "balanced",
    use_coinglass: bool = True,
) -> Dict[str, Any]:
    """
    Main Ultra Engine entrypoint.

    mode:
      - "safe"
      - "balanced"
      - "momentum"
    """
    mode = (mode or "balanced").lower()
    if mode not in ("safe", "balanced", "momentum"):
        mode = "balanced"

    symbol_norm = _normalize_symbol(symbol)
    tf_results: Dict[str, Dict[str, Any]] = {}

    last_close: Optional[float] = None

    try:
        arkham_intel = get_arkham_intel(symbol_norm)
    except Exception:
        arkham_intel = None

    try:
        orderbook_intel = analyse_orderbook(symbol_norm, limit=100)
    except Exception:
        orderbook_intel = None

    try:
        binance_sentiment = fetch_binance_sentiment(symbol_norm)
    except Exception:
        binance_sentiment = None

    try:
        global_intel = get_global_intel()
    except Exception as e:
        print("Global Intel error:", e)
        global_intel = {
            "btc_trend": "FLAT",
            "btc_regime": "CHOP",
            "shock_mode": False,
            "fear_greed_index": None,
            "global_mood_score": 50.0,
        }

    try:
        etherscan_key = os.getenv("ETHERSCAN_API_KEY")
        onchain_intel = get_onchain_intel(symbol_norm, etherscan_key)
    except Exception as e:
        print("On-chain Intel error:", e)
        onchain_intel = {"available": False, "dump_risk": "MEDIUM"}

    coinglass = None
    if use_coinglass:
        try:
            coinglass = get_coinglass_intel(symbol_norm)
            print(">>> COINGLASS DEBUG:", coinglass)
        except Exception as e:
            print("Coinglass error:", e)
            coinglass = None

    for name, interval in TIMEFRAMES.items():
        try:
            ohlcv = fetch_klines(symbol_norm, interval)
            tf_info = analyse_timeframe(ohlcv, name)
            tf_results[name] = tf_info

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

    combined = combine_timeframes(
        tf_results,
        arkham_intel=arkham_intel,
        orderbook_intel=orderbook_intel,
        binance_sentiment=binance_sentiment,
        mode=mode,
    )

    combined["global_intel"] = global_intel
    combined["onchain_intel"] = onchain_intel

    # =========================
    # ✅ Flow Engine External (مرة واحدة فقط)
    # =========================
    try:
        flow_engine = compute_flow_engine(
            symbol_norm,
            combined,
            coinglass or {"available": False},
            onchain_intel or {"available": False},
        )
    except Exception as e:
        print("Flow Engine error:", e)
        flow_engine = {"available": False}

    combined["flow_engine"] = flow_engine

    # ✅✅✅ هنا تحط Gate فلتر Flow Engine
    fe = combined.get("flow_engine") or {}
    if fe.get("available"):
        if fe.get("regime") == "EXHAUSTION" and mode != "momentum":
            combined["no_trade"] = True
            combined["action"] = "WAIT"
            # (اختياري) سبب واضح داخل الريزن
            combined["flow_gate_reason"] = "Flow Engine رجّح Exhaustion → No-Trade (غير مومنتوم)"


    # =========================
    # 3.25) تأثير Coinglass (اختياري)
    # =========================
    if coinglass and coinglass.get("available"):
        try:
            def _adj(key: str, delta: float):
                if key in combined and abs(delta) > 0:
                    combined[key] = max(0.0, min(100.0, combined.get(key, 50.0) + delta))

            oi = (coinglass or {}).get("open_interest") or {}
            oi_bias = oi.get("oi_bias", "NEUTRAL")
            oi_chg = oi.get("oi_change_24h")

            delta_long = 0.0
            delta_short = 0.0

            if oi_bias == "LEVERAGE_UP" and oi_chg is not None and oi_chg > 0:
                delta_long += 2.0
                delta_short -= 2.0
            elif oi_bias == "LEVERAGE_DOWN" and oi_chg is not None and oi_chg < 0:
                delta_short += 4.0
                delta_long -= 2.0

            _adj("long_score", delta_long)
            _adj("short_score", delta_short)

            act = combined.get("action")
            if act == "BUY":
                overall_delta = delta_long
            elif act == "SELL":
                overall_delta = delta_short
            else:
                overall_delta = (delta_long + delta_short) / 2.0
            _adj("score", overall_delta)

            funding = (coinglass or {}).get("funding") or {}
            f_rate = funding.get("rate")
            f_severity = funding.get("severity")

            if f_rate is not None and f_severity:
                if f_severity == "EXTREME" and mode != "momentum":
                    combined["no_trade"] = True
                elif f_severity in ("HIGH", "MEDIUM"):
                    if f_rate > 0:
                        _adj("long_score", -3.0)
                        if combined.get("action") == "BUY":
                            _adj("score", -2.0)
                    elif f_rate < 0:
                        _adj("short_score", -3.0)
                        if combined.get("action") == "SELL":
                            _adj("score", -2.0)

            liq = (coinglass or {}).get("liquidation") or {}
            liq_bias = liq.get("bias")
            liq_intensity = float(liq.get("intensity") or 0.0)
            liq_intensity = max(0.0, min(1.0, liq_intensity))

            if liq_bias and liq_intensity > 0:
                if liq_bias == "LONG_WASHOUT":
                    _adj("short_score", -4.0 * liq_intensity)
                    if combined.get("action") == "SELL":
                        _adj("score", -3.0 * liq_intensity)
                elif liq_bias == "SHORT_WASHOUT":
                    _adj("long_score", -4.0 * liq_intensity)
                    if combined.get("action") == "BUY":
                        _adj("score", -3.0 * liq_intensity)

        except Exception as e:
            print("Coinglass impact error:", e)

    # =========================
    # 3.4) ULTRA Hacker Filter
    # =========================
    try:
        ultra_ok = _is_ultra_hacker_signal(
            combined,
            tf_results,
            global_intel,
            coinglass if use_coinglass else None,
            onchain_intel,
        )
    except Exception as e:
        ultra_ok = False
        print("ULTRA filter error:", e)

    combined["is_ultra"] = ultra_ok
    if ultra_ok:
        combined["grade"] = "ULTRA"

    # =========================
    # 3.5) Performance Intel
    # =========================
    try:
        perf = performance_intel(symbol_norm, combined)
    except Exception:
        perf = {"score_delta": 0.0, "risk_multiplier": 1.0, "force_no_trade": False, "note": None}

    combined["score"] = max(0.0, min(100.0, combined.get("score", 50.0) + perf["score_delta"]))

    if perf.get("force_no_trade"):
        combined["no_trade"] = True
        combined["action"] = "WAIT"

    combined["mode"] = mode

    # ===========================
    # 🛡 B7A SHIELD (تحذير فقط)
    # ===========================
    combined = _apply_shield(
        combined,
        global_intel,
        coinglass if use_coinglass else None,
        onchain_intel,
    )

    tp: Optional[float] = None
    sl: Optional[float] = None
    rr: Optional[float] = None

    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    rr1: Optional[float] = None
    rr2: Optional[float] = None
    rr3: Optional[float] = None

    risk_pct: Optional[float] = None
    reward_pct: Optional[float] = None

    if last_close is not None:
        price = float(last_close)

        if combined["confidence"] == "HIGH":
            risk_pct = 2.0
        elif combined["confidence"] == "MEDIUM":
            risk_pct = 1.5
        else:
            risk_pct = 1.0

        if mode == "safe":
            risk_pct *= 0.7
        elif mode == "momentum":
            risk_pct *= 1.2

        risk_pct *= perf.get("risk_multiplier", 1.0)
        risk_pct = max(0.5, min(3.0, risk_pct))

        if combined["score"] >= 75:
            reward_mult = 2.5
        elif combined["score"] >= 65:
            reward_mult = 2.0
        else:
            reward_mult = 1.5

        if mode == "safe":
            reward_mult *= 0.9
        elif mode == "momentum":
            reward_mult *= 1.1

        reward_pct = risk_pct * reward_mult

        levels = compute_trade_levels_multi(
            decision=combined,
            symbol_norm=symbol_norm,
            price=price,
            risk_pct=risk_pct,
            mode=mode,
        )

        sl = levels["sl"]
        tp1 = levels["tp1"]
        tp2 = levels["tp2"]
        tp3 = levels["tp3"]
        rr1 = levels["rr1"]
        rr2 = levels["rr2"]
        rr3 = levels["rr3"]

        tp = tp2
        rr = rr2

# =========================
# (8) Translation Layer: B7A Ultra X Bot Logic
# =========================

def _clamp(x, a=0.0, b=100.0):
    try:
        x = float(x)
    except Exception:
        x = 50.0
    return max(a, min(b, x))

action = (combined.get("action") or "WAIT").upper()
mode = (combined.get("mode") or "balanced").lower()

score = _clamp(combined.get("score", 50))
long_score = _clamp(combined.get("long_score", score))
short_score = _clamp(combined.get("short_score", 100 - score))

bull_align = float(combined.get("bull_align") or 0.0)
bear_align = float(combined.get("bear_align") or 0.0)

liq_score = float(combined.get("liquidity_score") or 0.0)
pump_risk = (combined.get("pump_dump_risk") or "LOW").upper()
confidence = (combined.get("confidence") or "LOW").upper()

# Flow
flow = combined.get("flow") or {}
flow_score = _clamp(flow.get("flow_score", 50))
flow_bias = (flow.get("bias") or "NEUTRAL").upper()
flow_state = (flow.get("state") or "CALM").upper()

# Shield
shield_suggest = bool(combined.get("shield_suggest_no_trade", False))

# --- Compute EDGE Score (قوة الإشارة الفعلية) ---
# BUY: نركز على long_score
# SELL: نركز على short_score
edge = long_score if action == "BUY" else short_score if action == "SELL" else 50.0

# Align boost
edge += 10.0 * bull_align if action == "BUY" else 10.0 * bear_align if action == "SELL" else 0.0

# Liquidity quality
if liq_score >= 12:
    edge += 3.0
elif liq_score <= 4:
    edge -= 4.0

# Flow confirmation
if action == "BUY" and flow_bias == "BUY":
    edge += 4.0
if action == "SELL" and flow_bias == "SELL":
    edge += 4.0
if flow_state == "EXHAUSTION" and mode != "momentum":
    edge -= 6.0

# Pump risk penalty
if pump_risk == "MEDIUM":
    edge -= 3.0
elif pump_risk == "HIGH":
    edge -= 8.0

# Shield penalty (تحذيري)
if shield_suggest and mode != "momentum":
    edge -= 6.0

edge = _clamp(edge)

# --- Map EDGE to Signal Tier ---
if edge >= 85 and confidence in ("HIGH", "MEDIUM") and pump_risk == "LOW":
    tier = "S"
elif edge >= 75:
    tier = "A"
elif edge >= 65:
    tier = "B"
elif edge >= 55:
    tier = "C"
else:
    tier = "D"

combined["edge_score"] = round(edge, 2)   # ✅ أهم رقم للرادار
combined["tier"] = tier                  # ✅ تصنيف تشغيل

    # =========================
    # Reason Builder
    # =========================
    reason_lines: List[str] = []
    reason_lines.append(f"وضع الإشارة الحالي (Mode): {mode.upper()}")

    grade = combined.get("grade")
    no_trade = combined.get("no_trade", False)
    market_regime = combined.get("market_regime", "UNKNOWN")

    if grade:
        reason_lines.append(f"تصنيف الإشارة (Grade): {grade}")
    reason_lines.append(f"وضع السوق العام: {market_regime}")
    if no_trade:
        reason_lines.append("⚠️ هذه المنطقة مصنّفة حالياً كـ No-Trade Zone حسب فلتر B7A Ultra.")

    reason_lines.append(f"الاتجاه العام: {combined['trend']}")

    liq_bias = combined.get("liquidity_bias")
    liq_score = combined.get("liquidity_score", 0.0)
    if liq_bias == "UP":
        reason_lines.append(f"السيولة أقوى أعلى السعر (Liquidity Score ≈ {liq_score:.0f}) → احتمال سحب للأعلى.")
    elif liq_bias == "DOWN":
        reason_lines.append(f"السيولة أقوى أسفل السعر (Liquidity Score ≈ {liq_score:.0f}) → احتمال سحب للأسفل.")

    if combined.get("pump_dump_risk") != "LOW":
        reason_lines.append(f"تنبيه: Pump/Dump Risk = {combined['pump_dump_risk']}.")

    ob_bias = combined.get("orderbook_bias", "FLAT")
    ob_score = combined.get("orderbook_score", 0.0)
    if ob_score and ob_score > 0:
        if ob_bias == "BID":
            reason_lines.append(f"Orderbook: BID Pressure (Score ≈ {ob_score:.0f}).")
        elif ob_bias == "ASK":
            reason_lines.append(f"Orderbook: ASK Pressure (Score ≈ {ob_score:.0f}).")

    bs_bias = combined.get("binance_sentiment_bias", "NEUTRAL")
    bs_strength = combined.get("binance_sentiment_strength", 0.0)
    if bs_bias != "NEUTRAL" and bs_strength:
        reason_lines.append(f"Binance Sentiment: {bs_bias} (strength≈{bs_strength:.1f}).")

    # ✅ Flow Engine Summary (خارجي)
    fe = combined.get("flow_engine") or {}
    if fe and fe.get("available"):
        reason_lines.append(
            f"Flow Engine: Bias={fe.get('bias')} | Score={fe.get('flow_score')} | Regime={fe.get('regime')}"
        )

    if perf.get("note"):
        reason_lines.append(perf["note"])

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
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rr1": rr1,
        "rr2": rr2,
        "rr3": rr3,
        "performance": perf,
        "arkham_intel": arkham_intel,
        "coinglass": coinglass,
        "orderbook": orderbook_intel,
        "binance_sentiment": binance_sentiment,
        "mode": mode,
        "onchain_intel": combined.get("onchain_intel"),
        "global_intel": combined.get("global_intel"),
        "is_ultra": combined.get("is_ultra", False),
        # ✅ flow موحد
        "flow_engine": combined.get("flow_engine"),
        "flow": combined.get("flow_engine"),
    }

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
