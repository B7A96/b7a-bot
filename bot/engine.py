import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import requests
import csv
import os
from datetime import datetime

from .coinglass_client import get_coinglass_intel
from .analytics import performance_intel


# =========================
# إعدادات عامة
# =========================

BINANCE_SPOT_BASE_URL = "https://api.binance.com"
BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"

# الفريمات اللي نستخدمها للتحليل
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
        "whale_inflow_score": 0.0,      # قوة دخول حيتان (0 - 100)
        "whale_outflow_score": 0.0,     # قوة خروج حيتان (0 - 100)
        "smart_money_bias": "NEUTRAL",  # UP / DOWN / NEUTRAL
        "cex_inflow_score": 0.0,        # عملات داخلة للمنصات (احتمال بيع)
        "cex_outflow_score": 0.0,       # عملات خارجة من المنصات (احتمال تجميع)
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
        "period": "5m",   # فترة قصيرة تعطي إحساس لحظي
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

    # كل عنصر: [price, qty]
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

    # نستخدم الكميات (qty) لقياس الضغط
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

    # Imbalance بين جانب المشترين والبائعين
    imbalance = (total_bid - total_ask) / (total_bid + total_ask)

    if imbalance > 0.15:
        bias = "BID"
    elif imbalance < -0.15:
        bias = "ASK"
    else:
        bias = "FLAT"

    score = max(0.0, min(100.0, abs(imbalance) * 100.0))

    # نحدد "جدران" السيولة (Walls) – مستويات كميتها كبيرة جداً نسبياً
    bid_walls = []
    ask_walls = []
    if total_bid > 0:
        bid_threshold = 0.03 * total_bid  # أي مستوى > 3% من إجمالي جانب الـ Bids
        for price, qty in bids:
            if qty >= bid_threshold:
                bid_walls.append({"price": price, "qty": qty})

    if total_ask > 0:
        ask_threshold = 0.03 * total_ask
        for price, qty in asks:
            if qty >= ask_threshold:
                ask_walls.append({"price": price, "qty": qty})

    # نرتبهم من الأكبر للأصغر ونأخذ أهم 5 فقط
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
                    "result",  # WIN / LOSS (تترك فاضية الآن)
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
                data.get("result", ""),  # تقدر تضيفها لاحقاً لو حبيت
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
    """
    Average True Range لقياس تذبذب السعر.
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

def _detect_swings(
    high: np.ndarray,
    low: np.ndarray,
    left: int = 2,
    right: int = 2,
) -> Tuple[List[int], List[int]]:
    """
    يحدد swing highs و swing lows بسيطة.
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


def _cluster_levels(prices: List[float], tolerance: float = 0.0015) -> List[Dict[str, Any]]:
    """
    يجمع القمم / القيعان المتقاربة في مستوى واحد (zone).
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
    يبني خريطة سيولة بسيطة من القمم والقيعان.
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

    # -------- أوزان الفريمات --------
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

        # Pump/Dump
        risk = data.get("pump_dump_risk", "LOW")
        if risk == "HIGH":
            max_pump_risk = "HIGH"
        elif risk == "MEDIUM" and max_pump_risk != "HIGH":
            max_pump_risk = "MEDIUM"

        # السيولة
        liq_above_total += data.get("liq_above", 0.0) * w
        liq_below_total += data.get("liq_below", 0.0) * w

        # وضع السوق
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

    if total_weight > 0:
        base_score = score_sum / total_weight
    else:
        base_score = 50.0

    bull_align = bullish_votes / total_weight if total_weight > 0 else 0.0
    bear_align = bearish_votes / total_weight if total_weight > 0 else 0.0

    # اتجاه عام
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

    # انحياز السيولة
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

    # RSI عالي الفريمات
    rsi_1h = tf_data.get("1h", {}).get("rsi")
    rsi_4h = tf_data.get("4h", {}).get("rsi")
    rsi_1d = tf_data.get("1d", {}).get("rsi")

    def _is_overbought(x):
        return x is not None and not np.isnan(x) and x > 70.0

    def _is_oversold(x):
        return x is not None and not np.isnan(x) and x < 30.0

    overbought = any(_is_overbought(r) for r in [rsi_1h, rsi_4h, rsi_1d])
    oversold = any(_is_oversold(r)   for r in [rsi_1h, rsi_4h, rsi_1d])

    # -------- تعديل السكور الكلاسيكي --------
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

    # -------- Arkham / Orderbook / Sentiment (نفس منطقك السابق) --------
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

    # -------- تمدد عن EMA200 + فلتر حماية --------
    def _extended_side(tf_name: str) -> str:
        data = tf_data.get(tf_name, {})
        c = data.get("close")
        ema200 = data.get("ema200")
        if c is None or ema200 is None or np.isnan(ema200) or ema200 == 0:
            return "NONE"
        dist_pct = abs(c - ema200) / abs(ema200) * 100.0
        if dist_pct < 8.0:
            return "NONE"
        if c > ema200:
            return "UP"
        else:
            return "DOWN"

    ext_4h = _extended_side("4h")
    ext_1d = _extended_side("1d")
    extended_up = (ext_4h == "UP") or (ext_1d == "UP")
    extended_down = (ext_4h == "DOWN") or (ext_1d == "DOWN")

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

    safety_block_buy = False
    safety_block_sell = False

    # في SAFE فقط نفعّل أقوى حماية
    if mode == "safe":
        if global_trend == "BULLISH" and extended_up and overbought:
            safety_block_buy = True
        if global_trend == "BEARISH" and extended_down and oversold:
            safety_block_sell = True

    # -------- thresholds لكل مود --------
    if mode == "safe":
        buy_score_min = 68.0
        sell_score_min = 60.0   # قوة بيع أعلى شوي
        buy_align_min = 0.55
        sell_align_min = 0.50
        gray_low = 52.0
        gray_high = 65.0
    elif mode == "momentum":
        buy_score_min = 60.0
        sell_score_min = 55.0
        buy_align_min = 0.40
        sell_align_min = 0.40
        gray_low = 48.0
        gray_high = 65.0
    else:  # balanced
        buy_score_min = 65.0
        sell_score_min = 58.0
        buy_align_min = 0.50
        sell_align_min = 0.45
        gray_low = 50.0
        gray_high = 65.0

    # SELL يستخدم "bear_score" (معكوس السكور)
    # مثال: لو combined_score = 40 → bear_score = 60 (بيع قوي)
    bear_score = 100.0 - combined_score

    action = "WAIT"

    # --- BUY ---
    if (
        combined_score >= buy_score_min
        and bull_align >= buy_align_min
        and not overbought
        and max_pump_risk != "HIGH"
        and (
            strong_bull_anchor
            or (global_regime in ("TRENDING", "RANGING") and liquidity_bias in ("UP", "FLAT"))
        )
    ):
        action = "BUY"

    # --- SELL ---
    if (
        bear_score >= sell_score_min          # هنا استخدمنا bear_score بدل sell_score_max
        and bear_align >= sell_align_min
        and not oversold
        and (
            strong_bear_anchor
            or (global_trend == "BEARISH" and liquidity_bias in ("DOWN", "FLAT"))
        )
    ):
        action = "SELL"



    # --- المنطقة الرمادية (للطرفين) ---
    if action == "WAIT" and gray_low <= combined_score < gray_high and max_pump_risk != "HIGH":
        if (
            liquidity_bias == "UP"
            and bull_align >= max(0.45, buy_align_min - 0.05)
            and (strong_bull_anchor or breakout_up_weight > 0.20)
        ):
            action = "BUY"
        elif (
            liquidity_bias == "DOWN"
            and bear_align >= max(0.45, sell_align_min - 0.05)
            and (strong_bear_anchor or breakout_down_weight > 0.20)
        ):
            action = "SELL"

    # --- Pump Sniper للمومنتوم: فريم 15m ---
    pump_momentum = False
    if mode == "momentum" and max_pump_risk != "HIGH":
        tf15 = tf_data.get("15m", {})
        if (
            tf15.get("market_regime") == "TRENDING"
            and tf15.get("is_breakout_up")
            and tf15.get("volume_surge")
            and bull_align >= 0.35
        ):
            # حتى لو السكور مو مثالي، نعطي BUY مومنتوم
            action = "BUY"
            pump_momentum = True

    # --- الثقة ---
    distance = abs(combined_score - 50.0)
    if distance >= 22:
        confidence = "HIGH"
    elif distance >= 12:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # حماية من Pump/Dump HIGH
    if max_pump_risk == "HIGH" and action in ("BUY", "SELL"):
        action = "WAIT"

    # تطبيق فلتر الحماية
    if safety_block_buy and action == "BUY":
        action = "WAIT"
    if safety_block_sell and action == "SELL":
        action = "WAIT"

    # -------- Grade + No-Trade --------
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
    else:  # momentum
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
        "bear_score": round(float(100.0 - combined_score), 2),
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
    """
    يحسب SL + TP1/TP2/TP3 باستخدام ATR من فريم 1h (ولو فشل → 15m)
    ويعدل المسافات حسب Grade + Score + Mode (SAFE / BALANCED / MOMENTUM).
    """

    mode = (mode or "balanced").lower()
    if mode not in ("safe", "balanced", "momentum"):
        mode = "balanced"

    # نحاول ATR من 1h أولاً
    anchor_tf = "1h"
    interval = TIMEFRAMES.get(anchor_tf, "1h")
    atr_value: Optional[float] = None

    try:
        ohlcv = fetch_klines(symbol_norm, interval, limit=200)
        atr_vals = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
        atr_value = float(atr_vals[-1])
    except Exception:
        # نحاول 15m
        try:
            ohlcv = fetch_klines(symbol_norm, TIMEFRAMES["15m"], limit=200)
            atr_vals = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
            atr_value = float(atr_vals[-1])
        except Exception:
            atr_value = None

    if atr_value is None or atr_value <= 0:
        return {
            "sl": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "rr1": None,
            "rr2": None,
            "rr3": None,
        }

    action = decision.get("action")
    grade = decision.get("grade", "C")
    score = float(decision.get("score", 50.0) or 50.0)

    # === Multipliers الأساسية حسب Grade (قبل المود) ===
    if grade == "A+":
        tp_mults = (1.2, 2.0, 3.0)
        base_sl_mult = 1.3
    elif grade == "A":
        tp_mults = (1.0, 1.8, 2.5)
        base_sl_mult = 1.25
    else:
        tp_mults = (0.8, 1.5, 2.0)
        base_sl_mult = 1.1

    # لو السكور تحت 70 نخفف الأهداف شوي
    if score < 70:
        tp_mults = tuple(m * 0.9 for m in tp_mults)

    # === تأثير الـ Mode على شكل الصفقة ===
    # SAFE: SL أوسع، TP أقرب شوي
    # BALANCED: افتراضي
    # MOMENTUM: SL أضيق، TP أبعد (أكثر Aggressive)
    if mode == "safe":
        sl_mode_factor = 1.3   # SL أبعد
        tp_mode_factor = 0.85  # TP أقرب
    elif mode == "momentum":
        sl_mode_factor = 0.8   # SL أقرب
        tp_mode_factor = 1.2   # TP أبعد
    else:  # balanced
        sl_mode_factor = 1.0
        tp_mode_factor = 1.0

    # مسافة SL: نعتمد أكثر على ATR مع تأثير المود
    sl_dist_atr = atr_value * base_sl_mult * sl_mode_factor

    # ما نبي risk_pct يختفي تمامًا، فنستخدم متوسط بين ATR و نسبة السعر
    sl_dist_pct = price * (risk_pct / 100.0)
    sl_dist = max(sl_dist_atr * 0.7 + sl_dist_pct * 0.3, atr_value * 0.5)

    # مسافات TP بناءً على ATR + المود
    tp_dists = [atr_value * m * tp_mode_factor for m in tp_mults]

    if action not in ("BUY", "SELL"):
        return {
            "sl": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "rr1": None,
            "rr2": None,
            "rr3": None,
        }

    if action == "BUY":
        sl = round(price - sl_dist, 6)
        tp1 = round(price + tp_dists[0], 6)
        tp2 = round(price + tp_dists[1], 6)
        tp3 = round(price + tp_dists[2], 6)
    else:  # SELL
        sl = round(price + sl_dist, 6)
        tp1 = round(price - tp_dists[0], 6)
        tp2 = round(price - tp_dists[1], 6)
        tp3 = round(price - tp_dists[2], 6)

    rr1 = round(tp_dists[0] / sl_dist, 2) if sl_dist > 0 else None
    rr2 = round(tp_dists[1] / sl_dist, 2) if sl_dist > 0 else None
    rr3 = round(tp_dists[2] / sl_dist, 2) if sl_dist > 0 else None

    return {
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rr1": rr1,
        "rr2": rr2,
        "rr3": rr3,
    }



# =========================
# نقطة الدخول الرئيسية
# =========================

def generate_signal(symbol: str, mode: str = "balanced") -> Dict[str, Any]:
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

    # نحتفظ بآخر سعر واضح (نفضّل 1h ثم 15m)
    last_close: Optional[float] = None

    # 1) Arkham Intel (placeholder)
    try:
        arkham_intel = get_arkham_intel(symbol_norm)
    except Exception:
        arkham_intel = None

    # 1.25) Orderbook Intel
    try:
        orderbook_intel = analyse_orderbook(symbol_norm, limit=100)
    except Exception:
        orderbook_intel = None

    # 1.3) Binance Sentiment (بديل مجاني)
    try:
        binance_sentiment = fetch_binance_sentiment(symbol_norm)
    except Exception:
        binance_sentiment = None

    # 1.5) Coinglass Intel (من coinglass_client.py)
    try:
        coinglass = get_coinglass_intel(symbol_norm)
    except Exception:
        coinglass = None

    # 2) نجيب بيانات كل الفريمات
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

    # 3) ندمج الفريمات في قرار واحد
    combined = combine_timeframes(
        tf_results,
        arkham_intel=arkham_intel,
        orderbook_intel=orderbook_intel,
        binance_sentiment=binance_sentiment,
        mode=mode,
    )

    # 3.25) تأثير خفيف لـ Coinglass (Open Interest Bias)
    if coinglass and coinglass.get("available"):
        try:
            oi = (coinglass or {}).get("open_interest") or {}
            oi_bias = oi.get("oi_bias", "NEUTRAL")
            oi_chg = oi.get("oi_change_24h")

            delta = 0.0

            if oi_bias == "LEVERAGE_UP" and oi_chg is not None:
                if combined["action"] == "BUY":
                    delta += 2.0
                elif combined["action"] == "SELL":
                    delta -= 2.0
            elif oi_bias == "LEVERAGE_DOWN" and oi_chg is not None:
                if combined["action"] == "SELL":
                    delta += 2.0
                elif combined["action"] == "BUY":
                    delta -= 2.0

            combined["score"] = max(
                0.0, min(100.0, combined.get("score", 50.0) + delta)
            )
        except Exception as e:
            print("Coinglass impact error:", e)

    # 3.5) ذكاء الأداء: تعديل القرار بناءً على تاريخ الصفقات في اللوق
    try:
        perf = performance_intel(symbol_norm, combined)
    except Exception:
        perf = {
            "score_delta": 0.0,
            "risk_multiplier": 1.0,
            "force_no_trade": False,
            "note": None,
        }

    # نعدل السكور حسب أداء الزوج في الماضي
    combined["score"] = max(
        0.0, min(100.0, combined.get("score", 50.0) + perf["score_delta"])
    )

    # لو الفلتر يقول هذي المنطقة خطرة → نحولها No-Trade
    if perf.get("force_no_trade"):
        combined["no_trade"] = True
        combined["action"] = "WAIT"

    # نثبت الـ mode داخل decision
    combined["mode"] = mode

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

        # نسب المخاطرة حسب الثقة (الأساسية)
        if combined["confidence"] == "HIGH":
            risk_pct = 2.0
        elif combined["confidence"] == "MEDIUM":
            risk_pct = 1.5
        else:
            risk_pct = 1.0

        # تعديل الرسك حسب الـ mode
        if mode == "safe":
            risk_pct *= 0.7
        elif mode == "momentum":
            risk_pct *= 1.2

        # حجم الصفقة الذكي حسب أداء الزوج
        risk_pct *= perf.get("risk_multiplier", 1.0)
        risk_pct = max(0.5, min(3.0, risk_pct))

        # مضاعف هدف الربح حسب السكور
        if combined["score"] >= 75:
            reward_mult = 2.5
        elif combined["score"] >= 65:
            reward_mult = 2.0
        else:
            reward_mult = 1.5

        # تعديل الأهداف حسب المود
        if mode == "safe":
            reward_mult *= 0.9
        elif mode == "momentum":
            reward_mult *= 1.1

        reward_pct = risk_pct * reward_mult

        # حساب المستويات الديناميكية
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

        # التوافق مع البوت الحالي: نستخدم TP2 كهدف رئيسي
        tp = tp2
        rr = rr2

    # 4) نص توضيحي ذكي مختصر
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

    # 📘 Orderbook Intel Summary
    ob_bias = combined.get("orderbook_bias", "FLAT")
    ob_score = combined.get("orderbook_score", 0.0)
    if ob_score and ob_score > 0:
        if ob_bias == "BID":
            reason_lines.append(
                f"ضغط المشترين في دفتر الأوامر (Orderbook BID) ملحوظ (Score ≈ {ob_score:.0f}) → الطلبات متقدّمة حالياً."
            )
        elif ob_bias == "ASK":
            reason_lines.append(
                f"ضغط البائعين في دفتر الأوامر (Orderbook ASK) ملحوظ (Score ≈ {ob_score:.0f}) → العروض متقدّمة حالياً."
            )

    # 📊 Binance Sentiment Summary
    bs_bias = combined.get("binance_sentiment_bias", "NEUTRAL")
    bs_strength = combined.get("binance_sentiment_strength", 0.0)
    if bs_bias != "NEUTRAL" and bs_strength:
        if bs_bias == "LONG":
            reason_lines.append(
                f"Binance Sentiment → حسابات الفيوتشر تميل للـ LONG بقوة تقريبية {bs_strength:.1f} نقطة."
            )
        else:
            reason_lines.append(
                f"Binance Sentiment → حسابات الفيوتشر تميل للـ SHORT بقوة تقريبية {bs_strength:.1f} نقطة."
            )

    # 📊 Coinglass Intel (لو متوفر)
    if coinglass:
        try:
            oi = (coinglass or {}).get("open_interest") or {}
            fut = (coinglass or {}).get("futures_status") or {}
            spot_i = (coinglass or {}).get("spot_status") or {}
            etf = (coinglass or {}).get("btc_etf") or {}

            oi_parts = []
            if oi.get("available"):
                oi_usd = oi.get("oi_usd")
                oi_chg = oi.get("oi_change_24h")
                oi_bias = oi.get("oi_bias", "NEUTRAL")

                if oi_usd is not None:
                    try:
                        oi_parts.append(f"OI ≈ {oi_usd:,.0f} USD")
                    except Exception:
                        oi_parts.append(f"OI ≈ {oi_usd} USD")

                if oi_chg is not None:
                    oi_parts.append(f"24h ΔOI ≈ {oi_chg:.1f}%")

                if oi_bias and oi_bias != "NEUTRAL":
                    if oi_bias == "LEVERAGE_UP":
                        oi_parts.append("Leverage Bias: UP (مراكز جديدة تفتح)")
                    elif oi_bias == "LEVERAGE_DOWN":
                        oi_parts.append("Leverage Bias: DOWN (مراكز تتقفل)")
                    else:
                        oi_parts.append(f"Leverage Bias: {oi_bias}")

            status_parts = []
            if fut.get("available") and fut.get("listed") is not None:
                status_parts.append("Futures: LISTED" if fut["listed"] else "Futures: NOT LISTED")
            if spot_i.get("available") and spot_i.get("listed") is not None:
                status_parts.append("Spot: LISTED" if spot_i["listed"] else "Spot: NOT LISTED")

            etf_parts = []
            if etf.get("available") and etf.get("funds", 0) > 0:
                funds = etf.get("funds", 0)
                trading = etf.get("trading_count", 0)
                halted = etf.get("halted_count", 0)
                etf_parts.append(
                    f"BTC ETFs: {trading}/{funds} trading, {halted} halted → يعكس شهية المؤسسات على بيتكوين."
                )

            all_parts = []
            if oi_parts:
                all_parts.append("Open Interest: " + " | ".join(oi_parts))
            if status_parts:
                all_parts.append("Listings: " + " | ".join(status_parts))
            if etf_parts:
                all_parts.extend(etf_parts)

            if all_parts:
                reason_lines.append("📊 Coinglass Intel → " + " || ".join(all_parts))
        except Exception:
            pass

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
        # Multi-TP extra info
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
