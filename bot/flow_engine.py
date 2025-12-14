# bot/flow_engine.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import math


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        x = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, x))


def _safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _sigmoid(z: float) -> float:
    # stable sigmoid
    try:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)
    except Exception:
        return 0.5


def _norm_pct(x: float, cap: float) -> float:
    """
    Normalize percentage-like values into [-1..+1] then map to [0..1] later.
    """
    if cap <= 0:
        return 0.0
    x = max(-cap, min(cap, x))
    return x / cap


def _infer_price_dir(combined: Dict[str, Any]) -> Tuple[str, float]:
    """
    Direction + strength (0..1)
    """
    decision = _safe_dict(combined.get("decision"))
    trend = (decision.get("trend") or combined.get("trend") or "RANGING").upper()
    score = _to_float(decision.get("score"), 50.0)

    if trend == "BULLISH":
        return "UP", _clip((score - 50.0) / 30.0, 0.0, 1.0)
    if trend == "BEARISH":
        return "DOWN", _clip((50.0 - score) / 30.0, 0.0, 1.0)
    return "FLAT", _clip(abs(score - 50.0) / 40.0, 0.0, 1.0)


def compute_flow_engine(
    symbol_norm: str,
    combined: Optional[Dict[str, Any]] = None,
    coinglass_intel: Optional[Dict[str, Any]] = None,
    onchain_intel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    B7A Flow Engine v1 (خارجي):
    - ياخذ combined (قرار + فريمات + مخاطر)
    - ياخذ coinglass_intel
    - ياخذ onchain_intel
    ويرجع Flow State + Bias + Score + Shield
    """

    combined = _safe_dict(combined)
    cg = _safe_dict(coinglass_intel)
    oc = _safe_dict(onchain_intel)

    decision = _safe_dict(combined.get("decision"))
    tfs = _safe_dict(combined.get("timeframes"))

    # ---------------------------
    # 1) Price / Trend core
    # ---------------------------
    price_dir, price_strength = _infer_price_dir(combined)  # UP/DOWN/FLAT, 0..1
    base_score = _to_float(decision.get("score"), 50.0)
    confidence = (decision.get("confidence") or "LOW").upper()
    pump_risk = (decision.get("pump_dump_risk") or combined.get("pump_dump_risk") or "LOW").upper()

    # quick volatility proxy from 15m change_1 if موجود
    chg_15m = _to_float(_safe_dict(tfs.get("15m")).get("change_1"), 0.0)
    chg_1h = _to_float(_safe_dict(tfs.get("1h")).get("change_4"), 0.0)  # 4 candles on 1h = 4h-ish
    price_impulse = abs(chg_15m) + 0.5 * abs(chg_1h)

    # ---------------------------
    # 2) Coinglass: OI / Funding / Liquidations
    # ---------------------------
    oi = _safe_dict(cg.get("open_interest"))
    oi_change_24h = _to_float(oi.get("oi_change_24h"), 0.0)  # %
    oi_bias = (oi.get("oi_bias") or "NEUTRAL").upper()

    funding = _safe_dict(cg.get("funding"))
    funding_rate = _to_float(funding.get("rate"), 0.0)  # %
    funding_severity = (funding.get("severity") or "LOW").upper()
    funding_side = (funding.get("side_bias") or "NEUTRAL").upper()  # LONG/SHORT/NEUTRAL

    liq = _safe_dict(cg.get("liquidation"))
    liq_bias = (liq.get("bias") or "NEUTRAL").upper()  # LONG/SHORT/NEUTRAL
    liq_intensity = _to_float(liq.get("intensity"), 0.0)  # 0..?
    liq_usd = _to_float(liq.get("liquidation_usd"), 0.0)

    # ---------------------------
    # 3) Onchain: BTC dump risk / ETH gas / SOL load
    # ---------------------------
    btc = _safe_dict(oc.get("btc"))
    btc_dump_risk = (btc.get("dump_risk") or "MEDIUM").upper()

    eth_gas = _safe_dict(oc.get("eth_gas"))
    eth_congestion = _to_float(eth_gas.get("congestion_score"), 0.0)  # 0..100

    sol = _safe_dict(oc.get("solana"))
    sol_load = _to_float(sol.get("load_score"), 0.0)  # 0..100

    # ---------------------------
    # 4) Build Flow Score (0..100)
    # ---------------------------
    flow = 50.0

    # Price trend contribution (stronger weight)
    # base_score is already 0..100
    flow += (base_score - 50.0) * 0.55

    # OI change contribution (cap at ±8%)
    flow += _norm_pct(oi_change_24h, 8.0) * 10.0  # ±10

    # Funding crowding penalty/bonus
    # If funding severe and side matches direction => crowding => reduce
    # If funding severe and opposite => can support continuation/mean-revert depending; here slightly boost cautionary.
    if funding_severity == "HIGH":
        if (funding_side == "LONG" and price_dir == "UP") or (funding_side == "SHORT" and price_dir == "DOWN"):
            flow -= 6.0
        elif (funding_side == "LONG" and price_dir == "DOWN") or (funding_side == "SHORT" and price_dir == "UP"):
            flow += 2.0

    # Liquidations (contrarian + momentum hint)
    # If price UP and SHORT liqs high => continuation can happen (but also exhaustion later)
    liq_boost = _clip(liq_intensity / 3.0, 0.0, 1.0) * 6.0  # max 6
    if liq_bias == "SHORT" and price_dir == "UP":
        flow += liq_boost
    elif liq_bias == "LONG" and price_dir == "DOWN":
        flow -= liq_boost

    # Onchain macro risk penalty
    if btc_dump_risk == "HIGH":
        flow -= 5.0
    elif btc_dump_risk == "LOW":
        flow += 1.5

    # Network heat (minor effect)
    # very high congestion can reduce reliability for fast scalps
    if eth_congestion >= 80:
        flow -= 1.5
    if sol_load >= 80:
        flow -= 1.0

    # Clamp
    flow = max(0.0, min(100.0, flow))

    # ---------------------------
    # 5) Determine Bias
    # ---------------------------
    if flow >= 60:
        bias = "BUY"
    elif flow <= 40:
        bias = "SELL"
    else:
        bias = "NEUTRAL"

    # ---------------------------
    # 6) Determine Flow State (CALM/BUILD_UP/EXPLOSION/EXHAUSTION)
    # ---------------------------
    # Heuristics:
    # - EXPLOSION: strong impulse + OI rising
    # - EXHAUSTION: strong impulse but OI falling OR funding severe crowding OR pump risk high
    # - BUILD_UP: low impulse but OI rising
    # - CALM: low impulse and flat OI and low risk
    oi_up = oi_change_24h > 1.2
    oi_down = oi_change_24h < -1.2
    strong_impulse = price_impulse >= 3.0   # tune later
    medium_impulse = price_impulse >= 1.2
    severe_crowd = funding_severity == "HIGH" and (
        (funding_side == "LONG" and bias == "BUY") or (funding_side == "SHORT" and bias == "SELL")
    )

    if strong_impulse and (oi_down or severe_crowd or pump_risk == "HIGH"):
        state = "EXHAUSTION"
        regime = "RISKY"
    elif strong_impulse and oi_up:
        state = "EXPLOSION"
        regime = "TRENDING"
    elif (not medium_impulse) and oi_up:
        state = "BUILD_UP"
        regime = "BUILDING"
    else:
        state = "CALM"
        regime = "UNKNOWN" if confidence == "LOW" else "STABLE"

    # ---------------------------
    # 7) Shield (Test Mode)
    # ---------------------------
    shield_active = True
    shield_suggest_no_trade = False
    reasons = []

    # hard filters
    if pump_risk == "HIGH":
        shield_suggest_no_trade = True
        reasons.append("Pump/Dump risk HIGH")
    if state == "EXHAUSTION":
        shield_suggest_no_trade = True
        reasons.append("Flow State: EXHAUSTION (احتمال انعكاس/تعب)")
    if confidence == "LOW":
        reasons.append("Confidence LOW (الدقة أقل)")
    if severe_crowd:
        shield_suggest_no_trade = True
        reasons.append("Funding شديد بنفس اتجاه الصفقة (Crowded trade)")

    # neutral zone
    if bias == "NEUTRAL":
        reasons.append("Flow Bias NEUTRAL (أفضل انتظار/تأكيد)")

    # Optional: if BTC dump risk high -> caution
    if btc_dump_risk == "HIGH":
        reasons.append("BTC Onchain dump risk HIGH")

    return {
        "available": True,
        "symbol": symbol_norm,
        "flow_score": round(flow, 1),
        "bias": bias,              # BUY/SELL/NEUTRAL
        "state": state,            # CALM/BUILD_UP/EXPLOSION/EXHAUSTION
        "regime": regime,          # UNKNOWN/STABLE/TRENDING/...
        "components": {
            "price_dir": price_dir,
            "price_strength": round(price_strength, 3),
            "price_impulse": round(price_impulse, 3),
            "oi_change_24h": round(oi_change_24h, 3),
            "oi_bias": oi_bias,
            "funding_rate": round(funding_rate, 6),
            "funding_severity": funding_severity,
            "funding_side": funding_side,
            "liq_bias": liq_bias,
            "liq_intensity": round(liq_intensity, 3),
            "liq_usd": round(liq_usd, 2),
            "btc_dump_risk": btc_dump_risk,
            "eth_congestion": round(eth_congestion, 1),
            "sol_load": round(sol_load, 1),
            "pump_risk": pump_risk,
            "confidence": confidence,
        },
        "shield_active": shield_active,
        "shield_suggest_no_trade": shield_suggest_no_trade,
        "shield_reasons": reasons,
    }
