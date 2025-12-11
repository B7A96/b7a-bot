# ============================================
# B7A FLOW ENGINE v1  (Foundation Stage)
# ============================================

from typing import Dict, Any
import math


def _clip(v, low, high):
    return max(low, min(high, v))


def compute_flow_engine(
    symbol: str,
    combined: Dict[str, Any],
    coinglass: Dict[str, Any],
    onchain: Dict[str, Any],
) -> Dict[str, Any]:

    # إذا ما في بيانات نرجع NONE
    if not combined:
        return {"available": False}

    # -----------------------------------------
    # 1) Directional Flow (DF)
    # -----------------------------------------
    score = combined.get("score", 50)
    trend = combined.get("trend", "NEUTRAL")

    df = (score - 50) / 50     # يتحول إلى -1 → +1

    if trend == "BULLISH":
        df *= 1.1
    elif trend == "BEARISH":
        df *= -1.1

    # -----------------------------------------
    # 2) Leverage Pressure (LP)
    # -----------------------------------------
    lp = 0
    if coinglass and coinglass.get("available", False):
        oi = coinglass.get("oi_change_24h")
        funding = coinglass.get("funding", {})
        funding_rate = funding.get("rate", 0.0)
        funding_bias = funding.get("side_bias", "NEUTRAL")

        # OI
        if oi is not None:
            if oi > 4:
                lp += 0.25
            elif oi > 1:
                lp += 0.10
            elif oi < -4:
                lp -= 0.25
            elif oi < -1:
                lp -= 0.10

        # Funding
        if abs(funding_rate) > 0.03:
            if funding_bias == "LONG":
                lp -= 0.1      # مزدحم
            elif funding_bias == "SHORT":
                lp += 0.1

    # -----------------------------------------
    # 3) Liquidation Pressure (LiP)
    # -----------------------------------------
    lip = 0
    if coinglass:
        liq = coinglass.get("liquidations", {})
        inten = liq.get("intensity", 0)
        lbias = liq.get("bias", "NEUTRAL")

        if inten > 60:
            if lbias == "LONG":
                lip -= 0.15
            elif lbias == "SHORT":
                lip += 0.15
        elif inten > 30:
            if lbias == "LONG":
                lip -= 0.05
            elif lbias == "SHORT":
                lip += 0.05

    # -----------------------------------------
    # 4) Exhaustion Layer (EXH)
    # -----------------------------------------
    exh = 0
    pump = combined.get("pump_dump_risk", "LOW")
    dump_risk = onchain.get("dump_risk") if onchain else "MEDIUM"

    if pump == "HIGH":
        exh -= 0.30
    elif pump == "MEDIUM":
        exh -= 0.10

    if dump_risk == "HIGH":
        exh -= 0.20
    elif dump_risk == "MEDIUM":
        exh -= 0.05

    # -----------------------------------------
    # === Final Flow Score Calculation ===
    # -----------------------------------------
    raw_flow = df + lp + lip + exh
    flow_score = int(_clip((raw_flow + 1) * 50, 0, 100))

    # -----------------------------------------
    # Determine Regime
    # -----------------------------------------
    if flow_score < 35:
        regime = "CALM"
    elif flow_score < 60:
        regime = "BUILD_UP"
    elif flow_score < 80:
        regime = "EXPLOSION"
    else:
        regime = "EXHAUSTION"

    # -----------------------------------------
    # Determine Bias (BUY/SELL direction)
    # -----------------------------------------
    if flow_score >= 55 and trend == "BULLISH":
        bias = "BULLISH"
    elif flow_score >= 55 and trend == "BEARISH":
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {
        "available": True,
        "flow_score": flow_score,
        "regime": regime,
        "bias": bias,
        "debug": {
            "df": df,
            "lp": lp,
            "lip": lip,
            "exh": exh,
            "raw_flow": raw_flow,
        }
    }
