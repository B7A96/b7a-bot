import os
import math
from typing import Dict, Any, Optional

import requests


BLOCKCHAIN_STATS_URL = "https://api.blockchain.info/stats"
ETHERSCAN_GAS_URL = "https://api.etherscan.io/v2/api"
SOLSCAN_CHAININFO_URL = "https://public-api.solscan.io/chaininfo"


class OnchainError(Exception):
    pass


def _safe_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 8,
) -> Dict[str, Any]:
    """
    طلب GET بسيط مع فحص الـ status code وفحص أن الرد JSON/dict.
    يرفع OnchainError في حال وجود مشكلة.
    """
    resp = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
    if resp.status_code != 200:
        raise OnchainError(f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")
    data = resp.json()
    if not isinstance(data, dict):
        raise OnchainError(f"Unexpected JSON from {url}")
    return data


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _scale(value: Optional[float], low: float, high: float) -> Optional[float]:
    """
    يحول قيمة خام إلى نطاق [0, 1] بناءً على حدود تقريبية.
    """
    if value is None:
        return None
    if high <= low:
        return None
    return _clip01((value - low) / (high - low))


# =========================
# BTC On-chain Snapshot
# =========================
def _btc_onchain_snapshot() -> Dict[str, Any]:
    """
    Snapshot مبسط من شبكة بيتكوين (API مجاني من blockchain.info):

    يرجّع:
      - activity_score (0-100)
      - dump_risk: LOW / MEDIUM / HIGH
    """
    stats = _safe_get(BLOCKCHAIN_STATS_URL)

    n_tx = float(stats.get("n_tx") or 0.0)
    trade_vol = float(stats.get("trade_volume_usd") or 0.0)
    hash_rate = float(stats.get("hash_rate") or 0.0)
    minutes_between_blocks = float(stats.get("minutes_between_blocks") or 0.0)

    # Normalization تقريبية – تكفي كبوصلة عامة
    tx_score = _scale(n_tx, 150_000, 400_000) or 0.5
    vol_score = _scale(trade_vol, 100_000_000, 700_000_000) or 0.5
    hash_score = _scale(hash_rate, 100_000_000, 400_000_000) or 0.5
    # كلما كان البلوك أسرع من 10 دقائق يعتبر نشاط أعلى
    if minutes_between_blocks > 0:
        blk_speed_score = _clip01((10.0 / minutes_between_blocks))
    else:
        blk_speed_score = 0.5

    activity_score = round(
        100.0 * (0.35 * tx_score + 0.25 * vol_score + 0.25 * hash_score + 0.15 * blk_speed_score),
        1,
    )

    # Dump risk تقريبي:
    # نشاط ضعيف + بلوكات بطيئة => HIGH
    # نشاط متوسط => MEDIUM
    # نشاط قوي => LOW
    if activity_score < 40 or minutes_between_blocks > 11:
        dump_risk = "HIGH"
    elif activity_score < 65:
        dump_risk = "MEDIUM"
    else:
        dump_risk = "LOW"

    return {
        "available": True,
        "n_tx": n_tx,
        "trade_volume_usd": trade_vol,
        "hash_rate": hash_rate,
        "minutes_between_blocks": minutes_between_blocks,
        "activity_score": activity_score,
        "dump_risk": dump_risk,
    }


# =========================
# ETH Gas (Etherscan)
# =========================
def _eth_gas_snapshot(etherscan_api_key: str) -> Dict[str, Any]:
    """
    Snapshot بسيط لحالة الغاز على شبكة ETH من Etherscan Gas Oracle.
    يحتاج ETHERSCAN_API_KEY.
    """
    params = {
        "module": "gastracker",
        "action": "gasoracle",
        "apikey": etherscan_api_key,
    }
    data = _safe_get(ETHERSCAN_GAS_URL, params=params)

    # Etherscan v2 يضع النتيجة داخل result
    result = data.get("result") or {}
    # الحقول ترجع كنصوص
    safe_gwei = float(result.get("SafeGasPrice") or 0.0)
    propose_gwei = float(result.get("ProposeGasPrice") or 0.0)
    fast_gwei = float(result.get("FastGasPrice") or 0.0)
    base_fee = float(result.get("suggestBaseFee") or 0.0)

    # متوسط استهلاك البلوك من gasUsedRatio (قائمة نصية مفصولة بفاصلة)
    ratios_raw = str(result.get("gasUsedRatio") or "")
    ratios = []
    for part in ratios_raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ratios.append(float(part))
        except ValueError:
            continue

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.5
    congestion = _clip01(avg_ratio)  # 0 → هدوء , 1 → ضغط شديد
    congestion_score = round(100.0 * congestion, 1)

    # انحياز الغاز بالكلمات
    if congestion < 0.3:
        gas_bias = "CALM"
    elif congestion < 0.6:
        gas_bias = "NORMAL"
    elif congestion < 0.8:
        gas_bias = "BUSY"
    else:
        gas_bias = "EXTREME"

    return {
        "available": True,
        "safe_gwei": safe_gwei,
        "propose_gwei": propose_gwei,
        "fast_gwei": fast_gwei,
        "base_fee_gwei": base_fee,
        "congestion_score": congestion_score,
        "gas_bias": gas_bias,
    }


# =========================
# Solana – Solscan
# =========================
def _solana_chain_snapshot() -> Dict[str, Any]:
    """
    Snapshot بسيط من Solscan API.

    مهم:
    - يجب إضافة متغير بيئة في Render باسم SOLSCAN_API_KEY
      (القيمة هي الـ API Key من Solscan).
    """
    api_key = os.getenv("SOLSCAN_API_KEY")
    if not api_key:
        # لا نعتبره خطأ قاتل – فقط نعيد available=False
        return {"available": False, "reason": "no_api_key"}

    headers = {"token": api_key}
    data = _safe_get(SOLSCAN_CHAININFO_URL, headers=headers)

    # بعض الإصدارات ترجع success + data، والبعض قد يرجع البيانات مباشرة.
    if "success" in data and not data.get("success"):
        return {"available": False}

    info = data.get("data") if "data" in data else data

    tx_count = float(
        info.get("transactionCount")
        or info.get("transactionCount24h")
        or 0.0
    )
    block_height = float(info.get("blockHeight") or 0.0)
    current_epoch = float(info.get("currentEpoch") or 0.0)

    # Normalization تقريبية – الهدف فقط معرفة هل الشبكة "حارة"
    # هذه الأرقام تحتاج ضبط لاحقاً مع التجربة.
    tx_score = _scale(tx_count, 50_000, 600_000) or 0.5
    load_score = round(100.0 * _clip01(tx_score), 1)

    if load_score < 30:
        load_bias = "CALM"
    elif load_score < 60:
        load_bias = "NORMAL"
    elif load_score < 80:
        load_bias = "BUSY"
    else:
        load_bias = "HOT"

    return {
        "available": True,
        "block_height": block_height,
        "current_epoch": current_epoch,
        "transaction_count": tx_count,
        "load_score": load_score,
        "load_bias": load_bias,
    }


# =========================
# Public API
# =========================
def get_onchain_intel(symbol_norm: str, etherscan_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    واجهة موحّدة للـ Engine:

      - تعتمد دائماً على BTC stats (كشبكة رئيسية).
      - تضيف ETH gas intel لو توفر API Key.
      - تضيف Solana snapshot (حالياً دائماً، لكن يمكن ربطها لاحقاً فقط بعملات سولانا).

    ترجع:
      {
        "available": True/False,
        "btc": {...},
        "eth_gas": {...} أو None,
        "solana": {...} أو None,
        "dump_risk": "LOW/MEDIUM/HIGH"   # من BTC
      }
    """
    # -------- BTC دائماً --------
    try:
        btc = _btc_onchain_snapshot()
    except Exception as e:
        print("Onchain BTC error:", e)
        btc = {"available": False}

    # -------- ETH Gas --------
    if etherscan_api_key is None:
        etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")

    if etherscan_api_key:
        try:
            eth_gas = _eth_gas_snapshot(etherscan_api_key)
        except Exception as e:
            print("Onchain ETH gas error:", e)
            eth_gas = {"available": False}
    else:
        eth_gas = {"available": False}

    # -------- Solana --------
    try:
        solana = _solana_chain_snapshot()
    except Exception as e:
        print("Onchain Solana error:", e)
        solana = {"available": False}

    dump_risk = btc.get("dump_risk", "MEDIUM") if isinstance(btc, dict) else "MEDIUM"

    return {
        "available": True,
        "btc": btc,
        "eth_gas": eth_gas,
        "solana": solana,
        "dump_risk": dump_risk,
    }
