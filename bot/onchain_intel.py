import math
import requests
from typing import Dict, Any, Optional


BLOCKCHAIN_STATS_URL = "https://api.blockchain.info/stats"
ETHERSCAN_GAS_URL = "https://api.etherscan.io/v2/api"
SOLSCAN_CHAININFO_URL = "https://public-api.solscan.io/chaininfo"


# ملاحظة:
# - BTC on-chain لا يحتاج API key.
# - Etherscan يحتاج ETHERSCAN_API_KEY في متغير البيئة (تحطه في Render).
# - Solscan public endpoint لا يحتاج مفتاح لكن عليه Rate Limit معقول.


class OnchainError(Exception):
    pass


def _safe_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 8) -> Dict[str, Any]:
    resp = requests.get(url, params=params or {}, timeout=timeout)
    if resp.status_code != 200:
        raise OnchainError(f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")
    data = resp.json()
    if not isinstance(data, dict):
        raise OnchainError(f"Unexpected JSON from {url}")
    return data


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _scale(value: Optional[float], low: float, high: float) -> Optional[float]:
    if value is None:
        return None
    if high <= low:
        return None
    return _clip01((value - low) / (high - low))


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

    # تقدير بسيط لمستويات "الطبيعي"
    n_tx_s = _scale(n_tx, 150_000, 400_000) or 0.5
    vol_s = _scale(trade_vol, 10_000_000, 1_000_000_000) or 0.5
    hash_s = _scale(hash_rate, 200_000_000, 600_000_000) or 0.5

    activity = (n_tx_s + vol_s + hash_s) / 3.0
    activity_score = round(100.0 * _clip01(activity), 1)

    # Dump risk:
    if minutes_between_blocks > 14 or (activity < 0.3 and trade_vol < 20_000_000):
        dump_risk = "HIGH"
    elif minutes_between_blocks > 11 or activity < 0.5:
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


def _eth_gas_snapshot(etherscan_api_key: Optional[str]) -> Dict[str, Any]:
    """
    يستخدم Etherscan Gas Oracle:
      - congestion_score (0-100)
      - gas_bias: LOW / NORMAL / HIGH / EXTREME
    """
    if not etherscan_api_key:
        return {"available": False}

    params = {
        "chainid": "1",
        "module": "gastracker",
        "action": "gasoracle",
        "apikey": etherscan_api_key,
    }
    data = _safe_get(ETHERSCAN_GAS_URL, params=params)
    status = data.get("status")
    if status not in ("1", 1, "OK"):
        result = data.get("result", {})
    else:
        result = data.get("result", {})

    if not isinstance(result, dict):
        return {"available": False}

    try:
        safe_gwei = float(result.get("SafeGasPrice") or 0.0)
        propose_gwei = float(result.get("ProposeGasPrice") or 0.0)
        fast_gwei = float(result.get("FastGasPrice") or 0.0)
        base_fee = float(result.get("suggestBaseFee") or 0.0)
    except Exception:
        return {"available": False}

    # متوسط استهلاك البلوك من gasUsedRatio
    ratios_raw = str(result.get("gasUsedRatio") or "")
    ratios = []
    for p in ratios_raw.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            ratios.append(float(p))
        except ValueError:
            continue

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.5
    congestion = _clip01(avg_ratio)  # 0 → هدوء , 1 → ضغط شديد
    congestion_score = round(congestion * 100.0, 1)

    if congestion_score < 30:
        gas_bias = "LOW"
    elif congestion_score < 60:
        gas_bias = "NORMAL"
    elif congestion_score < 85:
        gas_bias = "HIGH"
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


def _solana_chain_snapshot() -> Dict[str, Any]:
    """
    Snapshot بسيط من Solscan public API.
    نستخدم transactionCount كإشارة على ضغط الشبكة.
    """
    data = _safe_get(SOLSCAN_CHAININFO_URL)
    if not data.get("success"):
        return {"available": False}

    info = data.get("data") or {}
    tx_count = float(info.get("transactionCount") or info.get("transactionCount24h") or 0.0)

    # Normalization تقريبية (تحتاج ضبط مع الوقت)
    tx_score = _scale(tx_count, 50_000, 500_000) or 0.5
    load_score = round(100.0 * _clip01(tx_score), 1)

    if load_score < 30:
        load_bias = "LOW"
    elif load_score < 60:
        load_bias = "NORMAL"
    elif load_score < 85:
        load_bias = "HIGH"
    else:
        load_bias = "EXTREME"

    return {
        "available": True,
        "transaction_count": tx_count,
        "load_score": load_score,
        "load_bias": load_bias,
    }


def get_onchain_intel(symbol_norm: str, etherscan_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    واجهة موحّدة للـ Engine:

      - تعتمد دائماً على BTC stats (كشبكة رئيسية).
      - تضيف ETH gas intel لو توفر API Key.
      - تضيف Solana snapshot لو العملة لها علاقة بسولانا.

    ترجع:
      {
        "available": True/False,
        "btc": {...},
        "eth_gas": {...} أو None,
        "solana": {...} أو None,
        "dump_risk": "LOW/MEDIUM/HIGH"   # من BTC
      }
    """
    symbol_norm = (symbol_norm or "").upper()

    try:
        btc = _btc_onchain_snapshot()
    except Exception as e:
        print("Onchain BTC error:", e)
        btc = {"available": False, "dump_risk": "MEDIUM"}

    eth_gas = None
    solana = None

    if etherscan_api_key:
        try:
            eth_gas = _eth_gas_snapshot(etherscan_api_key)
        except Exception as e:
            print("Onchain ETH gas error:", e)
            eth_gas = {"available": False}

    # لو العملة على سولانا (رمز ينتهي بـ /SOL أو اسم فيه SOL)
    if "SOL" in symbol_norm:
        try:
            solana = _solana_chain_snapshot()
        except Exception as e:
            print("Onchain Solana error:", e)
            solana = {"available": False}

    dump_risk = btc.get("dump_risk", "MEDIUM")

    return {
        "available": True,
        "btc": btc,
        "eth_gas": eth_gas,
        "solana": solana,
        "dump_risk": dump_risk,
    }
