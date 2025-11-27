import csv
import os
from collections import Counter, defaultdict
from typing import Dict, Any, List


LOG_FILE = "trades_log.csv"


def _read_trades() -> List[Dict[str, Any]]:
    """
    يقرأ كل الصفقات من ملف اللوق ويرجعها كـ list of dicts
    """
    if not os.path.isfile(LOG_FILE):
        return []

    rows: List[Dict[str, Any]] = []
    with open(LOG_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def get_trades_summary() -> str:
    trades = _read_trades()
    if not trades:
        return "📊 ما في بيانات في اللوق حالياً.\nجرّب تستخدم /signal كم مرة وبعدين استخدم /stats."

    total = len(trades)

    # عدّ الأكشنات والـ Grades
    actions = Counter(t["action"] for t in trades if t.get("action"))
    grades = Counter(t.get("grade", "C") for t in trades)
    regimes = Counter(t.get("market_regime", "UNKNOWN") for t in trades)
    liq_biases = Counter(t.get("liquidity_bias", "FLAT") for t in trades)

    # توزيع العملات
    symbols = Counter(t["symbol"] for t in trades if t.get("symbol"))

    # متوسطات رقمية
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    avg_score = sum(_safe_float(t.get("score")) for t in trades) / total

    rr_values = [_safe_float(t.get("rr")) for t in trades if t.get("rr")]
    avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0.0

    risk_vals = [_safe_float(t.get("risk_pct")) for t in trades if t.get("risk_pct")]
    reward_vals = [_safe_float(t.get("reward_pct")) for t in trades if t.get("reward_pct")]

    avg_risk = sum(risk_vals) / len(risk_vals) if risk_vals else 0.0
    avg_reward = sum(reward_vals) / len(reward_vals) if reward_vals else 0.0

    # أفضل 5 عملات من حيث عدد الإشارات
    top_symbols = symbols.most_common(5)

    # بناء النص
    lines: List[str] = []

    lines.append("📊 <b>B7A Ultra Analytics – ملخص أداء الإشارات</b>")
    lines.append("━━━━━━━━━━━━━━━━━━")

    # نظرة عامة
    lines.append("📌 <b>نظرة عامة:</b>")
    lines.append(f"• إجمالي الصفقات المسجّلة: <b>{total}</b>")
    lines.append(
        f"• BUY: <b>{actions.get('BUY', 0)}</b> | SELL: <b>{actions.get('SELL', 0)}</b>"
    )
    lines.append(f"• متوسط قوة الإشارة (Score): <b>{avg_score:.1f}/100</b>")
    if avg_rr > 0:
        lines.append(f"• متوسط نسبة R:R المسجّلة: <b>{avg_rr:.2f}</b>")
    if avg_risk > 0 and avg_reward > 0:
        lines.append(
            f"• متوسط المخاطرة: <b>{avg_risk:.1f}%</b> | "
            f"متوسط هدف الربح: <b>{avg_reward:.1f}%</b>"
        )

    # توزيع الـ Grades
    lines.append("")
    lines.append("🏆 <b>توزيع Grades:</b>")
    for g in ["A+", "A", "B", "C"]:
        if grades.get(g, 0) > 0:
            pct = grades[g] / total * 100
            lines.append(f"• {g}: <b>{grades[g]}</b> ({pct:.1f}%)")

    # وضع السوق العام
    lines.append("")
    lines.append("🌍 <b>أكثر أوضاع السوق تكراراً (Market Regime):</b>")
    for regime, cnt in regimes.most_common():
        pct = cnt / total * 100
        lines.append(f"• {regime}: <b>{cnt}</b> ({pct:.1f}%)")

    # Bias السيولة
    lines.append("")
    lines.append("💧 <b>انحياز السيولة (Liquidity Bias):</b>")
    for bias, cnt in liq_biases.most_common():
        pct = cnt / total * 100
        lines.append(f"• {bias}: <b>{cnt}</b> ({pct:.1f}%)")

    # العملات الأكثر ظهوراً
    if top_symbols:
        lines.append("")
        lines.append("🪙 <b>أكثر العملات ظهوراً في الإشارات:</b>")
        for sym, cnt in top_symbols:
            pct = cnt / total * 100
            lines.append(f"• {sym}: <b>{cnt}</b> إشارة ({pct:.1f}%)")

    lines.append("")
    lines.append("ℹ️ هذا التقرير يعتمد على بيانات ملف اللوق فقط (trades_log.csv).")
    lines.append("🔁 كل ما تستخدم /signal أكثر، التقرير يصير أذكى وأقوى.")

    return "\n".join(lines)
