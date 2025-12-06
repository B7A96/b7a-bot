# bot/analytics.py

import csv
import os
from collections import Counter
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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


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


# =========================
# B7A Performance Intel
# =========================

def performance_intel(symbol: str, decision: Dict[str, Any]) -> Dict[str, Any]:
    """
    ذكاء داخلي يعتمد على ملف اللوق:
    - Weapon 1: يتعلم من تاريخ نفس الزوج (WIN/LOSS).
    - Weapon 2: يراقب أوضاع السوق اللي تكررت فيها الخسارة ويمنعها.
    - Weapon 3: يضبط حجم المخاطرة (risk_multiplier).
    """
    trades = _read_trades()
    if not trades:
        # ما في بيانات → لا تغيير
        return {
            "score_delta": 0.0,
            "risk_multiplier": 1.0,
            "force_no_trade": False,
            "note": None,
        }

    action = decision.get("action")
    regime_now = decision.get("market_regime")
    liq_now = decision.get("liquidity_bias")
    grade_now = decision.get("grade")

    # نركز على آخر 100 صفقة لنفس الزوج ونفس الـ Action
    filtered = [
        t for t in trades
        if t.get("symbol") == symbol
        and t.get("action") == action
    ]
    filtered = filtered[-100:]

    # لو مافي صفقات سابقة → نرجع نيترال
    if not filtered:
        return {
            "score_delta": 0.0,
            "risk_multiplier": 1.0,
            "force_no_trade": False,
            "note": None,
        }

    # نحسب نتائج WIN / LOSS لو موجودة
    wins = [t for t in filtered if str(t.get("result", "")).upper() == "WIN"]
    losses = [t for t in filtered if str(t.get("result", "")).upper() == "LOSS"]
    total_with_result = len(wins) + len(losses)

    # لو ما تم تسجيل نتائج إلى الآن → نستخدم فقط R:R كمؤشر خفيف
    if total_with_result == 0:
        rr_vals = [_safe_float(t.get("rr")) for t in filtered if t.get("rr")]
        avg_rr = sum(rr_vals) / len(rr_vals) if rr_vals else 1.0

        if avg_rr < 0.9:
            return {
                "score_delta": -3.0,
                "risk_multiplier": 0.8,
                "force_no_trade": False,
                "note": "📉 Performance Filter: هذا الزوج أعطى تاريخياً R:R ضعيف، تم تقليل المخاطرة.",
            }
        else:
            return {
                "score_delta": 0.0,
                "risk_multiplier": 1.0,
                "force_no_trade": False,
                "note": None,
            }

    win_rate = len(wins) / total_with_result

    # إحصائيات حسب وضع السوق الحالي والسيولة
    regime_trades = [
        t for t in filtered
        if t.get("market_regime") == regime_now
        and t.get("liquidity_bias") == liq_now
        and str(t.get("result", "")).upper() in ("WIN", "LOSS")
    ]
    regime_wins = [t for t in regime_trades if str(t.get("result", "")).upper() == "WIN"]
    regime_losses = [t for t in regime_trades if str(t.get("result", "")).upper() == "LOSS"]
    regime_total = len(regime_wins) + len(regime_losses)
    regime_win_rate = (len(regime_wins) / regime_total) if regime_total > 0 else None

    # متوسط R:R لنفس الزوج
    rr_vals_all = [_safe_float(t.get("rr")) for t in filtered if t.get("rr")]
    avg_rr_all = sum(rr_vals_all) / len(rr_vals_all) if rr_vals_all else 1.0

    score_delta = 0.0
    risk_multiplier = 1.0
    force_no_trade = False
    note_parts: List[str] = []

    # Weapon 1: تعلم عام من أداء الزوج
    if total_with_result >= 12:
        if win_rate < 0.4:
            if grade_now in ("C", "B"):
                force_no_trade = True
                note_parts.append("⛔ Performance Filter: هذا الزوج خسر كثيراً في الماضي في إشارات مشابهة – تم حظره مؤقتاً.")
            else:
                score_delta -= 7.0
                risk_multiplier *= 0.6
                note_parts.append("⚠️ Performance Filter: نسبة نجاح هذا الزوج ضعيفة، تم تقليل السكور والمخاطرة.")
        elif win_rate > 0.65:
            score_delta += 4.0
            risk_multiplier *= 1.2
            note_parts.append("✅ Performance Boost: هذا الزوج أثبت أداء جيد تاريخياً، تم تعزيز السكور والمخاطرة قليلاً.")

    # Weapon 2: فلتر أوضاع السوق/السيولة
    if regime_total and regime_win_rate is not None:
        if regime_win_rate < 0.35 and regime_total >= 6:
            score_delta -= 5.0
            risk_multiplier *= 0.7
            note_parts.append(
                f"🧱 Market Memory: وضع السوق [{regime_now}/{liq_now}] سجل خسائر متكررة ({regime_win_rate*100:.0f}%)."
            )
            if grade_now in ("B", "C"):
                force_no_trade = True

    # Weapon 3: ضبط حجم الصفقة حسب R:R التاريخي
    if avg_rr_all < 0.9:
        risk_multiplier *= 0.8
        note_parts.append("📉 Historical R:R ضعيف، تم تخفيض حجم الصفقة.")
    elif avg_rr_all > 1.5 and win_rate and win_rate > 0.55:
        risk_multiplier *= 1.2
        note_parts.append("📈 Historical R:R ممتاز، تم رفع حجم الصفقة بشكل محسوب.")

    # حدود منطقية للمخاطرة
    risk_multiplier = max(0.5, min(1.8, risk_multiplier))

    note = " | ".join(note_parts) if note_parts else None

    return {
        "score_delta": float(score_delta),
        "risk_multiplier": float(risk_multiplier),
        "force_no_trade": bool(force_no_trade),
        "note": note,
    }
    
# =========================
# B7A Trade Result Trainer
# =========================

def mark_last_trade(symbol: str, result: str) -> bool:
    """
    يعلّم النظام نتيجة آخر صفقة على رمز معيّن (WIN / LOSS).

    symbol: مثل "BTCUSDT"
    result: "WIN" أو "LOSS"
    """
    rows = _read_trades()
    if not rows:
        return False

    symbol = (symbol or "").upper()
    target_idx = None

    # نبحث من آخر صف إلى أول صف عن آخر صفقة لنفس الرمز
    for i in range(len(rows) - 1, -1, -1):
        row = rows[i]
        if str(row.get("symbol", "")).upper() == symbol:
            target_idx = i
            break

    if target_idx is None:
        return False

    rows[target_idx]["result"] = result.upper()

    # نعيد كتابة الملف بالكامل مع التعديل
    fieldnames = list(rows[0].keys())
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return True
    
# ============================================
#  B7A Ultra — Stats Analyzer (Trainer + Win/Loss)
# ============================================

def analyze_stats(chat_data: dict) -> str:
    """
    يحسب نتائج المتداول بناءً على /win و /loss
    ويعرض ملخص أداء البوت + أداء العملات.
    """
    trainer = chat_data.get("trainer", {})
    if not trainer:
        return "📊 لا توجد بيانات بعد. استخدم /win و /loss لتسجيل نتائج الصفقات."

    total_wins = 0
    total_losses = 0

    lines = []
    lines.append("📊 <b>B7A Ultra — Performance Stats</b>\n")

    for sym, record in trainer.items():
        wins = record.get("wins", 0)
        losses = record.get("losses", 0)

        total_wins += wins
        total_losses += losses

        total = wins + losses
        if total > 0:
            win_rate = (wins / total) * 100
        else:
            win_rate = 0.0

        lines.append(
            f"• {sym}: {wins} ربح / {losses} خسارة — معدل نجاح <b>{win_rate:.1f}%</b>"
        )

    lines.append("\n— — — — —")

    total_all = total_wins + total_losses
    if total_all > 0:
        global_wr = (total_wins / total_all) * 100
    else:
        global_wr = 0.0

    lines.append(f"📈 إجمالي الأرباح: <b>{total_wins}</b>")
    lines.append(f"📉 إجمالي الخسائر: <b>{total_losses}</b>")
    lines.append(f"🏁 <b>المعدل العام للنجاح: {global_wr:.1f}%</b>")

    return "\n".join(lines)

