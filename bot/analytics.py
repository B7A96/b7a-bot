from typing import Dict, Any, List
import csv
import os
from statistics import mean


LOG_PATH = "trades_log.csv"


def _load_trades() -> List[Dict[str, Any]]:
    """
    يقرأ ملف trades_log.csv ويعيده كقائمة من الدكت.
    لو الملف مو موجود أو فاضي يرجع قائمة فاضية.
    """
    if not os.path.isfile(LOG_PATH):
        return []

    rows: List[Dict[str, Any]] = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # نحاول نحول الأرقام
            try:
                row["price"] = float(row.get("price") or 0)
                row["tp"] = float(row.get("tp") or 0)
                row["sl"] = float(row.get("sl") or 0)
                row["rr"] = float(row.get("rr") or 0)
                row["score"] = float(row.get("score") or 0)
            except Exception:
                pass
            rows.append(row)
    return rows


def get_trades_summary() -> str:
    """
    يرجع نص جاهز للإرسال في تيليجرام فيه ملخص أداء الإشارات.
    """
    trades = _load_trades()
    if not trades:
        return "📊 ما في بيانات في اللوق حالياً.\nجرّب تستخدم /signal كم مرة وبعدين استخدم /stats."

    total = len(trades)

    buy = sum(1 for t in trades if (t.get("action") or "").upper() == "BUY")
    sell = sum(1 for t in trades if (t.get("action") or "").upper() == "SELL")
    wait = sum(1 for t in trades if (t.get("action") or "").upper() == "WAIT")

    grades: Dict[str, int] = {}
    for t in trades:
        g = (t.get("grade") or "NA").upper()
        grades[g] = grades.get(g, 0) + 1

    # درجات السكور
    scores = [float(t.get("score") or 0) for t in trades]
    avg_score = mean(scores) if scores else 0.0
    high_conf = sum(1 for t in trades if (t.get("confidence") or "").upper() == "HIGH")
    med_conf = sum(1 for t in trades if (t.get("confidence") or "").upper() == "MEDIUM")
    low_conf = sum(1 for t in trades if (t.get("confidence") or "").upper() == "LOW")

    # أنظمة السوق
    trending = sum(1 for t in trades if (t.get("market_regime") or "").upper() == "TRENDING")
    ranging = sum(1 for t in trades if (t.get("market_regime") or "").upper() == "RANGING")
    mixed = sum(1 for t in trades if (t.get("market_regime") or "").upper() == "MIXED")

    # صفقات No-Trade
    no_trade = 0
    for t in trades:
        val = str(t.get("no_trade", "")).strip().lower()
        if val in ("1", "true", "yes"):
            no_trade += 1

    with_trade = total - no_trade

    # بناء النص
    lines: List[str] = []
    lines.append("📊 B7A Ultra Analytics – ملخص أداء الإشارات")
    lines.append("")
    lines.append(f"عدد الإشارات المسجّلة: {total}")
    lines.append(f"• صفقات فعلية (ليست No-Trade): {with_trade}")
    lines.append(f"• مناطق No-Trade Zone: {no_trade}")
    lines.append("")
    lines.append("توزيع نوع القرار:")
    lines.append(f"• BUY: {buy}")
    lines.append(f"• SELL: {sell}")
    lines.append(f"• WAIT فقط: {wait}")
    lines.append("")
    lines.append(f"متوسط Score الكلي: {avg_score:.1f}/100")
    lines.append("توزيع الثقة:")
    lines.append(f"• HIGH: {high_conf}")
    lines.append(f"• MEDIUM: {med_conf}")
    lines.append(f"• LOW: {low_conf}")
    lines.append("")
    lines.append("توزيع Grade:")
    for g, cnt in sorted(grades.items()):
        lines.append(f"• {g}: {cnt}")
    lines.append("")
    lines.append("وضع السوق أثناء الإشارات:")
    lines.append(f"• TRENDING: {trending}")
    lines.append(f"• RANGING: {ranging}")
    lines.append(f"• MIXED: {mixed}")
    lines.append("")
    lines.append("💡 ملاحظة:")
    lines.append("هذا الملخص لا يحسب الربح/الخسارة الفعلي،")
    lines.append("لكن يعطيك صورة عن جودة الفلتر وسلوك البوت.")
    lines.append("لاحقاً نستخدم نفس اللوق لتطوير استراتيجيات أدق.")

    return "\n".join(lines)
