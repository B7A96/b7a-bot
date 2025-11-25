from telegram import Update
from telegram.ext import ContextTypes

from .engine import generate_signal
from bot.market import get_price  # مثل ما هو

# ... start / help / price نفس ما عندك ...


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 1) العملة
    if len(context.args) == 0:
        await update.message.reply_text(
            "🚨 استخدم الأمر بالشكل التالي:\n"
            "/signal BTC\n"
            "/signal ETH\n"
            "/signal SOL"
        )
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(
        f"⏳ جارِ تحليل السوق لـ {symbol} عبر B7A Ultra Engine ..."
    )

    try:
        signal_data = generate_signal(symbol)
    except Exception as e:
        print("Signal error:", e)
        await update.message.reply_text(
            "❌ صار خطأ أثناء توليد الإشارة، جرّب بعد شوي أو مع عملة ثانية."
        )
        return

    # 2) تفكيك البيانات
    decision   = signal_data.get("decision", {})
    tf_data    = signal_data.get("timeframes", {})
    last_price = signal_data.get("last_price")
    reason     = signal_data.get("reason", "")

    action     = decision.get("action", "WAIT")
    score      = decision.get("score", 50)
    trend      = decision.get("trend", "RANGING")
    confidence = decision.get("confidence", "LOW")
    pump_risk  = decision.get("pump_dump_risk", "LOW")

    tp         = signal_data.get("tp")
    sl         = signal_data.get("sl")
    rr         = signal_data.get("rr")
    risk_pct   = signal_data.get("risk_pct")
    reward_pct = signal_data.get("reward_pct")

    # 3) ملخص الفريمات (مثل ما هو عندك تقريباً)
    lines = []
    for tf_name in ["15m", "1h", "4h", "1d"]:
        tf = tf_data.get(tf_name)
        if not tf:
            continue

        tf_trend    = tf.get("trend", "UNKNOWN")
        tf_score    = tf.get("trend_score", 50)
        tf_rsi      = tf.get("rsi")
        tf_change_1 = tf.get("change_1")
        tf_change_4 = tf.get("change_4")

        line = f"• {tf_name}: {tf_trend} | Score: {tf_score:.0f}"

        if tf_rsi is not None and not str(tf_rsi) == "nan":
            line += f" | RSI: {tf_rsi:.1f}"

        if tf_change_1 is not None:
            line += f" | تغير آخر شمعة: {tf_change_1:+.2f}%"

        if tf_change_4 is not None:
            line += f" | تغير قصير المدى: {tf_change_4:+.2f}%"

        lines.append(line)

    tf_summary = "\n".join(lines) if lines else "لا يوجد بيانات كافية لكل الفريمات."

    # 4) بناء الرسالة النهائية
    msg = f"📈 إشارة {signal_data.get('symbol', symbol)} من B7A Ultra Bot 🇰🇼\n\n"

    if last_price is not None:
        msg += f"USDT {last_price:.4f} السعر الحالي:\n\n"

    msg += (
        f"قرار النظام: {action}\n"
        f"الاتجاه العام: {trend}\n"
        f"قوة الإشارة (Score): {score}/100\n"
        f"درجة الثقة: {confidence}\n"
        f"مخاطرة حركة حادة (Pump/Dump): {pump_risk}\n"
    )

    # ✅ عرض TP/SL و R:R وخطة المخاطرة
    if tp is not None and sl is not None:
        msg += "\n🎯 خطة الصفقة (آلية):\n"
        msg += f"• وقف الخسارة (SL): {sl:.4f}\n"
        msg += f"• هدف الربح (TP): {tp:.4f}\n"
        if risk_pct is not None and reward_pct is not None:
            msg += f"• مخاطرة تقريبية: {risk_pct:.1f}% | هدف ربح: {reward_pct:.1f}%\n"
        if rr is not None:
            msg += f"• نسبة العائد إلى المخاطرة R:R ≈ {rr}:1\n"
    else:
        msg += "\n(لم يتم حساب TP/SL لهذه الإشارة بسبب نقص البيانات.)\n"

    msg += "\n🧠 ملخص الفريمات:\n" + tf_summary

    if reason:
        msg += "\n\n📌 سبب الإشارة (ملخص ذكي):\n" + reason

    msg += "\n\n⚠️ هذه ليست نصيحة استثمارية، استخدم إدارة مخاطر دائماً."

    await update.message.reply_text(msg)
