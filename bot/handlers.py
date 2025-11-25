from telegram import Update
from telegram.ext import ContextTypes

from .engine import generate_signal
from .market import get_price


# /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔥 B7A Trading Bot is LIVE! 🔥")


# /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
🤖 قائمة الأوامر:

/start – تشغيل البوت
/help – عرض هذه القائمة
/price BTC – سعر العملة (مثال: /price sol)
/signal BTC – إشارة تحليل احترافية (مثال: /signal eth)
"""
    await update.message.reply_text(text)


# /price
async def price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("استخدم: /price BTC أو /price sol")
        return

    symbol = context.args[0].upper()
    value = get_price(symbol)

    if value:
        await update.message.reply_text(f"💵 سعر {symbol}: {value} USDT")
    else:
        await update.message.reply_text("صار خطأ غير متوقع أثناء جلب السعر 😢")


# /signal  (Ultra AI)
async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 1) نقرأ العملة من الأمر
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
        # 2) نولّد إشارة من المحرك الذكي
        signal_data = generate_signal(symbol)
    except Exception as e:
        await update.message.reply_text(
            "❌ صار خطأ أثناء توليد الإشارة، جرّب بعد شوي أو مع عملة ثانية."
        )
        print("Signal error:", e)
        return

    # 3) نفكك البيانات الراجعة من المحرك
    decision = signal_data.get("decision", {})
    tf_data = signal_data.get("timeframes", {})
    last_price = signal_data.get("last_price")
    reason = signal_data.get("reason", "")

    action = decision.get("action", "WAIT")
    score = decision.get("score", 50)
    trend = decision.get("trend", "RANGING")
    confidence = decision.get("confidence", "LOW")
    pump_risk = decision.get("pump_dump_risk", "LOW")

    tp = signal_data.get("tp")
    sl = signal_data.get("sl")
    rr = signal_data.get("rr")
    risk_pct = signal_data.get("risk_pct")
    reward_pct = signal_data.get("reward_pct")

    # 4) ملخص الفريمات
    lines = []
    for tf_name in ["15m", "1h", "4h", "1d"]:
        tf = tf_data.get(tf_name)
        if not tf:
            continue

        tf_trend = tf.get("trend", "UNKNOWN")
        tf_score = tf.get("trend_score", 50)
        tf_rsi = tf.get("rsi")
        tf_change_1 = tf.get("change_1")
        tf_change_4 = tf.get("change_4")
        tf_vol_surge = tf.get("volume_surge", False)

        line = f"• {tf_name}: {tf_trend} | Score: {tf_score:.0f}"

        if tf_rsi is not None and not str(tf_rsi) == "nan":
            line += f" | RSI: {tf_rsi:.1f}"

        if tf_change_1 is not None:
            line += f" | تغير آخر شمعة: {tf_change_1:+.2f}%"

        if tf_change_4 is not None:
            line += f" | تغير قصير المدى: {tf_change_4:+.2f}%"

        if tf_vol_surge:
            line += " | حجم تداول قوي 🔥"

        lines.append(line)

    tf_summary = "\n".join(lines) if lines else "لا يوجد بيانات كافية لكل الفريمات."

    # 5) نبني الرسالة النهائية
    msg_lines = []

    msg_lines.append(
        f"📊 إشارة B7A Ultra لـ {signal_data.get('symbol', symbol)}"
    )
    msg_lines.append("")

    if last_price is not None:
        msg_lines.append(f"السعر الحالي: {last_price:.4f} USDT")
        msg_lines.append("")

    # قرار أساسي
    msg_lines.append(f"قرار النظام: {action}")
    msg_lines.append(f"الاتجاه العام: {trend}")
    msg_lines.append(f"قوة الإشارة (Score): {score}/100")
    msg_lines.append(f"درجة الثقة: {confidence}")
    msg_lines.append(f"مخاطرة حركة حادة (Pump/Dump): {pump_risk}")
    msg_lines.append("")

    # خطة الصفقة (TP / SL)
    if action in ("BUY", "SELL") and tp is not None and sl is not None:
        نوع_الصفقة = "شراء" if action == "BUY" else "بيع"
        msg_lines.append("🎯 خطة الصفقة:")
        msg_lines.append(f"• نوع الصفقة: {نوع_الصفقة}")
        msg_lines.append(f"• هدف الربح (TP): {tp} USDT")
        msg_lines.append(f"• وقف الخسارة (SL): {sl} USDT")

        if risk_pct is not None and reward_pct is not None:
            msg_lines.append(
                f"• المخاطرة التقريبية: -{risk_pct:.1f}% | الهدف التقريبي: +{reward_pct:.1f}%"
            )

        if rr is not None:
            msg_lines.append(f"• نسبة العائد للمخاطرة (R/R): {rr}:1")

        msg_lines.append("")

    # ملخص الفريمات
    msg_lines.append("🧠 ملخص الفريمات:")
    msg_lines.append(tf_summary)

    # سبب الإشارة الذكي
    if reason:
        msg_lines.append("")
        msg_lines.append("📌 سبب الإشارة (ملخص ذكي):")
        msg_lines.append(reason)

    msg_lines.append("")
    msg_lines.append("⚠️ هذه ليست نصيحة استثمارية، استخدم إدارة مخاطر دائماً.")

    await update.message.reply_text("\n".join(msg_lines))
