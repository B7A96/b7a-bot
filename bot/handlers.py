from telegram import Update
from telegram.ext import ContextTypes

from .engine import generate_signal
from bot.market import get_price  # أو from .market لو كنت مشغل البوت كحزمة داخلية


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


# /signal  (B7A Ultra Engine)
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
    await update.message.reply_text(f"⏳ جارِ تحليل السوق لـ {symbol} عبر B7A Ultra Engine ...")

    try:
        # المحرك يجلب بيانات Binance بنفسه
        signal_data = generate_signal(symbol)
    except Exception as e:
        await update.message.reply_text(
            "❌ صار خطأ أثناء توليد الإشارة، جرّب بعد شوي أو مع عملة ثانية."
        )
        print("Signal error:", e)
        return

    # نفكّك البيانات الراجعة من المحرك
    decision   = signal_data.get("decision", {})
    tf_data    = signal_data.get("timeframes", {})
    last_price = signal_data.get("last_price")
    reason     = signal_data.get("reason", "")

    action     = decision.get("action", "WAIT")
    score      = decision.get("score", 50)
    trend      = decision.get("trend", "RANGING")
    confidence = decision.get("confidence", "LOW")
    pump_risk  = decision.get("pump_dump_risk", "LOW")

    # ملخص الفريمات
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
            line += f" | 1C: {tf_change_1:+.2f}%"

        if tf_change_4 is not None:
            line += f" | 4C: {tf_change_4:+.2f}%"

        lines.append(line)

    tf_summary = "\n".join(lines) if lines else "لا يوجد بيانات كافية لكل الفريمات."

    # نبني الرسالة النهائية
    msg_lines = []

    msg_lines.append(f"📊 إشارة B7A Ultra لـ {signal_data.get('symbol', symbol)}\n")

    if last_price is not None:
        msg_lines.append(f"السعر الحالي: {last_price:.4f} USDT\n")

    msg_lines.append(f"قرار النظام: {action}")
    msg_lines.append(f"الإتجاه العام: {trend}")
    msg_lines.append(f"قوة الإشارة (Score): {score}/100")
    msg_lines.append(f"درجة الثقة: {confidence}")
    msg_lines.append(f"مخاطرة Pump/Dump: {pump_risk}\n")

    msg_lines.append("🧠 ملخص الفريمات:")
    msg_lines.append(tf_summary)

    if reason:
        msg_lines.append("\n📌 سبب الإشارة (ملخص ذكي):")
        msg_lines.append(reason)

    msg_lines.append("\n⚠️ هذه ليست نصيحة استثمارية، استخدم إدارة مخاطر دائماً.")

    await update.message.reply_text("\n".join(msg_lines))
