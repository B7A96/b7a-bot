from telegram import Update
from telegram.ext import ContextTypes

from .engine import generate_signal
from bot.market import get_price  # أو من .market إذا حبيت

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
    await update.message.reply_text(f"⏳ جارِ تحليل السوق لـ {symbol} عبر B7A Ultra Engine ...")

    try:
        # 2) نولّد إشارة من المحرك الذكي (ما نرسل السعر هنا، الدالة هي اللي تجيب كل البيانات)
        signal_data = generate_signal(symbol)
    except Exception as e:
        await update.message.reply_text(
            "❌ صار خطأ أثناء توليد الإشارة، جرّب بعد شوي أو مع عملة ثانية."
        )
        # optional: اطبع الخطأ في اللوجات
        print("Signal error:", e)
        return

    # 3) نفكك البيانات الراجعة من المحرك
    decision = signal_data.get("decision", {})
    tf_data  = signal_data.get("timeframes", {})
    last_price = signal_data.get("last_price")
    reason = signal_data.get("reason", "")

    action      = decision.get("action", "WAIT")
    score       = decision.get("score", 50)
    trend       = decision.get("trend", "RANGING")
    confidence  = decision.get("confidence", "LOW")
    pump_risk   = decision.get("pump_dump_risk", "LOW")

    # 4) ملخص الفريمات
    lines = []
    for tf_name in ["15m", "1h", "4h", "1d"]:
        tf = tf_data.get(tf_name)
        if not tf:
            continue

        tf_trend = tf.get("trend", "UNKNOWN")
        tf_score = tf.get("trend_score", 50)
        tf_rsi   = tf.get("rsi")
        tf_change_1 = tf.get("change_1")
        tf_change_4 = tf.get("change_4")

        line = f"• {tf_name}: {tf_trend} | Score: {tf_score:.0f}"

        if tf_rsi is not None and not str(tf_rsi) == "nan":
            line += f" | RSI: {tf_rsi:.1f}"

        if tf_change_1 is not None:
            line += f" | Δ1c: {tf_change_1:+.2f}%"

        if tf_change_4 is not None:
            line += f" | Δ4c: {tf_change_4:+.2f}%"

        lines.append(line)

    tf_summary = "\n".join(lines) if lines else "لا يوجد بيانات كافية لكل الفريمات."

    # 5) نبني الرسالة النهائية
    msg = f"📊 إشارة B7A Ultra لـ {signal_data.get('symbol', symbol)}\n\n"

    if last_price is not None:
        msg += f"السعر الحالي: {last_price:.4f} USDT\n\n"

    msg += (
        f"قرار النظام: {action}\n"
        f"الإتجاه العام: {trend}\n"
        f"قوة الإشارة (Score): {score}/100\n"
        f"درجة الثقة: {confidence}\n"
        f"مخاطرة Pump/Dump: {pump_risk}\n\n"
    )

    msg += "🧠 ملخص الفريمات:\n" + tf_summary

    if reason:
        msg += "\n\n📌 سبب الإشارة (ملخص ذكي):\n" + reason

    msg += "\n\n⚠️ هذه ليست نصيحة استثمارية، استخدم إدارة مخاطر دائماً."

    await update.message.reply_text(msg)
