from telegram import Update
from telegram.ext import ContextTypes

from .engine import generate_signal
from bot.market import get_price


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔥 B7A Trading Bot is LIVE! 🔥")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 قائمة الأوامر:\n\n"
        "/start – تشغيل البوت\n"
        "/help – عرض هذه القائمة\n"
        "/price BTC – سعر العملة (سبوت)\n"
        "/signal BTC – إشارة تحليل احترافية من Ultra Engine\n"
    )
    await update.message.reply_text(text)


async def price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("استخدم: /price BTC")
        return

    symbol = context.args[0].upper()
    price_value = get_price(symbol)

    if price_value:
        await update.message.reply_text(f"💵 سعر {symbol}: {price_value} USDT")
    else:
        await update.message.reply_text("صار خطأ غير متوقع أثناء جلب السعر 😢")


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # مثال: /signal BTC
    if len(context.args) == 0:
        await update.message.reply_text(
            "🚨 استخدم الأمر بالشكل التالي:\n"
            "/signal BTC\n"
            "/signal ETH\n"
            "/signal SOL"
        )
        return

    symbol = context.args[0].upper()

    # رسالة مبدئية
    await update.message.reply_text("⏳ جاري تحليل السوق باستخدام B7A Ultra Engine...")

    try:
        data = generate_signal(symbol)   # ✅ الآن ياخذ رمز واحد فقط

        decision = data["decision"]
        price_value = data.get("last_price")
        reason = data.get("reason", "")
        tfs = data.get("timeframes", {})

        # نحاول نبرز أهم الفريمات (1h و 4h مثلاً)
        tf_summary_lines = []
        for tf in ["15m", "1h", "4h", "1d"]:
            d = tfs.get(tf)
            if not d:
                continue
            tf_summary_lines.append(
                f"• {tf}: ترند {d.get('trend', 'N/A')} | سكور {int(d.get('trend_score', 50))}"
            )
        tf_summary = "\n".join(tf_summary_lines) if tf_summary_lines else "ما توفرت بيانات كافية من Binance."

        text = (
            f"📊 *B7A Ultra Signal*\n"
            f"العملة: *{symbol}*\n\n"
        )

        if price_value is not None:
            text += f"السعر الحالي: `{price_value}` USDT\n\n"

        text += (
            f"الاتجاه العام: *{decision['trend']}*\n"
            f"الإجراء المقترح: *{decision['action']}*\n"
            f"درجة الثقة: *{decision['confidence']}*\n"
            f"مخاطر Pump/Dump: *{decision['pump_dump_risk']}*\n\n"
            f"🕒 ملخص الفريمات:\n{tf_summary}\n\n"
        )

        if reason:
            text += f"📌 سبب الإشارة:\n{reason}"

        await update.message.reply_text(text, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"❌ فشل أثناء التحليل: {str(e)}")
