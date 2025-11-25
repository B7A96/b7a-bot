from telegram import Update
from telegram.ext import ContextTypes

from .engine import generate_signal
from bot.market import get_price


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔥 B7A Trading Bot is LIVE! 🔥")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
🤖 قائمة الأوامر:

/start – تشغيل البوت
/help – عرض هذه القائمة
/price BTC – سعر العملة
/signal BTC – إشارة ذكية (تجريبية)
"""
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
    if len(context.args) == 0:
        await update.message.reply_text(
            "🚨 استخدم الأمر بالشكل التالي:\n"
            "/signal BTC\n"
            "/signal ETH\n"
            "/signal SOL"
        )
        return

    symbol = context.args[0].upper()

    try:
        signal_data = generate_signal(symbol)
    except Exception as e:
        await update.message.reply_text(
            f"⚠️ صار خطأ أثناء توليد الإشارة:\n{e}"
        )
        return

    side = signal_data.get("side", "WAIT")
    last_price = signal_data.get("last_price")
    tp = signal_data.get("tp")
    sl = signal_data.get("sl")
    trend = signal_data.get("trend", "UNKNOWN")
    confidence = signal_data.get("confidence", "LOW")
    pump_risk = signal_data.get("pump_dump_risk", "LOW")
    reason = signal_data.get("reason", "")

    msg = f"📊 إشارة {symbol} من B7A Ultra Bot\n\n"

    if last_price is not None:
        msg += f"السعر الحالي التقريبي: {last_price:.4f} USDT\n"

    msg += f"الاتجاه العام: {trend}\n"
    msg += f"قرار البوت: {side} (ثقة: {confidence})\n"

    if tp is not None:
        msg += f"🎯 هدف الربح (TP): {tp:.4f} USDT\n"
    if sl is not None:
        msg += f"🛡 وقف الخسارة (SL): {sl:.4f} USDT\n"

    if pump_risk and pump_risk != "LOW":
        msg += f"\n⚠️ تحذير: احتمال Pump/Dump = {pump_risk}\n"

    msg += "\n📌 ملاحظة مهمة: الإشارة تجريبية للاختبار فقط، وليست نصيحة استثمارية.\n"

    if reason:
        msg += "\nسبب الإشارة:\n" + reason

    await update.message.reply_text(msg)
