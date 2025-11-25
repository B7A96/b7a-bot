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
/signal – إشارة تجريبية
"""
    await update.message.reply_text(text)

async def price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("استخدم: /price BTC")
        return

    symbol = context.args[0].upper()
    price = get_price(symbol)

    if price:
        await update.message.reply_text(f"💵 سعر {symbol}: {price} USDT")
    else:
        await update.message.reply_text("صار خطأ غير متوقع أثناء جلب السعر 😢")

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 1) نجيب العملة من رسالة المستخدم
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

    # 2) نجيب السعر الحالي من Binance
    price = get_price(symbol)
    if price is None:
        await update.message.reply_text(
            f"⚠️ ما قدرت أجيب سعر {symbol} من Binance حالياً."
        )
        return

    # 3) نولّد إشارة من المحرك الذكي
    signal_data = generate_signal(symbol, price)

    # تتوقع أن generate_signal يرجّع dict مثلاً:
    # {"side": "BUY" أو "SELL", "tp": ..., "sl": ..., "reason": "نص التحليل"}
    side   = signal_data.get("side", "N/A")
    tp     = signal_data.get("tp")
    sl     = signal_data.get("sl")
    reason = signal_data.get("reason", "")

    # 4) نرسل النتيجة للمستخدم
    msg = (
        f"📊 إشارة {symbol} من B7A Ultra Bot\n\n"
        f"السعر الحالي: {price:.4f} USDT\n"
        f"الإتجاه: {side}\n"
    )

    if tp is not None:
        msg += f"🎯 هدف الربح (TP): {tp}\n"
    if sl is not None:
        msg += f"🛡 وقف الخسارة (SL): {sl}\n"

    if reason:
        msg += "\n📌 سبب الإشارة:\n" + reason

    await update.message.reply_text(msg)
