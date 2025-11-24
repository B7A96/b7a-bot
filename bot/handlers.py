from telegram import Update
from telegram.ext import ContextTypes

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
    text = """
📈 إشارة تجريبية من B7A Ultra Bot:

العملة: BTC
الاتجاه: (LONG)
منطقة الدخول: 86,000 - 85,000
منطقة أخذ الربح: 90,000
منطقة وقف الخسارة: 83,500

⚠️ مثال تجريبي فقط. إشارات SniperFlow قادمة 🔥
"""
    await update.message.reply_text(text)
