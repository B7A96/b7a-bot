import logging
from telegram import Update
from telegram.ext import ContextTypes

from .market import get_price_usd, generate_demo_signal

logger = logging.getLogger(__name__)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "🔥 B7A Trading Bot is LIVE! 🔥\n\n"
        "استخدم /help لعرض قائمة الأوامر المتاحة."
    )
    await update.message.reply_text(text)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "🤖 أوامر B7A Ultra Bot:\n\n"
        "/start - تشغيل البوت والترحيب\n"
        "/help - عرض هذه القائمة\n"
        "/price <رمز العملة> - سعر العملة بالدولار (مثال: /price BTC)\n"
        "/signal - إشارة تجريبية (سنربطها لاحقًا مع SniperFlow)\n"
    )
    await update.message.reply_text(text)


async def cmd_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if not context.args:
            await update.message.reply_text(
                "استخدم الأمر بهالشكل:\n/price BTC\n/price ETH\n/price SOL"
            )
            return

        symbol = context.args[0].upper()
        price = get_price_usd(symbol)

        if price is None:
            await update.message.reply_text(
                f"ما عرفت العملة: {symbol} 😅\n"
                "جرّب مثل: BTC, ETH, SOL, BNB, XRP, DOGE, TON"
            )
            return

        await update.message.reply_text(
            f"💰 سعر {symbol} الحالي: {price:,.2f} دولار"
        )

    except Exception:
        logger.exception("Error in /price command")
        await update.message.reply_text("صار خطأ غير متوقع أثناء جلب السعر 😔")


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    sig = generate_demo_signal()

    text = (
        "📡 إشارة تجريبية من B7A Ultra Bot:\n\n"
        f"العملة: {sig['symbol']}\n"
        f"الاتجاه: {sig['direction']} (LONG)\n"
        f"منطقة الدخول: {sig['entry']}\n"
        f"منطقة جني الربح: {sig['take_profit']}\n"
        f"منطقة وقف الخسارة: {sig['stop_loss']}\n\n"
        "⚠️ هذه ليست نصيحة استثمارية، فقط مثال تجريبي.\n"
        "قريبًا سنربط البوت مع SniperFlow لإشارات حقيقية 🔥"
    )
    await update.message.reply_text(text)
