import os
import logging
import requests
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# ---------- الإعدادات الأساسية ----------
TOKEN = os.getenv("TELEGRAM_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# خريطة بسيطة من الرمز إلى CoinGecko ID
COIN_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "TON": "the-open-network",
}


# ---------- أوامر البوت ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """أمر /start"""
    text = (
        "🔥 B7A Trading Bot is LIVE! 🔥\n\n"
        "استخدم /help لعرض قائمة الأوامر المتاحة."
    )
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """أمر /help"""
    text = (
        "🤖 أوامر B7A Ultra Bot:\n\n"
        "/start - تشغيل البوت والترحيب\n"
        "/help - عرض هذه القائمة\n"
        "/price <رمز العملة> - سعر العملة بالدولار (مثال: /price BTC)\n"
        "/signal - إشارة تجريبية (سنربطها لاحقًا مع SniperFlow)\n"
    )
    await update.message.reply_text(text)


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """أمر /price"""
    try:
        if not context.args:
            await update.message.reply_text(
                "استخدم الأمر بهالشكل:\n/price BTC\n/price ETH\n/price SOL"
            )
            return

        symbol = context.args[0].upper()
        coin_id = COIN_MAP.get(symbol)

        if not coin_id:
            await update.message.reply_text(
                f"ما عرفت العملة: {symbol} 😅\n"
                "جرّب مثل: BTC, ETH, SOL, BNB, XRP, DOGE, TON"
            )
            return

        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": coin_id, "vs_currencies": "usd"}

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        price = data.get(coin_id, {}).get("usd")
        if price is None:
            await update.message.reply_text("ما قدرت أجيب السعر الآن، حاول بعد شوي 🙏")
            return

        await update.message.reply_text(
            f"💰 سعر {symbol} الحالي: {price:,.2f} دولار"
        )

    except Exception as e:
        logger.exception("Error in /price command")
        await update.message.reply_text("صار خطأ غير متوقع أثناء جلب السعر 😔")


async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """أمر /signal (نسخة تجريبية)"""
    text = (
        "📡 إشارة تجريبية من B7A Ultra Bot:\n\n"
        "العملة: BTC\n"
        "الاتجاه: صعود (LONG)\n"
        "منطقة الدخول: 85,000 - 86,000\n"
        "منطقة جني الربح: 90,000\n"
        "منطقة وقف الخسارة: 83,500\n\n"
        "⚠️ هذه ليست نصيحة استثمارية، فقط مثال تجريبي.\n"
        "قريبًا سنربط البوت مع SniperFlow لإشارات حقيقية 🔥"
    )
    await update.message.reply_text(text)


# ---------- تشغيل البوت ----------

def main() -> None:
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is not set in environment variables")

    app = ApplicationBuilder().token(TOKEN).build()

    # ربط الأوامر
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("price", price_command))
    app.add_handler(CommandHandler("signal", signal_command))

    print("B7A BOT starting Telegram service...")
    app.run_polling(drop_pending_updates=True)
    # drop_pending_updates=True يساعد يقلل مشاكل الـ Conflict مع الرسائل القديمة


if __name__ == "__main__":
    main()
