import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

TOKEN = os.getenv("TELEGRAM_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔥 B7A Trading Bot is LIVE! 🔥")

if __name__ == "__main__":
    print("B7A BOT starting Telegram service...")

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))

    print("B7A BOT is running on Telegram...")
   app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


