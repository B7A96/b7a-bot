import os
from telegram.ext import ApplicationBuilder, CommandHandler
from bot.handlers import start, help_command, price, signal

TOKEN = os.getenv("TELEGRAM_TOKEN")

if __name__ == "__main__":
    print("B7A BOT starting Telegram service...")

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("price", price))
    app.add_handler(CommandHandler("signal", signal))

    print("B7A BOT is running on Telegram...")
    app.run_polling(drop_pending_updates=True)
