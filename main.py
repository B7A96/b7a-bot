import os
import logging
from telegram.ext import ApplicationBuilder, CommandHandler

from bot.handlers import cmd_start, cmd_help, cmd_price, cmd_signal

TOKEN = os.getenv("TELEGRAM_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is not set in environment variables")

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("price", cmd_price))
    app.add_handler(CommandHandler("signal", cmd_signal))

    print("B7A BOT starting Telegram service...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
