import os

from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
)

from bot.handlers import (
    start,
    help_command,
    price,
    signal,
    scan,
    scan_watchlist,
    daily,
    refresh_signal,
    add_symbol,
    remove_symbol,
    list_watchlist,
    stats,
    radar,  # <-- أضفناها هنا
)

TOKEN = os.getenv("TELEGRAM_TOKEN")


if __name__ == "__main__":
    print("B7A BOT starting Telegram service...")

    app = ApplicationBuilder().token(TOKEN).build()

    # أوامر أساسية
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("price", price))
    app.add_handler(CommandHandler("signal", signal))

    # أوامر سكانية / تقارير
    app.add_handler(CommandHandler("scan", scan))
    app.add_handler(CommandHandler("scan_watchlist", scan_watchlist))
    app.add_handler(CommandHandler("daily", daily))
    app.add_handler(CommandHandler("radar", radar))  # <-- هنا

    # إدارة الـ Watchlist
    app.add_handler(CommandHandler("add", add_symbol))
    app.add_handler(CommandHandler("remove", remove_symbol))
    app.add_handler(CommandHandler("list", list_watchlist))

    # الإحصائيات
    app.add_handler(CommandHandler("stats", stats))

    # زر تحديث الإشارة
    app.add_handler(CallbackQueryHandler(refresh_signal, pattern=r"^refresh\|"))

    print("B7A BOT is running on Telegram...")
    app.run_polling(drop_pending_updates=True)
