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
    radar,
    radar_long,      # 🔵 رادار لونغ
    radar_short,     # 🔴 رادار شورت
    toggle_mode,
    mark_win,
    mark_loss,
)

# متغير التوكن من البيئة
TOKEN = os.getenv("TELEGRAM_TOKEN")


if __name__ == "__main__":
    print("B7A BOT starting Telegram service...")

    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is not set in environment")

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
    app.add_handler(CommandHandler("radar", radar))


    # إدارة قائمة المراقبة
    app.add_handler(CommandHandler("add", add_symbol))
    app.add_handler(CommandHandler("remove", remove_symbol))
    app.add_handler(CommandHandler("list", list_watchlist))

    # الإحصائيات
    app.add_handler(CommandHandler("stats", stats))

    # تعليم نتيجة الصفقة
    app.add_handler(CommandHandler("win", mark_win))
    app.add_handler(CommandHandler("loss", mark_loss))


    # أزرار الإشارة (Refresh + Mode)
    app.add_handler(CallbackQueryHandler(refresh_signal, pattern=r"^refresh\|"))
    app.add_handler(CallbackQueryHandler(toggle_mode, pattern=r"^mode\|"))

    print("B7A BOT is running on Telegram...")
    app.run_polling(drop_pending_updates=True)


