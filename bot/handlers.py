from typing import Dict, Any, List, Set

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from .engine import generate_signal
from bot.market import get_price
from bot.scanner import get_top_usdt_symbols
from bot.analytics import get_trades_summary


# قائمة مراقبة ديناميكية (في الذاكرة)
WATCHLIST: Set[str] = set(["BTC", "ETH", "SOL", "DOGE", "TON", "BNB"])


# /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔥 B7A Ultra Bot is LIVE! 🔥")


# /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
🤖 قائمة الأوامر:

/start – تشغيل البوت
/help – عرض هذه القائمة
/price BTC – سعر العملة (مثال: /price sol)
/signal BTC – إشارة تحليل احترافية (مثال: /signal eth)

/scan – فحص أعلى عملات USDT من حيث الفوليوم وإظهار أفضل الفرص
/scan_watchlist – فحص قائمة المراقبة الخاصة فيك فقط
/daily – تقرير يومي مختصر عن السوق

/add BTC – إضافة عملة إلى قائمة المراقبة
/remove BTC – حذف عملة من قائمة المراقبة
/list – عرض قائمة المراقبة الحالية

/stats – ملخص أداء الإشارات من اللوق (B7A Ultra Analytics)
"""
    await update.message.reply_text(text)



# /price
async def price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("استخدم: /price BTC أو /price sol")
        return

    symbol = context.args[0].upper()
    # لو كتب BTCUSDT نحوله BTC فقط
    if symbol.endswith("USDT"):
        symbol = symbol[:-4]

    value = get_price(symbol)

    if value:
        await update.message.reply_text(f"💵 سعر {symbol}: {value} USDT")
    else:
        await update.message.reply_text("صار خطأ غير متوقع أثناء جلب السعر 😢")


# ====== دالة داخلية تبني نص الإشارة ======

def _build_signal_message(signal_data: Dict[str, Any], symbol_fallback: str) -> str:
    decision = signal_data.get("decision", {})
    tf_data = signal_data.get("timeframes", {})
    last_price = signal_data.get("last_price")
    reason = signal_data.get("reason", "")

    # عناصر القرار
    action = decision.get("action", "WAIT")
    score = decision.get("score", 50)
    trend = decision.get("trend", "RANGING")
    confidence = decision.get("confidence", "LOW")
    pump_risk = decision.get("pump_dump_risk", "LOW")
    liquidity_bias = decision.get("liquidity_bias", "FLAT")
    liquidity_score = decision.get("liquidity_score", 0.0)

    # الجديد
    grade = decision.get("grade", "C")
    no_trade = decision.get("no_trade", False)
    market_regime = decision.get("market_regime", "UNKNOWN")

    tp = signal_data.get("tp")
    sl = signal_data.get("sl")
    rr = signal_data.get("rr")
    risk_pct = signal_data.get("risk_pct")
    reward_pct = signal_data.get("reward_pct")

    # ---------------------------
    # 🧠 عنوان الإشارة
    # ---------------------------
    msg = f"📈 إشارة {signal_data.get('symbol', symbol_fallback)} – B7A Ultra Engine\n\n"

    # السعر
    if last_price is not None:
        msg += f"💰 السعر الحالي: {last_price:.4f} USDT\n"

    # Grade + Market Regime
    msg += f"🏆 تصنيف الإشارة (Grade): {grade}\n"
    msg += f"🌍 وضع السوق العام: {market_regime}\n"

    if no_trade:
        msg += "⚠️ منطقة No-Trade: هذه الإشارة ضعيفة حسب فلتر B7A Ultra.\n"

    msg += "\n"

    # ---------------------------
    # 🔥 قرار البوت
    # ---------------------------
    msg += (
        f"🎯 قرار النظام: {action}\n"
        f"📊 قوة الإشارة (Score): {score}/100\n"
        f"📈 الاتجاه العام: {trend}\n"
        f"🧪 الثقة: {confidence}\n"
        f"🧨 Pump/Dump Risk: {pump_risk}\n"
    )

    # ---------------------------
    # 💧 السيولة
    # ---------------------------
    msg += f"\n💧 Liquidity Bias: {liquidity_bias} (Score ≈ {liquidity_score:.0f})\n"

    # ---------------------------
    # 🔍 فحص الفريمات (ملخص)
    # ---------------------------
    msg += "\n🧠 ملخص الفريمات:\n"
    order = ["15m", "1h", "4h", "1d"]
    for tf in order:
        fr = tf_data.get(tf, {})
        reg = fr.get("market_regime", "?")
        bo = "Break ↑" if fr.get("is_breakout_up") else ("Break ↓" if fr.get("is_breakout_down") else "–")
        div = (
            "Bullish Div" if fr.get("has_bull_div")
            else ("Bearish Div" if fr.get("has_bear_div") else "–")
        )
        msg += (
            f"• {tf} | Trend: {fr.get('trend')} "
            f"| Score: {fr.get('trend_score')} "
            f"| Regime: {reg} | {bo} | {div}\n"
        )

    # ---------------------------
    # 🎯 TP / SL
    # ---------------------------
    msg += "\n🎯 خطة الصفقة:\n"
    if tp and sl:
        msg += (
            f"• الهدف (TP): {tp}\n"
            f"• وقف الخسارة (SL): {sl}\n"
            f"• R:R ≈ {rr}\n"
        )
        if risk_pct and reward_pct:
            msg += f"• مخاطرة تقريبية: {risk_pct:.1f}% | هدف ربح: {reward_pct:.1f}%\n"
    else:
        msg += "لا توجد مستويات دخول واضحة (No-Trade).\n"

    # ---------------------------
    # 📌 ملخص ذكي
    # ---------------------------
    msg += "\n📌 ملخص ذكي:\n"
    msg += reason + "\n\n"

    msg += "⚠️ هذا تحليل آلي – استخدم إدارة مخاطر دائمًا.\n"
    msg += "— B7A Ultra Engine"

    return msg



# /signal
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
    if symbol.endswith("USDT"):
        symbol = symbol[:-4]

    await update.message.reply_text(
        f"⏳ جارِ تحليل السوق لـ {symbol} عبر B7A Ultra Engine ..."
    )

    try:
        signal_data = generate_signal(symbol)
    except Exception as e:
        print("Signal error:", e)
        await update.message.reply_text(
            "❌ صار خطأ أثناء توليد الإشارة، جرّب بعد شوي أو مع عملة ثانية."
        )
        return

    msg = _build_signal_message(signal_data, symbol)

    tv_symbol = signal_data.get("symbol", symbol)
    keyboard = [
        [
            InlineKeyboardButton(
                "🔄 تحديث الإشارة", callback_data=f"refresh|{tv_symbol}"
            ),
            InlineKeyboardButton(
                "📊 فتح الشارت",
                url=f"https://www.tradingview.com/chart/?symbol=BINANCE:{tv_symbol}",
            ),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(msg, reply_markup=reply_markup)


# زر تحديث الإشارة
async def refresh_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    try:
        _, symbol = query.data.split("|", 1)
    except Exception:
        await query.edit_message_text("حدث خطأ في قراءة بيانات التحديث.")
        return

    try:
        signal_data = generate_signal(symbol)
    except Exception as e:
        print("Refresh error:", e)
        await query.edit_message_text(
            "❌ صار خطأ أثناء تحديث الإشارة، جرّب مرة ثانية بعد شوي."
        )
        return

    msg = _build_signal_message(signal_data, symbol)

    tv_symbol = signal_data.get("symbol", symbol)
    keyboard = [
        [
            InlineKeyboardButton(
                "🔄 تحديث الإشارة", callback_data=f"refresh|{tv_symbol}"
            ),
            InlineKeyboardButton(
                "📊 فتح الشارت",
                url=f"https://www.tradingview.com/chart/?symbol=BINANCE:{tv_symbol}",
            ),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(msg, reply_markup=reply_markup)


# /scan – Smart Scanner (Top Volume)
async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 جارِ فحص أعلى عملات USDT من حيث الفوليوم...")

    try:
        symbols = get_top_usdt_symbols(limit=40)
    except Exception as e:
        print("Top volume error:", e)
        await update.message.reply_text("⚠️ ما قدرت أجيب قائمة العملات من Binance حالياً.")
        return

    results = []
    for symbol in symbols:
        try:
            data = generate_signal(symbol)
            decision = data.get("decision", {})
            action = decision.get("action", "WAIT")
            score = decision.get("score", 50)
            if action != "WAIT":
                results.append((symbol, action, score, decision))
        except Exception as e:
            print("Scan error for", symbol, ":", e)
            continue

    if not results:
        await update.message.reply_text("ما في فرص قوية واضحة حالياً في السوق حسب الفلتر.")
        return

    results.sort(key=lambda x: x[2], reverse=True)
    top = results[:5]

    lines = ["📊 أفضل الفرص الحالية (Top Volume Scanner):\n"]
    for symbol, action, score, decision in top:
        trend = decision.get("trend", "RANGING")
        pump = decision.get("pump_dump_risk", "LOW")
        lines.append(
            f"• {symbol}: {action} | Score: {score:.0f} | Trend: {trend} | Pump: {pump}"
        )

    lines.append("\nاستخدم /signal BTC مثلاً لعرض تحليل مفصل لأي عملة.")

    await update.message.reply_text("\n".join(lines))


# /scan_watchlist – فحص قائمة المراقبة الخاصة
async def scan_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not WATCHLIST:
        await update.message.reply_text("قائمة المراقبة فاضية. استخدم /add BTC مثلاً.")
        return

    await update.message.reply_text("🔍 جارِ فحص قائمة المراقبة الخاصة فيك...")

    results = []
    for symbol in sorted(WATCHLIST):
        try:
            data = generate_signal(symbol)
            decision = data.get("decision", {})
            action = decision.get("action", "WAIT")
            score = decision.get("score", 50)
            if action != "WAIT":
                results.append((symbol, action, score, decision))
        except Exception as e:
            print("Watchlist scan error for", symbol, ":", e)
            continue

    if not results:
        await update.message.reply_text("ما في فرص قوية واضحة حالياً داخل قائمة المراقبة.")
        return

    results.sort(key=lambda x: x[2], reverse=True)
    top = results[:5]

    lines = ["📌 أفضل الفرص داخل قائمة المراقبة:\n"]
    for symbol, action, score, decision in top:
        trend = decision.get("trend", "RANGING")
        pump = decision.get("pump_dump_risk", "LOW")
        lines.append(
            f"• {symbol}: {action} | Score: {score:.0f} | Trend: {trend} | Pump: {pump}"
        )

    lines.append("\nتقدر توسع التحليل باستخدام /signal BTC مثلاً.")
    await update.message.reply_text("\n".join(lines))


# /daily – تقرير يومي مختصر
async def daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📰 تجهيز تقرير يومي مختصر للسوق...")

    results = []
    try:
        symbols = get_top_usdt_symbols(limit=30)
    except Exception as e:
        print("Daily top volume error:", e)
        symbols = list(WATCHLIST) or ["BTC", "ETH", "SOL"]

    for symbol in symbols:
        try:
            data = generate_signal(symbol)
            decision = data.get("decision", {})
            action = decision.get("action", "WAIT")
            score = decision.get("score", 50)
            if action != "WAIT":
                results.append((symbol, action, score, decision))
        except Exception as e:
            print("Daily scan error for", symbol, ":", e)
            continue

    # تحليل BTC كقائد للسوق
    try:
        btc_data = generate_signal("BTC")
        btc_decision = btc_data.get("decision", {})
    except Exception:
        btc_decision = {}

    btc_trend = btc_decision.get("trend", "UNKNOWN")
    btc_action = btc_decision.get("action", "WAIT")
    btc_score = btc_decision.get("score", 50)

    msg_lines = [
        "📰 تقرير يومي من B7A Ultra Bot:",
        "",
        f"🪙 حالة BTC: {btc_trend} | Action: {btc_action} | Score: {btc_score}/100",
        "",
    ]

    if results:
        results.sort(key=lambda x: x[2], reverse=True)
        best = results[:3]
        msg_lines.append("🔥 أفضل 3 فرص اليوم:")
        for symbol, action, score, decision in best:
            trend = decision.get("trend", "RANGING")
            msg_lines.append(f"• {symbol}: {action} | Score: {score:.0f} | Trend: {trend}")
    else:
        msg_lines.append("ما في فرص قوية جداً اليوم حسب الفلتر الحالي (الكل تقريباً WAIT).")

    msg_lines.append("")
    msg_lines.append("تقدر تستخدم /signal BTC لأي عملة تبي تشوف تحليلها بالتفصيل.")
    await update.message.reply_text("\n".join(msg_lines))


# ========= أوامر إدارة الـ Watchlist =========

async def add_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("استخدم: /add BTC")
        return

    symbol = context.args[0].upper()
    if symbol.endswith("USDT"):
        symbol = symbol[:-4]

    if symbol in WATCHLIST:
        await update.message.reply_text(f"{symbol} موجودة مسبقاً في قائمة المراقبة ✅")
        return

    WATCHLIST.add(symbol)
    await update.message.reply_text(f"✅ تمت إضافة {symbol} إلى قائمة المراقبة.")


async def remove_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("استخدم: /remove BTC")
        return

    symbol = context.args[0].upper()
    if symbol.endswith("USDT"):
        symbol = symbol[:-4]

    if symbol not in WATCHLIST:
        await update.message.reply_text(f"{symbol} غير موجودة في قائمة المراقبة.")
        return

    WATCHLIST.remove(symbol)
    await update.message.reply_text(f"❌ تم حذف {symbol} من قائمة المراقبة.")


async def list_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not WATCHLIST:
        await update.message.reply_text("قائمة المراقبة فاضية حالياً. أضف عملة بـ /add BTC مثلاً.")
        return

    coins = ", ".join(sorted(WATCHLIST))
    await update.message.reply_text(f"👀 قائمة المراقبة الحالية:\n{coins}")
    
# /stats – ملخص أداء الإشارات من اللوق
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = get_trades_summary()
    await update.message.reply_text(text)
