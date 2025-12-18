import time
from typing import Dict, Any, List, Optional
from html import escape
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from bot.engine import generate_signal
from bot.market import (
    get_binance_price,
    get_top_volume_symbols,
    get_top_gainers,
    get_top_losers,
)
from bot.scanner import scan_market, scan_watchlist_symbols
from bot.analytics import analyze_stats
from bot.indicators import normalize_symbol as _normalize_symbol


# ==========================
# 1) Utility: Get Mode
# ==========================
def _get_current_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    mode = context.chat_data.get("mode", "balanced").lower()
    if mode not in ("balanced", "momentum", "safe"):
        mode = "balanced"
        context.chat_data["mode"] = mode
    return mode


# ==========================
# 2) Start Command
# ==========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👑 <b>مرحباً بك في B7A Ultra Bot</b>\n"
        "أقوى نظام تحليلات ذكي للكريبتو – مبني على محرك متعدد الفريمات + سيولة + مؤشرات احترافية.\n\n"
        "⚡ <b>ماذا يقدم لك البوت؟</b>\n"
        "• تحليل فوري لأي عملة (Multi-Timeframe Engine)\n"
        "• كشف اتجاه السوق العام والاتجاهات المخفية\n"
        "• خطة دخول كاملة: SL / TP / R:R\n"
        "• رادار ذكي لاكتشاف أفضل فرص BUY و SELL\n"
        "• فحص أعلى عملات USDT من حيث الفوليوم\n"
        "• دعم Coinglass (Open Interest / Funding / Liquidations)\n\n"
        "🛠 <b>اختر أسلوب التداول الخاص بك:</b>\n"
        "• BALANCED – أكثر وضع متزن\n"
        "• SAFE – أقل مخاطرة\n"
        "• MOMENTUM – بحث عن الانفجارات\n\n"
        "💡 <b>ابدأ الآن:</b>\n"
        "اكتب:\n"
        "• <b>/signal BTC</b> لتحليل عملة محددة\n"
        "• <b>/radar</b> لأقوى الفرص الآن\n"
        "• <b>/scan</b> لفحص عملات السوق\n\n"
        "📘 لا تعرف الأوامر؟ استخدم <b>/help</b>\n"
    )
    await update.message.reply_text(text, parse_mode="HTML")


# ==========================
# 3) Help Command
# ==========================
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🧾 <b>قائمة أوامر B7A Ultra Bot</b>\n\n"
        "💰 <b>الأسعار والإشارات</b>\n"
        "• <b>/price BTC</b> – عرض السعر الحالي\n"
        "• <b>/signal BTC</b> – إشارة تفصيلية (مع زر 🧠 تحليل مفصل)\n\n"
        "📡 <b>مسح السوق</b>\n"
        "• <b>/scan</b> – مسح أقوى العملات من حيث الفوليوم\n"
        "• <b>/scan_watchlist</b> – مسح قائمة المراقبة فقط\n"
        "• <b>/radar</b> – رادار الفرص (Top BUY و SELL)\n"
        "• <b>/daily</b> – ملخص يومي لأكبر الرابحين والخاسرين\n\n"
        "👀 <b>قائمة المراقبة</b>\n"
        "• <b>/add BTC</b> – إضافة عملة إلى قائمة المراقبة\n"
        "• <b>/remove BTC</b> – إزالة عملة من قائمة المراقبة\n"
        "• <b>/list</b> – عرض قائمة المراقبة الحالية\n\n"
        "📊 <b>الإحصائيات والتدريب</b>\n"
        "• <b>/stats</b> – ملخص أداء الصفقات (باستخدام /win و /loss)\n"
        "• <b>/win BTC</b> – تسجيل صفقة رابحة لعملة\n"
        "• <b>/loss BTC</b> – تسجيل صفقة خاسرة لعملة\n"
    )
    await update.message.reply_text(text, parse_mode="HTML")


# ==========================
# 4) Price Command
# ==========================
async def price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        symbol = context.args[0].upper()
    except Exception:
        await update.message.reply_text("❗ استخدم: /price BTC")
        return

    price_now = get_binance_price(symbol)
    if price_now is None:
        await update.message.reply_text("⚠️ لم يتم العثور على السعر.")
        return

    await update.message.reply_text(
        f"💰 السعر الحالي لـ <b>{symbol}</b>\n"
        f"<b>{price_now}</b> USDT",
        parse_mode="HTML",
    )


# =================================================
# 5) 🔥 Build Basic (Short) Signal Message
# =================================================
def _build_signal_message(signal_data: Dict[str, Any], symbol: str) -> str:
    decision = signal_data.get("decision", {})
    last_price = signal_data.get("last_price")
    mode = signal_data.get("mode", decision.get("mode", "balanced"))

    action = decision.get("action", "WAIT")
    score = float(decision.get("score", 50.0) or 50.0)
    trend = decision.get("trend", "RANGING")
    confidence = decision.get("confidence", "LOW")
    pump_risk = decision.get("pump_dump_risk", "LOW")
    grade = decision.get("grade", "C")

    sl = signal_data.get("sl")
    tp1 = signal_data.get("tp1")
    tp2 = signal_data.get("tp2")
    tp3 = signal_data.get("tp3")

    msg: List[str] = []
    msg.append(f"🏅 <b>B7A Ultra Signal – {symbol.upper()}</b>")
    if last_price is not None:
        msg.append(f"💰 السعر الحالي: <b>{last_price}</b> USDT")

    msg.append(f"🏆 Grade: <b>{grade}</b>")
    msg.append(f"🌍 وضع السوق العام: <b>{trend}</b>")
    msg.append(f"⚙️ Mode: <b>{str(mode).upper()}</b>")

    msg.append("")
    msg.append("📬 <b>قرار النظام</b>")
    msg.append(f"• Action: <b>{action}</b>")
    msg.append(f"• Score: <b>{score:.1f}/100</b>")
    msg.append(f"• Trend: <b>{trend}</b>")
    msg.append(f"• Confidence: <b>{confidence}</b>")
    msg.append(f"• Pump/Dump Risk: <b>{pump_risk}</b>")

    msg.append("")
    msg.append("📌 <b>خطة الصفقة</b>")
    msg.append(f"• نوع الصفقة: <b>{action}</b>")
    if sl is not None:
        msg.append(f"• SL (وقف الخسارة): <b>{sl}</b>")
    if tp1 is not None:
        msg.append(f"• TP1: <b>{tp1}</b>")
    if tp2 is not None:
        msg.append(f"• TP2: <b>{tp2}</b>")
    if tp3 is not None:
        msg.append(f"• TP3: <b>{tp3}</b>")

    msg.append("")
    msg.append("⚠️ هذا تحليل آلي — استخدم إدارة مخاطر صارمة.")

    # =========================
    # 🛡 B7A Shield – وضع الاختبار
    # =========================
    shield_active = bool(decision.get("shield_active"))
    shield_suggest_no_trade = bool(decision.get("shield_suggest_no_trade"))
    shield_reasons = decision.get("shield_reasons") or decision.get("no_trade_reasons") or []

    if shield_active:
        msg.append("")
        msg.append("🛡 <b>B7A Shield</b> (وضع اختبار)")
        if shield_suggest_no_trade:
            msg.append("• ⚠️ الشيلد يعتبر هذه الصفقة <b>عالية الخطورة</b> ولا ينصح بالدخول.")
        else:
            msg.append("• الشيلد فعّال لكنه <b>لم يمنع</b> هذه الصفقة.")
        for r in shield_reasons[:6]:
            msg.append(f"• {escape(str(r))}")

    # =========================
    # 🔄 B7A Flow Engine (FIXED KEYS)
    # =========================
    flow = decision.get("flow") or signal_data.get("flow")
    if flow:
        msg.append("")
        msg.append("🔄 <b>B7A Flow Engine</b>")
        flow_state = str(flow.get("flow_state", "UNKNOWN"))
        flow_bias = str(flow.get("flow_bias", "NEUTRAL"))
        msg.append(f"• Regime: <b>{escape(flow_state)}</b>")
        msg.append(f"• Bias: <b>{escape(flow_bias)}</b>")

    return "\n".join(msg)


# =================================================
# 6) 🧠 Build Detailed Analysis Block
# =================================================
def _build_analysis_block(signal_data: Dict[str, Any], mode: str) -> str:
    decision = signal_data.get("decision", {})
    tf_data = signal_data.get("timeframes", {})
    reason = signal_data.get("reason", "")

    flow = decision.get("flow") or signal_data.get("flow") or {}
    flow_score = flow.get("flow_score")
    flow_bias = flow.get("flow_bias")
    flow_state = flow.get("flow_state")

    liquidity_bias = decision.get("liquidity_bias") or signal_data.get("liquidity_bias")
    liquidity_score = decision.get("liquidity_score") or signal_data.get("liquidity_score")

    coinglass = signal_data.get("coinglass") or {}
    funding = coinglass.get("funding") or {}
    liquidation = coinglass.get("liquidation") or {}

    lines: List[str] = []

    # 🌊 B7A Flow Engine
    if flow:
        lines.append("<b>🌊 B7A Flow Engine</b>")
        try:
            fs = float(flow_score) if flow_score is not None else 50.0
        except Exception:
            fs = 50.0
        lines.append(
            f"• Flow Bias: <b>{escape(str(flow_bias))}</b> | Flow Score: <b>{fs:.0f}</b> | State: <b>{escape(str(flow_state))}</b>"
        )
        notes = flow.get("notes") or []
        if notes:
            lines.append("• Hint: " + escape(str(notes[0])))
        lines.append("")

    # 💧 السيولة
    lines.append("<b>💧 السيولة (Liquidity)</b>")
    try:
        ls = float(liquidity_score) if liquidity_score is not None else 0.0
    except Exception:
        ls = 0.0
    lines.append(f"• Bias: <b>{escape(str(liquidity_bias))}</b> | Liquidity Score ≈ <b>{ls:.0f}</b>")

    # 📊 Coinglass Intel
    if funding.get("available") or liquidation.get("available"):
        lines.append("")
        lines.append("<b>📊 Coinglass Intel</b>")

        if funding.get("available"):
            rate = funding.get("rate")
            severity = funding.get("severity")
            side = funding.get("side_bias")
            try:
                rate_txt = f"{float(rate):.4f}%"
            except Exception:
                rate_txt = "N/A"
            lines.append(
                f"• Funding: <b>{rate_txt}</b> | Severity: <b>{escape(str(severity))}</b> | Side: <b>{escape(str(side))}</b>"
            )

        if liquidation.get("available"):
            bias = liquidation.get("bias")
            intensity = liquidation.get("intensity")
            total = liquidation.get("liquidation_usd")
            try:
                it = float(intensity)
            except Exception:
                it = 0.0
            try:
                tt = float(total) if total is not None else 0.0
            except Exception:
                tt = 0.0
            lines.append(
                f"• Liquidations: Bias <b>{escape(str(bias))}</b> | Intensity: <b>{it:.2f}</b> | Total ≈ <b>{tt:,.0f}</b> USD"
            )

    # ملخص الفريمات
    lines.append("")
    lines.append("<b>🧠 ملخص الفريمات</b>")
    for tf in ["15m", "1h", "4h", "1d"]:
        tf_info = tf_data.get(tf)
        if not tf_info:
            continue
        t_trend = tf_info.get("trend")
        t_score = tf_info.get("score")
        regime = tf_info.get("regime")
        lines.append(
            f"• {tf} | Trend: <b>{escape(str(t_trend))}</b> | Score: <b>{escape(str(t_score))}</b> | Regime: <b>{escape(str(regime))}</b>"
        )

    # لماذا أعطى البوت الإشارة؟
    if reason:
        lines.append("")
        lines.append("<b>📝 لماذا أعطى البوت هذه الإشارة؟</b>")
        lines.append(escape(str(reason)))

    return "\n".join(lines)


# =================================================
# 7) Signal Handler
# =================================================
async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        symbol = context.args[0].upper()
    except Exception:
        await update.message.reply_text("❗ استخدم: /signal BTC")
        return

    symbol_norm = _normalize_symbol(symbol)
    mode = _get_current_mode(context)

    signal_data = generate_signal(symbol_norm, mode=mode, use_coinglass=True)
    text = _build_signal_message(signal_data, symbol_norm)

    # FIX: لا تضيف USDT مرة ثانية
    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol_norm}"

    keyboard = [
        [
            InlineKeyboardButton(f"⚙️ Mode: {mode}", callback_data=f"mode|{symbol_norm}"),
            InlineKeyboardButton("🔄 Refresh", callback_data=f"refresh|{symbol_norm}"),
            InlineKeyboardButton("📊 فتح الشارت", url=tv_url),
        ],
        [InlineKeyboardButton("🧠 تحليل مفصل", callback_data=f"analysis|{symbol_norm}")],
    ]

    await update.message.reply_text(
        text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML",
    )


# =================================================
# 8) Refresh Signal
# =================================================
async def refresh_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, symbol = query.data.split("|")
    mode = _get_current_mode(context)

    symbol_norm = _normalize_symbol(symbol)
    signal_data = generate_signal(symbol_norm, mode=mode, use_coinglass=True)
    text = _build_signal_message(signal_data, symbol_norm)

    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol_norm}"

    keyboard = [
        [
            InlineKeyboardButton(f"⚙️ Mode: {mode}", callback_data=f"mode|{symbol_norm}"),
            InlineKeyboardButton("🔄 Refresh", callback_data=f"refresh|{symbol_norm}"),
            InlineKeyboardButton("📊 فتح الشارت", url=tv_url),
        ],
        [InlineKeyboardButton("🧠 تحليل مفصل", callback_data=f"analysis|{symbol_norm}")],
    ]

    await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))


# =================================================
# 9) Detailed Analysis Button
# =================================================
async def show_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, symbol = query.data.split("|")
    symbol_norm = _normalize_symbol(symbol)
    mode = _get_current_mode(context)

    signal_data = generate_signal(symbol_norm, mode=mode, use_coinglass=True)
    text = _build_analysis_block(signal_data, mode)

    await query.message.reply_text(text, parse_mode="HTML")


# =================================================
# 10) Radar
# =================================================
async def radar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = _get_current_mode(context)
    symbols = get_top_volume_symbols(limit=40)

    waiting = await update.message.reply_text(
        "⏳ جاري تشغيل B7A Ultra Radar للبحث عن أقوى فرص BUY / SELL...",
        parse_mode="HTML",
    )

    result: List[str] = []
    result.append("🎯 <b>B7A Ultra Radar</b>\n")

    data = scan_market(symbols, mode=mode)

    buys = sorted(
        [x for x in data if x["signal"]["decision"]["action"] == "BUY"],
        key=lambda x: float(x["signal"]["decision"]["score"]),
        reverse=True,
    )[:5]

    sells = sorted(
        [x for x in data if x["signal"]["decision"]["action"] == "SELL"],
        key=lambda x: float(x["signal"]["decision"]["score"]),
        reverse=False,
    )[:5]

    if buys:
        result.append("🟢 أفضل فرص BUY:\n")
        for item in buys:
            sym = item["symbol"]
            sdata = item["signal"]["decision"]
            result.append(f"• {sym}: BUY | Grade: {sdata.get('grade')} | Score: {float(sdata.get('score') or 0):.0f}")

    if sells:
        result.append("\n🔴 أفضل فرص SELL:\n")
        for item in sells:
            sym = item["symbol"]
            sdata = item["signal"]["decision"]
            result.append(f"• {sym}: SELL | Grade: {sdata.get('grade')} | Score: {float(sdata.get('score') or 0):.0f}")

    if not buys and not sells:
        result.append("لا توجد فرص واضحة حالياً حسب شروط B7A Ultra.")

    await waiting.edit_text("\n".join(result), parse_mode="HTML")


# =================================================
# 11) Scan
# =================================================
async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = _get_current_mode(context)
    symbols = get_top_volume_symbols(limit=30)

    waiting = await update.message.reply_text(
        "⏳ جاري فحص أعلى عملات USDT من حيث الفوليوم...",
        parse_mode="HTML",
    )

    results = scan_market(symbols, mode=mode)

    msg: List[str] = ["🔍 فحص أعلى عملات USDT من حيث الفوليوم...\n"]
    for item in results[:10]:
        sym = item["symbol"]
        dec = item["signal"]["decision"]
        msg.append(
            f"• {sym}: {dec.get('action')} | Grade: {dec.get('grade')} | Score: {float(dec.get('score') or 0):.0f}"
        )

    if len(msg) == 1:
        msg.append("لا توجد بيانات كافية حالياً.")

    await waiting.edit_text("\n".join(msg), parse_mode="HTML")


# =================================================
# 12) Scan Watchlist
# =================================================
async def scan_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = _get_current_mode(context)
    watch = context.chat_data.get("watchlist", [])
    if not watch:
        await update.message.reply_text("⚠️ قائمة المراقبة فارغة.")
        return

    res = scan_watchlist_symbols(watch, mode=mode)
    msg = ["📊 نتائج مسح قائمة المراقبة:\n"]

    for item in res:
        sym = item["symbol"]
        dec = item["signal"]["decision"]
        msg.append(
            f"• {sym}: {dec.get('action')} | Grade: {dec.get('grade')} | Score: {float(dec.get('score') or 0):.0f}"
        )

    await update.message.reply_text("\n".join(msg), parse_mode="HTML")


# =================================================
# 13) Watchlist Add/Remove/List
# =================================================
async def add_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        symbol = context.args[0].upper()
    except Exception:
        await update.message.reply_text("❗ استخدم: /add BTC")
        return

    wl = context.chat_data.get("watchlist", [])
    if symbol not in wl:
        wl.append(symbol)
    context.chat_data["watchlist"] = wl

    await update.message.reply_text(f"تمت إضافة {symbol} إلى قائمة المراقبة.")


async def remove_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        symbol = context.args[0].upper()
    except Exception:
        await update.message.reply_text("❗ استخدم: /remove BTC")
        return

    wl = context.chat_data.get("watchlist", [])
    if symbol in wl:
        wl.remove(symbol)
    context.chat_data["watchlist"] = wl

    await update.message.reply_text(f"تمت إزالة {symbol} من قائمة المراقبة.")


async def list_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    wl = context.chat_data.get("watchlist", [])
    if not wl:
        await update.message.reply_text("⚠️ قائمة المراقبة فارغة.")
        return

    await update.message.reply_text("👀 قائمة المراقبة الحالية:\n" + ", ".join(wl))


# =================================================
# 14) Daily Summary
# =================================================
async def daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    gains = get_top_gainers()
    losses = get_top_losers()

    msg = ["📅 <b>الملخص اليومي</b>\n"]

    if gains:
        msg.append("🔼 أكبر الرابحين:")
        for s, pct in gains[:5]:
            msg.append(f"• {s}: +{pct:.2f}%")

    if losses:
        msg.append("\n🔽 أكبر الخاسرين:")
        for s, pct in losses[:5]:
            msg.append(f"• {s}: {pct:.2f}%")

    await update.message.reply_text("\n".join(msg), parse_mode="HTML")


# =================================================
# 15) Stats
# =================================================
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st = analyze_stats(context.chat_data)
    await update.message.reply_text(st, parse_mode="HTML")


# =================================================
# 16) Mode Toggle
# =================================================
async def toggle_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, symbol = query.data.split("|")
    symbol_norm = _normalize_symbol(symbol)

    current = _get_current_mode(context)
    modes = ["balanced", "momentum", "safe"]
    idx = modes.index(current)
    new_mode = modes[(idx + 1) % len(modes)]

    context.chat_data["mode"] = new_mode

    signal_data = generate_signal(symbol_norm, mode=new_mode, use_coinglass=True)
    text = _build_signal_message(signal_data, symbol_norm)

    tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol_norm}"
    keyboard = [
        [
            InlineKeyboardButton(f"⚙️ Mode: {new_mode}", callback_data=f"mode|{symbol_norm}"),
            InlineKeyboardButton("🔄 Refresh", callback_data=f"refresh|{symbol_norm}"),
            InlineKeyboardButton("📊 فتح الشارت", url=tv_url),
        ],
        [InlineKeyboardButton("🧠 تحليل مفصل", callback_data=f"analysis|{symbol_norm}")],
    ]

    await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))


# =================================================
# 17) Win / Loss Trainer
# =================================================
async def mark_win(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        symbol = context.args[0].upper()
    except Exception:
        await update.message.reply_text("❗ استخدم: /win BTC")
        return

    hist = context.chat_data.get("trainer", {})
    entry = hist.get(symbol, {"wins": 0, "losses": 0})
    entry["wins"] += 1
    hist[symbol] = entry
    context.chat_data["trainer"] = hist

    await update.message.reply_text(f"🎉 تم تسجيل ربح لعملة {symbol}!")


async def mark_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        symbol = context.args[0].upper()
    except Exception:
        await update.message.reply_text("❗ استخدم: /loss BTC")
        return

    hist = context.chat_data.get("trainer", {})
    entry = hist.get(symbol, {"wins": 0, "losses": 0})
    entry["losses"] += 1
    hist[symbol] = entry
    context.chat_data["trainer"] = hist

    await update.message.reply_text(f"⚠️ تم تسجيل خسارة لعملة {symbol}!")


# =================================================
# END OF FILE
# =================================================
