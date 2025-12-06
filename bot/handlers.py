from typing import Dict, Any, List, Set

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from .engine import generate_signal
from bot.market import get_price
from bot.scanner import get_top_usdt_symbols
from .analytics import get_trades_summary

# قائمة مراقبة ديناميكية (في الذاكرة فقط)
WATCHLIST: Set[str] = set(["BTC", "ETH", "SOL", "DOGE", "TON", "BNB"])

# مودات البوت
MODES = ["SAFE", "BALANCED", "MOMENTUM"]


# =========================
# Helpers
# =========================

def _normalize_symbol(symbol: str) -> str:
    symbol = (symbol or "").upper().strip()
    if symbol.endswith("USDT"):
        symbol = symbol[:-4]
    return symbol


def _get_current_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    mode = context.chat_data.get("mode")
    if mode not in MODES:
        mode = "BALANCED"
        context.chat_data["mode"] = mode
    return mode


def _set_current_mode(context: ContextTypes.DEFAULT_TYPE, mode: str) -> None:
    mode = (mode or "BALANCED").upper()
    if mode not in MODES:
        mode = "BALANCED"
    context.chat_data["mode"] = mode


def _build_debug_block(signal_data: Dict[str, Any], mode: str) -> str:
    """Debug info للمالك/المطور – يساعدنا نفهم قرار البوت."""
    decision = signal_data.get("decision", {})
    long_score = signal_data.get("long_score")
    short_score = signal_data.get("short_score")
    bull_align = signal_data.get("bull_align")
    bear_align = signal_data.get("bear_align")
    trend = decision.get("trend") or signal_data.get("trend")
    liq_bias = decision.get("liquidity_bias") or signal_data.get("liquidity_bias")
    pump_risk = decision.get("pump_dump_risk") or signal_data.get("pump_dump_risk")

    lines: List[str] = []
    lines.append("")
    lines.append("🧪 <b>DEBUG – B7A Ultra Engine</b>")
    lines.append(f"• Mode: <b>{(mode or '').upper()}</b>")

    if long_score is not None:
        lines.append(f"• LongScore: <b>{long_score:.1f}</b>")
    if short_score is not None:
        lines.append(f"• ShortScore: <b>{short_score:.1f}</b>")
    if bull_align is not None:
        lines.append(f"• Bull Align: <b>{bull_align:.2f}</b>")
    if bear_align is not None:
        lines.append(f"• Bear Align: <b>{bear_align:.2f}</b>")
    if trend:
        lines.append(f"• Global Trend: <b>{trend}</b>")
    if liq_bias:
        lines.append(f"• Liquidity Bias: <b>{liq_bias}</b>")
    if pump_risk:
        lines.append(f"• Pump/Dump Risk: <b>{pump_risk}</b>")

    return "\n".join(lines)


def _build_signal_message(signal_data: Dict[str, Any], symbol_fallback: str) -> str:
    decision = signal_data.get("decision", {})
    tf_data = signal_data.get("timeframes", {})
    last_price = signal_data.get("last_price")
    reason = signal_data.get("reason", "")

    # عناصر القرار
    action = decision.get("action", "WAIT")
    score = decision.get("score", 50.0) or 50.0
    trend = decision.get("trend") or signal_data.get("trend", "RANGING")
    confidence = decision.get("confidence", "LOW")
    pump_risk = decision.get("pump_dump_risk") or signal_data.get("pump_dump_risk", "LOW")
    liquidity_bias = decision.get("liquidity_bias") or signal_data.get("liquidity_bias", "FLAT")
    liquidity_score = decision.get("liquidity_score") or signal_data.get("liquidity_score", 0.0)
    market_regime = decision.get("market_regime") or signal_data.get("market_regime", "RANGING")
    no_trade = decision.get("no_trade") or signal_data.get("no_trade", False)

    grade = decision.get("grade", "C")

    # مستويات السعر
    sl = signal_data.get("sl")
    tp = signal_data.get("tp")
    tp1 = signal_data.get("tp1")
    tp2 = signal_data.get("tp2")
    tp3 = signal_data.get("tp3")
    rr = signal_data.get("rr")
    rr1 = signal_data.get("rr1")
    rr2 = signal_data.get("rr2")
    rr3 = signal_data.get("rr3")
    risk_pct = signal_data.get("risk_pct")
    reward_pct = signal_data.get("reward_pct")

    symbol_text = signal_data.get("symbol", symbol_fallback)

    lines: List[str] = []

    # =========================
    # الهيدر
    # =========================
    lines.append(f"⚜️ <b>B7A Ultra Signal – {symbol_text}USDT</b>")
    if last_price is not None:
        lines.append(f"💰 السعر الحالي: <b>{last_price}</b> USDT")
    lines.append(f"🏆 Grade: <b>{grade}</b>")
    lines.append(f"🌍 وضع السوق العام: <b>{market_regime}</b>")
    lines.append("")

    # =========================
    # قرار النظام
    # =========================
    lines.append("<b>🎯 قرار النظام</b>")
    lines.append(f"• Action: <b>{action}</b>")
    lines.append(f"• Score: <b>{score:.1f}/100</b>")
    lines.append(f"• Trend: <b>{trend}</b>")
    lines.append(f"• Confidence: <b>{confidence}</b>")
    lines.append(f"• Pump/Dump Risk: <b>{pump_risk}</b>")

    # =========================
    # خطة الصفقة (Multi-TP)
    # =========================
    lines.append("")
    lines.append("<b>📌 خطة الصفقة</b>")

    if action in ("BUY", "SELL") and not no_trade:
        direction = "شراء" if action == "BUY" else "بيع"
        lines.append(f"• نوع الصفقة: <b>{direction}</b>")

        if sl is not None:
            lines.append(f"• وقف الخسارة (SL): <b>{sl}</b>")

        # Multi-TP
        if tp1 is not None:
            rr1_text = f" (R:R ≈ {rr1})" if rr1 is not None else ""
            lines.append(f"• TP1: <b>{tp1}</b>{rr1_text}")
        if tp2 is not None:
            rr2_text = f" (R:R ≈ {rr2})" if rr2 is not None else ""
            lines.append(f"• TP2 (الهدف الرئيسي): <b>{tp2}</b>{rr2_text}")
        if tp3 is not None:
            rr3_text = f" (R:R ≈ {rr3})" if rr3 is not None else " (تمديد)"
            lines.append(f"• TP3 (تمديد): <b>{tp3}</b>{rr3_text}")

        if tp is not None and rr is not None:
            lines.append(f"• الهدف القياسي (TP): <b>{tp}</b> | R:R ≈ <b>{rr}</b>")

        if risk_pct is not None and reward_pct is not None:
            lines.append(
                f"• مخاطرة تقريبية: <b>{risk_pct:.1f}%</b> | "
                f"هدف ربح تقديري: <b>{reward_pct:.1f}%</b>"
            )
    else:
        lines.append("• لا توجد مستويات دخول واضحة – <b>No-Trade</b>.")

    # =========================
    # السيولة
    # =========================
    lines.append("")
    lines.append("<b>💧 السيولة (Liquidity)</b>")
    try:
        lines.append(
            f"• Bias: <b>{liquidity_bias}</b> | Liquidity Score ≈ <b>{float(liquidity_score):.0f}</b>"
        )
    except Exception:
        lines.append(f"• Bias: <b>{liquidity_bias}</b>")

    # =========================
    # ملخص الفريمات
    # =========================
    if tf_data:
        lines.append("")
        lines.append("<b>🧠 ملخص الفريمات</b>")
        order = ["15m", "1h", "4h", "1d"]
        for tf in order:
            info = tf_data.get(tf)
            if not info:
                continue
            tf_trend = info.get("trend", "RANGING")
            tf_score = info.get("trend_score", 50)
            tf_regime = info.get("market_regime", info.get("regime", "RANGING"))
            lines.append(
                f"• {tf} | Trend: <b>{tf_trend}</b> | Score: <b>{tf_score:.0f}</b> | Regime: <b>{tf_regime}</b>"
            )

    # =========================
    # السبب النصي
    # =========================
    if reason:
        lines.append("")
        lines.append("<b>📝 لماذا أعطى البوت هذه الإشارة؟</b>")
        lines.append(reason)

    lines.append("")
    lines.append("⚠️ هذا تحليل آلي – استخدم إدارة مخاطر دائماً.")
    lines.append("— <b>X: @B7Acrypto</b>")

    return "\n".join(lines)


# =========================
# أوامر البوت
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "مرحباً بك في <b>B7A Ultra X Bot</b> 👑\n\n"
        "أنا بوت تحليل ذكي متعدد الفريمات للكريبتو.\n\n"
        "أهم الأوامر:\n"
        "• /price BTC – يعرض السعر الحالي\n"
        "• /signal BTC – إشارة تفصيلية مع خطة دخول/خروج\n"
        "• /scan – مسح لأقوى الفرص في السوق\n"
        "• /scan_watchlist – مسح لقائمة المراقبة الخاصة بك\n"
        "• /radar – رادار الفرص (Long & Short)\n"
        "• /stats – ملخص أداء الإشارات\n"
        "• /add BTC – إضافة عملة لقائمة المراقبة\n"
        "• /list – عرض قائمة المراقبة\n"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)


async def price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("استخدم: /price BTC")
        return

    symbol = _normalize_symbol(context.args[0])
    p = get_price(symbol)
    if p is None:
        await update.message.reply_text("تعذر جلب السعر حالياً، جرّب مرة أخرى.")
        return

    await update.message.reply_text(f"💰 سعر {symbol}USDT الحالي ≈ <b>{p}</b>", parse_mode="HTML")


# /signal
async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) == 0:
        await update.message.reply_text(
            "🚨 استخدم الأمر بالشكل التالي:\n"
            "/signal BTC\n"
            "/signal ETH\n"
            "/signal SOL\n\n"
            "المود (SAFE/BALANCED/MOMENTUM) يتحدد من الزر داخل الإشارة."
        )
        return

    symbol = _normalize_symbol(context.args[0])

    mode = _get_current_mode(context)

    try:
        signal_data = generate_signal(symbol, mode=mode.lower(), use_coinglass=True)
    except Exception as e:
        print("Signal error:", e)
        await update.message.reply_text("❌ صار خطأ أثناء توليد الإشارة، جرّب مرة ثانية.")
        return

    msg = _build_signal_message(signal_data, symbol)
    msg += _build_debug_block(signal_data, mode)

    tv_symbol = signal_data.get("symbol", symbol)

    keyboard = [
        [
            InlineKeyboardButton(
                f"⚙️ Mode: {mode}",
                callback_data=f"mode|{tv_symbol}",
            ),
            InlineKeyboardButton(
                "🔄 Refresh",
                callback_data=f"refresh|{tv_symbol}",
            ),
        ]
    ]

    await update.message.reply_text(
        msg,
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def refresh_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    try:
        _, symbol = data.split("|", 1)
    except ValueError:
        await query.edit_message_text("حدث خطأ في بيانات التحديث.")
        return

    symbol = _normalize_symbol(symbol)
    mode = _get_current_mode(context)

    try:
        signal_data = generate_signal(symbol, mode=mode.lower(), use_coinglass=True)
    except Exception as e:
        print("Refresh error:", e)
        await query.edit_message_text("❌ صار خطأ أثناء تحديث الإشارة، جرّب مرة ثانية.")
        return

    msg = _build_signal_message(signal_data, symbol)
    msg += _build_debug_block(signal_data, mode)

    tv_symbol = signal_data.get("symbol", symbol)

    keyboard = [
        [
            InlineKeyboardButton(
                f"⚙️ Mode: {mode}",
                callback_data=f"mode|{tv_symbol}",
            ),
            InlineKeyboardButton(
                "🔄 Refresh",
                callback_data=f"refresh|{tv_symbol}",
            ),
        ]
    ]

    await query.edit_message_text(
        msg,
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def toggle_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    try:
        _, symbol = data.split("|", 1)
    except ValueError:
        await query.edit_message_text("حدث خطأ في قراءة بيانات الـ Mode.")
        return

    current_mode = _get_current_mode(context)
    try:
        idx = MODES.index(current_mode)
    except ValueError:
        idx = 1  # BALANCED
    new_mode = MODES[(idx + 1) % len(MODES)]
    _set_current_mode(context, new_mode)

    symbol = _normalize_symbol(symbol)

    try:
        signal_data = generate_signal(symbol, mode=new_mode.lower(), use_coinglass=True)
    except Exception as e:
        print("Toggle mode error:", e)
        await query.edit_message_text(
            "❌ صار خطأ أثناء إعادة توليد الإشارة بعد تغيير الـ Mode."
        )
        return

    msg = _build_signal_message(signal_data, symbol)
    msg += _build_debug_block(signal_data, new_mode)

    tv_symbol = signal_data.get("symbol", symbol)

    keyboard = [
        [
            InlineKeyboardButton(
                f"⚙️ Mode: {new_mode}",
                callback_data=f"mode|{tv_symbol}",
            ),
            InlineKeyboardButton(
                "🔄 Refresh",
                callback_data=f"refresh|{tv_symbol}",
            ),
        ]
    ]

    await query.edit_message_text(
        msg,
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


# ========= Scan & Radar =========

async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = _get_current_mode(context)
    await update.message.reply_text(f"⏳ مسح السوق ({mode}) ... انتظر قليلاً.")

    try:
        symbols = get_top_usdt_symbols(limit=40)
    except Exception as e:
        print("Scan symbols error:", e)
        await update.message.reply_text("❌ تعذر جلب قائمة العملات من Binance.")
        return

    candidates: List[Dict[str, Any]] = []

    for sym in symbols:
        try:
            data = generate_signal(sym, mode=mode.lower(), use_coinglass=False)
        except Exception:
            continue

        decision = data.get("decision", {})
        action = decision.get("action", "WAIT")
        if action in ("BUY", "SELL"):
            candidates.append(
                {
                    "symbol": data.get("symbol", sym),
                    "action": action,
                    "score": decision.get("score", data.get("score", 0.0)),
                    "grade": decision.get("grade", "C"),
                    "regime": decision.get("market_regime", data.get("market_regime", "RANGING")),
                    "liquidity_bias": decision.get("liquidity_bias", data.get("liquidity_bias", "FLAT")),
                    "rr": data.get("rr"),
                    "risk_pct": data.get("risk_pct"),
                    "reward_pct": data.get("reward_pct"),
                }
            )

    if not candidates:
        await update.message.reply_text("ما في فرص قوية حالياً – أغلب السوق WAIT.")
        return

    # ترتيب حسب السكور
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:10]

    lines: List[str] = []
    lines.append(f"📡 B7A Ultra Scan – Top {len(top)} فرص ({mode})\n")

    for c in top:
        line = (
            f"• {c['symbol']}: {c['action']} | Grade: {c['grade']} | Score: {c['score']:.0f} | "
            f"Regime: {c['regime']} | Liquidity: {c['liquidity_bias']}"
        )
        if c["rr"] is not None:
            line += f" | R:R ≈ {c['rr']}"
        if c["risk_pct"] is not None and c["reward_pct"] is not None:
            line += f" | Risk ~{c['risk_pct']:.1f}% / Reward ~{c['reward_pct']:.1f}%"
        lines.append(line)

    lines.append("\nاستخدم /signal BTC مثلاً لعرض تفاصيل أي عملة من القائمة.")
    await update.message.reply_text("\n".join(lines))


async def scan_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = _get_current_mode(context)
    if not WATCHLIST:
        await update.message.reply_text("قائمة المراقبة فاضية. استخدم /add BTC لإضافة عملة.")
        return

    await update.message.reply_text(f"⏳ مسح قائمة المراقبة ({mode}) ...")

    candidates: List[Dict[str, Any]] = []

    for sym in sorted(WATCHLIST):
        try:
            data = generate_signal(sym, mode=mode.lower(), use_coinglass=False)
        except Exception:
            continue

        decision = data.get("decision", {})
        action = decision.get("action", "WAIT")
        if action in ("BUY", "SELL"):
            candidates.append(
                {
                    "symbol": data.get("symbol", sym),
                    "action": action,
                    "score": decision.get("score", data.get("score", 0.0)),
                    "grade": decision.get("grade", "C"),
                    "regime": decision.get("market_regime", data.get("market_regime", "RANGING")),
                    "liquidity_bias": decision.get("liquidity_bias", data.get("liquidity_bias", "FLAT")),
                    "rr": data.get("rr"),
                    "risk_pct": data.get("risk_pct"),
                    "reward_pct": data.get("reward_pct"),
                }
            )

    if not candidates:
        await update.message.reply_text("ما في فرص قوية حالياً في قائمة المراقبة.")
        return

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:10]

    lines: List[str] = []
    lines.append(f"📡 B7A Watchlist Scan – Top {len(top)} فرص ({mode})\n")

    for c in top:
        line = (
            f"• {c['symbol']}: {c['action']} | Grade: {c['grade']} | Score: {c['score']:.0f} | "
            f"Regime: {c['regime']} | Liquidity: {c['liquidity_bias']}"
        )
        if c["rr"] is not None:
            line += f" | R:R ≈ {c['rr']}"
        if c["risk_pct"] is not None and c["reward_pct"] is not None:
            line += f" | Risk ~{c['risk_pct']:.1f}% / Reward ~{c['reward_pct']:.1f}%"
        lines.append(line)

    lines.append("\nاستخدم /signal BTC مثلاً لعرض تفاصيل أي عملة من القائمة.")
    await update.message.reply_text("\n".join(lines))


async def radar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """رادار موحّد يعرض أفضل الفرص لونغ + شورت معاً."""

    mode = _get_current_mode(context)
    await update.message.reply_text(f"📡 تشغيل الرادار ({mode}) ...")

    try:
        symbols = get_top_usdt_symbols(limit=80)
    except Exception as e:
        print("Radar symbols error:", e)
        await update.message.reply_text("❌ تعذر جلب قائمة العملات من Binance.")
        return

    long_candidates: List[Dict[str, Any]] = []
    short_candidates: List[Dict[str, Any]] = []

    for sym in symbols:
        try:
            data = generate_signal(sym, mode=mode.lower(), use_coinglass=False)
        except Exception:
            continue

        decision = data.get("decision", {})
        action = decision.get("action", "WAIT")
        if action not in ("BUY", "SELL"):
            continue

        entry = {
            "symbol": data.get("symbol", sym),
            "action": action,
            "score": decision.get("score", data.get("score", 0.0)),
            "grade": decision.get("grade", "C"),
            "regime": decision.get("market_regime", data.get("market_regime", "RANGING")),
            "liquidity_bias": decision.get("liquidity_bias", data.get("liquidity_bias", "FLAT")),
            "long_score": data.get("long_score"),
            "short_score": data.get("short_score"),
        }

        if action == "BUY":
            long_candidates.append(entry)
        elif action == "SELL":
            short_candidates.append(entry)

    lines: List[str] = []

    if not long_candidates and not short_candidates:
        await update.message.reply_text("ما في فرص قوية حالياً – الكل تقريباً WAIT.")
        return

    if long_candidates:
        long_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_long = long_candidates[:5]
        lines.append("🔵 أفضل فرص BUY:")
        for c in top_long:
            lines.append(
                f"• {c['symbol']}: BUY | Grade: {c['grade']} | Score: {c['score']:.0f} | "
                f"Regime: {c['regime']} | Liquidity: {c['liquidity_bias']}"
            )
        lines.append("")

    if short_candidates:
        short_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_short = short_candidates[:5]
        lines.append("🔴 أفضل فرص SELL:")
        for c in top_short:
            lines.append(
                f"• {c['symbol']}: SELL | Grade: {c['grade']} | Score: {c['score']:.0f} | "
                f"Regime: {c['regime']} | Liquidity: {c['liquidity_bias']}"
            )

    await update.message.reply_text("\n".join(lines))


async def radar_long(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يعرض أفضل 10 فرص BUY فقط."""
    mode = _get_current_mode(context)
    await update.message.reply_text("⏳ scanning market for BUY opportunities...")

    try:
        symbols = get_top_usdt_symbols(limit=80)
    except Exception as e:
        print("radar_long symbols error:", e)
        await update.message.reply_text("❌ تعذر جلب قائمة العملات من Binance.")
        return

    candidates: List[Dict[str, Any]] = []

    for sym in symbols:
        try:
            data = generate_signal(sym, mode=mode.lower(), use_coinglass=False)
        except Exception:
            continue

        decision = data.get("decision", {})
        action = decision.get("action", "WAIT")
        if action != "BUY":
            continue

        score = decision.get("score", data.get("score", 0.0))
        long_score = data.get("long_score", score)

        if score >= 70 or (long_score is not None and long_score >= 72):
            candidates.append(
                {
                    "symbol": data.get("symbol", sym),
                    "score": score,
                    "long_score": long_score,
                    "grade": decision.get("grade", "C"),
                }
            )

    if not candidates:
        await update.message.reply_text("😕 لا توجد فرص BUY قوية حالياً.")
        return

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:10]

    msg_lines: List[str] = []
    msg_lines.append("🔵 <b>B7A Ultra Radar – LONG ONLY</b>\n")
    for c in top:
        msg_lines.append(
            f"• {c['symbol']} → BUY | Score {c['score']:.0f} | LS {c['long_score']}"
        )

    await update.message.reply_text("\n".join(msg_lines), parse_mode="HTML")


async def radar_short(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يعرض أفضل 10 فرص SELL فقط."""
    mode = _get_current_mode(context)
    await update.message.reply_text("⏳ scanning market for SELL opportunities...")

    try:
        symbols = get_top_usdt_symbols(limit=80)
    except Exception as e:
        print("radar_short symbols error:", e)
        await update.message.reply_text("❌ تعذر جلب قائمة العملات من Binance.")
        return

    candidates: List[Dict[str, Any]] = []

    for sym in symbols:
        try:
            data = generate_signal(sym, mode=mode.lower(), use_coinglass=False)
        except Exception:
            continue

        decision = data.get("decision", {})
        action = decision.get("action", "WAIT")
        if action != "SELL":
            continue

        short_score = data.get("short_score")
        score = decision.get("score", data.get("score", 0.0))

        if short_score is not None and short_score >= 65:
            candidates.append(
                {
                    "symbol": data.get("symbol", sym),
                    "score": score,
                    "short_score": short_score,
                    "grade": decision.get("grade", "C"),
                }
            )

    if not candidates:
        await update.message.reply_text("😕 لا توجد فرص SELL قوية حالياً.")
        return

    candidates.sort(key=lambda x: x["short_score"], reverse=True)
    top = candidates[:10]

    msg_lines: List[str] = []
    msg_lines.append("🔴 <b>B7A Ultra Radar – SHORT ONLY</b>\n")
    for c in top:
        msg_lines.append(
            f"• {c['symbol']} → SELL | SS {c['short_score']} | Score {c['score']:.0f}"
        )

    await update.message.reply_text("\n".join(msg_lines), parse_mode="HTML")


# ========= Daily & Stats =========

async def daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تقرير يومي مبسط + أفضل 3 فرص."""

    mode = _get_current_mode(context)

    # تحليل BTC كمرجع للسوق
    try:
        btc_data = generate_signal("BTC", mode=mode.lower(), use_coinglass=True)
    except Exception as e:
        print("daily BTC error:", e)
        await update.message.reply_text("❌ تعذر جلب تحليل BTC حالياً.")
        return

    btc_decision = btc_data.get("decision", {})
    btc_trend = btc_decision.get("trend", "RANGING")
    btc_action = btc_decision.get("action", "WAIT")
    btc_score = btc_decision.get("score", btc_data.get("score", 50.0))

    # مسح سريع لبعض العملات
    try:
        symbols = get_top_usdt_symbols(limit=40)
    except Exception as e:
        print("daily symbols error:", e)
        symbols = []

    results: List[Any] = []
    for sym in symbols:
        try:
            d = generate_signal(sym, mode=mode.lower(), use_coinglass=False)
        except Exception:
            continue
        dec = d.get("decision", {})
        act = dec.get("action", "WAIT")
        score = dec.get("score", d.get("score", 0.0))
        if act in ("BUY", "SELL") and score >= 65:
            results.append((sym, act, score, dec))

    msg_lines: List[str] = []
    msg_lines.append("📰 تقرير يومي من B7A Ultra Bot:")
    msg_lines.append("")
    msg_lines.append(f"🪙 حالة BTC: {btc_trend} | Action: {btc_action} | Score: {btc_score:.0f}/100")
    msg_lines.append("")

    if results:
        results.sort(key=lambda x: x[2], reverse=True)
        best = results[:3]
        msg_lines.append("🔥 أفضل 3 فرص اليوم:")
        for symbol, action, score, decision in best:
            trend = decision.get("trend", "RANGING")
            grade = decision.get("grade", "C")
            msg_lines.append(
                f"• {symbol}: {action} | Grade: {grade} | Score: {score:.0f} | Trend: {trend}"
            )
    else:
        msg_lines.append("ما في فرص قوية جداً اليوم حسب الفلتر الحالي (الكل تقريباً WAIT).")

    msg_lines.append("")
    msg_lines.append("تقدر تستخدم /signal BTC لأي عملة تبي تشوف تحليلها بالتفصيل.")

    await update.message.reply_text("\n".join(msg_lines))


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يعرض ملخص أداء الإشارات من ملف اللوق."""
    summary = get_trades_summary()
    await update.message.reply_text(summary, parse_mode="HTML")


# ========= أوامر إدارة الـ Watchlist =========

async def add_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("استخدم: /add BTC")
        return

    symbol = _normalize_symbol(context.args[0])
    if symbol in WATCHLIST:
        await update.message.reply_text(f"{symbol} موجودة مسبقاً في قائمة المراقبة ✅")
        return

    WATCHLIST.add(symbol)
    await update.message.reply_text(f"✅ تمت إضافة {symbol} إلى قائمة المراقبة.")


async def remove_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("استخدم: /remove BTC")
        return

    symbol = _normalize_symbol(context.args[0])
    if symbol not in WATCHLIST:
        await update.message.reply_text(f"{symbol} غير موجودة في قائمة المراقبة.")
        return

    WATCHLIST.remove(symbol)
    await update.message.reply_text(f"🗑️ تمت إزالة {symbol} من قائمة المراقبة.")


async def list_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not WATCHLIST:
        await update.message.reply_text("قائمة المراقبة فاضية حالياً.")
        return

    symbols = ", ".join(sorted(WATCHLIST))
    await update.message.reply_text(f"👀 قائمة المراقبة الحالية:\n{symbols}")
