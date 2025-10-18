# FINAL BOT SCRIPT - V7 (yfinance Typo Fix)
import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
import os
import asyncio
from telegram.ext import Application, CommandHandler
import threading
from flask import Flask

# --- 1. LOAD SECRETS FROM ENVIRONMENT VARIABLES ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    print("FATAL ERROR: BOT_TOKEN and CHAT_ID environment variables not set.")
    exit()

# --- 2. WEB SERVER TO KEEP RENDER ALIVE ---
app = Flask(__name__)
@app.route('/')
def home():
    return "Bot is alive and running."

def run_web_server():
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

# --- 3. BOT SETTINGS & IN-MEMORY STATE ---
PAIRS = ['GBPUSD=X', 'GBPJPY=X', 'GBPAUD=X', 'EURGBP=X']
RISK_PER_TRADE_PCT = 0.5
RISK_REWARD_RATIO = 1.9
INITIAL_CAPITAL = 1000.0

portfolio = {
    'balance': INITIAL_CAPITAL, 'trades': [], 'peak_balance': INITIAL_CAPITAL,
    'max_drawdown': 0.0, 'start_time': datetime.now(timezone.utc)
}
trade_lock = threading.Lock()
last_checked_hour = -1
current_analysis = {pair: {"status": "Initializing...", "alert_sent": False} for pair in PAIRS}

# --- 4. TELEGRAM COMMAND HANDLERS (Unchanged) ---
async def start_command(update, context):
    await update.message.reply_text("Bot Online (v7: Typo Fixed). Commands: /start, /status, /stats, /analysis", parse_mode='Markdown')

async def status_command(update, context):
    uptime_delta = datetime.now(timezone.utc) - portfolio['start_time']
    await update.message.reply_text(f"✅ *Bot is ONLINE.*\n\nUptime: {str(uptime_delta).split('.')[0]}", parse_mode='Markdown')

async def stats_command(update, context):
    with trade_lock:
        total_trades = len(portfolio['trades']); wins = sum(1 for t in portfolio['trades'] if t['pnl'] > 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0; total_pnl = portfolio['balance'] - INITIAL_CAPITAL
        stats_msg = (f"*📊 STATS (Since Restart) 📊*\n"
                     f"Balance: ${portfolio['balance']:,.2f} (P/L: ${total_pnl:,.2f})\n"
                     f"Win Rate: {win_rate:.2f}% ({wins}/{total_trades})\n"
                     f"Max Drawdown: {portfolio['max_drawdown']*100:.2f}%\n"
                     f"--------------------\n")
        for pair_name in PAIRS:
            pair_trades = [t for t in portfolio['trades'] if t['pair'] == pair_name]; pair_total = len(pair_trades)
            pair_wins = sum(1 for t in pair_trades if t['pnl'] > 0); pair_win_rate = (pair_wins / pair_total * 100) if pair_total > 0 else 0
            stats_msg += f"{pair_name.replace('=X', '')}: {pair_win_rate:.2f}% ({pair_wins}/{pair_total})\n"
    await update.message.reply_text(stats_msg, parse_mode='Markdown')

async def analysis_command(update, context):
    message = "*🕵️ Live Market Analysis 🕵️*\n\n"
    for pair, analysis_data in current_analysis.items():
        message += f"*{pair.replace('=X', '')}:* {analysis_data['status']}\n"
    await update.message.reply_text(message, parse_mode='Markdown')

# --- 5. CORE BOT LOGIC (TYPO FIXED) ---
def update_portfolio(trade):
    with trade_lock:
        portfolio['balance'] += trade['pnl']; portfolio['trades'].append(trade); portfolio['peak_balance'] = max(portfolio['peak_balance'], portfolio['balance'])
        peak = portfolio['peak_balance']; current_dd = (peak - portfolio['balance']) / peak if peak > 0 else 0
        portfolio['max_drawdown'] = max(portfolio['max_drawdown'], current_dd)

async def check_for_signal(context):
    global last_checked_hour
    now = datetime.now(timezone.utc)
    if now.minute < 5 and now.hour != last_checked_hour:
        last_checked_hour = now.hour
        print(f"\n--- Running hourly check at {now.strftime('%Y-%m-%d %H:%M')} UTC ---")
        for pair in PAIRS:
            try:
                # --- THE FIX IS HERE ---
                # Changed 'showerrors' to the correct 'show_errors'
                h4_data = yf.download(pair, period='5d', interval='4h', progress=False, show_errors=False)
                h1_data = yf.download(pair, period='5d', interval='1h', progress=False, show_errors=False)
                
                if h1_data.empty or len(h1_data) < 2 or h4_data.empty or len(h4_data) < 1:
                    current_analysis[pair]["status"] = "⌛ Waiting for data."; current_analysis[pair]["alert_sent"] = False; continue
                prev_h1, last_h1 = h1_data.iloc[-2], h1_data.iloc[-1]
                relevant_h4 = h4_data[h4_data.index < last_h1.name].iloc[-1]
                h4_is_bullish = relevant_h4['Close'] > relevant_h4['Open']
                h4_trend = "BULLISH" if h4_is_bullish else "BEARISH"
                current_analysis[pair]["status"] = f"🔵 H4 is {h4_trend}. Looking for entry."; current_analysis[pair]["alert_sent"] = False
                signal_type = None
                is_buy_setup = h4_is_bullish and prev_h1['Close'] < prev_h1['Open']
                is_sell_setup = not h4_is_bullish and prev_h1['Close'] > prev_h1['Open']
                if is_buy_setup:
                    current_analysis[pair]["status"] = f"🟡 WAITING FOR BUY on {pair.replace('=X','')}. Need H1 close > {prev_h1['Close']:.5f}"
                    if not current_analysis[pair]["alert_sent"]:
                        await context.bot.send_message(chat_id=CHAT_ID, text=f" heads-up: Potential BUY setup on *{pair.replace('=X','')}*.", parse_mode='Markdown'); current_analysis[pair]["alert_sent"] = True
                    if last_h1['Close'] > prev_h1['Close']: signal_type, entry_price, stop_loss = 'BUY', last_h1['Close'], prev_h1['Low']
                elif is_sell_setup:
                    current_analysis[pair]["status"] = f"🟡 WAITING FOR SELL on {pair.replace('=X','')}. Need H1 close < {prev_h1['Close']:.5f}"
                    if not current_analysis[pair]["alert_sent"]:
                        await context.bot.send_message(chat_id=CHAT_ID, text=f" heads-up: Potential SELL setup on *{pair.replace('=X','')}*.", parse_mode='Markdown'); current_analysis[pair]["alert_sent"] = True
                    if last_h1['Close'] < prev_h1['Close']: signal_type, entry_price, stop_loss = 'SELL', last_h1['Close'], prev_h1['High']
                if signal_type:
                    current_analysis[pair]["status"] = f"🔥 {signal_type} SIGNAL TRIGGERED! 🔥"
                    risk_distance = abs(entry_price - stop_loss)
                    if risk_distance == 0: continue
                    take_profit = entry_price + (risk_distance * RISK_REWARD_RATIO) if signal_type == 'BUY' else entry_price - (risk_distance * RISK_REWARD_RATIO)
                    risk_amount = portfolio['balance'] * (RISK_PER_TRADE_PCT / 100.0); pnl = risk_amount * RISK_REWARD_RATIO
                    trade = {'pair': pair, 'type': signal_type, 'pnl': pnl}; update_portfolio(trade)
                    message = f"🚨 *{signal_type} Signal: {pair.replace('=X','')}*\n\nEntry: `{entry_price:.5f}`\nStop Loss: `{stop_loss:.5f}`\nTake Profit: `{take_profit:.5f}`"
                    await context.bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
                    current_analysis[pair]["alert_sent"] = False
            except Exception as e:
                current_analysis[pair]["status"] = f"Error: {e}"; print(f"Error processing {pair}: {e}")
        print("--- Hourly check complete ---")

# --- 6. MAIN APPLICATION SETUP (Unchanged) ---
async def main():
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("analysis", analysis_command))
    application.job_queue.run_repeating(check_for_signal, interval=60, first=10)
    async with application:
        print("Starting Telegram bot...")
        await application.start()
        await application.updater.start_polling()
        await application.bot.send_message(chat_id=CHAT_ID, text="✅ *Bot Online (v7: Typo Fixed)*")
        while True:
            await asyncio.sleep(3600)

if __name__ == "__main__":
    web_thread = threading.Thread(target=run_web_server)
    web_thread.daemon = True
    print("Starting web server thread...")
    web_thread.start()
    print("Starting main bot logic...")
    asyncio.run(main())
