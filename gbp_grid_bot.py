# FINAL BOT SCRIPT - SECURE & STABLE FOR RENDER
import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
import os
import json
import asyncio
from telegram import Bot
from telegram.ext import Application, CommandHandler, JobQueue
import threading
from flask import Flask

# --- 1. LOAD SECRETS FROM ENVIRONMENT VARIABLES ---
# This is the secure way. Render provides these to the script.
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

# --- 3. PERSISTENT DATA STORAGE ---
DATA_DIR = "/data"
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio_data.json")

# --- BOT SETTINGS ---
PAIRS = ['GBPUSD=X', 'GBPJPY=X', 'GBPAUD=X', 'EURGBP=X']
RISK_PER_TRADE_PCT = 0.5
RISK_REWARD_RATIO = 1.9
INITIAL_CAPITAL = 1000.0

# --- GLOBAL STATE & PORTFOLIO MANAGEMENT ---
trade_lock = threading.Lock()
last_checked_hour = -1
current_analysis = {pair: "Initializing..." for pair in PAIRS}

def load_portfolio():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            data = json.load(f)
            data['start_time'] = datetime.fromisoformat(data['start_time'])
            print("Portfolio data loaded.")
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        print("No portfolio data. Starting fresh.")
        return {'balance': INITIAL_CAPITAL, 'trades': [], 'peak_balance': INITIAL_CAPITAL, 'max_drawdown': 0.0, 'start_time': datetime.now(timezone.utc)}

def save_portfolio():
    with trade_lock:
        data_to_save = portfolio.copy()
        data_to_save['start_time'] = data_to_save['start_time'].isoformat()
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print("Portfolio data saved.")

portfolio = load_portfolio()

# --- 4. TELEGRAM COMMAND HANDLERS ---
async def start_command(update, context):
    await update.message.reply_text("Bot is running securely. Commands: /start, /status, /stats, /analysis", parse_mode='Markdown')

async def status_command(update, context):
    uptime_delta = datetime.now(timezone.utc) - portfolio['start_time']
    await update.message.reply_text(f"✅ *Bot is ONLINE.*\n\nTotal Uptime: {str(uptime_delta).split('.')[0]}", parse_mode='Markdown')

async def stats_command(update, context):
    await update.message.reply_text(get_stats_message(), parse_mode='Markdown')

async def analysis_command(update, context):
    message = "*🕵️ Live Market Analysis 🕵️*\n\n"
    for pair, status in current_analysis.items():
        message += f"*{pair.replace('=X', '')}:* {status}\n"
    await update.message.reply_text(message, parse_mode='Markdown')

# --- 5. CORE BOT LOGIC ---
def get_stats_message():
    # This function remains synchronous
    with trade_lock:
        # ... (stats calculation logic is unchanged)
        total_trades = len(portfolio['trades']); wins = sum(1 for t in portfolio['trades'] if t['pnl'] > 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0; total_pnl = portfolio['balance'] - INITIAL_CAPITAL
        stats_msg = f"*📊 STATS 📊*\nBalance: ${portfolio['balance']:,.2f} (P/L: ${total_pnl:,.2f})\nWin Rate: {win_rate:.2f}% ({wins}/{total_trades})\nMax Drawdown: {portfolio['max_drawdown']*100:.2f}%\n--------------------\n"
        for pair_name in PAIRS:
            pair_trades = [t for t in portfolio['trades'] if t['pair'] == pair_name]; pair_total = len(pair_trades)
            pair_wins = sum(1 for t in pair_trades if t['pnl'] > 0); pair_win_rate = (pair_wins / pair_total * 100) if pair_total > 0 else 0
            stats_msg += f"{pair_name.replace('=X', '')}: {pair_win_rate:.2f}% ({pair_wins}/{pair_total})\n"
        return stats_msg

def update_portfolio(trade):
    # This function remains synchronous
    with trade_lock:
        # ... (portfolio update logic is unchanged)
        portfolio['balance'] += trade['pnl']; portfolio['trades'].append(trade); portfolio['peak_balance'] = max(portfolio['peak_balance'], portfolio['balance'])
        peak = portfolio['peak_balance']; current_dd = (peak - portfolio['balance']) / peak if peak > 0 else 0
        portfolio['max_drawdown'] = max(portfolio['max_drawdown'], current_dd)
    save_portfolio()

async def check_for_signal(context):
    """Main checking function, now runs as a scheduled job."""
    global last_checked_hour
    now = datetime.now(timezone.utc)
    # Check at the top of the hour
    if now.minute < 5 and now.hour != last_checked_hour:
        last_checked_hour = now.hour
        print(f"\n--- Running hourly check at {now.strftime('%Y-%m-%d %H:%M')} UTC ---")
        for pair in PAIRS:
            # ... (The actual trading logic for checking a signal is unchanged)
            try:
                h4_data = yf.download(pair, period='5d', interval='4h', progress=False, show_errors=False)
                h1_data = yf.download(pair, period='5d', interval='1h', progress=False, show_errors=False)
                if h1_data.empty or len(h1_data) < 2 or h4_data.empty or len(h4_data) < 1: current_analysis[pair] = "Waiting for data."; continue
                prev_h1, last_h1 = h1_data.iloc[-2], h1_data.iloc[-1]
                relevant_h4 = h4_data[h4_data.index < last_h1.name].iloc[-1]
                h4_is_bullish = relevant_h4['Close'] > relevant_h4['Open']
                h4_trend = "BULLISH" if h4_is_bullish else "BEARISH"
                current_analysis[pair] = f"H4 is {h4_trend}. No H1 signal."
                signal_type, entry_price, stop_loss = None, 0, 0
                if h4_is_bullish and prev_h1['Close'] < prev_h1['Open'] and last_h1['Close'] > prev_h1['Close']: signal_type, entry_price, stop_loss = 'BUY', last_h1['Close'], prev_h1['Low']
                elif not h4_is_bullish and prev_h1['Close'] > prev_h1['Open'] and last_h1['Close'] < prev_h1['Close']: signal_type, entry_price, stop_loss = 'SELL', last_h1['Close'], prev_h1['High']
                if signal_type:
                    current_analysis[pair] = f"🔥 {signal_type} SIGNAL FOUND! 🔥"
                    risk_distance = abs(entry_price - stop_loss)
                    if risk_distance == 0: continue
                    take_profit = entry_price + (risk_distance * RISK_REWARD_RATIO) if signal_type == 'BUY' else entry_price - (risk_distance * RISK_REWARD_RATIO)
                    risk_amount = portfolio['balance'] * (RISK_PER_TRADE_PCT / 100.0); pnl = risk_amount * RISK_REWARD_RATIO
                    trade = {'pair': pair, 'type': signal_type, 'pnl': pnl}; update_portfolio(trade)
                    message = f"🚨 *{signal_type} Signal: {pair.replace('=X','')}*\n\nEntry: `{entry_price:.5f}`\nStop Loss: `{stop_loss:.5f}`\nTake Profit: `{take_profit:.5f}`"
                    await context.bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
            except Exception as e:
                current_analysis[pair] = f"Error: {e}"; print(f"Error processing {pair}: {e}")
        print("--- Hourly check complete ---")

# --- 6. MAIN APPLICATION SETUP ---
def main():
    """Main function to set up and run the bot."""
    # Start the web server in a separate thread to keep Render alive
    print("Starting web server in a separate thread...")
    web_thread = threading.Thread(target=run_web_server)
    web_thread.daemon = True
    web_thread.start()

    # --- JOB QUEUE FIX ---
    # Create the Application object first
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Now that the application exists, its job_queue also exists.
    # We can schedule the repeating task.
    job_queue = application.job_queue
    job_queue.run_repeating(check_for_signal, interval=60, first=10)

    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("analysis", analysis_command))

    # Start the bot
    print("Starting Telegram bot polling...")
    application.run_polling()

if __name__ == "__main__":
    main()
