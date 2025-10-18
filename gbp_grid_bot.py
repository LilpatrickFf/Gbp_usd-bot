import time
import threading
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
import os
import json
import asyncio
from telegram import Bot
from telegram.ext import Application, CommandHandler
from telegram.request import HTTPXRequest

# --- FILE FOR SAVING DATA ---
# Render uses an ephemeral filesystem, so we'll store data in a specific directory
DATA_DIR = "/var/data"
PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio_data.json")

# --- User Credentials & Strategy Settings ---
BOT_TOKEN = "8037184350:AAE8d_EErW9-BF1St1EUZMCuryZEVSNKsZE"
CHAT_ID = "7741540586"
PAIRS = ['GBPUSD=X', 'GBPJPY=X', 'GBPAUD=X', 'EURGBP=X']
RISK_PER_TRADE_PCT = 0.5
RISK_REWARD_RATIO = 1.9
INITIAL_CAPITAL = 1000.0

# --- Global State ---
trade_lock = threading.Lock()
last_checked_hour = -1
# This new dictionary will hold the live analysis status
current_analysis = {pair: "Initializing..." for pair in PAIRS}

# --- Portfolio Loading and Initialization ---
def load_portfolio():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            data = json.load(f)
            data['start_time'] = datetime.fromisoformat(data['start_time'])
            print("Portfolio data loaded successfully.")
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        print("No portfolio data found. Starting fresh.")
        return {
            'balance': INITIAL_CAPITAL, 'trades': [], 'peak_balance': INITIAL_CAPITAL,
            'max_drawdown': 0.0, 'start_time': datetime.now(timezone.utc)
        }

def save_portfolio():
    with trade_lock:
        data_to_save = portfolio.copy()
        data_to_save['start_time'] = data_to_save['start_time'].isoformat()
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print("Portfolio data saved.")

portfolio = load_portfolio()

# --- Telegram Command Handlers ---
async def start_command(update, context):
    await update.message.reply_text(
        "Welcome! Bot is running the final backtester strategy.\n\n"
        "Commands:\n"
        "`/start` - This message\n"
        "`/status` - Check bot uptime\n"
        "`/stats` - View performance statistics\n"
        "`/analysis` - See live market analysis",
        parse_mode='Markdown'
    )

async def status_command(update, context):
    uptime_delta = datetime.now(timezone.utc) - portfolio['start_time']
    await update.message.reply_text(f"✅ *Bot is ONLINE.*\n\nTotal Uptime (across restarts): {str(uptime_delta).split('.')[0]}", parse_mode='Markdown')

async def stats_command(update, context):
    await update.message.reply_text(get_stats_message(), parse_mode='Markdown')

async def analysis_command(update, context):
    """Handler for the new /analysis command."""
    message = "*🕵️ Live Market Analysis 🕵️*\n\n"
    for pair, status in current_analysis.items():
        message += f"*{pair.replace('=X', '')}:* {status}\n"
    await update.message.reply_text(message, parse_mode='Markdown')

# --- Core Logic Functions ---
def get_stats_message():
    with trade_lock:
        total_trades = len(portfolio['trades'])
        wins = sum(1 for t in portfolio['trades'] if t['pnl'] > 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_pnl = portfolio['balance'] - INITIAL_CAPITAL
        stats_msg = (f"*📊 PERFORMANCE STATS 📊*\n"
                     f"--------------------\n"
                     f"Balance: ${portfolio['balance']:,.2f} (P/L: ${total_pnl:,.2f})\n"
                     f"Win Rate: {win_rate:.2f}% ({wins}/{total_trades} trades)\n"
                     f"Max Drawdown: {portfolio['max_drawdown']*100:.2f}%\n"
                     f"--------------------\n*Win Rate by Pair:*\n")
        for pair_name in PAIRS:
            pair_trades = [t for t in portfolio['trades'] if t['pair'] == pair_name]
            pair_total = len(pair_trades)
            pair_wins = sum(1 for t in pair_trades if t['pnl'] > 0)
            pair_win_rate = (pair_wins / pair_total * 100) if pair_total > 0 else 0
            stats_msg += f"{pair_name.replace('=X', '')}: {pair_win_rate:.2f}% ({pair_wins}/{pair_total})\n"
        return stats_msg

def update_portfolio(trade):
    with trade_lock:
        portfolio['balance'] += trade['pnl']
        portfolio['trades'].append(trade)
        portfolio['peak_balance'] = max(portfolio['peak_balance'], portfolio['balance'])
        peak = portfolio['peak_balance']
        current_dd = (peak - portfolio['balance']) / peak if peak > 0 else 0
        portfolio['max_drawdown'] = max(portfolio['max_drawdown'], current_dd)
    save_portfolio()

async def check_for_signal(context):
    """Main checking function, now runs periodically."""
    global last_checked_hour
    now = datetime.now(timezone.utc)

    if now.minute in [2,3,4] and now.hour != last_checked_hour:
        last_checked_hour = now.hour
        print(f"\n--- Running hourly check at {now.strftime('%Y-%m-%d %H:%M')} UTC ---")
        
        for pair in PAIRS:
            try:
                h4_data = yf.download(pair, period='5d', interval='4h', progress=False, show_errors=False)
                h1_data = yf.download(pair, period='5d', interval='1h', progress=False, show_errors=False)
                
                if h1_data.empty or len(h1_data) < 2 or h4_data.empty or len(h4_data) < 1:
                    current_analysis[pair] = "Waiting for sufficient market data."
                    continue

                prev_h1, last_h1 = h1_data.iloc[-2], h1_data.iloc[-1]
                relevant_h4 = h4_data[h4_data.index < last_h1.name].iloc[-1]
                h4_is_bullish = relevant_h4['Close'] > relevant_h4['Open']
                
                h4_trend = "BULLISH" if h4_is_bullish else "BEARISH"
                current_analysis[pair] = f"H4 Trend is {h4_trend}. Looking for H1 reversal..."

                signal_type = None
                if h4_is_bullish and prev_h1['Close'] < prev_h1['Open'] and last_h1['Close'] > prev_h1['Close']:
                    signal_type = 'BUY'
                elif not h4_is_bullish and prev_h1['Close'] > prev_h1['Open'] and last_h1['Close'] < prev_h1['Close']:
                    signal_type = 'SELL'
                
                if signal_type:
                    current_analysis[pair] = f"🔥 {signal_type} SIGNAL FOUND! 🔥"
                    entry_price = last_h1['Close']
                    stop_loss = prev_h1['Low'] if signal_type == 'BUY' else prev_h1['High']
                    risk_distance = abs(entry_price - stop_loss)
                    if risk_distance == 0: continue
                    take_profit = entry_price + (risk_distance * RISK_REWARD_RATIO) if signal_type == 'BUY' else entry_price - (risk_distance * RISK_REWARD_RATIO)
                    risk_amount = portfolio['balance'] * (RISK_PER_TRADE_PCT / 100.0)
                    pnl = risk_amount * RISK_REWARD_RATIO
                    trade = {'pair': pair, 'type': signal_type, 'pnl': pnl}
                    update_portfolio(trade)
                    message = (f"🚨 *{signal_type} Signal: {pair.replace('=X','')}*\n\n"
                               f"Entry: `{entry_price:.5f}`\nStop Loss: `{stop_loss:.5f}`\nTake Profit: `{take_profit:.5f}`")
                    await context.bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
                else:
                    current_analysis[pair] = f"H4 Trend is {h4_trend}. No valid H1 signal found."

            except Exception as e:
                current_analysis[pair] = f"An error occurred during analysis: {e}"
                print(f"Error processing {pair}: {e}")
        print("--- Hourly check complete ---")

async def post_init(application):
    """Function to run after the bot has started."""
    await application.bot.send_message(chat_id=CHAT_ID, text="✅ *Bot online and ready on Render!*")
    # Start the periodic check
    application.job_queue.run_repeating(check_for_signal, interval=60, first=10)

def main():
    """Main function to set up and run the bot."""
    print("Starting bot...")
    # Using Application.builder for modern python-telegram-bot setup
    application = Application.builder().token(BOT_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("analysis", analysis_command))

    # Set the post_init function to run once at the start
    application.post_init = post_init

    # Run the bot until the user presses Ctrl-C
    application.run_polling()

if __name__ == "__main__":
    main()

