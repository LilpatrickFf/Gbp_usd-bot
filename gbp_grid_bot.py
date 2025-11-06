# deriv_multi_tf_backtester.py
# Multi-TF backtester implementing:
# - Main trendline on D1/H4/H1
# - Counter trendline on lower TF (D1->H4, H4->M15, H1->M5)
# - Counter trendline must have >=3 touches (past only) and not be near S/R
# - Breakout on counter TF, retest on counter TF, M5 price-action confirmation
# - LTF alignment: M1, M5, M15 must align with chosen HTF bias
# - No look-ahead: decisions use only candles up to the current bar
#
# Notes: This is complex — tune constants below to your taste.

import asyncio, json, time, csv
import websockets, nest_asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

nest_asyncio.apply()

# ---------------- CONFIG ----------------
APP_ID = "1089"
DERIV_TOKEN = "t6OJ2rMF5kEOZtL"   # dummy; replace if you want authorized endpoints
DERIV_WS = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

SYMBOLS = ["R_25", "R_75"]
MONTHS_HISTORY = 6
EXPORT_CSV = True
CSV_FILE = "multi_tf_trades.csv"

# granularities (seconds)
GR_M1 = 60
GR_M5 = 5*60
GR_M15 = 15*60
GR_H1 = 60*60
GR_H4 = 4*60*60
GR_D1 = 24*60*60

CHUNK_COUNT = 1000  # per request

# strategy params (tune)
SWING_WINDOW = 6
MIN_COUNTER_TOUCHES = 3
TOUCH_TOL_PCT = 0.0012     # tolerance for a candle touching a line (0.12%)
SR_DISTANCE_MIN_PCT = 0.0025  # min distance from nearby S/R (0.25%)
BREAK_TOL = 0.0015
RETEST_TOL = 0.0010
PA_LOOKAHEAD_M5 = 6
SL_BUFFER_PCT = 0.0008
LOOKAHEAD_M5_FOR_RESULT = 200

# ---------------- helpers ----------------
def df_from_candles_json(j):
    df = pd.DataFrame(j['candles'])
    df['time'] = pd.to_datetime(df['epoch'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df = df[['open','high','low','close','epoch']].astype(float)
    return df

def detect_swings(df, window=SWING_WINDOW):
    highs, lows = [], []
    arr_h = df['high'].values; arr_l = df['low'].values; n=len(df)
    for i in range(window, n-window):
        if arr_h[i] == arr_h[i-window:i+window+1].max():
            highs.append((i, float(arr_h[i])))
        if arr_l[i] == arr_l[i-window:i+window+1].min():
            lows.append((i, float(arr_l[i])))
    return highs, lows

def fit_line(points):
    if not points or len(points) < 2:
        return None
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    m,b = np.polyfit(x,y,1)
    return (float(m), float(b))

def price_on_line(mb, x):
    m,b = mb
    return m*x + b

def body_size(o,c): return abs(c-o)
def candle_range(h,l): return h-l
def is_strong_bull(prev_o, prev_c, o, c, h, l):
    body = body_size(o,c); rng = candle_range(h,l)
    return (body >= 0.6*rng) or ((c>prev_c) and (o<prev_o) and (c>o))
def is_strong_bear(prev_o, prev_c, o, c, h, l):
    body = body_size(o,c); rng = candle_range(h,l)
    return (body >= 0.6*rng) or ((c<prev_c) and (o>prev_o) and (c<o))

# map line defined in TF-A index space onto TF-B index positions by nearest timestamp mapping
def map_line_to_target(mb, src_df, target_df):
    if mb is None:
        return None
    m,b = mb
    src_positions = np.arange(len(src_df))
    src_prices = m*src_positions + b
    s = pd.Series(src_prices, index=src_df.index)
    nearest_idx = s.index.get_indexer(target_df.index, method='nearest')
    mapped = s.iloc[nearest_idx].values
    return mapped

# check touches count for a line on a df (past-only up to index_end)
def count_touches_on_line(line_mapped, df, upto_idx, is_support=True, tol_pct=TOUCH_TOL_PCT):
    # count distinct candles where low (for support) or high (for resistance) gets within tol of line
    touches = []
    for i in range(0, upto_idx+1):
        lp = float(line_mapped[i])
        if is_support:
            price = float(df['low'].iloc[i])
        else:
            price = float(df['high'].iloc[i])
        if abs(price - lp)/lp <= tol_pct:
            touches.append(i)
    # collapse touches that cluster (same touch window); return count of unique touches
    # We'll count unique indices (already)
    return len(touches), touches

# simple S/R candidate detection: use MTF swings as zones (returns nearest distance pct)
def nearest_sr_distance_pct(price, swings):
    # swings: list of (idx, price)
    if not swings:
        return 1.0
    diffs = [abs(price - p)/p for (_,p) in swings]
    return min(diffs)

# ----------------- WebSocket fetch (paginated) -----------------
async def fetch_candles_range(symbol, granularity, months=6):
    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(days=30*months)
    end_epoch = int(now.timestamp())
    all_frames = []
    while True:
        async with websockets.connect(DERIV_WS) as ws:
            if DERIV_TOKEN:
                await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
                try:
                    await ws.recv()
                except:
                    pass
            req = {"ticks_history": symbol, "style":"candles", "granularity":granularity, "count":CHUNK_COUNT, "end": end_epoch}
            await ws.send(json.dumps(req))
            raw = await ws.recv()
            j = json.loads(raw)
            if "error" in j:
                raise RuntimeError(f"Deriv error: {j['error']}")
            if "candles" not in j:
                raise RuntimeError("Unexpected response")
            df = df_from_candles_json(j)
            df = df.sort_index()
            all_frames.insert(0, df)
            earliest_epoch = int(df['epoch'].iloc[0]) if len(df)>0 else None
            if earliest_epoch is None or earliest_epoch <= int(start_dt.timestamp()):
                break
            end_epoch = earliest_epoch - 1
            await asyncio.sleep(0.15)
    if not all_frames:
        return pd.DataFrame(columns=['open','high','low','close','epoch'])
    full = pd.concat(all_frames)
    full = full[full.index >= pd.to_datetime(start_dt)]
    return full.sort_index()

# ----------------- Core backtest: forward-walking (no lookahead) -----------------
def determine_bias_from_df(df):
    # simple bias: last close vs midpoint of last N candle
    try:
        last_close = df['close'].iloc[-1]
        mid = (df['high'].iloc[-10:].max() + df['low'].iloc[-10:].min())/2.0
        return "BUY" if last_close > mid else "SELL"
    except:
        return "NEUTRAL"

def backtest_for_symbol(symbol, dfs):
    """
    dfs: dict with keys 'D1','H4','H1','M15','M5','M1' -> DataFrames (chronological ascending)
    We iterate along M15 timeline (since counter trend usually drawn on M15 or H4 depending).
    For each M15 index t, we build HTF main line using HTF candles <= t (mapped to their indices),
    then build counter line on lower TF using candles <= corresponding time, ensure 3+ touches,
    detect breakout at current M15 open/close, then retest in future M15 bars, then wait M5 confirmation.
    """
    df_d1 = dfs['D1']; df_h4 = dfs['H4']; df_h1 = dfs['H1']
    df_m15 = dfs['M15']; df_m5 = dfs['M5']; df_m1 = dfs['M1']

    trades = []

    # precompute swings for S/R usage (we'll recalc per-window for strictness but keep global)
    # We'll iterate M15 index from a safe start (skip initial SWING_WINDOW bars)
    start_idx = SWING_WINDOW + 50
    for t in range(start_idx, len(df_m15)-20):
        # timestamp at current M15 bar (we only use past candles up to t)
        t_time = df_m15.index[t]

        # For each possible main TF try to see if it produces a valid setup
        for main_tf in ['D1','H4','H1']:
            # map main TF df and counter TF mapping
            if main_tf == 'D1':
                htf_df = df_d1
                counter_tf = 'H4'
                counter_df = df_h4
                counter_gran = GR_H4
            elif main_tf == 'H4':
                htf_df = df_h4
                counter_tf = 'M15'
                counter_df = df_m15
                counter_gran = GR_M15
            else:  # H1
                htf_df = df_h1
                counter_tf = 'M5'
                counter_df = df_m5
                counter_gran = GR_M5

            # determine last available HTF index up to t_time
            htf_pos = htf_df.index.get_indexer([t_time], method='nearest')[0]
            # only use HTF candles up to that pos for line fitting
            htf_window = htf_df.iloc[:htf_pos+1]
            if len(htf_window) < 10:
                continue

            # detect htf swings and decide bias (BUY if support-based)
            htf_highs, htf_lows = detect_swings(htf_window, window=SWING_WINDOW)
            bias = determine_bias_from_df(htf_window)

            # choose main trendline points: support for BUY, resistance for SELL
            if bias == "BUY":
                pts = htf_lows[-8:]
                is_support_main = True
            else:
                pts = htf_highs[-8:]
                is_support_main = False

            if len(pts) < 2:
                continue
            main_line = fit_line(pts)
            if main_line is None:
                continue

            # Now we need counter trendline on counter_df but using only candles up to the current time
            # Find counter_df position corresponding to current M15 time (nearest)
            counter_pos = counter_df.index.get_indexer([t_time], method='nearest')[0]
            counter_window = counter_df.iloc[:counter_pos+1]
            # detect swings on counter window
            c_highs, c_lows = detect_swings(counter_window, window=SWING_WINDOW)
            # For counter trendline we want opposite orientation relative to main trendline direction:
            # If main is BUY (HTF support), counter trendline should be a descending counter-line (resistance) drawn on lower TF
            # We'll attempt to fit a line from the side that makes sense:
            if bias == "BUY":
                # counter trendline = resistance on counter TF (use highs)
                c_pts = c_highs[-12:]
                is_support_counter = False
            else:
                # counter trendline = support on counter TF (use lows)
                c_pts = c_lows[-12:]
                is_support_counter = True

            if len(c_pts) < MIN_COUNTER_TOUCHES:
                # not enough candidate points for counter line
                continue
            # Fit candidate counter line
            counter_line = fit_line(c_pts)
            if counter_line is None:
                continue
            # Map main HTF line into counter_df space if needed (for SR proximity checks)
            # Also map counter_line into counter_df (it's already defined in counter_df index space)
            # Count touches on counter line (past only up to counter_pos)
            touches_count, touches_idx = count_touches_on_line(
                # map counter_line trivially by computing line price at integer indices
                [price_on_line(counter_line, i) for i in range(len(counter_window))],
                counter_window, upto_idx=counter_pos, is_support=is_support_counter, tol_pct=TOUCH_TOL_PCT
            )
            if touches_count < MIN_COUNTER_TOUCHES:
                continue

            # Ensure counter line not near S/R: compute nearest distance to HTF swings (use htf swings)
            # take representative price = line at most recent index
            line_price_now = price_on_line(counter_line, counter_pos)
            # compute nearest htf swing distance
            htf_swings = (htf_highs + htf_lows)
            sr_dist = nearest_sr_distance_pct(line_price_now, htf_swings)
            if sr_dist <= SR_DISTANCE_MIN_PCT:
                # too close to a major HTF S/R -> skip (user requested not close to S/R)
                continue

            # Now check breakout: breakout must occur on counter TF. We'll check if at current M15 index t there is
            # a recent breakout of the counter line. For counter_tf==M15 the breakout should be at or before t.
            # We'll check last few candles (looking-back, not forward).
            breakout_detected = False
            breakout_idx = None
            # examine last 6 counter bars up to counter_pos to detect close beyond line by BREAK_TOL
            lookback = 6
            start_chk = max(0, counter_pos - lookback + 1)
            for chk in range(start_chk, counter_pos+1):
                lp = price_on_line(counter_line, chk)
                close_val = float(counter_window['close'].iloc[chk])
                if bias == "BUY" and close_val > lp*(1+BREAK_TOL):
                    breakout_detected = True; breakout_idx = chk; break
                if bias == "SELL" and close_val < lp*(1-BREAK_TOL):
                    breakout_detected = True; breakout_idx = chk; break
            if not breakout_detected:
                continue

            # After breakout, we require a RETEST on counter TF: price must come back within RETEST_TOL
            # Look for a retest in subsequent counter bars (but only those available up to t; no future)
            # Because we are forward-walking, retest must happen after breakout but before or at counter_pos
            found_retest = False
            retest_idx = None
            for r in range(breakout_idx+1, counter_pos+1):
                if is_support_counter:
                    # retest means high touches near line for support
                    price_r = float(counter_window['high'].iloc[r])
                else:
                    price_r = float(counter_window['low'].iloc[r])
                lp_r = price_on_line(counter_line, r)
                if abs(price_r - lp_r)/lp_r <= RETEST_TOL:
                    found_retest = True; retest_idx = r; break
            if not found_retest:
                continue

            # Now require LTF alignment: M1, M5, M15 bias must match HTF bias
            # For this trade we require M15, M5, M1 bias (computed using only candles up to times prior to t_time)
            # compute M15 bias (using window up to t)
            m15_bias = determine_bias_from_df(df_m15.iloc[:t+1])
            m5_pos_now = df_m5.index.get_indexer([t_time], method='nearest')[0]
            m5_bias = determine_bias_from_df(df_m5.iloc[:m5_pos_now+1])
            m1_pos_now = df_m1.index.get_indexer([t_time], method='nearest')[0]
            m1_bias = determine_bias_from_df(df_m1.iloc[:m1_pos_now+1])
            if not (m15_bias == bias == m5_bias == m1_bias):
                # alignment fails
                continue

            # Now we have breakout + retest (all on past data up to t_time) and LTF alignment.
            # Next we need to wait for a PA confirmation on M5 AFTER the retest.
            # We must only consider M5 bars that come after the M5 index corresponding to retest_idx
            retest_time = counter_window.index[retest_idx]
            m5_retest_pos = df_m5.index.get_indexer([retest_time], method='nearest')[0]
            # Look forward on available M5 bars but only up to current time (no future)
            # We'll consider M5 bars AFTER retest that appear at or before t_time
            max_m5_pos = df_m5.index.get_indexer([t_time], method='nearest')[0]
            confirmed = False; entry_m5_idx = None; entry_price = None
            for k in range(m5_retest_pos+1, min(m5_retest_pos+1+PA_LOOKAHEAD_M5, max_m5_pos+1)):
                prev_o = float(df_m5['open'].iloc[k-1]); prev_c = float(df_m5['close'].iloc[k-1])
                o = float(df_m5['open'].iloc[k]); c = float(df_m5['close'].iloc[k])
                high_k = float(df_m5['high'].iloc[k]); low_k = float(df_m5['low'].iloc[k])
                # compute corresponding counter pos for this m5 bar (nearest)
                counter_pos_for_m5 = counter_window.index.get_indexer([df_m5.index[k]], method='nearest')[0]
                lp_m5 = price_on_line(counter_line, counter_pos_for_m5)
                # require candle close beyond line and strong PA
                if bias == "BUY" and (c > lp_m5*(1+0.00025)) and is_strong_bull(prev_o, prev_c, o, c, high_k, low_k):
                    confirmed=True; entry_m5_idx=k; entry_price=float(df_m5['open'].iloc[k]); break
                if bias == "SELL" and (c < lp_m5*(1-0.00025)) and is_strong_bear(prev_o, prev_c, o, c, high_k, low_k):
                    confirmed=True; entry_m5_idx=k; entry_price=float(df_m5['open'].iloc[k]); break
            if not confirmed:
                continue

            # Determine TP1 (next M15 swing) using only M15 swings up to current t
            m15_highs, m15_lows = detect_swings(df_m15.iloc[:t+1], window=SWING_WINDOW)
            if bias == "BUY":
                tp1 = next_swing_after(retest_idx, m15_highs)
            else:
                tp1 = next_swing_after(retest_idx, m15_lows)
            # Determine TP2 (next HTF swing in HTF space after corresponding htf_pos)
            htf_highs, htf_lows = detect_swings(htf_window, window=SWING_WINDOW)
            if bias == "BUY":
                tp2 = next_swing_after(htf_pos, htf_lows)
            else:
                tp2 = next_swing_after(htf_pos, htf_highs)
            if tp1 is None and tp2 is None:
                # require at least one target
                continue

            sl = price_on_line(counter_line, retest_idx) * (1 - SL_BUFFER_PCT) if bias=="BUY" else price_on_line(counter_line, retest_idx)*(1+SL_BUFFER_PCT)
            # simulate trade result scanning forward M5 closes AFTER entry_m5_idx but only up to current available M5 bars (no future)
            outcome="NO_HIT"; exit_price=None
            for fut in range(entry_m5_idx, min(entry_m5_idx + LOOKAHEAD_M5_FOR_RESULT, max_m5_pos+1)):
                close_f = float(df_m5['close'].iloc[fut])
                if bias=="BUY":
                    if tp1 and close_f >= tp1:
                        outcome="TP1"; exit_price=tp1; break
                    if tp2 and close_f >= tp2:
                        outcome="TP2"; exit_price=tp2; break
                    if close_f <= sl:
                        outcome="SL"; exit_price=sl; break
                else:
                    if tp1 and close_f <= tp1:
                        outcome="TP1"; exit_price=tp1; break
                    if tp2 and close_f <= tp2:
                        outcome="TP2"; exit_price=tp2; break
                    if close_f >= sl:
                        outcome="SL"; exit_price=sl; break

            # record trade (timestamps reflect when entry occurred)
            trades.append({
                "symbol": symbol, "main_tf": main_tf, "counter_tf": counter_tf,
                "bias": bias, "retest_time": retest_time, "entry_time": df_m5.index[entry_m5_idx],
                "entry": entry_price, "sl": sl, "tp1": tp1, "tp2": tp2, "outcome": outcome, "exit_price": exit_price
            })

            # break after a valid trade for this t (avoid double-trading same M15 bar for different main_tf)
            break

    # produce summary
    wins = sum(1 for t in trades if str(t['outcome']).startswith("TP"))
    losses = sum(1 for t in trades if t['outcome']=="SL")
    total = len(trades)
    winrate = (wins/total*100) if total>0 else 0
    return {"symbol":symbol, "trades":trades, "summary":{"total":total,"wins":wins,"losses":losses,"winrate":winrate}}
# ---------------- runner ----------------
async def run_all():
    results=[]
    for sym in SYMBOLS:
        print(f"\nFetching data for {sym} (~{MONTHS_HISTORY} months)...")
        df_m1 = await fetch_candles_range(sym, GR_M1, months=MONTHS_HISTORY)
        df_m5 = await fetch_candles_range(sym, GR_M5, months=MONTHS_HISTORY)
        df_m15 = await fetch_candles_range(sym, GR_M15, months=MONTHS_HISTORY)
        df_h1 = await fetch_candles_range(sym, GR_H1, months=MONTHS_HISTORY)
        df_h4 = await fetch_candles_range(sym, GR_H4, months=MONTHS_HISTORY)
        df_d1 = await fetch_candles_range(sym, GR_D1, months=MONTHS_HISTORY)

        print(f"Fetched: M1={len(df_m1)}, M5={len(df_m5)}, M15={len(df_m15)}, H1={len(df_h1)}, H4={len(df_h4)}, D1={len(df_d1)}")

        dfs = {'M1':df_m1, 'M5':df_m5, 'M15':df_m15, 'H1':df_h1, 'H4':df_h4, 'D1':df_d1}
        res = backtest_for_symbol(sym, dfs)
        results.append(res)
        s=res['summary']
        print(f"RESULT {sym}: Trades={s['total']}, Wins={s['wins']}, Losses={s['losses']}, Winrate={s['winrate']:.2f}%")

        # export CSV
        if EXPORT_CSV and res['trades']:
            with open(CSV_FILE, "a", newline="") as f:
                w=csv.writer(f)
                if f.tell()==0:
                    w.writerow(["symbol","main_tf","counter_tf","bias","retest_time","entry_time","entry","sl","tp1","tp2","outcome","exit_price"])
                for t in res['trades']:
                    w.writerow([t.get(k) for k in ["symbol","main_tf","counter_tf","bias","retest_time","entry_time","entry","sl","tp1","tp2","outcome","exit_price"]])

    return results

# run
if __name__ == "__main__":
    out = asyncio.get_event_loop().run_until_complete(run_all())
    print("\nDone.")
    for r in out:
        print(r['symbol'], r['summary'])
