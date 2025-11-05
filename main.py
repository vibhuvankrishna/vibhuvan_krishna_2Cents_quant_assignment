

# ---------------------------- CONFIG ----------------------------
import alphas

api_key="mgMDvJNtetK1AukW4EOroL1Vq9dUIrkltYzBTATzXzCS9mX7a95nWIvvxoMAYcug"
api_secret="sBZlBhLd5G0wEVDX3v47lZuAAAsreY9PJDvPEZ1G7f11uCXPjb2likvAJwcbdeGb"

API_KEY = api_key
API_SECRET = api_secret

import time
import sqlite3
import json
import logging
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict

import pandas as pd
import requests
from binance.client import Client

# ----------------- CONFIG -----------------
# API_KEY = "YOUR_API_KEY"
# API_SECRET = "YOUR_API_SECRET"

REST_BASE = "https://testnet.binance.vision"

DB_FILE = "bot_data1.db"

CONFIG = [
    {"symbol": "BTCUSDT", "timeframe": "1m", "alpha": "momentum", "candles_to_keep": 40, "qty": 0.001},
    {"symbol": "SOLUSDT", "timeframe": "1m", "alpha": "mean_revert", "candles_to_keep": 60, "qty": 1},
    {"symbol": "ETHUSDT", "timeframe": "3m", "alpha": "st", "candles_to_keep": 50, "qty": 0.01},
    {"symbol": "BNBUSDT", "timeframe": "1m", "alpha": "random", "candles_to_keep": 60, "qty": 0.1},
    {"symbol": "ADAUSDT", "timeframe": "3m", "alpha": "bb", "candles_to_keep": 50, "qty": 10},
]


# ---------- Logging ----------
import logging
from logging.handlers import RotatingFileHandler

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s", "%Y-%m-%d %H:%M:%S")

# Console handler (optional, keeps your current console prints)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# File handler: write logs to bot.log with rotation
file_handler = RotatingFileHandler("bot.log", maxBytes=5*1024*1024, backupCount=3)  # 5 MB per file, keep 3 backups
file_handler.setFormatter(log_formatter)

# Apply handlers to root logger
logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])

# ---------- Database Setup ----------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT, timeframe TEXT, open_time INTEGER, open REAL, high REAL,
            low REAL, close REAL, volume REAL, close_time INTEGER
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY, symbol TEXT, side TEXT, qty REAL, price REAL,
            status TEXT, timestamp INTEGER, raw TEXT
        )
    ''')
    conn.commit()
    conn.close()
    
import numpy as np

ALPHAS = {
    "momentum": alphas.alpha_momentum,
    "mean_revert": alphas.alpha_mean_revert,
    "st" :alphas.alpha_supertrend,
    "bb": alphas.alpha_bollinger,
    "random" : alphas.alpha_random
}

# ---------- Binance Client ----------
client = Client(API_KEY, API_SECRET)
client.API_URL = REST_BASE + "/api"

# ---------- Buffers & Trackers ----------
buffers = {}
positions = defaultdict(float)
last_signal_time = {}  # per symbol, tracks last candle we processed

for cfg in CONFIG:
    buffers[cfg["symbol"]] = deque(maxlen=cfg["candles_to_keep"])

# ---------- Helper Functions ----------
def get_klines(symbol, interval, limit=50):
    url = f"{REST_BASE}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades', 
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        numeric_cols = ['open','high','low','close','volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        # Add IST columns for human readability
        df['open_time_ist'] = pd.to_datetime(df['open_time']) + timedelta(hours=5, minutes=30)
        df['close_time_ist'] = pd.to_datetime(df['close_time']) + timedelta(hours=5, minutes=30)
        return df[['open_time','open','high','low','close','volume','close_time','open_time_ist','close_time_ist']]
    except Exception as e:
        logging.error(f"Failed to fetch klines for {symbol}: {e}")
        return pd.DataFrame()

def persist_candle(symbol, timeframe, row):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''INSERT INTO candles(symbol,timeframe,open_time,open,high,low,close,volume,close_time)
                 VALUES (?,?,?,?,?,?,?,?,?)''',
              (symbol, timeframe, int(row['open_time'].timestamp()*1000), row['open'], row['high'],
               row['low'], row['close'], row['volume'], int(row['close_time'].timestamp()*1000)))
    conn.commit()
    conn.close()

def persist_order(resp):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    order_id = str(resp.get('orderId', str(time.time())))
    c.execute('''INSERT OR REPLACE INTO orders(id,symbol,side,qty,price,status,timestamp,raw)
                 VALUES (?,?,?,?,?,?,?,?)''',
              (order_id,
               resp.get('symbol'),
               resp.get('side'),
               float(resp.get('origQty', 0)),
               float(resp.get('price', 0)),
               resp.get('status', ''),
               int(time.time()*1000),
               json.dumps(resp)))
    conn.commit()
    conn.close()

# ---------- Order Execution ----------
def handle_signal(symbol, timeframe, signal, qty):
    current_qty = positions.get(symbol, 0)
    now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    logging.info(f"[{now_ist.strftime('%Y-%m-%d %H:%M:%S')}] Signal for {symbol} {timeframe}: {signal}, current_qty={current_qty}")

    if signal == 1 and current_qty == 0:
        try:
            resp = client.order_market_buy(symbol=symbol, quantity=qty)
            positions[symbol] += float(qty)
            logging.info(f"[{now_ist.strftime('%Y-%m-%d %H:%M:%S')}] Bought {qty} {symbol}")
            persist_order(resp)
        except Exception:
            logging.exception("Buy order failed")

    elif signal == -1 and current_qty > 0:
        sell_qty = current_qty
        try:
            resp = client.order_market_sell(symbol=symbol, quantity=sell_qty)
            positions[symbol] -= float(sell_qty)
            logging.info(f"[{now_ist.strftime('%Y-%m-%d %H:%M:%S')}] Sold {sell_qty} {symbol}")
            persist_order(resp)
        except Exception:
            logging.exception("Sell order failed")

# ---------- PnL Tracking ----------
def compute_pnl():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    pnl = {}
    for sym, qty in positions.items():
        if qty == 0:
            pnl[sym] = 0
            continue
        c.execute("SELECT close FROM candles WHERE symbol=? ORDER BY close_time DESC LIMIT 1", (sym,))
        row = c.fetchone()
        if row:
            last_price = row[0]
            pnl[sym] = qty * last_price
        else:
            pnl[sym] = 0
    conn.close()
    return pnl

# ---------- Main Loop ----------
INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
    "1M": 2592000,
}


def get_next_timestamp_start(timeframe_seconds):

    right_now = datetime.now()
    now = right_now

    # Calculate the elapsed seconds since the start of the day
    elapsed_seconds = (now.hour * 3600) + (now.minute * 60) + now.second

    # Calculate the next candle start time in seconds since the start of the day
    next_candle_seconds = ((elapsed_seconds // timeframe_seconds) + 1) * timeframe_seconds

    # Convert seconds back to hours, minutes, and seconds
    next_candle_time = datetime(now.year, now.month, now.day) + timedelta(seconds=next_candle_seconds)
    print(next_candle_time.replace(microsecond=0))

    return next_candle_time.replace(microsecond=0)

for cfg in CONFIG:
    cfg["next_min"] = get_next_timestamp_start(INTERVAL_SECONDS[cfg["timeframe"]])
    # print(cfg["symbol"], cfg["next_min"])

    
def main():

    init_db()
    while True:
        for cfg in CONFIG:
            if datetime.now() >= cfg["next_min"]:
                # print( cfg)
                # cfg["next_min"] = get_next_timestamp_start(INTERVAL_SECONDS[cfg["timeframe"]])+timedelta(seconds=INTERVAL_SECONDS[cfg["timeframe"]])
                cfg["next_min"] += timedelta(seconds=INTERVAL_SECONDS[cfg["timeframe"]])

                symbol = cfg["symbol"]
                timeframe = cfg["timeframe"]
                alpha_func = ALPHAS[cfg["alpha"]]
                print("lin3")
                df = get_klines(symbol, timeframe, cfg["candles_to_keep"])
                if df.empty:
                    print("df is empty")
                    continue

                # ---------- NEW CODE START ----------
                if symbol not in last_signal_time:  # first time fetching data for this symbol
                    logging.info(f"Storing initial {len(df)} candles for {symbol} ({timeframe}) in DB...")
                    for _, row in df.iterrows():
                        persist_candle(symbol, timeframe, row)
                    last_signal_time[symbol] = df['close_time'].iloc[-1]
                    continue  # skip signal generation this first time
                # ---------- NEW CODE END ----------
                
                                
                last_candle_time = df['close_time'].iloc[-1]
                if last_signal_time.get(symbol) != last_candle_time:
                    # Call alpha function (from patched algo 1) on df
                    alpha_func = ALPHAS[cfg["alpha"]]
                    signal = alpha_func(symbol, timeframe, df)

                    # Execute order if signal exists
                    if signal != 0:
                        print(f"signal={signal}")
                        handle_signal(symbol, timeframe, signal, cfg["qty"])
                        last_signal_time[symbol] = last_candle_time
                        persist_candle(symbol, timeframe, df.iloc[-1])

                print("line last")
            # else:
            #     print("waiting ")

                pnl = compute_pnl()
                now_ist = datetime.now() # i deleted something
                logging.info(f"[{now_ist.strftime('%Y-%m-%d %H:%M:%S')}] Current PnL: {pnl}")
        time.sleep(1)  # loop speed

if __name__ == "__main__":
    print("strating bot")
    main()


