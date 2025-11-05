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
REST_BASE = "https://testnet.binance.vision"
DB_FILE = "bot_data1.db"

CONFIG = [
    {"symbol": "BTCUSDT", "timeframe": "1m", "alpha": "momentum", "candles_to_keep": 40, "qty": 0.001},
    {"symbol": "SOLUSDT", "timeframe": "1m", "alpha": "mean_revert", "candles_to_keep": 60, "qty": 1},
    {"symbol": "ETHUSDT", "timeframe": "3m", "alpha": "momentum", "candles_to_keep": 50, "qty": 0.01},
]

# API_KEY = "YOUR_API_KEY"
# API_SECRET = "YOUR_API_SECRET"

api_key="mgMDvJNtetK1AukW4EOroL1Vq9dUIrkltYzBTATzXzCS9mX7a95nWIvvxoMAYcug"
api_secret="sBZlBhLd5G0wEVDX3v47lZuAAAsreY9PJDvPEZ1G7f11uCXPjb2likvAJwcbdeGb"


# from binance.client import Client

API_KEY = api_key
API_SECRET = api_secret

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

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
    logging.info("Database initialized")

# ---------- Alpha Functions (from Algo 1) ----------
def alpha_momentum(symbol, timeframe, df):
    if len(df) < 11:
        return 0
    last = df['close'].iloc[-1]
    mean10 = df['close'].iloc[-11:-1].mean()
    if last > mean10 * 1.001:
        return 1
    if last < mean10 * 0.999:
        return -1
    return 0

def alpha_mean_revert(symbol, timeframe, df):
    if len(df) < 21:
        return 0
    short = df['close'].rolling(5).mean().iloc[-1]
    long = df['close'].rolling(20).mean().iloc[-1]
    if short > long * 1.002:
        return -1
    if short < long * 0.998:
        return 1
    return 0

ALPHAS = {"momentum": alpha_momentum, "mean_revert": alpha_mean_revert}

# ---------- Binance Client ----------
client = Client(API_KEY, API_SECRET)
client.API_URL = REST_BASE + "/api"

# ---------- Buffers & Trackers ----------
buffers = {}
positions = defaultdict(float)
last_signal_time = {}

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
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''
            INSERT INTO candles(symbol,timeframe,open_time,open,high,low,close,volume,close_time)
            VALUES (?,?,?,?,?,?,?,?,?)
        ''', (
            symbol, timeframe, int(row['open_time'].timestamp()*1000), row['open'], row['high'],
            row['low'], row['close'], row['volume'], int(row['close_time'].timestamp()*1000)
        ))
        conn.commit()
        conn.close()
        logging.info(f"Candle persisted: {symbol} {timeframe} {row['close_time']}")
    except Exception:
        logging.exception("Persist candle failed")

def persist_order(resp):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        order_id = str(resp.get('orderId', str(time.time())))
        c.execute('''
            INSERT OR REPLACE INTO orders(id,symbol,side,qty,price,status,timestamp,raw)
            VALUES (?,?,?,?,?,?,?,?)
        ''', (
            order_id, resp.get('symbol'), resp.get('side'), float(resp.get('origQty',0)),
            float(resp.get('price',0)), resp.get('status',''), int(time.time()*1000), json.dumps(resp)
        ))
        conn.commit()
        conn.close()
        logging.info(f"Order persisted: {resp.get('symbol')} {resp.get('side')} {resp.get('origQty',0)}")
    except Exception:
        logging.exception("Persist order failed")

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
        pnl[sym] = qty * row[0] if row else 0
    conn.close()
    return pnl

# ---------- Time Helper ----------
INTERVAL_SECONDS = {"1m":60,"3m":180,"5m":300,"15m":900,"30m":1800,"1h":3600,"2h":7200,
                    "4h":14400,"6h":21600,"8h":28800,"12h":43200,"1d":86400,"3d":259200,
                    "1w":604800,"1M":2592000}

def get_next_timestamp_start(timeframe_seconds):
    now = datetime.now()
    elapsed = now.hour*3600 + now.minute*60 + now.second
    next_sec = ((elapsed // timeframe_seconds) + 1) * timeframe_seconds
    next_time = datetime(now.year, now.month, now.day) + timedelta(seconds=next_sec)
    return next_time.replace(microsecond=0)

for cfg in CONFIG:
    cfg["next_min"] = get_next_timestamp_start(INTERVAL_SECONDS[cfg["timeframe"]])

# ---------- Main Loop ----------
def main():
    init_db()
    while True:
        for cfg in CONFIG:
            if datetime.now() >= cfg["next_min"]:
                cfg["next_min"] += timedelta(seconds=INTERVAL_SECONDS[cfg["timeframe"]])
                symbol = cfg["symbol"]
                timeframe = cfg["timeframe"]
                alpha_func = ALPHAS[cfg["alpha"]]
                df = get_klines(symbol, timeframe, cfg["candles_to_keep"])
                if df.empty:
                    logging.warning(f"No data for {symbol} {timeframe}")
                    continue

                last_candle_time = df['close_time'].iloc[-1]
                if last_signal_time.get(symbol) != last_candle_time:
                    signal = alpha_func(symbol, timeframe, df)
                    if signal != 0:
                        handle_signal(symbol, timeframe, signal, cfg["qty"])
                        last_signal_time[symbol] = last_candle_time
                        persist_candle(symbol, timeframe, df.iloc[-1])
                        logging.info(f"Processed candle for {symbol} {timeframe} at {last_candle_time}")

                pnl = compute_pnl()
                now_ist = datetime.now() + timedelta(hours=5, minutes=30)
                # logging.info(f"[{now_ist.strftime('%Y-%m-%d %H:%M:%S')}] Current PnL: {pnl}")
        time.sleep(1)

if __name__ == "__main__":
    logging.info("Starting Binance Bot...")
    main()
