import asyncio
import json
import time
import sqlite3
from collections import deque, defaultdict
from datetime import datetime
import logging
import pandas as pd
import websockets
from binance.client import Client



# ---------------- CONFIG ----------------


api_key="mgMDvJNtetK1AukW4EOroL1Vq9dUIrkltYzBTATzXzCS9mX7a95nWIvvxoMAYcug"
api_secret= "sBZlBhLd5G0wEVDX3v47lZuAAAsreY9PJDvPEZ1G7f11uCXPjb2likvAJwcbdeGb"


# from binance.client import Client

API_KEY = api_key
API_SECRET = api_secret
# API_KEY = "your_testnet_api_key"
# API_SECRET = "your_testnet_api_secret"

REST_BASE = "https://testnet.binance.vision"
WS_BASE = "wss://stream.testnet.binance.vision/stream?streams="

client = Client(api_key, api_secret)
client.API_URL = 'https://testnet.binancefuture.com/fapi'
# ---------- Define alpha functions ----------

def alpha_momentum(symbol, timeframe, candles_df):
    """Simple momentum alpha: buy if last close > 10-period mean."""
    if len(candles_df) < 11:
        return 0
    last = candles_df['Close'].iloc[-1]
    mean10 = candles_df['Close'].iloc[-11:-1].mean()
    if last > mean10 * 1.001:
        return 1
    if last < mean10 * 0.999:
        return -1
    return 0


def alpha_mean_revert(symbol, timeframe, candles_df):
    """Mean reversion alpha: sell if short MA > long MA."""
    if len(candles_df) < 21:
        return 0
    short = candles_df['Close'].rolling(5).mean().iloc[-1]
    long = candles_df['Close'].rolling(20).mean().iloc[-1]
    if short > long * 1.002:
        return -1
    if short < long * 0.998:
        return 1
    return 0


# ---------- Per-symbol configuration ----------
CONFIG = [
    {"symbol": "BTCUSDT", "timeframe": "1m", "alpha": alpha_momentum, "candles_to_keep": 40},
    {"symbol": "SOLUSDT", "timeframe": "1m", "alpha": alpha_mean_revert, "candles_to_keep": 60},
    # Add more entries if needed
]

# -----------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

DB_FILE = "bot_data.db"

# -------- Database Setup --------
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

# -------- Client & Buffers --------
client = Client(API_KEY, API_SECRET)
client.API_URL = REST_BASE + "/api"

buffers = {}
for cfg in CONFIG:
    symbol = cfg["symbol"]
    timeframe = cfg["timeframe"]
    buffers[(symbol, timeframe)] = deque(maxlen=cfg["candles_to_keep"])

positions = defaultdict(float)

# -------- Helper Functions --------
def kline_payload_to_row(k):
    return {
        'open_time': int(k['t']),
        'open': float(k['o']),
        'high': float(k['h']),
        'low': float(k['l']),
        'close': float(k['c']),
        'volume': float(k['v']),
        'close_time': int(k['T'])
    }

async def persist_candle(symbol, timeframe, row):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''INSERT INTO candles(symbol,timeframe,open_time,open,high,low,close,volume,close_time)
                 VALUES (?,?,?,?,?,?,?,?,?)''',
              (symbol, timeframe, row['open_time'], row['open'], row['high'],
               row['low'], row['close'], row['volume'], row['close_time']))
    conn.commit()
    conn.close()

async def persist_order(resp):
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

# -------- Order Execution Logic --------
async def handle_signal(symbol, timeframe, signal):
    current_qty = positions.get(symbol, 0)
    logging.info(f"Signal for {symbol} {timeframe}: {signal}, current_qty={current_qty}")

    if signal == 1 and current_qty == 0:
        qty = 0.001 if symbol.startswith("BTC") else 1
        try:
            resp = client.order_market_buy(symbol=symbol, quantity=qty)
            positions[symbol] += float(qty)
            logging.info(f"Bought {qty} {symbol}")
            await persist_order(resp)
        except Exception:
            logging.exception("Buy order failed")

    elif signal == -1 and current_qty > 0:
        qty = current_qty
        try:
            resp = client.order_market_sell(symbol=symbol, quantity=qty)
            positions[symbol] -= float(qty)
            logging.info(f"Sold {qty} {symbol}")
            await persist_order(resp)
        except Exception:
            logging.exception("Sell order failed")

# -------- Stream Builder --------
def make_streams(config):
    streams = []
    for cfg in config:
        s = cfg["symbol"].lower()
        tf = cfg["timeframe"]
        streams.append(f"{s}@kline_{tf}")
    return "/".join(streams)

# -------- WebSocket Main Loop --------
async def ws_loop(config):
    streams = make_streams(config)
    url = WS_BASE + streams
    logging.info(f"Connecting to {url}")

    symbol_to_cfg = {(c["symbol"], c["timeframe"]): c for c in config}

    while True:
        try:
            async with websockets.connect(url, ping_interval=60) as ws:
                logging.info("Websocket connected")
                async for message in ws:
                    data = json.loads(message)
                    payload = data.get('data')
                    if payload is None or payload.get('e') != 'kline':
                        continue

                    k = payload['k']
                    symbol = payload['s']
                    timeframe = k['i']
                    key = (symbol, timeframe)

                    if key not in symbol_to_cfg:
                        continue  # ignore unwanted streams

                    cfg = symbol_to_cfg[key]
                    row = kline_payload_to_row(k)
                    buffers[key].append(row)
                    await persist_candle(symbol, timeframe, row)

                    df = pd.DataFrame(list(buffers[key]))
                    df = df.sort_values('open_time')
                    df['Close'] = df['close']

                    try:
                        sig = cfg["alpha"](symbol, timeframe, df)
                        if sig != 0:
                            await handle_signal(symbol, timeframe, sig)
                    except Exception:
                        logging.exception(f"Alpha {cfg['alpha'].__name__} failed for {symbol} {timeframe}")

        except Exception:
            logging.exception("Websocket connection failed - retrying in 5s")
            await asyncio.sleep(5)

# -------- Account Poller --------
async def account_poller():
    while True:
        try:
            acc = client.get_account()
            for b in acc['balances']:
                free = float(b['free'])
                if free > 0:
                    positions[b['asset']+'USDT'] = positions.get(b['asset']+'USDT', 0)
            logging.debug("Account polled")
        except Exception:
            logging.exception("Account poll failed")
        await asyncio.sleep(30)

# -------- Main Entry --------
async def main():
    init_db()
    # Preload candle buffers via REST
    for cfg in CONFIG:
        s, tf, n = cfg["symbol"], cfg["timeframe"], cfg["candles_to_keep"]
        try:
            kl = client.get_klines(symbol=s, interval=tf, limit=n)
            for k in kl:
                row = {
                    'open_time': int(k[0]),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'close_time': int(k[6])
                }
                buffers[(s, tf)].append(row)
        except Exception:
            logging.exception(f"Prefill failed for {s} {tf}")

    await asyncio.gather(ws_loop(CONFIG), account_poller())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutting down bot")
