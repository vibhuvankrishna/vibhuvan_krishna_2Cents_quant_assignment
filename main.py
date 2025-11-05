

import alphas

api_key="mgMDvJNtetK1AukW4EOroL1Vq9dUIrkltYzBTATzXzCS9mX7a95nWIvvxoMAYcug"
api_secret= "sBZlBhLd5G0wEVDX3v47lZuAAAsreY9PJDvPEZ1G7f11uCXPjb2likvAJwcbdeGb"


from binance.client import Client

API_KEY = api_key
API_SECRET = api_secret
import asyncio
import sqlite3
from datetime import datetime, timezone
import random

# ---------------------------- CONFIG ----------------------------
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
timeframes = [1, 5, 15]  # in minutes
alphas = ["alpha1", "alpha2", "alpha3"]
CANDLES_TO_KEEP = [100, 50, 30]

# ---------------------------- DATABASE ----------------------------
DB_NAME = "trading_bot.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timestamp TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            timeframe INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            order_type TEXT,
            price REAL,
            quantity REAL,
            timestamp TEXT,
            alpha TEXT
        )
    """)

    conn.commit()
    conn.close()

# ---------------------------- SIMULATION: WEBSOCKET ----------------------------
async def fake_websocket_feed(symbol):
    """Simulates live candle updates for each symbol."""
    while True:
        yield {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "open": random.uniform(100, 200),
            "high": random.uniform(200, 250),
            "low": random.uniform(90, 150),
            "close": random.uniform(150, 210),
            "volume": random.uniform(10, 100)
        }
        await asyncio.sleep(1)

# ---------------------------- DATA STORAGE ----------------------------
def store_candle(data, timeframe):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO candles (symbol, timestamp, open, high, low, close, volume, timeframe)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["symbol"], data["timestamp"], data["open"], data["high"],
        data["low"], data["close"], data["volume"], timeframe
    ))
    conn.commit()
    conn.close()

def store_order(symbol, order_type, price, qty, alpha):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO orders (symbol, order_type, price, quantity, timestamp, alpha)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (symbol, order_type, price, qty, datetime.now(timezone.utc).isoformat(), alpha))
    conn.commit()
    conn.close()

# ---------------------------- STRATEGY PLACEHOLDER ----------------------------
def run_alpha(symbol, alpha_name, data):
    """Replace this with your real alpha logic."""
    print(f"[{symbol}] Running {alpha_name} on close={data['close']:.2f}")
    # Dummy trading logic
    if data["close"] > data["open"]:
        store_order(symbol, "BUY", data["close"], 0.01, alpha_name)
    else:
        store_order(symbol, "SELL", data["close"], 0.01, alpha_name)

# ---------------------------- MAIN LOOP ----------------------------
async def trade_symbol(symbol, timeframe, alpha, candles_to_keep):
    feed = fake_websocket_feed(symbol)
    candles = []

    async for data in feed:
        utc_minute = datetime.now(timezone.utc).minute
        if utc_minute % timeframe == 0:
            # Save candle to DB
            store_candle(data, timeframe)
            candles.append(data)
            candles = candles[-candles_to_keep:]  # keep limited candles

            # Run strategy
            run_alpha(symbol, alpha, data)
        else:
            pass
            # print(f"[{symbol}] Skipping minute {utc_minute}, waiting for timeframe {timeframe}")
        await asyncio.sleep(1)

# ---------------------------- RUNNER ----------------------------
async def main():
    init_db()
    tasks = []
    for i in range(len(symbols)):
        tasks.append(asyncio.create_task(
            trade_symbol(symbols[i], timeframes[i], alphas[i], CANDLES_TO_KEEP[i])
        ))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
