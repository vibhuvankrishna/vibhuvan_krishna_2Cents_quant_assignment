import sqlite3
import pandas as pd
import numpy as np
import vectorbt as vbt

# --- 1. Read SQLite tables ---
conn = sqlite3.connect("bot_data1.db")
df_candles = pd.read_sql_query("SELECT * FROM candles", conn)
df_orders  = pd.read_sql_query("SELECT * FROM orders", conn)
conn.close()

# --- 2. Convert timestamps ---
df_candles['open_time']  = pd.to_datetime(df_candles['open_time'], unit='ms')
df_candles['close_time'] = pd.to_datetime(df_candles['close_time'], unit='ms')
df_orders['timestamp']    = pd.to_datetime(df_orders['timestamp'], unit='ms')

# --- 3. Group candles by symbol and remove duplicates ---
symbols = df_candles['symbol'].unique()
candles_dict = {}
for sym in symbols:
    df_sym = df_candles[df_candles['symbol'] == sym].copy()
    df_sym = df_sym.groupby('open_time').last()  # remove duplicates
    candles_dict[sym] = df_sym

# --- 4. Prepare buy/sell signals per symbol ---
signals_dict = {}
for sym in symbols:
    df_orders_sym = df_orders[df_orders['symbol'] == sym].copy()
    df_orders_sym['buy_signal']  = df_orders_sym['side'] == 'BUY'
    df_orders_sym['sell_signal'] = df_orders_sym['side'] == 'SELL'

    buy_signal = df_orders_sym.set_index('timestamp')['buy_signal'].reindex(
        candles_dict[sym].index, method='ffill', fill_value=False
    ).astype(np.bool_)

    sell_signal = df_orders_sym.set_index('timestamp')['sell_signal'].reindex(
        candles_dict[sym].index, method='ffill', fill_value=False
    ).astype(np.bool_)

    signals_dict[sym] = pd.DataFrame({'buy': buy_signal, 'sell': sell_signal})

# --- 5. Run individual backtests ---
results_dict = {}
for sym in symbols:
    price = candles_dict[sym]['close'].astype(float)
    signals = signals_dict[sym]

    pf = vbt.Portfolio.from_signals(
        close=price,
        entries=signals['buy'],
        exits=signals['sell'],
        init_cash=10000,
        fees=0.001,
        size=np.inf,
        freq='1min'  # 1-minute candle frequency
    )

    results_dict[sym] = pf
    print(f"\n=== {sym} Backtest Results ===")
    print(pf.stats())
    pf.plot(title=f"{sym} Equity Curve").show()

# --- 6. Prepare combined portfolio ---
price_df = pd.DataFrame({sym: candles_dict[sym]['close'] for sym in symbols}).sort_index().ffill()
price_df = price_df.astype(float)

entries_df = pd.DataFrame({
    sym: signals_dict[sym]['buy'].reindex(price_df.index, method='ffill', fill_value=False).astype(np.bool_)
    for sym in symbols
})

exits_df = pd.DataFrame({
    sym: signals_dict[sym]['sell'].reindex(price_df.index, method='ffill', fill_value=False).astype(np.bool_)
    for sym in symbols
})

combined_pf = vbt.Portfolio.from_signals(
    close=price_df,
    entries=entries_df,
    exits=exits_df,
    init_cash=10000,
    fees=0.001,
    size=np.inf,
    freq='1min'
)

print("\n=== Combined Portfolio Results ===")
print(combined_pf.stats())

# --- 7. Plot combined portfolio equity curve ---
combined_pf.plot(title="Combined Portfolio Equity Curve",plot_orders=False)

# Optional: plot individual symbol equity curves from combined portfolio
for sym in symbols:
    combined_pf[sym].plot(title=f"{sym} Equity Curve from Combined Portfolio")
