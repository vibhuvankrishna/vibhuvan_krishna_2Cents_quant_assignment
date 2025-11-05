"""
backtester.py - Safe version with plots
"""

from typing import Optional, Dict, Any
import sqlite3
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timezone
import matplotlib.pyplot as plt

# --- vectorbt import ---
try:
    import vectorbt as vbt
except Exception as e:
    raise ImportError(
        "vectorbt is required for backtesting. Install with `pip install vectorbt`. "
        f"Original error: {e}"
    )

# --- Import user's bot config & alpha definitions ---
try:
    import main as bot
except Exception as e:
    raise ImportError(
        "Could not import tester2.py. Ensure tester2.py is in the same folder. "
        f"Original error: {e}"
    )

DB_FILE = getattr(bot, 'DB_FILE', 'bot_data1.db')
CONFIG = getattr(bot, 'CONFIG', []) or []
ALPHAS = getattr(bot, 'ALPHAS', {}) or {}

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ======================================================
# Database loader
# ======================================================
def load_candles_from_db(symbol: str, timeframe: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    query = (
        "SELECT open_time, open, high, low, close, volume, close_time "
        "FROM candles WHERE symbol=? AND timeframe=? ORDER BY close_time"
    )
    df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
    conn.close()

    if df.empty:
        return df

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    df = df.set_index('close_time')

    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df.dropna()


# ======================================================
# Signal generation
# ======================================================
def generate_signals_causal(alpha_func, symbol: str, timeframe: str, df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    signals = np.zeros(n, dtype=int)

    for i in range(n):
        window = df.iloc[: i + 1].copy()
        try:
            s = alpha_func(symbol, timeframe, window)
            if s is None:
                s_int = 0
            elif isinstance(s, (int, float, np.integer, np.floating)):
                s_int = int(np.sign(s))
            elif isinstance(s, bool):
                s_int = 1 if s else -1
            else:
                s_int = 0
        except Exception:
            s_int = 0
        signals[i] = s_int

    # Debug print first few signals
    print(f"First 20 signals for {symbol}: {signals[:20]}")
    return signals


# ======================================================
# Single backtest
# ======================================================
def backtest_single(cfg: Dict[str, Any], commission_per_trade: float = 100.0) -> Optional[Dict[str, Any]]:
    symbol = cfg.get('symbol')
    timeframe = cfg.get('timeframe')
    qty = cfg.get('qty', 1)
    alpha_name = cfg.get('alpha')

    if not symbol or not timeframe or not alpha_name:
        print(f"Invalid cfg entry: {cfg}")
        return None

    if alpha_name not in ALPHAS:
        print(f"Alpha '{alpha_name}' not found for {symbol} - skipping.")
        return None

    df = load_candles_from_db(symbol, timeframe)
    if df.empty:
        print(f"No historical data for {symbol} {timeframe} - skipping.")
        return None

    close = df['close']
    alpha_func = ALPHAS[alpha_name]

    print(f"\nRunning causal signal generation for {symbol} {timeframe} ({len(df)} bars) using alpha `{alpha_name}`...")
    signals = generate_signals_causal(alpha_func, symbol, timeframe, df)

    entries = signals == 1
    exits = signals == -1

    try:
        pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            init_cash=10000.0,
            size=qty,
            freq=None,
            cash_sharing=False,
            accumulate=False
        )
    except Exception as e:
        print(f"vectorbt error for {symbol} {timeframe}: {e}")
        return None

    # --- Portfolio stats ---
    stats = {}
    try:
        stats_obj = pf.stats()
        if stats_obj is not None:
            if hasattr(stats_obj, "to_dict"):
                stats = stats_obj.to_dict()
            elif isinstance(stats_obj, dict):
                stats = stats_obj
    except Exception as e:
        print(f"Warning: could not compute stats for {symbol}: {e}")

    # --- Count trades ---
    n_trades = 0
    try:
        trades_obj = getattr(pf, 'trades', None)
        if trades_obj is not None:
            recs = getattr(trades_obj, 'records_readable', None) or getattr(trades_obj, 'records', None)
            if recs is not None:
                n_trades = len(recs)
            else:
                n_trades = int(getattr(pf, 'total_trades', 0) or 0)
    except Exception:
        n_trades = 0

    commission_total = commission_per_trade * n_trades

    # --- Final value ---
    try:
        val_series = pf.value()
        if hasattr(val_series, 'iloc'):
            final_value = float(val_series.iloc[-1]) - commission_total
        else:
            final_value = float(np.asarray(val_series)[-1]) - commission_total
    except Exception:
        final_value = None

    # --- Print summary ---
    print("\n--- Backtest summary ---")
    print(f"Symbol: {symbol}  Timeframe: {timeframe}  Alpha: {alpha_name}")
    print(f"Bars used: {len(df)}")
    print(f"Trades: {n_trades}  Flat commission/trade: {commission_per_trade}  Commission total: {commission_total}")
    if final_value is not None:
        pnl = final_value - 10000.0
        print(f"Start Cash: 10000.00   Final Value (after flat fees): {final_value:.2f}   Net PnL: {pnl:.2f}")
    total_return = stats.get("Total Return [%]", "N/A")
    sharpe = stats.get("Sharpe Ratio", "N/A")
    win_rate = stats.get("Win Rate [%]", "N/A")
    print(f"Total Return: {total_return}")
    print(f"Sharpe Ratio: {sharpe}")
    print(f"Win Rate: {win_rate}")
    print("------------------------\n")

    # --- Plots ---
    try:
        # Equity curve
        pf.value().vbt.plot(title=f"{symbol} {timeframe} Equity Curve").savefig(f"{PLOT_DIR}/{symbol}_{timeframe}_equity.png")
    except Exception as e:
        print(f"Could not save equity plot for {symbol}: {e}")

    try:
        # Price + trade markers
        plt.figure(figsize=(12,6))
        plt.plot(close.index, close.values, label='Close')
        entry_idx = close.index[entries.values] if np.any(entries.values) else []
        exit_idx = close.index[exits.values] if np.any(exits.values) else []

        if len(entry_idx) > 0:
            plt.scatter(entry_idx, close.loc[entry_idx], marker='^', color='g', s=80, label='Entry')
        if len(exit_idx) > 0:
            plt.scatter(exit_idx, close.loc[exit_idx], marker='v', color='r', s=80, label='Exit')

        plt.title(f"{symbol} {timeframe} Price & Trades")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/{symbol}_{timeframe}_price_trades.png")
        plt.close()
    except Exception as e:
        print(f"Could not save price/trade plot for {symbol}: {e}")

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "alpha": alpha_name,
        "portfolio": pf,
        "stats": stats,
        "n_trades": n_trades,
        "commission_total": commission_total,
        "final_value": final_value
    }


# ======================================================
# Run all backtests
# ======================================================
def run_all(only_symbol: Optional[str] = None):
    results = []
    for cfg in CONFIG:
        if only_symbol is not None and cfg.get('symbol') != only_symbol:
            continue
        res = backtest_single(cfg)
        if res is not None:
            results.append(res)
    return results


# ======================================================
# CLI Entry Point
# ======================================================
if __name__ == "__main__":
    only_symbol: Optional[str] = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Starting backtests (DB: {DB_FILE}) at {datetime.now(timezone.utc).isoformat()}")
    run_all(only_symbol)
    print("All backtests finished.")
