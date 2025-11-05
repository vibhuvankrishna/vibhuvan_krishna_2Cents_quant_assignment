import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# ----------------- Existing Momentum & Mean Reversion -----------------
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

def alpha_random(symbol, timeframe, df):
    return np.random.randint(-1,2)

def alpha_bollinger(symbol, timeframe, df, window=20, std_multiplier=2):
    """
    Returns signal based on Bollinger Bands:
    1 = Buy (price below lower band)
    -1 = Sell (price above upper band)
    0 = Hold
    """
    if len(df) < window:
        return 0

    close = df['close']
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    
    upper_band = sma + std_multiplier * std
    lower_band = sma - std_multiplier * std
    
    last_close = close.iloc[-1]
    
    if last_close < lower_band.iloc[-1]:
        return 1
    elif last_close > upper_band.iloc[-1]:
        return -1
    else:
        return 0



def alpha_supertrend(symbol, timeframe, df, period=10, multiplier=3):
    """
    Returns signal based on SuperTrend:
    1 = Buy, -1 = Sell, 0 = Hold
    """
    if len(df) < period:
        return 0

    high = df['high']
    low = df['low']
    close = df['close']

    # ATR calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    hl2 = (high + low) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    
    # Supertrend direction
    supertrend = pd.Series(index=df.index)
    direction = 1  # 1 = bullish, -1 = bearish
    
    for i in range(1, len(df)):
        if close.iloc[i-1] <= upperband.iloc[i-1]:
            direction = 1
        elif close.iloc[i-1] >= lowerband.iloc[i-1]:
            direction = -1
        supertrend.iloc[i] = direction
    
    last_direction = supertrend.iloc[-1]
    
    return last_direction


# ----------------- Update ALPHAS dict -----------------
ALPHAS = {
    "momentum": alpha_momentum,
    "mean_revert": alpha_mean_revert,
    "random": alpha_random,
    "bb": alpha_bollinger,
    "st": alpha_supertrend
}
