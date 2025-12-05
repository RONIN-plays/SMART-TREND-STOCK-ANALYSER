# utils/preprocess.py
import pandas as pd
import numpy as np

def ensure_dateindex(df, date_col='Date'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df.set_index(date_col, inplace=True)
    return df

def add_sma(df, window=20, column='Close', col_name=None):
    if col_name is None:
        col_name = f"SMA_{window}"
    df[col_name] = df[column].rolling(window=window, min_periods=1).mean()
    return df

def add_ema(df, span=20, column='Close', col_name=None):
    if col_name is None:
        col_name = f"EMA_{span}"
    df[col_name] = df[column].ewm(span=span, adjust=False).mean()
    return df

def add_rsi(df, period=14, column='Close', col_name='RSI_14'):
    delta = df[column].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=period, min_periods=1).mean()
    roll_down = down.rolling(window=period, min_periods=1).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    df[col_name] = rsi.fillna(0)
    return df

def add_macd(df, fast=12, slow=26, signal=9, column='Close'):
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def add_bollinger(df, window=20, column='Close', n_std=2):
    sma = df[column].rolling(window=window, min_periods=1).mean()
    std = df[column].rolling(window=window, min_periods=1).std().fillna(0)
    df['Bollinger_Mid'] = sma
    df['Bollinger_Upper'] = sma + (n_std * std)
    df['Bollinger_Lower'] = sma - (n_std * std)
    return df

def add_volume_ma(df, window=20, column='Volume'):
    df[f'Volume_MA_{window}'] = df[column].rolling(window=window, min_periods=1).mean()
    return df

def add_all_indicators(df):
    """
    Input: DataFrame with Date column and Close column.
    Returns DataFrame with added indicator columns.
    """
    working = df.copy()
    if 'Date' in working.columns:
        working = ensure_dateindex(working, 'Date')

    working = add_sma(working, 10)
    working = add_sma(working, 20)
    working = add_ema(working, 12)
    working = add_ema(working, 26)
    working = add_rsi(working, period=14)
    working = add_macd(working, fast=12, slow=26, signal=9)
    working = add_bollinger(working, window=20, n_std=2)
    working = add_volume_ma(working, window=20)
    # reset index so downstream code that expects 'Date' column still works
    working = working.reset_index().rename_axis(None, axis=1)
    return working
