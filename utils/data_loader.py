import yfinance as yf
import pandas as pd

def download_stock(ticker, period="2y", interval="1d"):
    print(f"[{pd.Timestamp.now()}] Downloading {ticker} | period={period} interval={interval}")

    df = yf.download(ticker, period=period, interval=interval)

    # Reset index to get Date as a column
    df = df.reset_index()

    # Clean and normalize column names
    df.columns = df.columns.astype(str)          # ensure all are strings
    df.columns = df.columns.str.strip()          # remove leading/trailing spaces
    df.columns = df.columns.str.replace(" ", "") # remove inner spaces
    df.columns = [c.capitalize() for c in df.columns]  # standardize naming

    # Debugging print (optional)
    print(f"Columns after cleaning for {ticker}: {df.columns.tolist()}")

    # Critical check for Volume column
    if "Volume" not in df.columns:
        raise ValueError(
            f"\n❌ ERROR: 'Volume' column NOT FOUND for {ticker}!\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Yahoo Finance may have returned an unusual structure.\n"
        )

    return df
