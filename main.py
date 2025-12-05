# main.py
import os
from datetime import datetime
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def download_stock(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Download historical data for `ticker` using yfinance.
    period example: '1y', '2y', '5y', 'max'
    interval example: '1d', '1wk', '1mo'
    """
    print(f"[{datetime.now()}] Downloading {ticker} | period={period} interval={interval}")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    df.reset_index(inplace=True)
    return df


def save_dataframe(df: pd.DataFrame, filename: str) -> str:
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved CSV → {path}")
    return path


def plot_close_price(df: pd.DataFrame, ticker: str) -> str:
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Close'], label=f"{ticker} Close")
    plt.title(f"{ticker} — Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    out_path = os.path.join(OUT_DIR, f"{ticker}_close.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot → {out_path}")
    return out_path


def quick_stats(df: pd.DataFrame) -> None:
    print("\n=== Quick stats ===")
    print(df[['Date','Open','High','Low','Close','Volume']].describe(datetime_is_numeric=False))
    print("\nLast 5 rows:")
    print(df.tail())


def main():
    # example tickers: 'AAPL', 'TSLA', 'NSEI' (some markets), 'GOOGL'
    tickers = ["AAPL", "TSLA"]    # change to what you want
    for t in tickers:
        try:
            df = download_stock(t, period="2y", interval="1d")
            csv_name = f"{t}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            save_dataframe(df, csv_name)
            plot_close_price(df, t)
            quick_stats(df)
        except Exception as e:
            print(f"Error for {t}: {e}")


if __name__ == "__main__":
    main()
