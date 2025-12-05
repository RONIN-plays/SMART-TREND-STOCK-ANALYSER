# main.py
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from utils.preprocess import add_all_indicators
from utils.forecaster import make_forecast
import yfinance as yf

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
    plt.plot(pd.to_datetime(df['Date']), df['Close'], label=f"{ticker} Close")
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
    """
    Safe quick statistics that works across pandas versions.
    - Prints date range
    - Prints numeric describe()
    - Shows last 5 rows
    """
    print("\n=== Quick stats ===")
    # Ensure Date is datetime
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            print("Date range:", df['Date'].min(), "to", df['Date'].max())
        except Exception:
            pass

    # Numeric summary: select numeric columns only
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        print("\nNumeric summary:")
        print(numeric_df.describe().round(4))
    else:
        print("\nNo numeric columns to summarize.")

    print("\nLast 5 rows:")
    print(df.tail())


def main():
    # change tickers as you like
    tickers = ["AAPL", "TSLA"]

    for t in tickers:
        try:
            # 1) Download
            df = download_stock(t, period="2y", interval="1d")

            # 2) Add indicators
            df_ind = add_all_indicators(df)

            # 3) Save processed CSV
            csv_name = f"{t}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            save_dataframe(df_ind, csv_name)

            # 4) Plot close price
            plot_close_price(df_ind, t)

            # 5) Quick stats
            quick_stats(df_ind)

            # 6) Forecast (Prophet primary, SARIMAX fallback)
            forecast_df, model_path, forecast_plot, method = make_forecast(
                df_ind,
                ticker=t,
                periods=30
            )

            print(f"\nForecast completed for {t}:")
            print(" - Method:", method)
            print(" - Model saved at:", model_path)
            print(" - Forecast CSV:", os.path.join(OUT_DIR, f"{t}_forecast.csv"))
            print(" - Forecast plot:", forecast_plot)
            print("--------------------------------------\n")

        except Exception as e:
            print(f"Error for {t}: {e}")


if __name__ == "__main__":
    main()
