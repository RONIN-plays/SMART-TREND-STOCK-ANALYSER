# utils/eda.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def eda_summary(df, ticker="TICKER"):
    print("\n====== EDA SUMMARY ======")
    print(df.head())
    print("\n--- Statistical Summary ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(df.isna().sum())


def plot_price_trend(df, ticker="TICKER"):
    plt.figure(figsize=(10,5))
    plt.plot(df["Date"], df["Close"], label="Close Price")
    plt.title(f"{ticker} - Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    out = os.path.join(OUT_DIR, f"{ticker}_eda_price.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_moving_averages(df, ticker="TICKER"):
    plt.figure(figsize=(10,5))
    plt.plot(df["Date"], df["Close"], label="Close", alpha=0.7)
    plt.plot(df["Date"], df["SMA_20"], label="SMA 20", linestyle="--")
    plt.plot(df["Date"], df["SMA_10"], label="SMA 10", linestyle="--")
    plt.title(f"{ticker} - Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    out = os.path.join(OUT_DIR, f"{ticker}_eda_ma.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_volume(df, ticker):
    # SAFETY CHECK – this is the ONLY fix needed
    if "Volume" not in df.columns:
        print(f"[INFO] Volume column missing for {ticker}. Skipping volume plot.")
        return

    plt.figure(figsize=(10,4))
    plt.bar(df["Date"], df["Volume"], color="gray")
    plt.title(f"{ticker} - Volume Trend")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.savefig(f"outputs/{ticker}_eda_volume.png")
    plt.close()




def plot_candlestick(df, ticker="TICKER"):
    df_candle = df.copy()
    df_candle.index = pd.to_datetime(df_candle["Date"])
    cols = ["Open", "High", "Low", "Close"]

    out = os.path.join(OUT_DIR, f"{ticker}_eda_candlestick.png")

    mpf.plot(
        df_candle[cols],
        type="candle",
        style="charles",
        title=f"{ticker} - Candlestick Chart",
        volume=True,
        savefig=out
    )
    return out


def plot_correlation(df, ticker="TICKER"):
    plt.figure(figsize=(10,8))
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title(f"{ticker} - Correlation Heatmap")
    out = os.path.join(OUT_DIR, f"{ticker}_eda_corr.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def run_full_eda(df, ticker="TICKER"):
    print(f"\n==== Running EDA for {ticker} ====")

    eda_summary(df, ticker)
    p1 = plot_price_trend(df, ticker)
    p2 = plot_moving_averages(df, ticker)
    p3 = plot_volume(df, ticker)
    p4 = plot_candlestick(df, ticker)
    p5 = plot_correlation(df, ticker)

    print("\nSaved EDA plots:")
    print(p1)
    print(p2)
    print(p3)
    print(p4)
    print(p5)
