# utils/forecaster.py
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Try Prophet first (new package name "prophet")
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except Exception:
        PROPHET_AVAILABLE = False

# statsmodels fallback
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

def _ensure_datecol(df, date_col="Date"):
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)

    else:
        df = df.reset_index()
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")
    return df

def forecast_with_prophet(df, periods=30, freq='D', model_dir="models", ticker='TICKER'):
    os.makedirs(model_dir, exist_ok=True)
    df = _ensure_datecol(df)
    df_prop = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    m = Prophet()
    m.fit(df_prop)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    model_path = os.path.join(model_dir, f"{ticker}_prophet.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(m, f)
    out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    if 'y' in forecast.columns:
        out['y'] = forecast['y']
    return out, model_path

def forecast_with_sarimax(df, periods=30, freq='D', order=(1,1,1), seasonal_order=(0,0,0,0), model_dir="models", ticker='TICKER'):
    if not STATSMODELS_AVAILABLE:
        raise RuntimeError("statsmodels not available.")
    os.makedirs(model_dir, exist_ok=True)
    df = _ensure_datecol(df)
    ts = df.set_index('Date')['Close'].asfreq('D').fillna(method='ffill')
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    start = ts.index[-1] + pd.Timedelta(1, unit='D')
    future_idx = pd.date_range(start=start, periods=periods, freq=freq)
    preds = res.get_forecast(steps=periods)
    mean = preds.predicted_mean
    conf = preds.conf_int()
    out = pd.DataFrame({
        'ds': future_idx,
        'yhat': mean.values,
        'yhat_lower': conf.iloc[:, 0].values,
        'yhat_upper': conf.iloc[:, 1].values
    })
    model_path = os.path.join(model_dir, f"{ticker}_sarimax.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(res, f)
    return out, model_path

def save_forecast_plot(df_history, df_forecast, ticker, out_dir="outputs", horizon_days=30):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    hist = df_history.copy()
    hist['Date'] = pd.to_datetime(hist['Date'])
    plt.plot(hist['Date'], hist['Close'], label='History', linewidth=1)
    plt.plot(df_forecast['ds'], df_forecast['yhat'], label='Forecast', linestyle='--', linewidth=1.5)
    plt.fill_between(df_forecast['ds'].astype('datetime64[ns]'),
                     df_forecast['yhat_lower'],
                     df_forecast['yhat_upper'],
                     color='gray', alpha=0.2, label='Confidence')
    plt.title(f"{ticker} — Forecast (next {horizon_days} days)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    out_path = os.path.join(out_dir, f"{ticker}_forecast.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def make_forecast(df, ticker, periods=30, freq='D', model_dir="models", out_dir="outputs"):
    df = _ensure_datecol(df)
    method = None
    forecast_df = None
    model_path = None
    if PROPHET_AVAILABLE:
        try:
            forecast_df, model_path = forecast_with_prophet(df, periods=periods, freq=freq, model_dir=model_dir, ticker=ticker)
            method = 'prophet'
        except Exception as e:
            print(f"Prophet failed ({e}) — falling back to SARIMAX if available.")
            if STATSMODELS_AVAILABLE:
                forecast_df, model_path = forecast_with_sarimax(df, periods=periods, freq=freq, model_dir=model_dir, ticker=ticker)
                method = 'sarimax'
            else:
                raise
    else:
        if STATSMODELS_AVAILABLE:
            forecast_df, model_path = forecast_with_sarimax(df, periods=periods, freq=freq, model_dir=model_dir, ticker=ticker)
            method = 'sarimax'
        else:
            raise RuntimeError("Neither Prophet nor SARIMAX available.")
    os.makedirs(out_dir, exist_ok=True)
    forecast_csv = os.path.join(out_dir, f"{ticker}_forecast.csv")
    forecast_df.to_csv(forecast_csv, index=False)
    plot_path = save_forecast_plot(df, forecast_df, ticker, out_dir=out_dir, horizon_days=periods)
    return forecast_df, model_path, plot_path, method
