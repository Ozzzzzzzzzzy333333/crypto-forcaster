import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from numpy import log, std, subtract
import warnings
warnings.filterwarnings("ignore")

# SMA
def calculate_sma(df, window=20):
    df['sma'] = df['close'].rolling(window=window).mean()
    return df

# Bollinger Bands
def calculate_bollinger_bands(df, window=20):
    sma = df['close'].rolling(window=window).mean()
    std_dev = df['close'].rolling(window=window).std()
    df['bb_upper'] = sma + 2 * std_dev
    df['bb_lower'] = sma - 2 * std_dev
    return df

# RSI
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = rsi
    return df

# Hurst Exponent
def calculate_hurst(df):
    ts = df['close'].dropna().values
    lags = range(2, 20)

    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    hurst = np.polyfit(log(lags), log(tau), 1)[0]
    df['hurst'] = hurst
    return df

# ARIMA
def predict_arima(df, steps=20):
    model = ARIMA(df['close'], order=(5, 1, 0))
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    future_df = pd.DataFrame({'timestamp': pd.date_range(start=df['timestamp'].iloc[-1], periods=steps+1, freq='H')[1:],
                              'arima_pred': forecast})
    return future_df
