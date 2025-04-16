# imports
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import requests
from indicators import (
    calculate_sma,
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_hurst,
    predict_arima
)

# page setup
st.set_page_config(page_title="Crypto Predictor", layout="wide")
st.title("Crypto Report")
st.header("a simple and easy to use prediction app")
# sidebar
st.sidebar.header(" Technical Indicators")
indicators_selected = st.sidebar.multiselect(
    "Select Indicators to Display",
    ['SMA', 'Bollinger Bands', 'RSI', 'Hurst Exponent', 'ARIMA'],
    default=['SMA', 'Bollinger Bands']
)


crypto = st.sidebar.selectbox("Select Cryptocurrency", ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'])
interval = st.sidebar.selectbox("Select Time Interval", ['5m', '15m', '30m', '1h', '4h', '1d'])

if st.sidebar.button("Run Prediction"):
    st.session_state.run_model = True
else:
    st.session_state.run_model = False

# Binance API 
def fetch_binance_data(symbol='BTCUSDT', interval='1h', limit=200):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    return df[['timestamp', 'close']]

binance_symbol = crypto.replace('/', '')
df = fetch_binance_data(symbol=binance_symbol, interval=interval, limit=100)

# Prediction Time 
freq_map = {'5m': '5T', '15m': '15T', '30m': '30T', '1h': '1H', '4h': '4H','1d': '1D'}
freq = freq_map[interval]

future_steps = len(df)
last_time = df['timestamp'].iloc[-1]
future_times = pd.date_range(start=last_time + pd.Timedelta(freq), periods=future_steps, freq=freq)
future_df = pd.DataFrame({'timestamp': future_times, 'close': [None] * future_steps})
combined_df = pd.concat([df, future_df], ignore_index=True)

#  Plot 
fig = go.Figure()
fig.add_trace(go.Scatter(x=combined_df['timestamp'], y=combined_df['close'], mode='lines', name='Price'))

# Indicators
if 'SMA' in indicators_selected:
    df = calculate_sma(df)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma'], name='SMA'))

if 'Bollinger Bands' in indicators_selected:
    df = calculate_bollinger_bands(df)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='Upper BB', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='Lower BB', line=dict(dash='dot')))

if 'RSI' in indicators_selected:
    df = calculate_rsi(df)
    st.line_chart(df.set_index('timestamp')['rsi'], use_container_width=True)

if 'Hurst Exponent' in indicators_selected:
    df = calculate_hurst(df)
    st.write(f"Hurst Exponent: `{df['hurst'].iloc[-1]:.4f}`")

if 'ARIMA' in indicators_selected:
    arima_future = predict_arima(df, steps=len(df))
    fig.add_trace(go.Scatter(x=arima_future['timestamp'], y=arima_future['arima_pred'], name='ARIMA Forecast', line=dict(dash='dash')))


st.subheader(f"Live {crypto} Chart ({interval})")
st.plotly_chart(fig, use_container_width=True)
