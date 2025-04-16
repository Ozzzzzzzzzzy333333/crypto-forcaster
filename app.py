#imports
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import requests

#page
st.set_page_config(page_title="Crypto Predictor", layout="wide")
st.title("🧠 Crypto Prediction App")

#sidebar
st.sidebar.header("⚙️ Settings")
crypto_pair = st.sidebar.selectbox("Select Cryptocurrency", ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'])
interval = st.sidebar.selectbox("Select Time Interval", ['5m', '15m', '30m', '1h', '4h'])
model_choice = st.sidebar.selectbox("Select Prediction Model", ['Random Forest', 'LSTM'])
show_bb = st.sidebar.checkbox("Show Bollinger Bands")
show_sma = st.sidebar.checkbox("Show SMA")

if st.sidebar.button("Run Prediction"):
    st.session_state.run_model = True
else:
    st.session_state.run_model = False

# binance api
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
binance_symbol = crypto_pair.replace('/', '')

df = fetch_binance_data(symbol=binance_symbol, interval=interval, limit=100)

freq_map = {
    '5m': '5T', '15m': '15T', '30m': '30T', '1h': '1H', '4h': '4H'
}
freq = freq_map[interval]

future_steps = len(df)
last_time = df['timestamp'].iloc[-1]
future_times = pd.date_range(start=last_time + pd.Timedelta(freq), periods=future_steps, freq=freq)
future_df = pd.DataFrame({'timestamp': future_times, 'close': [None] * future_steps})

combined_df = pd.concat([df, future_df], ignore_index=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=combined_df['timestamp'], y=combined_df['close'], mode='lines', name='Price'))

# SMA
if show_sma:
    df['sma'] = df['close'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma'], mode='lines', name='SMA'))

# Bollinger Bands
if show_bb:
    sma = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    upper = sma + std * 2
    lower = sma - std * 2
    fig.add_trace(go.Scatter(x=df['timestamp'], y=upper, line=dict(dash='dot'), name='Upper BB'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=lower, line=dict(dash='dot'), name='Lower BB'))

fig.update_layout(
    title=f"{crypto_pair} Price with Prediction Space",
    xaxis_title='Time',
    yaxis_title='Price (USDT)',
    template='plotly_white',
    autosize=True
)

st.subheader(f"📈 Live {crypto_pair} Chart ({interval})")
st.plotly_chart(fig, use_container_width=True)
