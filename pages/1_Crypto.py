# imports
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import requests
import time
from indicators import (
    calculate_sma,
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_hurst,
    predict_arima,
    calculate_ema,  
    calculate_macd,  
    calculate_obv,   
    calculate_atr 
)

# page setup
st.set_page_config(page_title="Crypto Predictor", layout="wide")
st.title("Crypto Report")
st.header("a simple and easy to use prediction app")
# sidebar
st.sidebar.header(" Technical Indicators")
indicators_selected = st.sidebar.multiselect(
    "Select Indicators to Display",
    ['SMA', 'EMA', 'Bollinger Bands', 'RSI', 'MACD', 'OBV', 'ATR', 'Hurst Exponent', 'ARIMA'],
    default=['SMA', 'Bollinger Bands']
)

cryptos = st.sidebar.multiselect(
    "Select Cryptocurrencies", 
    ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'], 
    default=['BTC/USDT']
)

interval = st.sidebar.selectbox("Select Time Interval", ['5m', '15m', '30m', '1h', '4h', '1d'])

if st.sidebar.button("Run Prediction"):
    st.session_state.run_model = True
else:
    st.session_state.run_model = False


# Button to toggle the indicator guide
if st.sidebar.button("📘 What do these indicators mean?"):

    st.sidebar.markdown("Technical Indicator Guide")

    st.sidebar.markdown("SMA")
    st.sidebar.markdown("is the avarge price, showing trend direction.")

    st.sidebar.markdown("EMA")
    st.sidebar.markdown("Similar to SMA but gives more weight to recent prices.")

    st.sidebar.markdown("RSI")
    st.sidebar.markdown("Shows if something is overbought or oversold (0–100)." \
    "Generally, values above 70 are considered overbought," \
    " while values below 30 are considered oversold. ")

    st.sidebar.markdown("MACD")
    st.sidebar.markdown("Measures momentum using two moving averages. " \
    "which can help identify potential buy/sell signals. " )

    st.sidebar.markdown("Bollinger Bands")
    st.sidebar.markdown("Tracks volatility using bands around the price." \
    "The bands widen when volatility increases and narrow when it decreases.")

    st.sidebar.markdown("ARIMA")
    st.sidebar.markdown("Uses past data to forecast future prices.")

    st.sidebar.markdown("OBV")
    st.sidebar.markdown("Measures buying/selling pressure using volume." \
    " A rising OBV indicates buying pressure, while a falling OBV indicates selling pressure.")

    st.sidebar.markdown("ATR")
    st.sidebar.markdown("Measures how much the price moves on average." \
    " A higher ATR indicates more volatility, while a lower ATR indicates less.")

    st.sidebar.markdown("Hurst Exponent")
    st.sidebar.markdown("Tells if price movements are random or trending." \
    " A value of 0.5 indicates a more 'random' movement, while values above 0.5 " \
    "indicate a clearer trend.")

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

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/London')

    # Convert relevant columns to numeric
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


for crypto in cryptos:
    binance_symbol = crypto.replace('/', '')
    df = fetch_binance_data(symbol=binance_symbol, interval=interval, limit=100)
    

# time 
freq_map = {'5m': '5T', '15m': '15T', '30m': '30T', '1h': '1H', '4h': '4H','1d': '1D'}
freq = freq_map[interval]

future_steps = len(df)
last_time = df['timestamp'].iloc[-1]
future_times = pd.date_range(start=last_time + pd.Timedelta(freq), periods=future_steps, freq=freq)
future_df = pd.DataFrame({'timestamp': future_times, 'close': [None] * future_steps})
combined_df = pd.concat([df, future_df], ignore_index=True)
chart_type = st.radio("Select chart type", ["Line Chart", "Candlestick"], horizontal=True)
# plot 
fig = go.Figure()
# plot line graph
if chart_type == "Line Chart":
    fig.add_trace(go.Scatter(
        x=combined_df['timestamp'],
        y=combined_df['close'],
        mode='lines',
        name='Price'
    ))
else:  # plot candlestick
    candle_df = df.copy()
    candle_data = fetch_binance_data(symbol=binance_symbol, interval=interval, limit=100)

    candle_data_full = pd.DataFrame(candle_data)
    candle_data_full['open'] = candle_data['open'].astype(float)
    candle_data_full['high'] = candle_data['high'].astype(float)
    candle_data_full['low'] = candle_data['low'].astype(float)
    candle_data_full['close'] = candle_data['close'].astype(float)
    candle_data_full['timestamp'] = pd.to_datetime(candle_data['timestamp'], unit='ms')

    fig.add_trace(go.Candlestick(
    x=candle_data_full['timestamp'],
    open=candle_data_full['open'],
    high=candle_data_full['high'],
    low=candle_data_full['low'],
    close=candle_data_full['close'],
    name='Candlestick',
    increasing=dict(line=dict(color='green')),
    decreasing=dict(line=dict(color='red'))
))


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

if 'EMA' in indicators_selected:
    df = calculate_ema(df)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema'], name='EMA', line=dict(dash='dot')))

if 'MACD' in indicators_selected:
    df = calculate_macd(df)
    st.line_chart(df.set_index('timestamp')[['macd', 'macd_signal']], use_container_width=True)

if 'OBV' in indicators_selected:
    df = calculate_obv(df)
    st.line_chart(df.set_index('timestamp')['obv'], use_container_width=True)

if 'ATR' in indicators_selected:
    df = calculate_atr(df)
    st.line_chart(df.set_index('timestamp')['atr'], use_container_width=True)



st.subheader(f"Live {crypto} Chart ({interval})")
st.plotly_chart(fig, use_container_width=True)
while True:
    time.sleep(60)
    st.rerun()
