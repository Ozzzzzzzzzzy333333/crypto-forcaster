# imports
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
import time
from indicators import (
    calculate_sma,
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_hurst,
    predict_arima
)


# page setup
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("Stock Report")
st.header("A simple and easy-to-use prediction app")

# sidebar
st.sidebar.header("Technical Indicators")
indicators_selected = st.sidebar.multiselect(
    "Select Indicators to Display",
    ['SMA', 'Bollinger Bands', 'RSI', 'Hurst Exponent', 'ARIMA'],
    default=['SMA', 'Bollinger Bands']
)

selected_stocks = st.sidebar.multiselect(
    "Select Tech Stocks", 
    ['AAPL', 'GOOGL', 'MSFT'],
    default=['AAPL']
)

interval = st.sidebar.selectbox("Select Time Interval", ['5m', '15m', '30m', '1h', '1d'])

if st.sidebar.button("Run Prediction"):
    st.session_state.run_model = True
else:
    st.session_state.run_model = False

# interval map
interval_map = {
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '60m',
    '1d': '1d'
}

# fetch stock data
def fetch_stock_data(ticker='AAPL', interval='60m', period='7d'):
    stock = yf.Ticker(ticker)
    df = stock.history(interval=interval, period=period)
    df.reset_index(inplace=True)
    df.rename(columns={'Datetime': 'timestamp'}, inplace=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/London')

    df = df[['timestamp', 'Open', 'High', 'Low', 'Close']].rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'
    })
    return df

for stock in selected_stocks:
    df = fetch_stock_data(ticker=stock, interval=interval_map[interval], period='7d')

    # future projection (dummy timestamps for visualization)
    freq_map = {'5m': '5T', '15m': '15T', '30m': '30T', '1h': '1H', '1d': '1D'}
    freq = freq_map[interval]
    future_steps = len(df)
    last_time = df['timestamp'].iloc[-1]
    future_times = pd.date_range(start=last_time + pd.Timedelta(freq), periods=future_steps, freq=freq)
    future_df = pd.DataFrame({'timestamp': future_times, 'close': [None] * future_steps})
    combined_df = pd.concat([df, future_df], ignore_index=True)

    # chart type toggle
    chart_type = st.radio("Select chart type", ["Line Chart", "Candlestick"], horizontal=True, key=f"chart_{stock}")

    fig = go.Figure()

    if chart_type == "Line Chart":
        fig.add_trace(go.Scatter(
            x=combined_df['timestamp'],
            y=combined_df['close'],
            mode='lines',
            name='Price'
        ))
    else:  # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candlestick',
            increasing=dict(line=dict(color='green')),
            decreasing=dict(line=dict(color='red'))
        ))

    # indicators
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
        st.write(f"Hurst Exponent for {stock}: `{df['hurst'].iloc[-1]:.4f}`")

    if 'ARIMA' in indicators_selected:
        arima_future = predict_arima(df, steps=len(df))
        fig.add_trace(go.Scatter(x=arima_future['timestamp'], y=arima_future['arima_pred'], name='ARIMA Forecast', line=dict(dash='dash')))

    # output chart
    st.subheader(f"📈 Live {stock} Chart ({interval})")
    st.plotly_chart(fig, use_container_width=True)
while True:
    time.sleep(60)
    st.rerun()
