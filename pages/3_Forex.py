# imports
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
from indicators import (
    calculate_sma,
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_hurst,
    predict_arima
)

# page setup
st.set_page_config(page_title="Forex Predictor", layout="wide")
st.title("Forex Report")
st.header("A simple and easy-to-use prediction app")

# sidebar
st.sidebar.header("Technical Indicators")
indicators_selected = st.sidebar.multiselect(
    "Select Indicators to Display",
    ['SMA', 'Bollinger Bands', 'RSI', 'Hurst Exponent', 'ARIMA'],
    default=['SMA', 'Bollinger Bands']
)

selected_pairs = st.sidebar.multiselect(
    "Select Forex Pairs", 
    ['EUR/USD', 'GBP/USD', 'USD/JPY'],  # Top 3 Forex pairs
    default=['EUR/USD']
)

if st.sidebar.button("Run Prediction"):
    st.session_state.run_model = True
else:
    st.session_state.run_model = False

# Fixed interval set to '1h'
fixed_interval = '1h'

# fetch forex data
def fetch_forex_data(pair='EUR/USD', interval=fixed_interval, period='7d'):
    # Map Forex pairs to Yahoo Finance tickers
    forex_ticker_map = {
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'USD/JPY': 'USDJPY=X'
    }
    
    ticker = forex_ticker_map.get(pair)
    if not ticker:
        st.error(f"Unsupported Forex pair: {pair}")
        return None

    # Fetch data using yfinance
    forex = yf.Ticker(ticker)
    df = forex.history(interval=interval, period=period)

    if df.empty:
        st.error(f"No data found for {pair} with interval {interval} and period {period}.")
        return None

    # Reset index and rename columns for consistency
    df.reset_index(inplace=True)
    df.rename(columns={'Datetime': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df[['timestamp', 'Open', 'High', 'Low', 'Close']].rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'
    })
    return df

for pair in selected_pairs:
    df = fetch_forex_data(pair=pair, interval=fixed_interval, period='7d')

    if df is None:
        continue

    # chart type toggle
    chart_type = st.radio("Select chart type", ["Line Chart", "Candlestick"], horizontal=True, key=f"chart_{pair}")

    fig = go.Figure()

    if chart_type == "Line Chart":
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['close'],
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
        st.write(f"Hurst Exponent for {pair}: `{df['hurst'].iloc[-1]:.4f}`")

    if 'ARIMA' in indicators_selected:
        arima_future = predict_arima(df, steps=len(df))
        fig.add_trace(go.Scatter(x=arima_future['timestamp'], y=arima_future['arima_pred'], name='ARIMA Forecast', line=dict(dash='dash')))

    # output chart
    st.subheader(f"📈 Live {pair} Chart ({fixed_interval})")
    st.plotly_chart(fig, use_container_width=True)
