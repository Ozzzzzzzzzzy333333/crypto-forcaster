# imports
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import yfinance as yf
from indicators import (
    calculate_sma,
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_hurst,
    predict_arima,
    calculate_macd,
    calculate_atr
)

# page setup
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("Stock Report")
st.header("A simple and easy-to-use prediction app")
st.header("Note: The Crypto option is recomeneded, this is due to API limits and the current stage of development")

# sidebar
st.sidebar.header("Technical Indicators")
indicators_selected = st.sidebar.multiselect(
    "Select Indicators to Display",
    ['SMA', 'Bollinger Bands', 'RSI', 'ATR', 'Hurst Exponent', 'ARIMA'],
    default=['SMA', 'Bollinger Bands']
)

selected_stocks = st.sidebar.multiselect(
    "Select Tech Stocks", 
    ['AAPL', 'GOOGL', 'MSFT'],
    default=['AAPL']
)

interval = st.sidebar.selectbox("Select Time Interval", ['5m', '15m', '30m', '1h', '1d'])

# fetch stock data
def fetch_stock_data(ticker='AAPL', interval='60m', period='7d'):
    stock = yf.Ticker(ticker)
    df = stock.history(interval=interval, period=period)

    if df.empty:
        st.error(f"No data found for {ticker} with interval {interval} and period {period}.")
        return None

    # Reset index and handle the timestamp column
    df.reset_index(inplace=True)
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'timestamp'}, inplace=True)
    elif 'Date' in df.columns:
        df.rename(columns={'Date': 'timestamp'}, inplace=True)

    # Ensure the timestamp column is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Rename columns for consistency
    df = df[['timestamp', 'Open', 'High', 'Low', 'Close']].rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'
    })
    return df

for stock in selected_stocks:
    df = fetch_stock_data(ticker=stock, interval=interval, period='7d')

    if df is None:
        continue

    # Chart type toggle
    chart_type = st.radio("Select chart type", ["Line Chart", "Candlestick"], horizontal=True, key=f"chart_{stock}")

    # Calculate the number of rows for indicators
    indicator_rows = sum(1 for ind in indicators_selected if ind in ['RSI', 'ATR'])

    # Ensure indicator_rows is at least 1 to avoid division by zero
    if indicator_rows == 0:
        indicator_rows = 1

    # Setup subplot layout
    row_count = 1 + indicator_rows  # 1 for price + indicators
    row_index = 2  # Start adding indicators from row 2

    fig = make_subplots(
        rows=row_count, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5] + [0.5 / indicator_rows] * indicator_rows,
        subplot_titles=["Price Chart"] +
            [ind for ind in indicators_selected if ind in ['RSI', 'ATR']]
    )

    # Add price chart
    if chart_type == "Line Chart":
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['close'],
            mode='lines',
            name='Price'
        ), row=1, col=1)
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
        ), row=1, col=1)

    # Add indicators
    if 'SMA' in indicators_selected:
        df = calculate_sma(df)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma'], name='SMA'), row=1, col=1)

    if 'Bollinger Bands' in indicators_selected:
        df = calculate_bollinger_bands(df)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='Upper BB', line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='Lower BB', line=dict(dash='dot')), row=1, col=1)

    if 'RSI' in indicators_selected:
        df = calculate_rsi(df)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI'), row=row_index, col=1)
        row_index += 1

    if 'ATR' in indicators_selected:
        df = calculate_atr(df)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['atr'], name='ATR'), row=row_index, col=1)
        row_index += 1

    if 'Hurst Exponent' in indicators_selected:
        df = calculate_hurst(df)
        st.write(f"Hurst Exponent for {stock}: `{df['hurst'].iloc[-1]:.4f}`")

    if 'ARIMA' in indicators_selected:
        arima_future = predict_arima(df, steps=len(df))
        fig.add_trace(go.Scatter(x=arima_future['timestamp'], y=arima_future['arima_pred'], name='ARIMA Forecast', line=dict(dash='dash')), row=1, col=1)

    # Output chart
    st.subheader(f"Live {stock} Chart ({interval})")
    st.plotly_chart(fig, use_container_width=True)

# Trend Summary Section
st.subheader("Indicator Summary & Trend Signals")

def display_signal(name, trend, details):
    color = 'green' if trend == 'up' else 'red'
    arrow = '⬆️' if trend == 'up' else '⬇️'
    st.markdown(
        f"<div style='padding:8px;border-radius:10px;background-color:#1e1e1e;margin-bottom:10px;'>"
        f"<strong style='color:{color};'>{arrow} {name}</strong><br>"
        f"<span style='color:#ccc;'>{details}</span>"
        f"</div>", unsafe_allow_html=True
    )

for indicator in indicators_selected:
    if indicator == 'SMA':
        trend = 'up' if df['sma'].iloc[-1] > df['sma'].iloc[-2] else 'down'
        display_signal("SMA", trend, "SMA is trending " + trend + ". Possible trend continuation.")
        
    elif indicator == 'EMA':
        trend = 'up' if df['ema'].iloc[-1] > df['ema'].iloc[-2] else 'down'
        display_signal("EMA", trend, "EMA is moving " + trend + ". Short-term price momentum.")

    elif indicator == 'RSI':
        last_rsi = df['rsi'].iloc[-1]
        if last_rsi > 70:
            display_signal("RSI", 'down', f"RSI at {last_rsi:.2f}. Overbought, potential pullback.")
        elif last_rsi < 30:
            display_signal("RSI", 'up', f"RSI at {last_rsi:.2f}. Oversold, possible rebound.")
        else:
            trend = 'up' if last_rsi > df['rsi'].iloc[-2] else 'down'
            display_signal("RSI", trend, f"RSI trending {trend}. Neutral zone.")

    elif indicator == 'MACD':
        trend = 'up' if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else 'down'
        display_signal("MACD", trend, "MACD " + ("above" if trend == 'up' else "below") + " signal line.")

    elif indicator == 'ATR':
        atr_change = df['atr'].iloc[-1] - df['atr'].iloc[-2]
        trend = 'up' if atr_change > 0 else 'down'
        display_signal("ATR", trend, "Volatility is increasing." if trend == 'up' else "Volatility is decreasing.")

    elif indicator == 'Bollinger Bands':
        last_close = df['close'].iloc[-1]
        if last_close > df['bb_upper'].iloc[-1]:
            display_signal("Bollinger Bands", 'down', "Price above upper band — possible overbought.")
        elif last_close < df['bb_lower'].iloc[-1]:
            display_signal("Bollinger Bands", 'up', "Price below lower band — possible oversold.")
        else:
            display_signal("Bollinger Bands", 'up', "Price within bands — stable trend.")

    elif indicator == 'ARIMA':
        try:
            last_pred = predict_arima(df, steps=2)['arima_pred']
            trend = 'up' if last_pred.iloc[-1] > last_pred.iloc[-2] else 'down'
            display_signal("ARIMA", trend, f"Forecasting {trend} movement.")
        except:
            display_signal("ARIMA", 'down', "ARIMA forecast not available.")

st.markdown('<h3 class="centered-text">This data is provided by yfinance</h3>', unsafe_allow_html=True)

