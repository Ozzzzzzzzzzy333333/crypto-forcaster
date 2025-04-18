# imports
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import requests
import time
from plotly.subplots import make_subplots
import plotly.graph_objs as go
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

# Determine if any indicators are selected
indicator_rows = sum(1 for ind in indicators_selected if ind in ['RSI', 'MACD', 'OBV', 'ATR'])

# Ensure at least one row for the price chart
row_count = 1 + indicator_rows  # 1 for price + indicators
row_index = 2  # Start adding indicators from row 2

# Setup subplot layout
if indicator_rows > 0:
    fig = make_subplots(
        rows=row_count, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5] + [0.5 / indicator_rows] * indicator_rows,
        subplot_titles=["Price Chart"] +
            [ind for ind in indicators_selected if ind in ['RSI', 'MACD', 'OBV', 'ATR']]
    )
else:
    # If no indicators are selected, create a single-row layout
    fig = make_subplots(
        rows=1, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Price Chart"]
    )

# Add price chart
if chart_type == "Line Chart":
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['close'],
        name='Close Price',
        line=dict(color='white')
    ), row=1, col=1)
else:
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Candlestick',
        increasing=dict(line=dict(color='green')),
        decreasing=dict(line=dict(color='red'))
    ), row=1, col=1)

# Add overlays to main chart
if 'SMA' in indicators_selected:
    df = calculate_sma(df)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma'], name='SMA'), row=1, col=1)

if 'EMA' in indicators_selected:
    df = calculate_ema(df)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema'], name='EMA', line=dict(dash='dot')), row=1, col=1)

if 'Bollinger Bands' in indicators_selected:
    df = calculate_bollinger_bands(df)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='Upper BB', line=dict(dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='Lower BB', line=dict(dash='dot')), row=1, col=1)

# Add indicators with their own subplots
if 'RSI' in indicators_selected:
    df = calculate_rsi(df)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI', line=dict(color='orange')), row=row_index, col=1)
    fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=row_index, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=row_index, col=1)
    fig.update_yaxes(title_text='RSI', row=row_index, col=1)
    row_index += 1

if 'MACD' in indicators_selected:
    df = calculate_macd(df)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'], name='MACD', line=dict(color='cyan')), row=row_index, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'], name='Signal', line=dict(color='magenta', dash='dot')), row=row_index, col=1)
    fig.update_yaxes(title_text='MACD', row=row_index, col=1)
    row_index += 1

if 'OBV' in indicators_selected:
    df = calculate_obv(df)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['obv'], name='OBV', line=dict(color='blue')), row=row_index, col=1)
    fig.update_yaxes(title_text='OBV', row=row_index, col=1)
    row_index += 1

if 'ATR' in indicators_selected:
    df = calculate_atr(df)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['atr'], name='ATR', line=dict(color='yellow')), row=row_index, col=1)
    fig.update_yaxes(title_text='ATR', row=row_index, col=1)
    row_index += 1

# General layout
fig.update_layout(
    height=300 * row_count,
    title=f"{crypto} Price & Indicators ({interval})",
    template="plotly_dark",
    showlegend=True,
    xaxis=dict(
        title="Time",
        type="date",
        tickformat="%Y-%m-%d %H:%M",
        rangeslider=dict(visible=False)
    ),
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# Trend Summary Section
st.subheader("📊 Indicator Summary & Trend Signals")

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

    elif indicator == 'OBV':
        trend = 'up' if df['obv'].iloc[-1] > df['obv'].iloc[-2] else 'down'
        display_signal("OBV", trend, "OBV is rising." if trend == 'up' else "OBV is falling.")

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

    elif indicator == 'Hurst Exponent':
        hurst_value = calculate_hurst(df)  # Ensure this function returns a numeric value or Series
        if isinstance(hurst_value, pd.Series):  # If it's a Series, extract the last value
            hurst_value = hurst_value.iloc[-1]
        elif isinstance(hurst_value, pd.DataFrame):  # If it's a DataFrame, extract the relevant column and value
            hurst_value = hurst_value.iloc[-1, 0]  # Adjust column index as needed

        if isinstance(hurst_value, (int, float)):  # Ensure it's a numeric value
            if hurst_value > 0.5:
                display_signal("Hurst Exponent", 'up', f"Hurst={hurst_value:.2f} suggests trending behavior.")
            else:
                display_signal("Hurst Exponent", 'down', f"Hurst={hurst_value:.2f} suggests randomness.")
        else:
            display_signal("Hurst Exponent", 'down', "Hurst Exponent calculation failed.")

    elif indicator == 'ARIMA':
        try:
            last_pred = predict_arima(df, steps=2)['arima_pred']
            trend = 'up' if last_pred.iloc[-1] > last_pred.iloc[-2] else 'down'
            display_signal("ARIMA", trend, f"Forecasting {trend} movement.")
        except:
            display_signal("ARIMA", 'down', "ARIMA forecast not available.")

st.markdown('<h3 class="centered-text">This data is provided by Binance</h3>', unsafe_allow_html=True)

while True:
    time.sleep(60)
    st.rerun()

