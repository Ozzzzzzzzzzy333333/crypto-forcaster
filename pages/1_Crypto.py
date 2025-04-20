# imports
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import requests
import time
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
from indicators import (
    calculate_sma, calculate_bollinger_bands, calculate_rsi,
    calculate_ema, calculate_macd, calculate_obv, calculate_atr
)
from predictor import make_prediction

# Initialize session state
if 'run_model' not in st.session_state:
    st.session_state.run_model = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'prediction_df' not in st.session_state:
    st.session_state.prediction_df = None

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Page setup
    st.set_page_config(page_title="Crypto Predictor", layout="wide")
    st.title("Crypto Report")
    st.header("A simple and easy to use prediction app")

    # Sidebar controls
    with st.sidebar:
        st.header("Chart Settings")
        chart_type = st.radio("Chart Type", ["Candlestick", "Line"], index=0)
        
        st.header("Technical Indicators")
        indicators_selected = st.multiselect(
            "Select Indicators to Display",
            ['SMA', 'EMA', 'Bollinger Bands', 'RSI', 'MACD', 'OBV', 'ATR'],
            default=['SMA', 'Bollinger Bands']
        )
        
        crypto = st.selectbox(
            "Select Cryptocurrency", 
            ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'], 
            index=0
        )
        
        interval = st.selectbox("Select Time Interval", ['5m', '15m', '30m', '1h', '4h', '1d'])
        
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.7)
        
        if st.button("Run Prediction"):
            st.session_state.run_model = True
            st.session_state.predictions = None
            st.session_state.prediction_df = None
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

    # Fetch data
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_data(symbol, interval, limit=100):
        url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
        response = requests.get(url)
        data = response.json()

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        logger.info(f"Fetched {len(df)} rows of data for {symbol} at interval {interval}")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    binance_symbol = crypto.replace('/', '')
    df = fetch_data(binance_symbol, interval)

    # Calculate indicators
    if 'SMA' in indicators_selected:
        df = calculate_sma(df)
    if 'EMA' in indicators_selected:
        df = calculate_ema(df)
    if 'Bollinger Bands' in indicators_selected:
        df = calculate_bollinger_bands(df)
    if 'RSI' in indicators_selected:
        df = calculate_rsi(df)
    if 'MACD' in indicators_selected:
        df = calculate_macd(df)
    if 'OBV' in indicators_selected:
        df = calculate_obv(df)
    if 'ATR' in indicators_selected:
        df = calculate_atr(df)

    # Determine subplot layout based on selected indicators
    indicator_rows = sum(1 for ind in indicators_selected if ind in ['RSI', 'MACD', 'OBV', 'ATR'])
    row_count = 2 + indicator_rows  # 1 for price, 1 for volume, plus indicators
    row_heights = [0.5, 0.2] + [0.3/indicator_rows]*indicator_rows if indicator_rows > 0 else [0.7, 0.3]
    
    # Create subplot titles
    subplot_titles = ["Price Chart", "Volume"]
    if 'RSI' in indicators_selected:
        subplot_titles.append("RSI")
    if 'MACD' in indicators_selected:
        subplot_titles.append("MACD")
    if 'OBV' in indicators_selected:
        subplot_titles.append("OBV")
    if 'ATR' in indicators_selected:
        subplot_titles.append("ATR")

    # Create the figure with dynamic rows
    fig = make_subplots(
        rows=row_count, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )

    # Add price chart based on selected type
    if chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['close'],
            name='Price',
            line=dict(color='white')
        ), row=1, col=1)
    else:
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            name='Price',
            increasing=dict(line=dict(color='green')),
            decreasing=dict(line=dict(color='red'))
        ), row=1, col=1)

    # Add volume
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['volume'],
        name='Volume',
        marker_color='rgba(100, 100, 255, 0.6)'
    ), row=2, col=1)

    # Add overlays to main chart
    if 'SMA' in indicators_selected:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['sma'],
            name='SMA',
            line=dict(color='orange', width=2)
        ), row=1, col=1)

    if 'EMA' in indicators_selected:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['ema'],
            name='EMA',
            line=dict(color='cyan', width=2, dash='dot')
        ), row=1, col=1)

    if 'Bollinger Bands' in indicators_selected:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['bb_upper'],
            name='Upper BB',
            line=dict(color='purple', width=1, dash='dot')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['bb_lower'],
            name='Lower BB',
            line=dict(color='purple', width=1, dash='dot')
        ), row=1, col=1)

    # Track current row for indicators (starting after price and volume)
    current_row = 3

    # Add indicators with their own subplots
    if 'RSI' in indicators_selected:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['rsi'],
            name='RSI',
            line=dict(color='orange')
        ), row=current_row, col=1)
        fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=current_row, col=1)
        fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=current_row, col=1)
        current_row += 1

    if 'MACD' in indicators_selected:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['macd'],
            name='MACD',
            line=dict(color='cyan')
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['macd_signal'],
            name='Signal',
            line=dict(color='magenta', dash='dot')
        ), row=current_row, col=1)
        current_row += 1

    if 'OBV' in indicators_selected:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['obv'],
            name='OBV',
            line=dict(color='blue')
        ), row=current_row, col=1)
        current_row += 1

    if 'ATR' in indicators_selected:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['atr'],
            name='ATR',
            line=dict(color='yellow')
        ), row=current_row, col=1)
        current_row += 1

    # Generate prediction if requested
    if st.session_state.run_model:
        with st.spinner("Generating prediction..."):
            try:
                if st.session_state.predictions is None:
                    # Generate single prediction
                    pred, conf = make_prediction(df, interval=interval)  # Pass the selected interval
                    last_price = df['close'].iloc[-1]
                    
                    # Create single prediction point
                    freq_map = {'5m': '5T', '15m': '15T', '30m': '30T', 
                                '1h': '1H', '4h': '4H', '1d': '1D'}
                    freq = freq_map[interval]
                    future_time = df['timestamp'].iloc[-1] + pd.Timedelta(freq)
                    
                    # Convert prediction to price value
                    predicted_price = last_price * 1.0015 if pred == 1 else last_price * 0.9995
                    
                    # Store in session state
                    st.session_state.predictions = pred
                    st.session_state.prediction_df = pd.DataFrame({
                        'timestamp': [future_time],
                        'predicted_price': [predicted_price],
                        'confidence': [conf]
                    })
                
                # Add prediction to chart
                if st.session_state.prediction_df is not None:
                    pred_df = st.session_state.prediction_df
                    
                    fig.add_trace(go.Scatter(
                        x=pred_df['timestamp'],
                        y=pred_df['predicted_price'],
                        mode='markers',
                        name='Prediction',
                        marker=dict(
                            color='gold',
                            size=12,
                            symbol='diamond'
                        )
                    ), row=1, col=1)
                    
                    # Add confidence annotation
                    if pred_df['confidence'].iloc[0] is not None:
                        fig.add_annotation(
                            x=pred_df['timestamp'].iloc[0],
                            y=pred_df['predicted_price'].iloc[0],
                            text=f"Prediction: {'UP' if st.session_state.predictions == 1 else 'DOWN'}<br>Confidence: {pred_df['confidence'].iloc[0]:.0%}",
                            showarrow=True,
                            arrowhead=2,
                            ax=0,
                            ay=-40,
                            font=dict(color="gold", size=12)
                        )
                
                st.success("Prediction generated successfully!")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.session_state.run_model = False

    # Update layout
    fig.update_layout(
        height=800,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    # Update y-axis titles for indicators
    current_row = 3
    if 'RSI' in indicators_selected:
        fig.update_yaxes(title_text="RSI", row=current_row, col=1)
        current_row += 1
    if 'MACD' in indicators_selected:
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        current_row += 1
    if 'OBV' in indicators_selected:
        fig.update_yaxes(title_text="OBV", row=current_row, col=1)
        current_row += 1
    if 'ATR' in indicators_selected:
        fig.update_yaxes(title_text="ATR", row=current_row, col=1)
        current_row += 1

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Display prediction results if available
    if st.session_state.prediction_df is not None:
        st.subheader("Prediction Result")
        pred = st.session_state.predictions
        conf = st.session_state.prediction_df['confidence'].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Direction", "UP" if pred == 1 else "DOWN")
        with col2:
            st.metric("Confidence", f"{conf:.0%}")
        
        st.caption(f"Predicted for: {st.session_state.prediction_df['timestamp'].iloc[0]}")

if __name__ == "__main__":
    main()