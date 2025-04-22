# Set the page configuration (must be the first Streamlit command)
import streamlit as st
st.set_page_config(page_title="Crypto Predictor", layout="wide")

# Imports
import pandas as pd
import plotly.graph_objs as go
import requests
import time
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
import numpy as np
from indicators import (
    calculate_sma, calculate_bollinger_bands, calculate_rsi,
    calculate_ema, calculate_macd, calculate_obv, calculate_atr
)
from predictor import make_prediction
from lstm import LivePredictionSystem, initial_training
import json

# Initialize session state
if 'run_model' not in st.session_state:
    st.session_state.run_model = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'prediction_df' not in st.session_state:
    st.session_state.prediction_df = None
if 'show_indicator_guide' not in st.session_state:
    st.session_state.show_indicator_guide = False
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'last_prediction_time' not in st.session_state:
    st.session_state.last_prediction_time = None
if 'rf_trained' not in st.session_state:
    st.session_state.rf_trained = False


@st.cache_data
def set_rf_trained_state(state):
    return state

# Set the state
st.session_state.rf_trained = set_rf_trained_state(True)

# Function to update the RF log file
def update_rf_log(data):
    with open('rf_log.json', 'w') as f:
        json.dump(data, f)

# Function to update the LSTM log file
def update_lstm_log(data):
    try:
        # Convert NumPy data types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert NumPy arrays to lists
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)  # Convert NumPy floats to Python floats
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)  # Convert NumPy integers to Python integers
            elif isinstance(obj, np.bool_):
                return bool(obj)  # Convert NumPy booleans to Python booleans
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open('lstm_log.json', 'w') as f:
            json.dump(data, f, default=convert_numpy)
    except Exception as e:
        print(f"Error writing to LSTM log file: {e}")

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Page setup
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
        
        # Allow the user to select the cryptocurrency
        crypto = st.selectbox(
            "Select Cryptocurrency", 
            ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'], 
            index=0
        )
        
        interval = st.selectbox("Select Time Interval", ['5m', '15m', '30m', '1h', '4h', '1d'])
        
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.7)
        
        # Add an info button next to the "Run Random Forest Prediction" button
        col1, col2 = st.sidebar.columns([3, 1])

        with col1:
            if st.button("Run Random Forest Prediction"):
                st.session_state.run_model = True
                st.session_state.predictions = None
                st.session_state.prediction_df = None
                st.session_state.model_type = "rf"  # Set model type to "rf"
                
                # Display a warning message
                st.warning("This will take several minutes to generate. Please be patient.")
                
                # Display an additional message if the interval is 5m
                if interval == "5m":
                    st.info("It may create a delay of a few minutes in the data. An interval greater than 5 minutes is therefore recommended.")

        with col2:
            st.markdown(
                """
                <a href="/RFinfo" target="_blank">
                    <button style="border: none; padding: 5px 10px; cursor: pointer;">
                        ℹ️ Info
                    </button>
                </a>
                """,
                unsafe_allow_html=True
            )
        
        # Add an info button next to the "Run LSTM Prediction" button
        col1, col2 = st.sidebar.columns([3, 1])

        with col1:
            if st.button("Run LSTM Prediction"):
                st.session_state.run_model = True
                st.session_state.predictions = None
                st.session_state.prediction_df = None
                st.session_state.model_type = "lstm"  # Set model type to "lstm"
                
                # Display a warning message
                st.warning("This will take up to 5 minutes to generate. Please be patient.")
                
                # Display an additional message if the interval is 5m
                if interval == "5m":
                    st.info("It may create a delay of a few minutes in the data. An interval greater than 5 minutes is therefore recommended.")

        with col2:
            st.markdown(
                """
                <a href="/LSTMinfo" target="_blank">
                    <button style=" border: none; padding: 5px 10px; cursor: pointer;">
                        ℹ️ Info
                    </button>
                </a>
                """,
                unsafe_allow_html=True
            )
        # Add a button to trigger LSTM predictions
        if st.sidebar.button("Trigger LSTM Prediction"):
            st.session_state.run_model = True
            st.session_state.model_type = "lstm"

        # Button to toggle the indicator guide
        if st.sidebar.button("📘 What do these indicators mean?"):
            st.session_state.show_indicator_guide = not st.session_state.show_indicator_guide

        if st.session_state.show_indicator_guide:
            st.sidebar.markdown("### Technical Indicator Guide")

            st.sidebar.markdown("**SMA**: The average price, showing trend direction.")
            st.sidebar.markdown("**EMA**: Similar to SMA but gives more weight to recent prices.")
            st.sidebar.markdown("**RSI**: Shows if something is overbought or oversold (0–100). Values above 70 are overbought, below 30 are oversold.")
            st.sidebar.markdown("**MACD**: Measures momentum using two moving averages, helping identify potential buy/sell signals.")
            st.sidebar.markdown("**Bollinger Bands**: Tracks volatility using bands around the price. Bands widen with increased volatility.")
            st.sidebar.markdown("**ARIMA**: Uses past data to forecast future prices.")
            st.sidebar.markdown("**OBV**: Measures buying/selling pressure using volume. Rising OBV indicates buying pressure, falling indicates selling.")
            st.sidebar.markdown("**ATR**: Measures average price movement. Higher ATR indicates more volatility.")
            st.sidebar.markdown("**Hurst Exponent**: Indicates if price movements are random or trending. Values above 0.5 indicate a clearer trend.")

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
        
        logging.info(f"Fetched {len(df)} rows of data for {symbol} at interval {interval}")
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

    # Add technical indicators required by the LSTM system
    df = calculate_sma(df)
    df = calculate_ema(df)
    df = calculate_bollinger_bands(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_obv(df)
    df = calculate_atr(df)

    # Add additional features required by the LSTM system
    df['prev_close'] = df['close'].shift(1)
    df['log_return'] = np.log(df['prev_close'] / df['close'].shift(2))
    df['price_change'] = df['close'].diff()
    for window in [5, 15, 30]:
        df[f'close_ma_{window}'] = df['close'].rolling(window).mean()
        df[f'volatility_{window}'] = df['close'].rolling(window).std()
    df['volume_ma_15'] = df['volume'].rolling(15).mean()
    df['volume_change'] = df['volume'].diff()
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Add missing features
    df['buy_ratio'] = df['volume'] / (df['volume'].rolling(15).sum())
    df['bollinger_upper'] = df['close'] + (2 * df['close'].rolling(20).std())
    df['bollinger_lower'] = df['close'] - (2 * df['close'].rolling(20).std())
    df['stoch_k'] = ((df['close'] - df['low'].rolling(14).min()) /
                     (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['williams_r'] = ((df['high'].rolling(14).max() - df['close']) /
                        (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * -100

    df = df.dropna()

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
                if st.session_state.model_type == "lstm":
                    # Train the LSTM model if not already trained
                    if 'lstm_system' not in st.session_state:
                        models, scaler_features, scaler_target, features, seq_length, pred_length = initial_training(
                            symbol=crypto.replace('/', ''), interval=interval
                        )
                        st.session_state.lstm_system = LivePredictionSystem(
                            models, scaler_features, scaler_target, features, seq_length, pred_length,
                            symbol=crypto.replace('/', ''), interval=interval
                        )
                    
                    # Use the LSTM system to generate predictions
                    lstm_system = st.session_state.lstm_system
                    
                    # Check if it's time to generate a new prediction
                    current_time = datetime.now()
                    if st.session_state.last_prediction_time is None or (current_time - st.session_state.last_prediction_time).total_seconds() >= 300:
                        logging.info("Generating a new LSTM prediction...")
                        
                        # Fetch the latest data
                        df = fetch_data(crypto.replace('/', ''), interval)
                        logging.info(f"Fetched {len(df)} rows of data for {crypto} at interval {interval}")
                        
                        # Add required features
                        df = calculate_sma(df)
                        df = calculate_ema(df)
                        df = calculate_bollinger_bands(df)
                        df = calculate_rsi(df)
                        df = calculate_macd(df)
                        df = calculate_obv(df)
                        df = calculate_atr(df)
                        df['prev_close'] = df['close'].shift(1)
                        df['log_return'] = np.log(df['prev_close'] / df['close'].shift(2))
                        df['price_change'] = df['close'].diff()
                        for window in [5, 15, 30]:
                            df[f'close_ma_{window}'] = df['close'].rolling(window).mean()
                            df[f'volatility_{window}'] = df['close'].rolling(window).std()
                        df['volume_ma_15'] = df['volume'].rolling(15).mean()
                        df['volume_change'] = df['volume'].diff()
                        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
                        df['buy_ratio'] = df['volume'] / (df['volume'].rolling(15).sum())
                        df['bollinger_upper'] = df['close'] + (2 * df['close'].rolling(20).std())
                        df['bollinger_lower'] = df['close'] - (2 * df['close'].rolling(20).std())
                        df['stoch_k'] = ((df['close'] - df['low'].rolling(14).min()) /
                                         (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
                        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
                        df['williams_r'] = ((df['high'].rolling(14).max() - df['close']) /
                                            (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * -100
                        df = df.dropna()
                        
                        # Ensure all required features are present
                        last_sequence = df[lstm_system.features].values[-lstm_system.seq_length:]
                        logging.info(f"Last sequence shape: {last_sequence.shape}")
                        scaled_sequence = lstm_system.scaler_features.transform(last_sequence)
                        scaled_sequence = scaled_sequence.reshape(1, lstm_system.seq_length, -1)
                        logging.info(f"Scaled sequence shape: {scaled_sequence.shape}")
                        
                        # Generate predictions
                        avg_prediction, avg_classification = lstm_system.predict(scaled_sequence)
                        logging.info(f"Generated predictions: avg_prediction={avg_prediction}, avg_classification={avg_classification}")
                        
                        last_price = df['close'].iloc[-1]
                        
                        # Calculate the predicted price based on the predicted price change
                        predicted_price = last_price + avg_prediction
                        
                        # Map freq for timestamp calculation
                        freq_map = {'5m': '5T', '15m': '15T', '30m': '30T', 
                                    '1h': '1H', '4h': '4H', '1d': '1D'}
                        freq = freq_map[interval]
                        future_time = df['timestamp'].iloc[-1] + pd.Timedelta(freq)
                        prediction_start_time = df['timestamp'].iloc[-1]
                        
                        # Store the prediction in session state
                        st.session_state.predictions = avg_classification
                        st.session_state.prediction_df = pd.DataFrame({
                            'timestamp': [future_time],
                            'predicted_price': [predicted_price],
                            'confidence': [avg_classification]
                        })
                        
                        # Write to the LSTM log file
                        lstm_log_data = {
                            "lstm_trained": True,
                            "accuracy": 92.5,  # Replace with actual accuracy
                            "training_points": 15000,  # Replace with actual training points
                            "features": lstm_system.features,
                            "crypto": crypto,
                            "interval": interval,
                            "current_price": last_price,
                            "predictions": [
                                {
                                    "prediction_start_time": prediction_start_time.isoformat(),
                                    "prediction_end_time": future_time.isoformat(),
                                    "movement": "UP" if avg_classification > 0.5 else "DOWN",
                                    "predicted_price": predicted_price,
                                    "current_price": last_price,
                                    "confidence": avg_classification
                                }
                            ]
                        }
                        update_lstm_log(lstm_log_data)
                        
                        # Update the last prediction time
                        st.session_state.last_prediction_time = current_time
                        st.success("LSTM prediction generated successfully!")
                    else:
                        st.warning("It's not time for a new prediction yet. Please wait for the 5-minute interval.")
                elif st.session_state.model_type == "rf":
                    # Random Forest prediction logic
                    pred, conf = make_prediction(df, interval=interval, symbol=crypto)
                    last_price = df['close'].iloc[-1]
                    
                    # Create single prediction point
                    freq_map = {'5m': '5T', '15m': '15T', '30m': '30T', 
                                '1h': '1H', '4h': '4H', '1d': '1D'}
                    freq = freq_map[interval]
                    future_time = df['timestamp'].iloc[-1] + pd.Timedelta(freq)
                    prediction_start_time = df['timestamp'].iloc[-1]
                    
                    # Convert prediction to price value
                    predicted_price = last_price * 1.0015 if pred == 1 else last_price * 0.9995
                    
                    # Store in session state
                    st.session_state.predictions = pred
                    st.session_state.prediction_df = pd.DataFrame({
                        'timestamp': [future_time],
                        'predicted_price': [predicted_price],
                        'confidence': [conf]
                    })
                    
                    # Mark the RF model as trained
                    st.session_state.rf_trained = True
                    
                    # Write to the RF log file
                    rf_log_data = {
                        "rf_trained": True,
                        "accuracy": 94.18,  # Replace with actual accuracy
                        "training_points": 10000,  # Replace with actual training points
                        "features": ['Open', 'High', 'Low', 'Close', 'Volume'],  # Replace with actual features
                        "crypto": crypto,
                        "interval": interval,
                        "current_price": last_price,
                        "predictions": [
                            {
                                "prediction_start_time": prediction_start_time.isoformat(),
                                "prediction_end_time": future_time.isoformat(),
                                "movement": "UP" if pred == 1 else "DOWN",
                                "predicted_price": predicted_price,
                                "current_price": last_price
                            }
                        ]
                    }
                    update_rf_log(rf_log_data)
                    
                    st.success("Random Forest prediction generated successfully!")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.session_state.run_model = False

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
        model_type = getattr(st.session_state, 'model_type', 'rf')
        if pred_df['confidence'].iloc[0] is not None:
            if model_type == "lstm":
                # LSTM-specific annotation
                fig.add_annotation(
                    x=pred_df['timestamp'].iloc[0],
                    y=pred_df['predicted_price'].iloc[0],
                    text=f"LSTM Prediction: {'UP' if st.session_state.predictions > 0.5 else 'DOWN'}<br>Confidence: {pred_df['confidence'].iloc[0]:.0%}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40,
                    font=dict(color="gold", size=12)
                )
            else:
                # RF-specific annotation
                fig.add_annotation(
                    x=pred_df['timestamp'].iloc[0],
                    y=pred_df['predicted_price'].iloc[0],
                    text=f"RF Prediction: {'UP' if st.session_state.predictions == 1 else 'DOWN'}<br>Confidence: {pred_df['confidence'].iloc[0]:.0%}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40,
                    font=dict(color="gold", size=12)
                )
    
    # Ensure model_type is set before displaying the success message
    if st.session_state.model_type:
        st.success(f"{st.session_state.model_type.upper()} prediction generated successfully!")
    else:
        st.warning("Prediction generated, but model type is not set.")

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
        pred = st.session_state.predictions  # This is avg_classification
        conf = st.session_state.prediction_df['confidence'].iloc[0]
        
        # Determine direction based on classification output
        direction = "UP" if pred > 0.5 else "DOWN"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Direction", direction)
        with col2:
            st.metric("Confidence", f"{conf:.0%}")
        
        st.caption(f"Predicted for: {st.session_state.prediction_df['timestamp'].iloc[0]}")
    # Trend Summary Section
    st.subheader("Indicator Summary & Trend Signals")
    display_trend_signals(df, indicators_selected)

def display_signal(name, trend, details):
    color = 'green' if trend == 'up' else 'red'
    arrow = '⬆️' if trend == 'up' else '⬇️'
    st.markdown(
        f"<div style='padding:8px;border-radius:10px;background-color:#1e1e1e;margin-bottom:10px;'>"
        f"<strong style='color:{color};'>{arrow} {name}</strong><br>"
        f"<span style='color:#ccc;'>{details}</span>"
        f"</div>", unsafe_allow_html=True
    )

def display_trend_signals(df, indicators_selected):
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

if __name__ == "__main__":
    main()