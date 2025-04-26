# imports
import streamlit as st
st.set_page_config(page_title="Crypto Predictor", layout="wide")
import traceback
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

# session state 
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
if 'lstm_system' not in st.session_state:
    st.session_state.lstm_system = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

@st.cache_data
def set_rf_trained_state(state):
    return state

st.session_state.rf_trained = set_rf_trained_state(True)

# update the RF log file
def update_rf_log(data):
    with open('rf_log.json', 'w') as f:
        json.dump(data, f)

# update the LSTM log file
def update_lstm_log(data):
    try:
        # NumPy to Python types
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist() 
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)  
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)  
            elif isinstance(obj, np.bool_):
                return bool(obj)  
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open('lstm_log.json', 'w') as f:
            json.dump(data, f, default=convert_numpy)
    except Exception as e:
        print(f"Error writing to LSTM log file: {e}")

def find_turning_points(series):
    """Enhanced turning point detection with confirmation."""
    if len(series) < 5:  # turning piont daata
        return []
    
    turning_points = []
    for i in range(2, len(series) - 2):
        prev_diff = series[i - 1] - series[i - 2]
        curr_diff = series[i] - series[i - 1]
        next_diff = series[i + 1] - series[i]
        next_next_diff = series[i + 2] - series[i + 1]
        
        if curr_diff * next_diff < 0:
            if (next_diff > 0 and next_next_diff > 0) or (next_diff < 0 and next_next_diff < 0):
                turning_points.append(i)
    return turning_points

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    st.title("Crypto Report")
    st.header("A simple and easy to use prediction app")

    # sidebar 
    with st.sidebar:
        st.header("Chart Settings")
        chart_type = st.radio("Chart Type", ["Candlestick", "Line"], index=0)
        
        st.header("Technical Indicators")
        indicators_selected = st.multiselect(
            "Select Indicators to Display",
            ['SMA', 'EMA', 'Bollinger Bands', 'RSI', 'MACD', 'OBV', 'ATR'],
            default=['SMA', 'Bollinger Bands']
        )
        
        # the user can select the cryptocurrency
        crypto = st.selectbox(
            "Select Cryptocurrency", 
            ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'], 
            index=0
        )
        #time selection
        interval = st.selectbox("Select Time Interval", ['5m', '15m', '30m', '1h', '4h', '1d'])
                
        #buttons for lstm and rf
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            if st.button("Run Random Forest Prediction"):
                st.session_state.run_model = True
                st.session_state.predictions = None
                st.session_state.prediction_df = None
                st.session_state.model_type = "rf"  
                st.warning("This will take several minutes to generate. Please be patient.")
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

        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            if st.button("Run LSTM Prediction"):
                st.session_state.run_model = True
                st.session_state.predictions = None
                st.session_state.prediction_df = None
                st.session_state.model_type = "lstm"  
                st.warning("This will take up to 5 minutes to generate. Please be patient.")
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
        if st.sidebar.button("Trigger LSTM Prediction"):
            if st.session_state.lstm_system is None:
                st.warning("No pre-trained LSTM model found. Please run 'Run LSTM Prediction' first.")
            else:
                st.session_state.run_model = True
                st.session_state.model_type = "lstm"
                st.session_state.predictions = None
                st.session_state.prediction_df = None

        # button for indicator guide
        if st.sidebar.button("What do these indicators mean?"):
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

    # fetch data
    @st.cache_data(ttl=300)  
    def fetch_data(symbol, interval, limit=200):  
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

    if df.empty or 'close' not in df.columns:
        st.error("Live data is empty")
        return

    #indicators 
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
    df = calculate_sma(df)
    df = calculate_ema(df)
    df = calculate_bollinger_bands(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_obv(df)
    df = calculate_atr(df)
    df['prev_close'] = df['close'].shift(1)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))  
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

    indicator_rows = sum(1 for ind in indicators_selected if ind in ['RSI', 'MACD', 'OBV', 'ATR'])
    row_count = 2 + indicator_rows  
    row_heights = [0.5, 0.2] + [0.3/indicator_rows]*indicator_rows if indicator_rows > 0 else [0.7, 0.3]
    
    subplot_titles = ["Price Chart", "Volume"]
    if 'RSI' in indicators_selected:
        subplot_titles.append("RSI")
    if 'MACD' in indicators_selected:
        subplot_titles.append("MACD")
    if 'OBV' in indicators_selected:
        subplot_titles.append("OBV")
    if 'ATR' in indicators_selected:
        subplot_titles.append("ATR")

    # plot
    fig = make_subplots(
        rows=row_count, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )

    # chart
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

    # volume chart
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['volume'],
        name='Volume',
        marker_color='rgba(100, 100, 255, 0.6)'
    ), row=2, col=1)

    # main chart indicators
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

    current_row = 3

    # indicators with their own chart
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

    # LSTM prediction
    if st.session_state.run_model:
        with st.spinner("Generating prediction..."):
            try:
                if st.session_state.model_type == "lstm":
                    # check if pre trained LSTM
                    if st.session_state.lstm_system is not None:
                        lstm_system = st.session_state.lstm_system
                        seq_length = lstm_system.seq_length
                        st.info("Using pre-trained LSTM model for prediction.")
                    else:
                        st.warning("Initializing LSTM model (this may take a few minutes)...")
                        try:
                            models, scaler_features, scaler_target, features, seq_length, pred_length, overall_acc, up_acc, down_acc = initial_training(
                                symbol=binance_symbol, interval=interval
                            )
                            st.session_state.lstm_system = LivePredictionSystem(
                                models, scaler_features, scaler_target, features, 
                                seq_length, pred_length,
                                symbol=binance_symbol, interval=interval,
                                overall_accuracy=overall_acc, up_accuracy=up_acc, down_accuracy=down_acc
                            )
                            lstm_system = st.session_state.lstm_system
                            st.success("LSTM model initialized successfully!")
                        except Exception as e:
                            st.error(f"Failed to initialize LSTM: {str(e)}")
                            st.error(traceback.format_exc())
                            st.session_state.run_model = False
                            return
                    try:
                        required_features = lstm_system.features
                        missing_features = [f for f in required_features if f not in df.columns]
                        if missing_features:
                            st.warning(f"Adding missing features: {missing_features}")

                        df['buy_ratio'] = df['volume'] / (df['volume'].rolling(15).sum().fillna(df['volume']))
                        df['bollinger_upper'] = df['close'] + (2 * df['close'].rolling(20).std())
                        df['bollinger_lower'] = df['close'] - (2 * df['close'].rolling(20).std())
                        df['stoch_k'] = df.apply(
                            lambda x: 100 * (x['close'] - df['low'].rolling(14).min().loc[x.name]) / 
                                    max(0.0001, df['high'].rolling(14).max().loc[x.name] - df['low'].rolling(14).min().loc[x.name]),
                            axis=1
                        )
                        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
                        df['williams_r'] = df.apply(
                            lambda x: -100 * (df['high'].rolling(14).max().loc[x.name] - x['close']) / 
                                    max(0.0001, df['high'].rolling(14).max().loc[x.name] - df['low'].rolling(14).min().loc[x.name]),
                            axis=1
                        )
                        df = df.dropna()
                        if len(df) < seq_length:
                            st.error(f"Not enough data for prediction after preprocessing. Need {seq_length} points, have {len(df)}")
                            st.session_state.run_model = False
                            return

                        last_sequence = df[required_features].values[-seq_length:]
                        scaled_sequence = lstm_system.scaler_features.transform(last_sequence)
                        scaled_sequence = scaled_sequence.reshape(1, seq_length, -1)
                        
                        try:
                            predictions = []
                            for model in lstm_system.models:
                                pred = model.predict(scaled_sequence, verbose=0)
                                predictions.append(pred)

                            regression_outputs = [pred[0][0][0] for pred in predictions]
                            classification_outputs = [pred[1][0][0] for pred in predictions]
                            avg_prediction = np.mean(regression_outputs)
                            avg_classification = np.mean(classification_outputs)
                            last_price = df['close'].iloc[-1]
                            predicted_price = last_price * (1 + (0.0015 if avg_classification > 0.5 else -0.0015))
                            
                            freq_map = {'5m': '5T', '15m': '15T', '30m': '30T', 
                                    '1h': '1H', '4h': '4H', '1d': '1D'}
                            future_time = df['timestamp'].iloc[-1] + pd.Timedelta(freq_map[interval])
                            
                            st.session_state.predictions = avg_classification
                            st.session_state.prediction_df = pd.DataFrame({
                                'timestamp': [future_time],
                                'predicted_price': [predicted_price],
                                'confidence': [max(0.1, abs(avg_classification - 0.5) * 2)]
                            })
                            
                            # Update LSTM log
                            lstm_log_data = {
                                "lstm_trained": True,
                                "features": required_features,
                                "crypto": crypto,
                                "interval": interval,
                                "current_price": float(last_price),
                                "predictions": [
                                    {
                                        "prediction_start_time": df['timestamp'].iloc[-1].isoformat(),
                                        "prediction_end_time": future_time.isoformat(),
                                        "movement": "UP" if avg_classification > 0.5 else "DOWN",
                                        "predicted_price": float(predicted_price),
                                        "current_price": float(last_price),
                                        "confidence": float(max(0.1, abs(avg_classification - 0.5) * 2)),
                                        "training_overall_accuracy": float(lstm_system.overall_accuracy),
                                        "training_up_accuracy": float(lstm_system.up_accuracy),
                                        "training_down_accuracy": float(lstm_system.down_accuracy)
                                    }
                                ]
                            }
                            update_lstm_log(lstm_log_data)                           
                            st.success("LSTM prediction generated successfully!")
                            st.markdown("### LSTM Model Training Accuracy")
                            
                            # message for overall accuracy
                            accuracy_percentage = lstm_system.overall_accuracy * 100
                            if accuracy_percentage >= 60:
                                st.success(f"Overall Accuracy: {accuracy_percentage:.2f}%")
                            elif accuracy_percentage >= 55:
                                st.warning(f"Overall Accuracy: {accuracy_percentage:.2f}%")
                            else:
                                st.error(f"⚠️ Low Overall Accuracy: {accuracy_percentage:.2f}% - Predictions may be unreliable!")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Overall Accuracy", f"{lstm_system.overall_accuracy*100:.2f}%")
                            with col2:
                                st.metric("Upward Movement Accuracy", f"{lstm_system.up_accuracy*100:.2f}%")
                            with col3:
                                st.metric("Downward Movement Accuracy", f"{lstm_system.down_accuracy*100:.2f}%")
                            
                            if st.session_state.prediction_history:
                                for idx, prev_pred in enumerate(st.session_state.prediction_history):
                                    if 'verified' not in prev_pred or not prev_pred['verified']:
                                        pred_end_time = pd.to_datetime(prev_pred['timestamp'])
                                        if pred_end_time <= df['timestamp'].iloc[-1]:
                                            closest_idx = df['timestamp'].searchsorted(pred_end_time)
                                            if closest_idx < len(df):
                                                actual_price = df['close'].iloc[closest_idx]
                                                starting_price = prev_pred['starting_price']
                                                

                                                actual_direction = "UP" if actual_price > starting_price else "DOWN"
                                                predicted_direction = prev_pred['direction']
                                                st.session_state.prediction_history[idx]['verified'] = True
                                                st.session_state.prediction_history[idx]['actual_price'] = float(actual_price)
                                                st.session_state.prediction_history[idx]['correct'] = (actual_direction == predicted_direction)

                            new_prediction = {
                                'timestamp': future_time,  
                                'predicted_price': float(predicted_price),
                                'confidence': float(max(0.1, abs(avg_classification - 0.5) * 2)),
                                'direction': "UP" if avg_classification > 0.5 else "DOWN",
                                'starting_price': float(last_price),
                                'verified': False 
                            }

                            st.session_state.prediction_history.append(new_prediction)
                                                        
                        except Exception as e:
                            st.error(f"LSTM prediction calculation failed: {str(e)}")
                            st.error(traceback.format_exc())
                            st.session_state.run_model = False
                            
                    except Exception as e:
                        st.error(f"LSTM data preparation failed: {str(e)}")
                        st.error(traceback.format_exc())
                        st.session_state.run_model = False
                #logic for the rf
                elif st.session_state.model_type == "rf":
                    try:
                        pred, conf = make_prediction(df, interval=interval, symbol=crypto)
                        last_price = df['close'].iloc[-1]
                        
                        freq_map = {'5m': '5T', '15m': '15T', '30m': '30T', 
                                    '1h': '1H', '4h': '4H', '1d': '1D'}
                        freq = freq_map[interval]
                        future_time = df['timestamp'].iloc[-1] + pd.Timedelta(freq)
                        prediction_start_time = df['timestamp'].iloc[-1]
                        # the rf just gives mvement so ive pre set the price change
                        predicted_price = last_price * 1.0015 if pred == 1 else last_price * 0.9985 
                        st.session_state.predictions = pred
                        st.session_state.prediction_df = pd.DataFrame({
                            'timestamp': [future_time],
                            'predicted_price': [predicted_price],
                            'confidence': [conf]
                        })
                    
                        st.session_state.rf_trained = True

                        rf_log_data = {
                            "rf_trained": True,
                            "crypto": crypto,
                            "interval": interval,
                            "current_price": float(last_price),
                            "predictions": [
                                {
                                    "prediction_start_time": prediction_start_time.isoformat(),
                                    "prediction_end_time": future_time.isoformat(),
                                    "movement": "UP" if pred == 1 else "DOWN",
                                    "predicted_price": float(predicted_price),
                                    "current_price": float(last_price),
                                    "confidence": float(conf)
                                }
                            ]
                        }
                        update_rf_log(rf_log_data)
                        st.success("Random Forest prediction generated successfully!")                       
                    except Exception as e:
                        st.error(f"Random Forest prediction failed: {str(e)}")
                        st.error(traceback.format_exc())
                        st.session_state.run_model = False
                        pass                       
            except Exception as e:
                st.error(f"Unexpected error in prediction system: {str(e)}")
                st.error(traceback.format_exc())
                st.session_state.run_model = False                   

    if st.session_state.prediction_history:
        correct_preds = [p for p in st.session_state.prediction_history 
                        if 'verified' in p and p['verified'] and p['correct']]
        incorrect_preds = [p for p in st.session_state.prediction_history 
                        if 'verified' in p and p['verified'] and not p['correct']]
    if st.session_state.prediction_df is not None:
        pred_df = st.session_state.prediction_df
        
        fig.add_trace(go.Scatter(
            x=pred_df['timestamp'],
            y=pred_df['predicted_price'],
            mode='markers',
            name='Current Prediction',
            marker=dict(
                color='gold',
                size=12,
                symbol='diamond'
            )
        ), row=1, col=1)
        
        if st.session_state.prediction_history:
            correct_preds = [p for p in st.session_state.prediction_history 
                            if 'verified' in p and p['verified'] and p['correct']]
            incorrect_preds = [p for p in st.session_state.prediction_history 
                            if 'verified' in p and p['verified'] and not p['correct']]
            if correct_preds:
                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(p['timestamp']) for p in correct_preds],
                    y=[p['predicted_price'] for p in correct_preds],
                    mode='markers',
                    name='Correct Predictions',
                    marker=dict(
                        color='green',
                        size=10,
                        symbol='circle',
                        line=dict(width=2, color='white')
                    )
                ), row=1, col=1)
            if incorrect_preds:
                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(p['timestamp']) for p in incorrect_preds],
                    y=[p['predicted_price'] for p in incorrect_preds],
                    mode='markers',
                    name='Incorrect Predictions', 
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='x',
                        line=dict(width=2, color='white')
                    )
                ), row=1, col=1)
            model_type = getattr(st.session_state, 'model_type', 'rf')
            if len(pred_df) > 0 and 'confidence' in pred_df.columns and pred_df['confidence'].iloc[0] is not None:
                if model_type == "lstm":
                    predicted_direction = 'UP' if st.session_state.predictions > 0.5 else 'DOWN'
                    confidence_pct = min(100, max(0, int(pred_df['confidence'].iloc[0] * 100)))
                    
                    fig.add_annotation(
                        x=pred_df['timestamp'].iloc[0],
                        y=pred_df['predicted_price'].iloc[0],
                        text=f"LSTM Prediction: {predicted_direction}<br>Confidence: {confidence_pct}%",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-40,
                        font=dict(color="gold", size=12)
                    )
                else:
                    predicted_direction = 'UP' if st.session_state.predictions == 1 else 'DOWN'
                    confidence_pct = min(100, max(0, int(pred_df['confidence'].iloc[0] * 100)))
    fig.update_layout(
        height=800,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    if st.session_state.prediction_history:
        st.subheader("Prediction History & Performance")

        verified_preds = [p for p in st.session_state.prediction_history if 'verified' in p and p['verified']]
        correct_preds = [p for p in verified_preds if p['correct']]       
        total_verified = len(verified_preds)
        total_correct = len(correct_preds)       
        if total_verified > 0:
            accuracy = (total_correct / total_verified) * 100
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", f"{total_verified}")
            with col2:
                st.metric("Correct Predictions", f"{total_correct}")
            with col3:
                st.metric("Live Accuracy", f"{accuracy:.2f}%")
            with st.expander("View Prediction History"):
                history_df = pd.DataFrame(verified_preds)
                history_df['Time'] = pd.to_datetime(history_df['timestamp'])
                history_df['Start Price'] = history_df['starting_price']
                history_df['Predicted'] = history_df['direction']
                history_df['Actual Price'] = history_df['actual_price']
                history_df['Result'] = history_df['correct'].apply(lambda x: "✅ Correct" if x else "❌ Incorrect")
                
                st.dataframe(
                    history_df[['Time', 'Start Price', 'Predicted', 'Actual Price', 'Result']],
                    use_container_width=True
                )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Indicator Summary & Trend Signals")
    # direction based on indicators
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

    display_trend_signals(df, indicators_selected)    
if __name__ == "__main__":
    main()