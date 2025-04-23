# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import TimeSeriesSplit
import types  # For method type binding
import os
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data(symbol='BTCUSDT', interval='5m', limit=2000, is_live=False):
    """
    Fetch historical or live data from Binance API with comprehensive error handling.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        interval (str): Candlestick interval (e.g., '5m', '1h')
        limit (int): Maximum number of records to retrieve
        is_live (bool): If True, uses more extensive error handling for live trading
        
    Returns:
        pd.DataFrame: Processed data frame with proper types
    """
    try:
        url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
        data = requests.get(url).json()
        
        # Create DataFrame with all columns as strings initially
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert numeric columns - handle potential non-numeric values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'quote_asset_volume', 'trades',
                       'taker_buy_base', 'taker_buy_quote']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Drop any rows with NaN values in key columns
        df = df.dropna(subset=['close', 'volume', 'taker_buy_base'])
        
        return df
    
    except Exception as e:
        error_msg = f"Error fetching {'live' if is_live else 'historical'} data: {e}"
        if is_live:
            print(error_msg)
            return pd.DataFrame()  # Return empty DataFrame on error for live mode
        else:
            raise Exception(error_msg)  # Raise exception for historical mode
        
def add_technical_indicators(df, is_live=False):
    """
    Add technical indicators to price data with error handling.
    
    Args:
        df (pd.DataFrame): Price data
        is_live (bool): If True, uses more extensive error handling for live trading
        
    Returns:
        pd.DataFrame: DataFrame with technical indicators added
    """
    if df.empty:
        return df
        
    try:
        df = df.copy()
        
        # Price features
        df['prev_close'] = df['close'].shift(1)
        df['log_return'] = np.log(df['prev_close'] / df['close'].shift(2))
        df['price_change'] = df['close'].diff()
        
        # Moving averages
        for window in [5, 15, 30]:
            df[f'close_ma_{window}'] = df['close'].rolling(window).mean()
            df[f'volatility_{window}'] = df['close'].rolling(window).std()
        
        # Volume features
        df['volume_ma_15'] = df['volume'].rolling(15).mean()
        df['volume_change'] = df['volume'].diff()
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Oscillators
        df['rsi'] = compute_rsi(df['close'], 14)
        df['macd'], df['macd_signal'] = compute_macd(df['close'])
        df['bollinger_upper'] = df['close_ma_15'] + (2 * df['volatility_15'])
        df['bollinger_lower'] = df['close_ma_15'] - (2 * df['volatility_15'])
        
        # Stochastic Oscillator
        df['stoch_k'] = ((df['close'] - df['low'].rolling(window=14).min()) /
                        (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())) * 100
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Williams %R
        df['williams_r'] = ((df['high'].rolling(window=14).max() - df['close']) /
                           (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())) * -100
        
        # ADX
        df['adx'] = compute_adx(df['high'], df['low'], df['close'], 14)
        
        # Buy ratio (ensure no division by zero)
        df['buy_ratio'] = df['taker_buy_base'] / df['volume']
        df['buy_ratio'] = df['buy_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return df.dropna()
        
    except Exception as e:
        if is_live:
            print(f"Error calculating indicators: {e}")
            return df  # Return whatever we have for live mode
        else:
            raise Exception(f"Error calculating indicators: {e}")  # Raise for historical
        
def compute_rsi(series, window):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

# Add ADX computation
def compute_adx(high, low, close, window):
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    atr = tr.rolling(window).mean()
    
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window).mean()
    return adx

def create_sequences(features, targets, seq_length, pred_length, is_live=False):
    """
    Create sequences for regression and classification tasks.
    
    Args:
        features: Scaled feature data (numpy array).
        targets: Scaled target data (numpy array).
        seq_length: Number of time steps in each input sequence.
        pred_length: Number of time steps ahead to predict.
        is_live: If True, uses more strict bounds checking for live trading.
    
    Returns:
        X: Input sequences (numpy array).
        y_reg: Regression targets (numpy array).
        y_class: Classification targets (numpy array).
    """
    X, y_reg, y_class = [], [], []
    
    # Check if we have enough data
    if is_live and len(features) < seq_length + pred_length:
        print(f"Warning: Not enough data. Need {seq_length + pred_length}, have {len(features)}")
        return np.array([]), np.array([]), np.array([])
    
    for i in range(seq_length, len(features) - pred_length):
        X.append(features[i-seq_length:i])
        y_reg.append(targets[i + pred_length - 1])  # Regression target: future price
        # Classification target: 1 if price increases, 0 otherwise
        y_class.append(1 if targets[i + pred_length - 1] > targets[i + pred_length - 2] else 0)
    
    return np.array(X), np.array(y_reg), np.array(y_class)


# Model Architecture
def build_model(input_shape):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    
    # Shared LSTM layers
    x = LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001))(input_layer)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Regression output
    reg_output = Dense(1, name='regression_output')(x)
    
    # Classification output (for direction)
    class_output = Dense(1, activation='sigmoid', name='classification_output')(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=[reg_output, class_output])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss={
            'regression_output': 'mse',
            'classification_output': 'binary_crossentropy'
        },
        metrics={
            'regression_output': 'mae',
            'classification_output': 'accuracy'
        }
    )
    
    return model

def find_turning_points(series):
    """Enhanced turning point detection with confirmation"""
    turning_points = []
    for i in range(2, len(series)-2):
        # Current and previous differences
        prev_diff = series[i-1] - series[i-2]
        curr_diff = series[i] - series[i-1]
        next_diff = series[i+1] - series[i]
        next_next_diff = series[i+2] - series[i+1]
        
        # Potential turning point (sign change in first differences)
        if curr_diff * next_diff < 0:
            # Confirmation: check if the move continues in the new direction
            if (next_diff > 0 and next_next_diff > 0) or (next_diff < 0 and next_next_diff < 0):
                turning_points.append(i)
    return turning_points

# ======================
# 3. Initial Model Training
# ======================
def initial_training(symbol='BTCUSDT', interval='5m'):
    print(f"Starting initial model training for {symbol} with {interval} interval...")
    
    # Fetch historical data
    btc_data = fetch_data(symbol=symbol, interval=interval)
    btc_data = add_technical_indicators(btc_data)
    
    features = ['prev_close', 'log_return', 'price_change', 
                'close_ma_5', 'close_ma_15', 'close_ma_30',
                'volatility_5', 'volatility_15', 'volatility_30',
                'volume_ma_15', 'volume_change', 'obv',
                'rsi', 'macd', 'macd_signal', 'buy_ratio',
                'bollinger_upper', 'bollinger_lower', 'stoch_k', 'stoch_d', 'williams_r']
    
    # Scale features and target separately
    scaler_features = RobustScaler()
    scaler_target = RobustScaler()
    
    scaled_features = scaler_features.fit_transform(btc_data[features])
    scaled_target = scaler_target.fit_transform(btc_data[['close']])
    
    # Define sequence length and prediction length
    seq_length = 60  # Number of time steps in each input sequence
    pred_length = 1  # Number of time steps ahead to predict
    
    # Create sequences
    X, y_reg, y_class = create_sequences(scaled_features, scaled_target, seq_length, pred_length)
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    train_indices, test_indices = [], []
    for train_index, test_index in tscv.split(X):
        train_indices.append(train_index)
        test_indices.append(test_index)
    
    # Train models
    models = []
    regression_mae_scores = []
    classification_accuracy_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(zip(train_indices, test_indices)):
        print(f"\nTraining Fold {fold + 1}")
        
        X_train, y_reg_train, y_class_train = X[train_idx], y_reg[train_idx], y_class[train_idx]
        X_val, y_reg_val, y_class_val = X[val_idx], y_reg[val_idx], y_class[val_idx]
        
        # Use the existing build_model function
        model = build_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
        ]
        
        history = model.fit(
            X_train, {'regression_output': y_reg_train, 'classification_output': y_class_train},
            epochs=100,
            batch_size=64,
            validation_data=(X_val, {'regression_output': y_reg_val, 'classification_output': y_class_val}),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate the model on the validation set
        val_predictions = model.predict(X_val, verbose=0)
        val_regression_predictions = val_predictions[0].flatten()
        val_classification_predictions = (val_predictions[1].flatten() > 0.5).astype(int)
        
        # Calculate regression MAE
        regression_mae = mean_absolute_error(y_reg_val, val_regression_predictions)
        regression_mae_scores.append(regression_mae)
        
        # Calculate classification accuracy
        classification_accuracy = np.mean(val_classification_predictions == y_class_val)
        classification_accuracy_scores.append(classification_accuracy)
        
        print(f"Fold {fold + 1} - Regression MAE: {regression_mae:.4f}, Classification Accuracy: {classification_accuracy:.4f}")
        
        models.append(model)
    
    # Output overall metrics
    print("\nTraining Completed.")
    print(f"Average Regression MAE: {np.mean(regression_mae_scores):.4f}")
    print(f"Average Classification Accuracy: {np.mean(classification_accuracy_scores):.4f}")
    
    return models, scaler_features, scaler_target, features, seq_length, pred_length

class LivePredictionSystem:
    """
    System that uses LSTM predictions to generate outputs like direction, probability, and turning points.
    """
    def __init__(self, models, scaler_features, scaler_target, features, seq_length, pred_length, 
                 symbol='BTCUSDT', interval='5m'):
        # Model and scaling parameters
        self.models = models
        self.scaler_features = scaler_features
        self.scaler_target = scaler_target
        self.features = features
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        # Metadata
        self.symbol = symbol
        self.interval = interval
        
        print(f"LSTM-based Prediction System initialized for {symbol} with {interval} interval")
    
    def predict(self, sequence, live_data):
        """
        Predict the price change, direction, and turning points.
        
        Args:
            sequence: Input sequence for prediction.
            live_data: DataFrame containing the live data for turning point detection.
        
        Returns:
            avg_prediction: Average predicted price change.
            avg_classification: Average probability of upward movement.
            turning_points: List of detected turning points.
        """
        print(f"Predicting with input sequence shape: {sequence.shape}")
        
        # Make predictions with all models
        predictions = []
        for model in self.models:
            pred = model.predict(sequence, verbose=0)
            predictions.append(pred)
        
        # Extract regression and classification outputs
        regression_outputs = [pred[0][0][0] for pred in predictions]  # First array, first value
        classification_outputs = [pred[1][0][0] for pred in predictions]  # Second array, first value
        
        avg_prediction = np.mean(regression_outputs)  # Average regression output
        avg_classification = np.mean(classification_outputs)  # Average classification output
        
        # Detect turning points in the live data
        turning_points = find_turning_points(live_data['close'])
        
        print(f"Averaged prediction: {avg_prediction}, classification: {avg_classification}")
        print(f"Detected turning points: {turning_points}")
        return avg_prediction, avg_classification, turning_points
    
def get_interval_seconds(interval):
    """Convert interval string to seconds."""
    unit = interval[-1]
    value = int(interval[:-1])
    
    if unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 60 * 60
    elif unit == 'd':
        return value * 24 * 60 * 60
    else:
        return 300  # Default to 5 minutes
    
def run_parallel_systems(symbol, interval, run_duration_hours):
    """
    Run the prediction system for the specified symbol and interval.

    Args:
        symbol (str): Trading symbol (e.g., 'BTCUSDT').
        interval (str): Time interval for predictions (e.g., '5m').
        run_duration_hours (int): Duration to run the system in hours.
    """
    print(f"Running prediction system for {symbol} with {interval} interval for {run_duration_hours} hours.")

    # Fetch initial data
    data = fetch_data(symbol=symbol, interval=interval, limit=2000)
    if data.empty:
        print("No data fetched. Exiting.")
        return

    # Add technical indicators
    data = add_technical_indicators(data)

    # Initialize the prediction system
    models, scaler_features, scaler_target, features, seq_length, pred_length = initial_training(
        symbol=symbol, interval=interval
    )
    lp_system = LivePredictionSystem(
        models, scaler_features, scaler_target, features, seq_length, pred_length,
        symbol=symbol, interval=interval
    )

    # Determine sleep time based on interval
    sleep_seconds = get_interval_seconds(interval)

    # Simulate live predictions
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=run_duration_hours)

    while datetime.now() < end_time:
        print(f"\n[{datetime.now()}] Fetching live data...")
        live_data = fetch_data(symbol=symbol, interval=interval, limit=2000, is_live=True)
        if live_data.empty:
            print(f"[{datetime.now()}] No live data fetched. Retrying in 60 seconds...")
            time.sleep(60)  # Wait for 1 minute before retrying
            continue

        print(f"[{datetime.now()}] Adding technical indicators to live data...")
        live_data = add_technical_indicators(live_data, is_live=True)

        # Ensure input to scaler has feature names
        last_sequence = live_data[features].values[-seq_length:]
        scaled_sequence = scaler_features.transform(last_sequence)
        scaled_sequence = scaled_sequence.reshape(1, seq_length, -1)

        print(f"[{datetime.now()}] Generating predictions...")
        avg_prediction, avg_classification, turning_points = lp_system.predict(scaled_sequence, live_data)

        # Output predictions
        current_price = live_data['close'].iloc[-1]
        print(f"Current price: {current_price}")
        print(f"Predicted price change: {avg_prediction}")
        print(f"Direction probability: {avg_classification}")
        print(f"Turning points: {turning_points}")

        # Wait for the next interval
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    # Run the prediction system for BTC/USDT with 5-minute intervals for 24 hours
    run_parallel_systems(symbol='BTCUSDT', interval='5m', run_duration_hours=24)