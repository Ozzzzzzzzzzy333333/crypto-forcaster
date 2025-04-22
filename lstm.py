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

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

# ======================
# 1. Data Fetching - Updated with better type handling
# ======================
def fetch_binance_data(symbol='BTCUSDT', interval='5m', limit=2000):
    """Fetch historical data from Binance API."""
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
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, invalid as NaN
        
    # Drop any rows with NaN values in key columns
    df = df.dropna(subset=['close', 'volume', 'taker_buy_base'])
    
    return df

def fetch_live_data(symbol='BTCUSDT', interval='5m', limit=2000):
    """Match historic data format exactly with error handling"""
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
        print(f"Error fetching live data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# ======================
# 2. Enhanced Feature Engineering with safer calculations
# ======================
# Add momentum and trend strength indicators
def add_technical_indicators(df):
    """Identical to historic version's indicators"""
    df = df.copy()
    
    # Price features
    df['prev_close'] = df['close'].shift(1)
    df['log_return'] = np.log(df['prev_close'] / df['close'].shift(2))
    df['price_change'] = df['close'].diff()
    
    # Moving averages (same windows)
    for window in [5, 15, 30]:
        df[f'close_ma_{window}'] = df['close'].rolling(window).mean()
        df[f'volatility_{window}'] = df['close'].rolling(window).std()
    
    # Volume features
    df['volume_ma_15'] = df['volume'].rolling(15).mean()
    df['volume_change'] = df['volume'].diff()
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Oscillators (same parameters)
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
    
    # ADX (from historic version)
    df['adx'] = compute_adx(df['high'], df['low'], df['close'], 14)
    
    # Buy ratio (ensure no division by zero)
    df['buy_ratio'] = df['taker_buy_base'] / df['volume']
    df['buy_ratio'] = df['buy_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df.dropna()

def add_live_technical_indicators(df):
    """Identical to historic version's indicators with safety checks"""
    if df.empty:
        return df
        
    df = df.copy()
    
    try:
        # Price features
        df['prev_close'] = df['close'].shift(1)
        df['log_return'] = np.log(df['prev_close'] / df['close'].shift(2))
        df['price_change'] = df['close'].diff()
        
        # Moving averages (same windows)
        for window in [5, 15, 30]:
            df[f'close_ma_{window}'] = df['close'].rolling(window).mean()
            df[f'volatility_{window}'] = df['close'].rolling(window).std()
        
        # Volume features
        df['volume_ma_15'] = df['volume'].rolling(15).mean()
        df['volume_change'] = df['volume'].diff()
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Oscillators (same parameters)
        df['rsi'] = compute_rsi(df['close'], 14)
        df['macd'], df['macd_signal'] = compute_macd(df['close'])
        df['stoch_k'] = 100 * (df['close'] - df['low'].rolling(14).min()) / \
                       (df['high'].rolling(14).max() - df['low'].rolling(14).min())
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Additional features from historic
        df['buy_ratio'] = df['taker_buy_base'] / df['volume']
        df['bollinger_upper'] = df['close_ma_15'] + (2 * df['volatility_15'])
        df['bollinger_lower'] = df['close_ma_15'] - (2 * df['volatility_15'])
        df['williams_r'] = -100 * (df['high'].rolling(14).max() - df['close']) / \
                          (df['high'].rolling(14).max() - df['low'].rolling(14).min())
        
        return df.dropna()
    
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return df  # Return whatever we have

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

# ======================
# 3. Sequence Creation
# ======================
def create_live_sequences(features, targets, seq_length, pred_length):
    """Match historic sequence generation exactly with bounds checking"""
    X, y_reg, y_class = [], [], []
    
    if len(features) < seq_length + pred_length:
        print(f"Warning: Not enough data. Need {seq_length + pred_length}, have {len(features)}")
        return np.array([]), np.array([]), np.array([])
    
    for i in range(seq_length, len(features) - pred_length):
        X.append(features[i-seq_length:i])
        y_reg.append(targets[i + pred_length - 1])  # Same regression target
        # Same classification logic
        y_class.append(1 if targets[i + pred_length - 1] > targets[i + pred_length - 2] else 0)
    
    return np.array(X), np.array(y_reg), np.array(y_class)

def create_live_sequences(features, targets, seq_length, pred_length):
    """Match historic sequence generation exactly"""
    X, y_reg, y_class = [], [], []
    for i in range(seq_length, len(features) - pred_length):
        X.append(features[i-seq_length:i])
        y_reg.append(targets[i + pred_length - 1])  # Same regression target
        # Same classification logic
        y_class.append(1 if targets[i + pred_length - 1] > targets[i + pred_length - 2] else 0)
    return np.array(X), np.array(y_reg), np.array(y_class)

def create_sequences_with_direction(features, targets, seq_length, pred_length):
    """
    Create sequences for regression and classification tasks.
    
    Args:
        features: Scaled feature data (numpy array).
        targets: Scaled target data (numpy array).
        seq_length: Number of time steps in each input sequence.
        pred_length: Number of time steps ahead to predict.
    
    Returns:
        X: Input sequences (numpy array).
        y_reg: Regression targets (numpy array).
        y_class: Classification targets (numpy array).
    """
    X, y_reg, y_class = [], [], []
    
    for i in range(seq_length, len(features) - pred_length):
        X.append(features[i-seq_length:i])
        y_reg.append(targets[i + pred_length - 1])  # Regression target: future price
        # Classification target: 1 if price increases, 0 otherwise
        y_class.append(1 if targets[i + pred_length - 1] > targets[i + pred_length - 2] else 0)
    
    return np.array(X), np.array(y_reg), np.array(y_class)

# ======================
# 4. Signal Generation
# ======================
def generate_live_signal(prediction, classification, current_price, current_time, strategy):
    """Match historic trading rules with enhanced risk management"""
    # Same trend confirmation logic
    if classification > 0.6 and strategy.position <= 0:  # Strong buy
        if strategy.position == -1:  # Close short first
            close_position(current_price, current_time, strategy, "SWITCH")
        enter_position(1, current_price, current_time, strategy)
    
    elif classification < 0.4 and strategy.position >= 0:  # Strong sell
        if strategy.position == 1:  # Close long first
            close_position(current_price, current_time, strategy, "SWITCH")
        enter_position(-1, current_price, current_time, strategy)
    
    # Same TP/SL logic but with position tracking
    elif strategy.position != 0:
        pl_pct = (current_price - strategy.entry_price) / strategy.entry_price * strategy.position
        if pl_pct >= 0.015:  # 1.5% TP
            close_position(current_price, current_time, strategy, "TP")
        elif pl_pct <= -0.01:  # 1% SL
            close_position(current_price, current_time, strategy, "SL")

def enter_position(direction, price, timestamp, strategy):
    """Enhanced position entry matching historic risk parameters"""
    # Calculate position size (10% of current balance)
    position_value = strategy.current_balance * 0.10
    position_size = position_value / price
    
    strategy.position = direction
    strategy.entry_price = price
    strategy.entry_time = timestamp
    strategy.position_size = position_size
    
    print(f"{timestamp} - ENTER {'LONG' if direction == 1 else 'SHORT'} at ${price:.2f}")
    print(f"Position size: {position_size:.6f} BTC, Value: ${position_value:.2f}")

def close_position(price, timestamp, strategy, reason):
    """Enhanced position closing with tracking"""
    if strategy.position == 0:
        return
        
    # Calculate profit/loss
    pl = strategy.position_size * (price - strategy.entry_price) * strategy.position
    pl_pct = (price - strategy.entry_price) / strategy.entry_price * strategy.position * 100
    
    # Update balance
    strategy.current_balance += pl
    
    # Record trade
    trade = {
        'entry_time': strategy.entry_time,
        'entry_price': strategy.entry_price,
        'exit_time': timestamp,
        'exit_price': price,
        'position': 'long' if strategy.position == 1 else 'short',
        'size': strategy.position_size,
        'profit_loss': pl,
        'return_pct': pl_pct,
        'reason': reason
    }
    strategy.trades.append(trade)
    
    print(f"{timestamp} - CLOSE {'LONG' if strategy.position == 1 else 'SHORT'} at ${price:.2f}")
    print(f"Profit/Loss: ${pl:.2f} ({pl_pct:.2f}%)")
    print(f"New Balance: ${strategy.current_balance:.2f}")
    
    # Reset position
    strategy.position = 0
    strategy.entry_price = None
    strategy.entry_time = None
    strategy.position_size = None

# ======================
# 5. Metrics Calculation
# ======================
def calculate_live_metrics(strategy):
    """Match historic performance metrics exactly"""
    metrics = {
        'final_balance': strategy.current_balance,
        'total_return_pct': (strategy.current_balance - strategy.initial_balance) / strategy.initial_balance * 100,
        'total_trades': len(strategy.trades),
        'winning_trades': sum(1 for t in strategy.trades if t['profit_loss'] > 0),
        'losing_trades': sum(1 for t in strategy.trades if t['profit_loss'] <= 0),
    }
    
    # Same additional metrics as historic
    metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades'] * 100 
                          if metrics['total_trades'] > 0 else 0)
    
    winning_amount = sum(t['profit_loss'] for t in strategy.trades if t['profit_loss'] > 0)
    losing_amount = abs(sum(t['profit_loss'] for t in strategy.trades if t['profit_loss'] <= 0))
    
    metrics['profit_factor'] = winning_amount / losing_amount if losing_amount > 0 else float('inf')
    
    # Calculate max drawdown
    equity_curve = [strategy.initial_balance]
    for trade in strategy.trades:
        equity_curve.append(equity_curve[-1] + trade['profit_loss'])
    
    peak = equity_curve[0]
    max_drawdown = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    metrics['max_drawdown_pct'] = max_drawdown * 100
    
    return metrics

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
    btc_data = fetch_binance_data(symbol=symbol, interval=interval)
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
    X, y_reg, y_class = create_sequences_with_direction(scaled_features, scaled_target, seq_length, pred_length)
    
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
    System that uses LSTM predictions to generate outputs like direction and probability.
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
    
    def _predict(self, model, sequence):
        """Internal method to make predictions with a single model."""
        return model.predict(sequence)

    def predict(self, sequence):
        print(f"Predicting with input sequence shape: {sequence.shape}")
        predictions = [model.predict(sequence, verbose=0) for model in self.models]
        print(f"Raw predictions: {predictions}")
        avg_prediction = np.mean([pred[0][0] for pred in predictions])  # Average regression output
        avg_classification = np.mean([pred[1][0] for pred in predictions])  # Average classification output
        print(f"Averaged prediction: {avg_prediction}, classification: {avg_classification}")
        return avg_prediction, avg_classification

def run_parallel_systems(symbol, interval, run_duration_hours, lookback_period):
    """
    Run the prediction system for the specified symbol and interval.

    Args:
        symbol (str): Trading symbol (e.g., 'BTCUSDT').
        interval (str): Time interval for predictions (e.g., '5m').
        run_duration_hours (int): Duration to run the system in hours.
        lookback_period (int): Lookback period for analysis.
    """
    print(f"Running prediction system for {symbol} with {interval} interval for {run_duration_hours} hours.")

    # Fetch initial data
    data = fetch_binance_data(symbol=symbol, interval=interval, limit=2000)
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

    # Simulate live predictions
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=run_duration_hours)

    while datetime.now() < end_time:
        print(f"\n[{datetime.now()}] Fetching live data...")
        live_data = fetch_live_data(symbol=symbol, interval=interval, limit=2000)
        if live_data.empty:
            print(f"[{datetime.now()}] No live data fetched. Retrying in 60 seconds...")
            time.sleep(60)  # Wait for 1 minute before retrying
            continue

        print(f"[{datetime.now()}] Adding technical indicators to live data...")
        live_data = add_live_technical_indicators(live_data)

        # Ensure input to scaler has feature names
        last_sequence = live_data[features].values[-seq_length:]
        scaled_sequence = scaler_features.transform(last_sequence)
        scaled_sequence = scaled_sequence.reshape(1, seq_length, -1)

        print(f"[{datetime.now()}] Generating predictions...")
        avg_prediction, avg_classification = lp_system.predict(scaled_sequence)

        # Output predictions
        current_price = live_data['close'].iloc[-1]
        print(f"Current price: {current_price}")
        print(f"Predicted price change: {avg_prediction}")
        print(f"Direction probability: {avg_classification}")

        # Wait for the next interval
        time.sleep(300)  # Wait for 5 minutes


if __name__ == "__main__":
    # Run the prediction system for BTC/USDT with 5-minute intervals for 24 hours
    run_parallel_systems(symbol='BTCUSDT', interval='5m', run_duration_hours=24, lookback_period=5)