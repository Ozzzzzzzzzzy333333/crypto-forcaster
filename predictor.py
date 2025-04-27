# imports
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands
import warnings
import time
import logging

logger = logging.getLogger()

warnings.filterwarnings("ignore")

class LiveCryptoPredictor:
    def __init__(self, symbol='BTC/USDT', timeframe='5m'):
        self.exchange = ccxt.binance()
        self.symbol = symbol  
        self.timeframe = timeframe
        self.train_days = 7
        self.gap_minutes = 10
        self.prediction_horizon = self._get_prediction_horizon(timeframe)
        self.confidence_threshold = 0.5
        self.model = RandomForestClassifier()
        self.live_predictions = {}
        self.current_price = None

    def _get_prediction_horizon(self, timeframe):

        mapping = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
        return mapping.get(timeframe, 5) 

    def fetch_data(self, start_dt, end_dt):
        since = int(start_dt.timestamp() * 1000)
        data = []
        try:
            while since < int(end_dt.timestamp() * 1000):
                candles = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, since=since, limit=1000)
                if not candles:
                    break
                data.extend(candles)
                since = candles[-1][0] + 1
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            self.current_price = df['close'].iloc[-1]
            return df
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            return pd.DataFrame()

    def prepare_features(self, df):
        df = df.copy()
        df['returns'] = df['close'].pct_change().shift(1)
        df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator().shift(1)
        df['rsi_14'] = RSIIndicator(df['close'], window=14).rsi().shift(1)
        df['bb_width'] = BollingerBands(df['close']).bollinger_wband().shift(1)
        df['volatility'] = df['close'].rolling(20).std().shift(1)
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(100).mean()
        df['volatility_ratio'] = df['volatility_ratio'].fillna(0)

        for lag in range(1, 6):
            df[f'returns_lag{lag}'] = df['returns'].shift(lag)

        df['target'] = (df['close'].shift(-self.prediction_horizon) > df['close']).astype(int)
        df = df.dropna()
        return df

    def optimize_model(self, X_train, y_train):
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [10, 20],
            'class_weight': ['balanced'] 
        }
        grid_search = GridSearchCV(
            RandomForestClassifier(),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        logger.info(f"Best model parameters: {grid_search.best_params_}")

    def train_model(self):
        end_dt = datetime.utcnow() - timedelta(minutes=self.gap_minutes)
        start_dt = end_dt - timedelta(days=self.train_days)

        logger.info(f"using data from {start_dt} to {end_dt}")
        df = self.fetch_data(start_dt, end_dt)
        if df.empty:
            return False

        df = self.prepare_features(df)
        if df.empty:
            return False

        features = df.columns.difference(['target'])
        X_train, y_train = df[features], df['target']

        self.optimize_model(X_train, y_train)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_train)
        acc = accuracy_score(y_train, y_pred)
        logger.info(f"accuracy: {acc:.2%}")
        return True

    def make_prediction(self):
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(hours=24)

        logger.info("Making live prediction...")
        df = self.fetch_data(start_dt, end_dt)
        if df.empty:
            return None

        df = self.prepare_features(df)
        if df.empty:
            return None

        X_latest = df[df.columns.difference(['target'])].iloc[-1:].copy()

        try:
            prediction = self.model.predict(X_latest)[0]
            prob = self.model.predict_proba(X_latest)[0]
            confidence = prob[prediction]
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            self.live_predictions[now] = {
                'prediction': prediction,
                'confidence': confidence,
                'price': self.current_price,
                'expected_check_time': datetime.utcnow() + timedelta(minutes=self.prediction_horizon),
                'verified': False,
                'actual': None
            }

            direction = "UP" if prediction == 1 else "DOWN"
            logger.info(f"[{now}] Prediction: {direction} | Confidence: {confidence:.2f}")
            return prediction, confidence
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

    def verify_predictions(self):
        now = datetime.utcnow()
        for timestamp, pred_info in list(self.live_predictions.items()):
            if pred_info['verified']:
                continue
            if now >= pred_info['expected_check_time']:
                logger.info(f"Verifying prediction from {timestamp}...")
                try:
                    df = self.fetch_data(pred_info['expected_check_time'] - timedelta(minutes=5), pred_info['expected_check_time'])
                    if df.empty:
                        logger.warning("No data available for verification.")
                        continue
                    end_price = df['close'].iloc[-1]
                    actual = 1 if end_price > pred_info['price'] else 0
                    pred_info['actual'] = actual
                    pred_info['verified'] = True
                    result = "CORRECT" if actual == pred_info['prediction'] else "WRONG"
                    logger.info(f"Prediction was {result} | Start: {pred_info['price']} End: {end_price}")
                except Exception as e:
                    logger.error(f"Verification error: {e}")

    def run_live(self):
        logger.info(f"Starting Live Predictor for {self.symbol} ({self.timeframe})")

        for attempt in range(3):
            if self.train_model():
                break
            logger.warning("Model training failed")
            time.sleep(60)
        else:
            logger.error("Failed to train 3 times")
            return

        try:
            while True:
                logger.info("cycle")
                self.make_prediction()
                self.verify_predictions()
                now = datetime.utcnow()
                next_tick = now + timedelta(minutes=self.prediction_horizon)
                next_tick = next_tick.replace(minute=(next_tick.minute // 5) * 5, second=0, microsecond=0)
                wait_time = (next_tick - now).total_seconds()
                time.sleep(max(1, wait_time))
        except KeyboardInterrupt:
            logger.info("Live mode interrupted.")

def make_prediction(df, interval='5m', symbol='BTC/USDT'):
    predictor = LiveCryptoPredictor(symbol=symbol, timeframe=interval) 
    predictor.train_model()  
    prediction, confidence = predictor.make_prediction()
    if prediction is not None:
        return prediction, confidence
    else:
        return None, None

