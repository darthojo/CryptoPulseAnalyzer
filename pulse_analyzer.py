import ccxt
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

class CryptoPulseAnalyzer:
    def __init__(self, symbol='BTC/USDT', timeframe='1h', lookback_days=7, api_key=None, api_secret=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.exchange = ccxt.binance({'apiKey': api_key, 'secret': api_secret})
        self.scaler = MinMaxScaler()
        
    def fetch_ohlcv(self):
        """Получение исторических данных с биржи."""
        since = int((datetime.now() - timedelta(days=self.lookback_days)).timestamp() * 1000)
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def fetch_social_signals(self):
        """Получение активности в X (заглушка для API X)."""
        # Здесь должен быть реальный API-запрос к X, но для примера возвращаем случайные данные
        return np.random.rand(len(self.fetch_ohlcv())) * 100

    def prepare_data(self, df):
        """Подготовка данных для модели."""
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=12).std()
        df['social_signal'] = self.fetch_social_signals()
        features = df[['close', 'volume', 'volatility', 'social_signal']].dropna()
        
        scaled_data = self.scaler.fit_transform(features)
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i])
            y.append(1 if scaled_data[i, 2] > np.percentile(scaled_data[:, 2], 90) else 0)  # Импульсный всплеск
        return np.array(X), np.array(y)

    def build_model(self):
        """Создание LSTM-модели."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 4)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X, y):
        """Обучение модели."""
        model = self.build_model()
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
        return model

    def predict_pulse(self, model, X):
        """Прогноз импульсных всплесков."""
        predictions = model.predict(X)
        return (predictions > 0.5).astype(int)

    def visualize_results(self, df, predictions):
        """Визуализация тепловой карты и прогнозов."""
        df = df.iloc[60:].copy()
        df['pulse_prediction'] = predictions
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[['close', 'volume', 'volatility', 'pulse_prediction']].corr(), annot=True, cmap='coolwarm')
        plt.title(f'Correlation Heatmap for {self.symbol}')
        plt.savefig('data/sample_output/pulse_heatmap.png')
        plt.show()

    def run(self):
        """Основной метод анализа."""
        df = self.fetch_ohlcv()
        X, y = self.prepare_data(df)
        model = self.train_model(X, y)
        predictions = self.predict_pulse(model, X)
        self.visualize_results(df, predictions)
        print(f"Pulse spikes predicted: {np.sum(predictions)} out of {len(predictions)} periods.")

if __name__ == "__main__":
    analyzer = CryptoPulseAnalyzer(symbol='BTC/USDT', timeframe='1h', lookback_days=7)
    analyzer.run()
