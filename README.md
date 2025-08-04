# CryptoPulseAnalyzer

**CryptoPulseAnalyzer** is a Python tool for detecting and predicting "pulse spikes" — short-term, high-volatility price movements in cryptocurrencies. It combines historical market data from Binance with social signals (e.g., activity on X) and uses an LSTM neural network to forecast potential price surges. The tool generates insightful visualizations, including correlation heatmaps, to help traders and analysts make informed decisions.

## Features
- Fetches real-time OHLCV data from Binance.
- Incorporates social signals (e.g., X activity) for enhanced predictions.
- Uses LSTM to predict high-volatility "pulse spikes."
- Visualizes correlations between price, volume, volatility, and predictions.
- Easy-to-use CLI interface.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CryptoPulseAnalyzer.git
   cd CryptoPulseAnalyzer
