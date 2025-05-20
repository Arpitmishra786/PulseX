import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Streamlit page config
st.set_page_config(page_title="Crypto Trading Platform", layout="wide")
st.title("🚀 Real-Time Crypto Trading with LSTM & Candlestick Signals")

# Sidebar for user input
symbol = st.sidebar.selectbox("Select Cryptocurrency", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"])
timeframe = st.sidebar.selectbox("Select Timeframe", ["1m", "5m", "15m", "30m", "1h"])
predict_button = st.sidebar.button("📈 Predict Now")

# Binance API for fetching historical data
def fetch_data(symbol, interval, limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close"]].astype(float)
    return df

# Detect candlestick patterns (simplified)
def detect_candlestick_patterns(df):
    signals = []
    for i in range(1, len(df)):
        if df['close'][i] > df['open'][i] and df['close'][i-1] < df['open'][i-1]:
            signals.append((df.index[i], 'Buy'))
        elif df['close'][i] < df['open'][i] and df['close'][i-1] > df['open'][i-1]:
            signals.append((df.index[i], 'Sell'))
    return signals

# LSTM training and prediction
def train_lstm_model(data):
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(data['close'].values.reshape(-1, 1))
    seq_len = 60
    X, y = [], []
    for i in range(seq_len, len(scaled_close)):
        X.append(scaled_close[i-seq_len:i])
        y.append(scaled_close[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(X, y, epochs=10, batch_size=8, verbose=0)

    last_seq = scaled_close[-seq_len:]
    last_seq = last_seq.reshape((1, seq_len, 1))
    prediction = model.predict(last_seq)
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    return predicted_price, history.history['loss']

# Real-time chart update
placeholder = st.empty()

if predict_button:
    df = fetch_data(symbol, timeframe)
    prediction, loss_history = train_lstm_model(df)

    # Plot candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Candles')
    ])

    # Add Buy/Sell signals
    signals = detect_candlestick_patterns(df)
    for time, signal in signals:
        price = df.loc[time]['close']
        color = 'green' if signal == 'Buy' else 'red'
        fig.add_trace(go.Scatter(x=[time], y=[price], mode='markers+text', name=signal,
                                 text=[signal], textposition='top center',
                                 marker=dict(color=color, size=10)))

    # Add prediction line
    fig.add_trace(go.Scatter(x=[df.index[-1], df.index[-1] + pd.Timedelta(minutes=1)],
                             y=[df['close'].iloc[-1], prediction],
                             mode='lines+markers', name='Prediction',
                             line=dict(color='blue', dash='dash')))

    fig.update_layout(title=f"{symbol} Price Chart ({timeframe}) with Predictions and Signals",
                      xaxis_title="Time", yaxis_title="Price (USDT)",
                      xaxis_rangeslider_visible=False, height=600)
    placeholder.plotly_chart(fig, use_container_width=True)

    # Show training loss chart
    st.subheader("📉 LSTM Training Loss")
    st.line_chart(loss_history)

# Auto-refresh chart every second
else:
    while True:
        df = fetch_data(symbol, timeframe)
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='Candles')
        ])
        fig.update_layout(title=f"{symbol} Live Chart ({timeframe})",
                          xaxis_title="Time", yaxis_title="Price (USDT)",
                          xaxis_rangeslider_visible=False, height=600)
        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(1)
