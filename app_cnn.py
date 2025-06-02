
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import MinMaxScaler
from binance.client import Client
import datetime
import os

# --------- Config ---------
st.set_page_config(page_title="Exness Demo with CNN + Portfolio", layout="wide")

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
client = Client(API_KEY, API_SECRET)

# --------- Utils ---------

def get_binance_data(symbol="BTCUSDT", interval="1m", limit=500):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def calculate_support_resistance(df, window=10):
    support = df['low'].rolling(window=window).min().iloc[-1]
    resistance = df['high'].rolling(window=window).max().iloc[-1]
    return support, resistance

def get_order_book(symbol="BTCUSDT", limit=5):
    try:
        ob = client.get_order_book(symbol=symbol, limit=limit)
        bids_df = pd.DataFrame(ob['bids'], columns=['Price', 'Quantity']).astype(float)
        asks_df = pd.DataFrame(ob['asks'], columns=['Price', 'Quantity']).astype(float)
        return bids_df, asks_df
    except Exception as e:
        st.error(f"Failed to fetch order book: {e}")
        return pd.DataFrame(), pd.DataFrame()

def build_cnn_model(seq_len):
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(seq_len, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_trend(df):
    if len(df) < 4:
        return "sideways"
    if df['close'][-3] < df['close'][-2] < df['close'][-1]:
        return "up"
    elif df['close'][-3] > df['close'][-2] > df['close'][-1]:
        return "down"
    else:
        return "sideways"

# --------- Session State Initialization ---------
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'balance' not in st.session_state:
    st.session_state.balance = 1000.0

# --------- UI ---------
st.title("📈 Exness Demo Trading with CNN + Portfolio")

col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.selectbox("Symbol", ['BTCUSDT', 'ETHUSDT'])
with col2:
    interval = st.selectbox("Interval", ['1m', '5m', '15m'])
with col3:
    limit = st.slider("Data Points", 100, 1000, 300, 100)

df = get_binance_data(symbol, interval, limit)
if df.empty:
    st.stop()

st.subheader("Live Price Chart")
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['open'], high=df['high'], low=df['low'], close=df['close']
)])
fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

support, resistance = calculate_support_resistance(df)
st.markdown(f"**Support:** {support:.2f} | **Resistance:** {resistance:.2f}")

trend = predict_trend(df)
if trend == "up":
    st.markdown("⬆️ Uptrend")
elif trend == "down":
    st.markdown("⬇️ Downtrend")
else:
    st.markdown("➡️ Sideways")

bids_df, asks_df = get_order_book(symbol)
colb, cola = st.columns(2)
colb.dataframe(bids_df, use_container_width=True)
cola.dataframe(asks_df, use_container_width=True)

st.subheader("🔮 CNN Price Prediction")
scaled_close = MinMaxScaler().fit_transform(df[['close']].values)
X, y = create_sequences(scaled_close, 10)
if len(X) < 10:
    st.warning("Not enough data")
    st.stop()

X = X.reshape((X.shape[0], 10, 1))
model = build_cnn_model(10)
model.fit(X, y, epochs=5, verbose=0)
pred = model.predict(scaled_close[-10:].reshape(1, 10, 1), verbose=0)
pred_price = MinMaxScaler().fit(df[['close']].values).inverse_transform(pred)[0][0]
st.success(f"Next predicted close price: ${pred_price:.2f}")

st.subheader("⚡ Trade Signal")
last_close = df['close'].iloc[-1]
signal = "Buy" if pred_price > last_close else "Sell"
st.info(f"**Suggested Signal:** {signal}")

st.download_button("📥 Export Signal to CSV",
                   pd.DataFrame([[datetime.datetime.now(), signal, pred_price]], columns=["Time", "Signal", "Price"]).to_csv(index=False),
                   file_name="signal.csv")

# Trade placement form
st.subheader("📋 Place a Trade")
with st.form("trade_form"):
    ttype = st.selectbox("Type", ["Long", "Short"])
    entry = st.number_input("Entry Price", value=float(last_close))
    sl = st.number_input("Stop Loss", value=entry - 10)
    tgt = st.number_input("Target", value=entry + 10)
    qty = st.number_input("Quantity", value=0.001)
    submit = st.form_submit_button("Place Trade")
if submit:
    st.session_state.portfolio.append({
        "type": ttype, "entry": entry, "sl": sl, "tgt": tgt, "qty": qty,
        "status": "open", "timestamp": datetime.datetime.now()
    })
    st.success("Trade added!")

# Portfolio display
st.subheader("📊 Active Trades")
port_df = pd.DataFrame(st.session_state.portfolio)
if not port_df.empty:
    port_df['timestamp'] = pd.to_datetime(port_df['timestamp']).dt.strftime("%Y-%m-%d %H:%M:%S")
    port_df['PnL'] = port_df.apply(lambda row: (df['close'].iloc[-1] - row['entry']) * row['qty']
                                   if row['type'] == 'Long' else (row['entry'] - df['close'].iloc[-1]) * row['qty'], axis=1)
    st.dataframe(port_df)

# Reset section
st.subheader("💰 Balance Settings")
st.write(f"Current balance: ${st.session_state.balance:.2f}")
colb1, colb2 = st.columns(2)
with colb1:
    if st.button("Reset Portfolio"):
        st.session_state.portfolio = []
        st.success("Portfolio reset!")
with colb2:
    if st.button("Reset Balance"):
        st.session_state.balance = 1000.0
        st.success("Balance reset!")
