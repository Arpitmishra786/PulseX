import streamlit as st

# 1. Set page config MUST be first Streamlit command
st.set_page_config(page_title="Crypto Trading Platform", layout="wide")

# 2. Import rest of libraries
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import os

# ----- Auto-refresh every 15 seconds -----
st_autorefresh(interval=15000, key="refresh")

st.title("🚀 Crypto Trading Platform with Demo Trading & Analytics")

# ----- Functions -----

def get_price(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
	if 'price' in data:
        return float(data['price'])
	else:
		st.error(f"Unexpected API response: {data}")
		return None
    except Exception as e:
        st.error(f"Error fetching price data: {e}")
        return None

def get_ohlcv(symbol="BTCUSDT", interval="1m", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        st.error(f"Error fetching OHLCV data: {e}")
        return pd.DataFrame()

def add_macd(df):
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = signal
    return df

def add_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def generate_signals(df):
    df['signal'] = np.where((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 'Buy',
                     np.where((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 'Sell', ''))
    return df

def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# -------- Portfolio Save/Load --------

PORTFOLIO_FILE = "portfolio.json"

def save_portfolio():
    data = {
        "balance": st.session_state.balance,
        "holdings": st.session_state.holdings,
        "pnl_history": st.session_state.pnl_history
    }
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f)

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            data = json.load(f)
            st.session_state.balance = data.get("balance", 10000.0)
            st.session_state.holdings = data.get("holdings", 0.0)
            st.session_state.pnl_history = data.get("pnl_history", [])
    else:
        st.session_state.balance = 10000.0  # USD
        st.session_state.holdings = 0.0     # Crypto units
        st.session_state.pnl_history = []

# -------- Authentication --------

def authenticate(username, password):
    return username == "arpit" and password == "crypto123"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid credentials")
    st.stop()

# -------- Load Portfolio --------
load_portfolio()

# -------- User Controls --------

symbol = st.selectbox("Select Cryptocurrency", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"])
interval = st.selectbox("Select Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])

price = get_price(symbol)
if price is not None:
    st.metric(label=f"💰 {symbol} Live Price", value=f"${price:,.2f}")
else:
    st.warning("Price data not available.")

ohlcv_df = get_ohlcv(symbol, interval=interval, limit=200)
if ohlcv_df.empty:
    st.warning("OHLCV data not available. Please try again later.")
    st.stop()

ohlcv_df = add_macd(ohlcv_df)
ohlcv_df = add_rsi(ohlcv_df)
ohlcv_df = generate_signals(ohlcv_df)

# -------- Charts --------
st.subheader("📊 Live Candlestick Chart with MACD Signals")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=ohlcv_df['timestamp'], open=ohlcv_df['open'], high=ohlcv_df['high'],
    low=ohlcv_df['low'], close=ohlcv_df['close'], name='Candles'))

buy_signals = ohlcv_df[ohlcv_df['signal'] == 'Buy']
sell_signals = ohlcv_df[ohlcv_df['signal'] == 'Sell']

fig.add_trace(go.Scatter(x=buy_signals['timestamp'], y=buy_signals['close'],
                         mode='markers', marker=dict(color='green', size=10), name='Buy Signal'))
fig.add_trace(go.Scatter(x=sell_signals['timestamp'], y=sell_signals['close'],
                         mode='markers', marker=dict(color='red', size=10), name='Sell Signal'))

fig.update_layout(xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)

# -------- LSTM Prediction --------
close_prices = ohlcv_df['close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(close_prices)

seq_len = 10
X, y = create_sequences(scaled_close, seq_len)
if len(X) > 0:
    X = X.reshape((X.shape[0], seq_len, 1))
    model = Sequential()
    model.add(Input(shape=(seq_len, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("Training LSTM model..."):
        model.fit(X, y, epochs=10, batch_size=8, verbose=0)

    last_seq = scaled_close[-seq_len:].reshape((1, seq_len, 1))
    pred_scaled = model.predict(last_seq, verbose=0)
    predicted_price = scaler.inverse_transform(pred_scaled)[0][0]
    st.success(f"📈 Predicted Next Close Price: ${predicted_price:,.2f}")

    st.subheader("🔮 Historical Predictions (last 50 points)")
    y_pred = model.predict(X, verbose=0)
    y_true = scaler.inverse_transform(y.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=y_true[-50:].flatten(), name='Actual'))
    fig2.add_trace(go.Scatter(y=y_pred_inv[-50:].flatten(), name='Predicted'))
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("❗Not enough data to train LSTM model. Try increasing limit or wait for more candles.")

# -------- Demo Trading --------

st.write("---")
st.subheader("💸 Demo Trading Panel")

col1, col2 = st.columns(2)

with col1:
    buy_amount = st.number_input(f"Buy amount in USD (Balance: ${st.session_state.balance:,.2f})", min_value=0.0, max_value=st.session_state.balance, value=0.0, step=10.0)
    if st.button("Buy"):
        if price is None:
            st.error("Cannot buy: price data unavailable.")
        elif buy_amount <= 0:
            st.error("Enter valid buy amount.")
        elif buy_amount > st.session_state.balance:
            st.error("Insufficient balance.")
        else:
            crypto_bought = buy_amount / price
            st.session_state.balance -= buy_amount
            st.session_state.holdings += crypto_bought
            st.success(f"Bought {crypto_bought:.6f} {symbol} for ${buy_amount:,.2f}")
            save_portfolio()

with col2:
    sell_amount = st.number_input(f"Sell amount in {symbol} (Holdings: {st.session_state.holdings:.6f} {symbol})", min_value=0.0, max_value=st.session_state.holdings, value=0.0, step=0.0001, format="%.6f")
    if st.button("Sell"):
        if price is None:
            st.error("Cannot sell: price data unavailable.")
        elif sell_amount <= 0:
            st.error("Enter valid sell amount.")
        elif sell_amount > st.session_state.holdings:
            st.error("Insufficient holdings.")
        else:
            proceeds = sell_amount * price
            st.session_state.balance += proceeds
            st.session_state.holdings -= sell_amount
            st.success(f"Sold {sell_amount:.6f} {symbol} for ${proceeds:,.2f}")
            save_portfolio()

# -------- Portfolio Summary --------
st.write("---")
st.subheader("💰 Portfolio Summary")

st.write(f"Balance (USD): ${st.session_state.balance:,.2f}")
st.write(f"Holdings ({symbol}): {st.session_state.holdings:.6f}")

if price is not None:
    current_value = st.session_state.holdings * price
    total_value = st.session_state.balance + current_value
    st.write(f"Current Crypto Value: ${current_value:,.2f}")
    st.write(f"Total Portfolio Value: ${total_value:,.2f}")

    # Calculate P&L
    initial_capital = 10000.0
    profit_loss = total_value - initial_capital
    color = "green" if profit_loss >= 0 else "red"
    sign = "+" if profit_loss >= 0 else ""
    st.markdown(f"### Profit / Loss: <span style='color:{color};'>{sign}${profit_loss:,.2f}</span>", unsafe_allow_html=True)

# -------- Logout --------
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.experimental_rerun()
