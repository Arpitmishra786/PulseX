# ⚡ PulseX – Real-Time Crypto Tracker with Basic Price Prediction

**PulseX** is a beginner-friendly crypto dashboard built using **Streamlit**. It fetches live price data from **Binance**, shows technical analysis using **MACD & RSI**, and makes basic **LSTM-based future price predictions** — all in one auto-refreshing dashboard.

---

## 🚀 Features

- 💰 **Live Price Feed** – Real-time prices for major cryptocurrencies from Binance
- 📊 **Candlestick Chart** – Visualizes recent market trends using OHLC data
- 📈 **MACD & RSI Indicators** – Displays Buy/Sell signals using MACD crossovers
- 🔮 **LSTM Prediction** – Forecasts the next close price using a simple deep learning model
- ⏱️ **Multi-Timeframe Support** – Choose from 1m, 5m, 15m, 1h, 4h, and 1d intervals
- 🔁 **Auto-Refreshing UI** – Updates every 15 seconds to keep your data fresh

---

## 🛠️ Tech Stack

- **Frontend/UI:** [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/python/)
- **APIs:** [Binance REST API](https://binance-docs.github.io/apidocs/spot/en/)
- **Machine Learning:** [TensorFlow (LSTM)](https://www.tensorflow.org/), [scikit-learn](https://scikit-learn.org/)
- **Data Handling:** `pandas`, `numpy`, `requests`

---

## ⚠️ Disclaimer

> This app is a **learning project** and not intended for real-world financial use or trading decisions.  
> The prediction model is trained on-the-fly on limited data and does **not guarantee accuracy**.

---

## 🧪 Ideal For

- 🧑‍🎓 Students learning real-time APIs + ML
- 🛠️ Developers exploring Streamlit dashboards
- 🧠 Crypto hobbyists who want simple, live visual insights

---

## 🚀 Live App

👉 [pulsex.streamlit.app](https://pulsex.streamlit.app) *(Replace with actual link)*

---

## 📷 Demo Preview

![PulseX Demo Screenshot](screenshot.png) *(optional)*

---

## 📦 Installation (Local)

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/pulsex.git
   cd pulsex
