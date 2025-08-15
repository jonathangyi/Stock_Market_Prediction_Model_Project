import streamlit as st
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib

st.title("📈 Multi-Stock Price Prediction")
st.markdown("This app uses a multi-output deep learning model to predict prices for multiple stocks at once.")

# Load model and scaler
model = keras.models.load_model("multi_output_stock_model.keras")
scaler = joblib.load("multi_stock_scaler.pkl")

# Define stock symbols
stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMD", "NVDA", "INTC"]
selected_symbol = st.selectbox("Select a stock to predict:", stock_symbols)

# Fetch recent historical data for display
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=90)
df = yf.download(selected_symbol, start=start_date, end=end_date)

if df.empty:
    st.error("Failed to load stock data.")
else:
    # Show the last 60 closing prices
    st.subheader(f"{selected_symbol} - Last 60 Closing Prices")
    closing_prices = df['Close'].values[-60:]
    dates = df.index[-60:]
    st.line_chart(closing_prices)

# --- Feature Construction ---

def fetch_recent_data():
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=180)  # fetch more for safety
    ohlcv_dict = {}

    for symbol in stock_symbols:
        df = yf.download(symbol, start=start_date, end=end_date)
        df = df[['Open', 'High', 'Low', 'Close']].dropna()
        df = df.tail(120)
        if len(df) < 120:
            raise ValueError(f"Not enough data for {symbol}")
        ohlcv_dict[symbol] = df

    return ohlcv_dict

def combine_features(ohlcv_dict):
    all_features = []
    for symbol in stock_symbols:
        df = ohlcv_dict[symbol][['Open', 'High', 'Low', 'Close']]
        all_features.append(df.values)  # shape (120, 4)

    base = np.concatenate(all_features, axis=1)  # shape: (120, 24)

    # 3 synthetic features
    avg_close = np.mean([ohlcv_dict[s]['Close'].values for s in stock_symbols], axis=0).reshape(-1, 1)
    high_low = np.mean([ohlcv_dict[s]['High'].values - ohlcv_dict[s]['Low'].values for s in stock_symbols], axis=0).reshape(-1, 1)
    open_close = np.mean([ohlcv_dict[s]['Open'].values - ohlcv_dict[s]['Close'].values for s in stock_symbols], axis=0).reshape(-1, 1)

    extra = np.hstack([avg_close, high_low, open_close])  # shape: (120, 3)
    combined = np.hstack([base, extra])  # final shape: (120, 27)

    return combined

# --- Predict Button ---

if st.button("Predict"):
    try:
        ohlcv_dict = fetch_recent_data()
        features = combine_features(ohlcv_dict)
        features_scaled = scaler.transform(features)
        X_input = features_scaled.reshape(1, 120, 27)

        predictions = model.predict(X_input)
        pred_dict = dict(zip(stock_symbols, predictions.flatten()))
        predicted_value = float(pred_dict[selected_symbol])

        # Plot result
        fig, ax = plt.subplots()
        ax.plot(dates, closing_prices, label="Historical Prices", marker='o')
        ax.scatter(dates[-1] + datetime.timedelta(days=1), predicted_value,
                   label="Predicted Next Price", color='red', marker='x')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.set_title(f"{selected_symbol} Price Forecast")
        ax.legend()
        st.pyplot(fig)

        st.markdown(f"### 🔮 Predicted next price for **{selected_symbol}**: `{predicted_value:.2f}` USD")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
