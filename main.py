import streamlit as st
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
from alpha_vantage.timeseries import TimeSeries
import joblib
import pandas as pd

# Replace with your actual Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "JO6S4PKBXUZMZWE1"

# Load the model
model = keras.models.load_model("AAPL_best_enhanced_model.keras")

st.title("Multi-Stock Price Prediction")

st.markdown("This app uses a multi-output deep learning model to predict prices for multiple stocks at once.")

# Define stock symbols and selection
stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMD", "NVDA", "INTC"]
selected_symbol = st.selectbox("Select a stock to predict:", stock_symbols)

def fetch_data(symbol):
    try:
        end_date = datetime.datetime.today()
        start_date = end_date - datetime.timedelta(days=1825)  # 5 years of data
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if not df.empty:
            return df, "Yahoo Finance"
    except Exception as e:
        st.warning(f"yfinance failed: {e}")

    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
        data = data.rename(columns={
            "1. open": "Open", "2. high": "High", "3. low": "Low",
            "4. close": "Close", "5. volume": "Volume"
        })
        data.index = data.index.tz_localize(None)
        return data.sort_index(), "Alpha Vantage"
    except Exception as e:
        st.error(f"Alpha Vantage also failed: {e}")
        return None, None

def prepare_input(df, scaler_path="AAPL_scaler.pkl"):
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler from {scaler_path}: {e}")

    # Feature Engineering
    df_feat = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_feat['Return'] = df_feat['Close'].pct_change()
    df_feat['MA_5'] = df_feat['Close'].rolling(window=5).mean()
    df_feat['MA_10'] = df_feat['Close'].rolling(window=10).mean()
    df_feat['MA_20'] = df_feat['Close'].rolling(window=20).mean()

    # Fill missing values early to preserve length
    df_feat = df_feat.fillna(method='ffill').fillna(method='bfill')

    # Final shape check
    if len(df_feat) < 120:
        raise ValueError(f"After filling, only {len(df_feat)} rows available. Need at least 120.")

    # Scale and reshape
    input_scaled = scaler.transform(df_feat)
    return np.expand_dims(input_scaled[-120:], axis=0), df_feat.index[-1]


# Fetch data
df, source = fetch_data(selected_symbol)

if df is None or df.empty:
    st.error("Failed to load stock data from both sources.")
else:
    st.success(f"Data loaded from **{source}**")

    # Show chart
    st.subheader(f"{selected_symbol} - Last 60 Closing Prices")
    closing_prices = df['Close'].values[-60:]
    dates = df.index[-60:]
    st.line_chart(closing_prices)

    if st.button("Predict"):
        try:
            input_data, last_date = prepare_input(df)
            predictions = model.predict(input_data)
            pred_dict = dict(zip(stock_symbols, predictions))
            predicted_value = float(pred_dict[selected_symbol])

            # Plot
            fig, ax = plt.subplots()
            ax.plot(dates, closing_prices, label="Historical Prices", marker='o')
            ax.plot(last_date + datetime.timedelta(days=1), predicted_value,
                    label="Predicted Next Price", marker='x', color='red')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.set_title(f"{selected_symbol} Price Forecast")
            ax.legend()

            st.pyplot(fig)
            st.markdown(f"### 📈 Predicted next price for **{selected_symbol}**: `{predicted_value:.2f}`")
        except Exception as e:
            st.error(f"Prediction failed: {e}")