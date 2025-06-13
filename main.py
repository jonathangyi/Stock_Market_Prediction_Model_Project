import streamlit as st
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
from multi_stock_model_combination import MultiStockModelCombiner

# Load the model
model = keras.models.load_model("multi_output_stock_model.keras")

st.title("Multi-Stock Price Prediction")

st.markdown("This app uses a multi-output deep learning model to predict prices for multiple stocks at once.")

# Define stock symbols and selection
stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMD", "NVDA", "INTC"]
selected_symbol = st.selectbox("Select a stock to predict:", stock_symbols)

# Fetch recent historical data
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=90)
df = yf.download(selected_symbol, start=start_date, end=end_date)

if df.empty:
    st.error("Failed to load stock data.")
else:
    # Show the last 60 closing prices always (optional)
    st.subheader(f"{selected_symbol} - Last 60 Closing Prices")
    closing_prices = df['Close'].values[-60:]  # Last 60 days
    dates = df.index[-60:]
    st.line_chart(closing_prices)  # Optional: show chart of recent prices

    # Predict button triggers the prediction and plotting
    if st.button("Predict"):
        # Dummy input for prediction (replace with real scaled/processed input)
        dummy_input = np.random.random((1, 120, 27))  # shape: (1, timesteps, features)

        # Make prediction
        predictions = model.predict(dummy_input)
        pred_dict = dict(zip(stock_symbols, predictions))
        predicted_value = float(pred_dict[selected_symbol])

        # Plot historical + predicted
        fig, ax = plt.subplots()
        ax.plot(dates, closing_prices, label="Historical Prices", marker='o')
        ax.plot(dates[-1] + datetime.timedelta(days=1), predicted_value,
                label="Predicted Next Price", marker='x', color='red')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.set_title(f"{selected_symbol} Price Forecast")
        ax.legend()

        st.pyplot(fig)

        st.markdown(f"### 📈 Predicted next price for **{selected_symbol}**: `{predicted_value:.2f}`")
