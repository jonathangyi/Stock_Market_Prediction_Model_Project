import streamlit as st
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from alpha_vantage.timeseries import TimeSeries
import warnings
warnings.filterwarnings('ignore')

# Replace with your actual Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "JO6S4PKBXUZMZWE1"

class StockPredictor:
    def __init__(self, learning_rate=0.01, epochs=1000, regularization=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = []
        self.training_history = {'loss': [], 'r2': []}
        
    def create_features(self, df):
        """Create technical indicators and features for stock prediction"""
        df = df.copy()
        # Handle timezone-aware datetime conversion
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
            except:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date').reset_index(drop=True)
        else:
            # If Date is index, reset it
            df = df.reset_index()
            try:
                df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
            except:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_{window}_Ratio'] = df['Close'] / df[f'MA_{window}']
        
        # Volatility features
        df['Volatility_5'] = df['Close'].rolling(window=5).std()
        df['Volatility_20'] = df['Close'].rolling(window=20).std()
        
        # Volume features
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']
        
        # RSI-like momentum indicator
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Lag features (previous prices)
        for lag in range(1, 11):
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            
        # Price position within recent range
        df['Price_Position'] = (df['Close'] - df['Low'].rolling(20).min()) / \
                              (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        
        return df
    
    def predict(self, X):
        """Make predictions using the trained model"""
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = np.dot(X_scaled, self.weights) + self.bias
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred
    
    def predict_next_10_days(self, recent_data):
        """Predict next 10 days using the most recent data"""
        # Prepare the most recent data point
        df_recent = self.create_features(recent_data)
        
        # Ensure we only select the same features used in training
        available_features = [col for col in self.feature_names if col in df_recent.columns]
        missing_features = set(self.feature_names) - set(available_features)
        
        if missing_features:
            st.warning(f"Missing features: {missing_features}")
        
        X_recent = df_recent[available_features].iloc[-1:]
        
        # Add missing features as zeros if any
        for feature in missing_features:
            X_recent[feature] = 0
        
        # Reorder columns to match training order
        X_recent = X_recent[self.feature_names]
        
        # Handle any remaining missing values
        X_recent = X_recent.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        predictions = self.predict(X_recent)
        return predictions[0]  # Return the 10-day predictions
    
    def load_model(self, filename):
        """Load a trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.bias = model_data['bias']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.feature_names = model_data['feature_names']
        self.learning_rate = model_data['learning_rate']
        self.epochs = model_data['epochs']
        self.regularization = model_data['regularization']
        self.training_history = model_data['training_history']

@st.cache_data
def load_prediction_model():
    """Load the trained stock prediction model"""
    try:
        model = StockPredictor()
        model.load_model('stock_prediction_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'stock_prediction_model.pkl' not found. Please train the model first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def fetch_data(symbol):
    """Fetch stock data from Yahoo Finance or Alpha Vantage"""
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

def main():
    st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
    
    st.title("📈 Multi-Stock Price Prediction System")
    st.markdown("This app uses a custom linear regression model to predict stock prices for the next 10 days.")
    
    # Load model
    model = load_prediction_model()
    if model is None:
        st.stop()
    
    # Sidebar for stock selection
    st.sidebar.header("Stock Selection")
    stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMD", "NVDA", "INTC"]
    selected_symbol = st.sidebar.selectbox("Select a stock to predict:", stock_symbols)
    
    # Display model information
    st.sidebar.header("Model Information")
    st.sidebar.info(f"""
    **Model Details:**
    - Features: {len(model.feature_names)}
    - Learning Rate: {model.learning_rate}
    - Epochs: {model.epochs}
    - Regularization: {model.regularization}
    """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 {selected_symbol} Stock Analysis")
        
        # Fetch data
        with st.spinner("Fetching stock data..."):
            df, source = fetch_data(selected_symbol)
        
        if df is None or df.empty:
            st.error("Failed to load stock data from both sources.")
            st.stop()
        
        st.success(f"✅ Data loaded from **{source}**")
        
        # Display recent data
        st.subheader("Recent Stock Data")
        st.dataframe(df.tail(10))
        
        # Show price chart
        st.subheader(f"{selected_symbol} - Last 60 Days Closing Prices")
        closing_prices = df['Close'].values[-60:]
        chart_data = pd.DataFrame(closing_prices, columns=['Close Price'])
        st.line_chart(chart_data)
        
        # Current price info - FIXED VERSION
        current_price = df['Close'].iloc[-1]  # Get the scalar value directly
        previous_price = df['Close'].iloc[-2]  # Get previous day's price
        price_change = current_price - previous_price
        price_change_pct = (price_change / previous_price) * 100

        st.metric(
            label=f"Current {selected_symbol} Price",
            value=f"${current_price:.2f}",  # Now current_price is a scalar
            delta=f"{price_change_pct:+.2f}%"  # Now price_change_pct is a scalar
        )
    
    with col2:
        st.subheader("🔮 Prediction Controls")
        
        if st.button("🚀 Predict Next 10 Days", type="primary"):
            try:
                with st.spinner("Making predictions..."):
                    # Make predictions
                    predictions = model.predict_next_10_days(df)
                    
                    # Display predictions
                    st.subheader("📈 10-Day Price Predictions")
                    
                    # Create prediction DataFrame
                    last_date = df.index[-1]
                    prediction_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 11)]
                    
                    pred_df = pd.DataFrame({
                        'Date': prediction_dates,
                        'Predicted_Price': predictions,
                        'Change_from_Current': predictions - current_price,
                        'Change_Percent': ((predictions - current_price) / current_price) * 100
                    })
                    
                    # Display prediction table
                    st.dataframe(pred_df.style.format({
                        'Predicted_Price': '${:.2f}',
                        'Change_from_Current': '${:.2f}',
                        'Change_Percent': '{:+.2f}%'
                    }))
                    
                    # Create prediction chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Historical prices (last 30 days)
                    hist_dates = df.index[-30:]
                    hist_prices = df['Close'].values[-30:]
                    
                    ax.plot(hist_dates, hist_prices, label="Historical Prices", 
                           marker='o', linewidth=2, color='blue')
                    ax.plot(prediction_dates, predictions, label="Predicted Prices", 
                           marker='s', linewidth=2, color='red', linestyle='--')
                    
                    # Add current price point
                    ax.plot(df.index[-1], current_price, marker='o', 
                           markersize=10, color='green', label='Current Price')
                    
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price (USD)")
                    ax.set_title(f"{selected_symbol} - Historical vs Predicted Prices")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Rotate x-axis labels for better readability
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Summary statistics
                    avg_predicted = np.mean(predictions)
                    max_predicted = np.max(predictions)
                    min_predicted = np.min(predictions)
                    
                    col1_summary, col2_summary, col3_summary = st.columns(3)
                    
                    with col1_summary:
                        st.metric("Average Predicted Price", f"${avg_predicted:.2f}")
                    
                    with col2_summary:
                        st.metric("Highest Predicted", f"${max_predicted:.2f}")
                    
                    with col3_summary:
                        st.metric("Lowest Predicted", f"${min_predicted:.2f}")
                    
                    # Investment recommendation
                    if avg_predicted > current_price:
                        st.success(f"📈 **Bullish Outlook**: Average predicted price is ${avg_predicted - current_price:.2f} higher than current price")
                    else:
                        st.warning(f"📉 **Bearish Outlook**: Average predicted price is ${current_price - avg_predicted:.2f} lower than current price")
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        
        # Additional info
        st.subheader("ℹ️ About the Model")
        st.info("""
        This model uses:
        - Technical indicators (RSI, Moving Averages)
        - Price patterns and volatility
        - Volume analysis
        - Historical price lags
        
        **Disclaimer**: This is for educational purposes only. 
        Always consult financial advisors before making investment decisions.
        """)

if __name__ == "__main__":
    main()