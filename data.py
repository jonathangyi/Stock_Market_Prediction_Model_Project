import yfinance as yf
import os
import pandas as pd
import time

def get_stock_data(ticker, period="5y", interval="1d", cache_dir="stock_cache", force_reload=False):
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.join(cache_dir, f"{ticker}_{period}_{interval}.csv")

    if not force_reload and os.path.exists(filename):
        print(f"[CACHE] {ticker} loaded from cache")
        return pd.read_csv(filename, index_col=0, parse_dates=True)

    print(f"[FETCH] Downloading {ticker}...")
    for attempt in range(3):
        try:
            data = yf.download(ticker, period=period, interval=interval, threads=False)
            if data.empty:
                raise ValueError("Empty data returned")
            data.to_csv(filename)
            print(f"[SAVED] {ticker} saved to {filename}")
            time.sleep(10)  # longer delay
            return data
        except Exception as e:
            wait_time = 5 * (attempt + 1)
            print(f"[RETRY {attempt+1}] Failed to download {ticker}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    print(f"[FAILED] Giving up on {ticker}")
    return None

tickers = ['AAPL', 'MSFT', 'AMD', 'NVDA', 'INTC', 'GOOGL']

for ticker in tickers:
    df = get_stock_data(ticker)
    if df is not None:
        print(df.tail(1))
