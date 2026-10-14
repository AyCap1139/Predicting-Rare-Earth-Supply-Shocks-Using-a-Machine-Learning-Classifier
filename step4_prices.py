import yfinance as yf
import pandas as pd

TICKERS = ["REMX", "BEL.NS", "DIXON.NS"]

print("Downloading price data...")

all_data = []

for ticker in TICKERS:
    print(f"  Fetching {ticker}...")
    df = yf.download(ticker, start="2020-01-01", end="2025-12-31", auto_adjust=True)
    df = df[["Close"]].copy()
    df.columns = [ticker]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    all_data.append(df)

# merge all tickers on date
prices = pd.concat(all_data, axis=1)
prices.index = prices.index.strftime("%Y%m%d")
prices = prices.reset_index()

print(f"\nDone. Got {len(prices)} rows.")
print(prices.head(10))

# compute log returns
import numpy as np
for ticker in TICKERS:
    prices[f"{ticker}_logret"] = np.log(prices[ticker] / prices[ticker].shift(1))

prices.to_csv("price_data_daily.csv", index=False)
print("\nSaved to price_data_daily.csv")