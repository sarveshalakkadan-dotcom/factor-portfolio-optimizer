"""
Data Pipeline: Pulls historical prices + fundamental data for a stock universe.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os
import time

# ---- Configuration ----
# A diversified universe across sectors (~60 large/mid caps)
UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "ADBE", "CRM", "ORCL", "CSCO", "INTC",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "SCHW", "AXP", "BLK", "SPGI", "C",
    # Healthcare
    "UNH", "JNJ", "PFE", "MRK", "ABBV", "TMO", "ABT", "LLY", "BMY", "AMGN",
    # Consumer
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "PG", "KO",
    # Industrials
    "BA", "CAT", "GE", "HON", "UPS", "LMT", "MMM", "RTX", "DE", "UNP",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Utilities/Other
    "NEE", "DUK", "SO", "AEP",
]

BENCHMARK = "SPY"
START_DATE = "2018-01-01"
END_DATE = None  # None = today

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def fetch_prices(tickers, start=START_DATE, end=END_DATE):
    """Download adjusted close prices for all tickers + benchmark."""
    all_tickers = tickers + [BENCHMARK]
    print(f"Downloading price data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Close"]
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.9))  # drop tickers with too much missing data
    print(f"Retained {prices.shape[1]} tickers after quality filter.")
    return prices


def fetch_fundamentals(tickers):
    """Pull static fundamental snapshot (P/E, P/B, sector, market cap, ROE) per ticker."""
    records = []
    for i, t in enumerate(tickers):
        try:
            info = yf.Ticker(t).info
            records.append({
                "ticker": t,
                "sector": info.get("sector", "Unknown"),
                "market_cap": info.get("marketCap", np.nan),
                "pe_ratio": info.get("trailingPE", np.nan),
                "pb_ratio": info.get("priceToBook", np.nan),
                "roe": info.get("returnOnEquity", np.nan),
                "profit_margin": info.get("profitMargins", np.nan),
            })
        except Exception as e:
            print(f"  Failed on {t}: {e}")
        if i % 10 == 0:
            print(f"  Fetched fundamentals {i+1}/{len(tickers)}")
        time.sleep(0.05)  # be polite to the API
    return pd.DataFrame(records).set_index("ticker")


def build_dataset(save=True):
    os.makedirs(DATA_DIR, exist_ok=True)

    prices = fetch_prices(UNIVERSE)
    valid_tickers = [t for t in prices.columns if t != BENCHMARK]
    fundamentals = fetch_fundamentals(valid_tickers)

    if save:
        prices.to_csv(os.path.join(DATA_DIR, "prices.csv"))
        fundamentals.to_csv(os.path.join(DATA_DIR, "fundamentals.csv"))
        print(f"\nSaved prices.csv ({prices.shape}) and fundamentals.csv ({fundamentals.shape}) to {DATA_DIR}")

    return prices, fundamentals


if __name__ == "__main__":
    prices, fundamentals = build_dataset()
    print("\nSample prices:")
    print(prices.tail())
    print("\nSample fundamentals:")
    print(fundamentals.head())
