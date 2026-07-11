"""
Synthetic Data Generator.

Purpose: This sandbox environment can't reach live market data APIs (network is
restricted to package registries). This module generates REALISTIC synthetic
data with the same structure/shape as data_pipeline.py would produce, so the
rest of the project (factor model, optimizer, backtest, dashboard) can be
built and tested end-to-end right now.

On your own machine, just run `python src/data_pipeline.py` instead — it pulls
real data into the same prices.csv / fundamentals.csv format, and everything
downstream (factor_model.py, optimizer.py, backtest.py, dashboard) works
unchanged.
"""
import numpy as np
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SECTORS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "ADBE", "CRM", "ORCL", "CSCO", "INTC"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "SCHW", "AXP", "BLK", "SPGI", "C"],
    "Healthcare": ["UNH", "JNJ", "PFE", "MRK", "ABBV", "TMO", "ABT", "LLY", "BMY", "AMGN"],
    "Consumer": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "PG", "KO"],
    "Industrials": ["BA", "CAT", "GE", "HON", "UPS", "LMT", "MMM", "RTX", "DE", "UNP"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Utilities": ["NEE", "DUK", "SO", "AEP"],
}
BENCHMARK = "SPY"
START_DATE = "2018-01-01"
END_DATE = "2026-07-10"


def generate_prices(seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range(START_DATE, END_DATE)
    n_days = len(dates)

    tickers = [t for group in SECTORS.values() for t in group]
    n_stocks = len(tickers)

    # Market factor (common to all stocks) - drives correlation
    market_daily_ret = np.random.normal(0.0003, 0.011, n_days)

    # Sector factors - add sector-level co-movement
    sector_rets = {s: np.random.normal(0.0, 0.006, n_days) for s in SECTORS}

    # Each stock: beta to market + sector effect + idiosyncratic noise + a
    # random "quality drift" so some stocks are structurally better than others
    prices = pd.DataFrame(index=dates)
    stock_meta = {}

    for sector, tickers_in_sector in SECTORS.items():
        for t in tickers_in_sector:
            beta = np.random.uniform(0.6, 1.5)
            idio_vol = np.random.uniform(0.008, 0.022)
            drift = np.random.normal(0.0002, 0.0003)  # stock-specific alpha/skill

            noise = np.random.normal(0, idio_vol, n_days)
            daily_ret = beta * market_daily_ret + 0.5 * sector_rets[sector] + drift + noise

            price_path = 100 * np.exp(np.cumsum(daily_ret))
            prices[t] = price_path

            stock_meta[t] = {"sector": sector, "beta": beta, "drift": drift, "idio_vol": idio_vol}

    # Benchmark = market cap weighted-ish average (just use market factor directly)
    benchmark_price = 100 * np.exp(np.cumsum(market_daily_ret))
    prices[BENCHMARK] = benchmark_price

    return prices, stock_meta


def generate_fundamentals(stock_meta, seed=42):
    np.random.seed(seed + 1)
    records = []
    for t, meta in stock_meta.items():
        # Make fundamentals loosely correlated with the "drift" (quality) so
        # the factor model actually has real signal to find
        quality_signal = meta["drift"] * 3000 + np.random.normal(0, 1)

        records.append({
            "ticker": t,
            "sector": meta["sector"],
            "market_cap": np.random.uniform(20e9, 800e9),
            "pe_ratio": np.clip(25 - quality_signal * 2 + np.random.normal(0, 4), 5, 60),
            "pb_ratio": np.clip(4 - quality_signal + np.random.normal(0, 1.5), 0.5, 15),
            "roe": np.clip(0.12 + quality_signal * 0.03 + np.random.normal(0, 0.05), -0.1, 0.6),
            "profit_margin": np.clip(0.1 + quality_signal * 0.02 + np.random.normal(0, 0.05), -0.1, 0.4),
        })
    return pd.DataFrame(records).set_index("ticker")


def build_dataset(save=True):
    os.makedirs(DATA_DIR, exist_ok=True)
    prices, stock_meta = generate_prices()
    fundamentals = generate_fundamentals(stock_meta)

    if save:
        prices.to_csv(os.path.join(DATA_DIR, "prices.csv"))
        fundamentals.to_csv(os.path.join(DATA_DIR, "fundamentals.csv"))
        print(f"Saved SYNTHETIC prices.csv {prices.shape} and fundamentals.csv {fundamentals.shape} to {DATA_DIR}")
        print("(Swap in real data anytime by running src/data_pipeline.py on a machine with internet access)")

    return prices, fundamentals


if __name__ == "__main__":
    prices, fundamentals = build_dataset()
    print("\nSample prices:\n", prices.iloc[-5:, :5])
    print("\nSample fundamentals:\n", fundamentals.head())
