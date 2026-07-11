"""
Factor Model: Scores every stock on 5 classic factors and combines them
into one composite score used by the optimizer.

Factors:
  - Value:     cheaper stocks (low P/E, low P/B) score higher
  - Momentum:  stocks with strong 12-1 month returns score higher
  - Size:      smaller market cap scores higher (small-cap premium)
  - Quality:   higher ROE / profit margin scores higher
  - Low-Vol:   lower realized volatility scores higher
"""
import pandas as pd
import numpy as np


def zscore(series):
    """Standardize a series to mean 0, std 1 - makes factors comparable."""
    return (series - series.mean()) / (series.std() + 1e-9)


def compute_momentum(prices, lookback=252, skip=21):
    """
    12-1 month momentum: return over the past year, EXCLUDING the most recent
    month. Skipping the last month is standard practice - momentum famously
    partially reverses in the short term, so including it hurts the signal.
    """
    past_price = prices.shift(skip)
    lookback_price = prices.shift(lookback)
    momentum = (past_price / lookback_price) - 1
    return momentum


def compute_volatility(prices, window=60):
    """Rolling annualized volatility from daily returns."""
    daily_returns = prices.pct_change()
    vol = daily_returns.rolling(window).std() * np.sqrt(252)
    return vol


def compute_factor_scores(prices, fundamentals, as_of_date=None):
    """
    Compute composite factor scores for every stock as of a given date.
    Uses only data available up to that date (no lookahead).

    Returns a DataFrame indexed by ticker with individual factor z-scores
    and a combined 'composite_score' column.
    """
    if as_of_date is None:
        as_of_date = prices.index[-1]

    tickers = [c for c in prices.columns if c in fundamentals.index]
    price_hist = prices.loc[:as_of_date, tickers]

    momentum_series = compute_momentum(price_hist).iloc[-1]
    vol_series = compute_volatility(price_hist).iloc[-1]

    df = pd.DataFrame(index=tickers)
    df["momentum_raw"] = momentum_series
    df["volatility_raw"] = vol_series
    df["pe_ratio"] = fundamentals.loc[tickers, "pe_ratio"]
    df["pb_ratio"] = fundamentals.loc[tickers, "pb_ratio"]
    df["market_cap"] = fundamentals.loc[tickers, "market_cap"]
    df["roe"] = fundamentals.loc[tickers, "roe"]
    df["profit_margin"] = fundamentals.loc[tickers, "profit_margin"]
    df["sector"] = fundamentals.loc[tickers, "sector"]

    df = df.dropna(subset=["momentum_raw", "volatility_raw"])

    # --- Value: cheaper = better, so invert P/E and P/B before z-scoring ---
    df["value_score"] = zscore(-df["pe_ratio"]) * 0.5 + zscore(-df["pb_ratio"]) * 0.5

    # --- Momentum: higher 12-1 month return = better ---
    df["momentum_score"] = zscore(df["momentum_raw"])

    # --- Size: smaller market cap = better (small-cap premium) ---
    df["size_score"] = zscore(-np.log(df["market_cap"]))

    # --- Quality: higher ROE + profit margin = better ---
    df["quality_score"] = zscore(df["roe"]) * 0.5 + zscore(df["profit_margin"]) * 0.5

    # --- Low-Vol: lower volatility = better ---
    df["lowvol_score"] = zscore(-df["volatility_raw"])

    # --- Composite: equal-weight blend of all 5 factors ---
    factor_cols = ["value_score", "momentum_score", "size_score", "quality_score", "lowvol_score"]
    df["composite_score"] = df[factor_cols].mean(axis=1)

    return df.sort_values("composite_score", ascending=False)


if __name__ == "__main__":
    import os
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    prices = pd.read_csv(os.path.join(DATA_DIR, "prices.csv"), index_col=0, parse_dates=True)
    fundamentals = pd.read_csv(os.path.join(DATA_DIR, "fundamentals.csv"), index_col=0)

    scores = compute_factor_scores(prices, fundamentals)
    print("Top 10 stocks by composite factor score:")
    print(scores[["sector", "composite_score", "value_score", "momentum_score",
                   "size_score", "quality_score", "lowvol_score"]].head(10).round(3))
    print("\nBottom 5:")
    print(scores[["sector", "composite_score"]].tail(5).round(3))
