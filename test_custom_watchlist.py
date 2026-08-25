"""
test_custom_watchlist.py

Quick standalone check that the self-serve pipeline works end-to-end:
custom ticker list -> live fetch -> factor scores (default AND custom
weights) -> optimized portfolio.

Run this from the project root:
    python test_custom_watchlist.py

This does NOT touch the Streamlit app or your main dashboard data at all
-- it's purely to confirm Stage A (the backend changes) work before we
wire up the UI.
"""
from data_pipeline import fetch_custom_universe
from factor_model import compute_factor_scores
from optimizer import optimize_portfolio

# A small, deliberately mixed test watchlist
test_tickers = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "COST"]

print(f"Fetching live data for: {test_tickers}")
prices, fundamentals = fetch_custom_universe(test_tickers)
print(f"\nGot prices for {prices.shape[1]} tickers, {prices.shape[0]} trading days")
print(f"Got fundamentals for {fundamentals.shape[0]} tickers")

# --- Test 1: default equal-weight factor scores (should behave exactly
#     like the main dashboard's scoring) ---
print("\n--- Equal-weight factor scores ---")
scores_default = compute_factor_scores(prices, fundamentals)
print(scores_default[["sector", "composite_score", "value_score",
                       "momentum_score", "size_score", "quality_score", "lowvol_score"]].round(3))

# --- Test 2: custom factor weights -- heavily tilted toward Value ---
print("\n--- Custom weights: Value-tilted (Value=5, others=1) ---")
custom_weights = {
    "value_score": 5,
    "momentum_score": 1,
    "size_score": 1,
    "quality_score": 1,
    "lowvol_score": 1,
}
scores_custom = compute_factor_scores(prices, fundamentals, factor_weights=custom_weights)
print(scores_custom[["sector", "composite_score", "value_score"]].round(3))

# --- Test 3: feed either scoring into the optimizer ---
print("\n--- Optimized portfolio (using value-tilted scores) ---")
weights = optimize_portfolio(scores_custom, prices, risk_aversion=3.0)
print(weights.round(4))

print("\nAll steps completed without errors.")
