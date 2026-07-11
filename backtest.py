"""
Backtest Engine: Walk-forward simulation of the factor strategy.

Critical design principle - NO LOOKAHEAD BIAS:
At each rebalance date, we only use price/fundamental data available up to
that date. We never let the model "see the future" when making a decision.
This is the single most common mistake in amateur backtests and the first
thing a quant interviewer will probe for.
"""
import pandas as pd
import numpy as np
import os

from factor_model import compute_factor_scores
from optimizer import optimize_portfolio

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def get_rebalance_dates(prices, freq="ME"):
    """
    Actual trading dates closest to month-end (or other freq) on which we
    rebalance. Uses real dates present in the price index - calendar month-end
    (e.g. a Sunday) may not be a trading day.
    """
    period_groups = prices.groupby(prices.index.to_period(freq[0]))
    last_trading_days = period_groups.apply(lambda x: x.index.max())
    return pd.DatetimeIndex(last_trading_days.values)


def run_backtest(prices, fundamentals, risk_aversion=3.0, freq="ME",
                  lookback_buffer=270, cost_bps=10):
    """
    Walk forward through time, rebalancing monthly.

    Parameters
    ----------
    cost_bps : transaction cost in basis points applied to turnover each
               rebalance (10 bps = 0.10%, a realistic institutional estimate)

    Returns
    -------
    dict with 'portfolio_value', 'benchmark_value', 'weights_history', 'turnover_history'
    """
    rebalance_dates = get_rebalance_dates(prices, freq=freq)
    # Need enough history for momentum (252d) + vol (60d) calc before first rebalance
    valid_rebalance_dates = [d for d in rebalance_dates if d >= prices.index[lookback_buffer]]

    portfolio_value = [1.0]
    benchmark_value = [1.0]
    value_dates = [valid_rebalance_dates[0]]
    weights_history = {}
    turnover_history = {}

    current_weights = pd.Series(dtype=float)
    benchmark_ticker = "SPY"

    for i in range(len(valid_rebalance_dates) - 1):
        rebal_date = valid_rebalance_dates[i]
        next_date = valid_rebalance_dates[i + 1]

        # --- Score stocks using ONLY data up to rebal_date (no lookahead) ---
        price_hist_to_date = prices.loc[:rebal_date]
        try:
            scores = compute_factor_scores(price_hist_to_date, fundamentals, as_of_date=rebal_date)
            new_weights = optimize_portfolio(scores, price_hist_to_date, risk_aversion=risk_aversion)
        except Exception as e:
            print(f"  Skipping rebalance at {rebal_date.date()} due to error: {e}")
            new_weights = current_weights if len(current_weights) > 0 else pd.Series(dtype=float)

        # --- Turnover & transaction cost ---
        all_tickers = set(current_weights.index) | set(new_weights.index)
        turnover = sum(abs(new_weights.get(t, 0) - current_weights.get(t, 0)) for t in all_tickers) / 2
        cost = turnover * (cost_bps / 10000)
        turnover_history[rebal_date] = turnover

        current_weights = new_weights
        weights_history[rebal_date] = current_weights

        # --- Compute forward return from rebal_date to next_date ---
        period_prices = prices.loc[rebal_date:next_date, current_weights.index]
        period_returns = period_prices.iloc[-1] / period_prices.iloc[0] - 1
        portfolio_period_return = (current_weights * period_returns).sum() - cost

        bench_period_return = prices.loc[next_date, benchmark_ticker] / prices.loc[rebal_date, benchmark_ticker] - 1

        portfolio_value.append(portfolio_value[-1] * (1 + portfolio_period_return))
        benchmark_value.append(benchmark_value[-1] * (1 + bench_period_return))
        value_dates.append(next_date)

    results = {
        "dates": value_dates,
        "portfolio_value": pd.Series(portfolio_value, index=value_dates),
        "benchmark_value": pd.Series(benchmark_value, index=value_dates),
        "weights_history": weights_history,
        "turnover_history": pd.Series(turnover_history),
    }
    return results


if __name__ == "__main__":
    prices = pd.read_csv(os.path.join(DATA_DIR, "prices.csv"), index_col=0, parse_dates=True)
    fundamentals = pd.read_csv(os.path.join(DATA_DIR, "fundamentals.csv"), index_col=0)

    print("Running walk-forward backtest (monthly rebalance, 2018-2026)...")
    results = run_backtest(prices, fundamentals, risk_aversion=3.0)

    port = results["portfolio_value"]
    bench = results["benchmark_value"]

    total_return_port = port.iloc[-1] / port.iloc[0] - 1
    total_return_bench = bench.iloc[-1] / bench.iloc[0] - 1
    years = (port.index[-1] - port.index[0]).days / 365.25

    cagr_port = (port.iloc[-1] / port.iloc[0]) ** (1 / years) - 1
    cagr_bench = (bench.iloc[-1] / bench.iloc[0]) ** (1 / years) - 1

    print(f"\n--- Results over {years:.1f} years ---")
    print(f"Factor Strategy : {total_return_port*100:+.1f}% total | {cagr_port*100:+.2f}% CAGR")
    print(f"Benchmark (SPY) : {total_return_bench*100:+.1f}% total | {cagr_bench*100:+.2f}% CAGR")
    print(f"Avg monthly turnover: {results['turnover_history'].mean()*100:.1f}%")

    results["portfolio_value"].to_csv(os.path.join(DATA_DIR, "backtest_portfolio.csv"))
    results["benchmark_value"].to_csv(os.path.join(DATA_DIR, "backtest_benchmark.csv"))
    print(f"\nSaved backtest results to {DATA_DIR}")
