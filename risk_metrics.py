"""
Risk Analytics: The metrics a finance professional actually looks at beyond
raw returns - because a strategy that returns 20% but loses 40% in one month
is not the same as one that returns 20% smoothly.
"""
import numpy as np
import pandas as pd


def to_returns(value_series):
    """Convert a cumulative value series (e.g. portfolio NAV) to period returns."""
    return value_series.pct_change().dropna()


def annualize_return(returns, periods_per_year=12):
    """CAGR from a periodic returns series."""
    cumulative = (1 + returns).prod()
    n_periods = len(returns)
    return cumulative ** (periods_per_year / n_periods) - 1


def annualize_vol(returns, periods_per_year=12):
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns, risk_free_rate=0.04, periods_per_year=12):
    """Return per unit of total risk (volatility). Higher is better; >1 is good, >2 is excellent."""
    excess_return = annualize_return(returns, periods_per_year) - risk_free_rate
    vol = annualize_vol(returns, periods_per_year)
    return excess_return / vol if vol > 0 else np.nan


def sortino_ratio(returns, risk_free_rate=0.04, periods_per_year=12):
    """Like Sharpe, but only penalizes DOWNSIDE volatility - upside swings don't count against you."""
    excess_return = annualize_return(returns, periods_per_year) - risk_free_rate
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(periods_per_year)
    return excess_return / downside_vol if downside_vol > 0 else np.nan


def max_drawdown(value_series):
    """
    Largest peak-to-trough decline. This is the number that answers
    'what's the worst it ever got, from a high point?' - the metric that
    matters most to someone deciding whether they could stomach this strategy.
    """
    running_max = value_series.cummax()
    drawdown = (value_series - running_max) / running_max
    return drawdown.min(), drawdown


def value_at_risk(returns, confidence=0.95):
    """
    VaR: 'In the worst (1-confidence)% of periods, how much do we lose at least?'
    e.g. 95% VaR of -5% means: 95% of the time, losses won't exceed 5% in a period.
    Historical (non-parametric) method - uses actual observed return distribution.
    """
    return np.percentile(returns, (1 - confidence) * 100)


def conditional_value_at_risk(returns, confidence=0.95):
    """
    CVaR (aka Expected Shortfall): 'GIVEN that we're in that worst 5% tail,
    what's the AVERAGE loss?' More informative than VaR because it captures
    how bad the tail actually is, not just where it starts.
    """
    var_threshold = value_at_risk(returns, confidence)
    tail_losses = returns[returns <= var_threshold]
    return tail_losses.mean()


def beta_alpha(portfolio_returns, benchmark_returns, periods_per_year=12, risk_free_rate=0.04):
    """
    Beta: portfolio's sensitivity to benchmark moves (1.0 = moves with market).
    Alpha: excess return NOT explained by market exposure - the actual value-add
    of the strategy after accounting for how much market risk it took.
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["portfolio", "benchmark"]

    covariance = aligned.cov().iloc[0, 1]
    benchmark_var = aligned["benchmark"].var()
    beta = covariance / benchmark_var if benchmark_var > 0 else np.nan

    port_annual_return = annualize_return(aligned["portfolio"], periods_per_year)
    bench_annual_return = annualize_return(aligned["benchmark"], periods_per_year)
    alpha = port_annual_return - (risk_free_rate + beta * (bench_annual_return - risk_free_rate))

    return beta, alpha


def factor_exposure(weights, scores_df):
    """
    Portfolio-level exposure to each factor - answers 'am I secretly making
    one big bet without realizing it?' A well-constructed portfolio should
    show intentional, not accidental, factor tilts.
    """
    factor_cols = ["value_score", "momentum_score", "size_score", "quality_score", "lowvol_score"]
    exposures = {}
    for col in factor_cols:
        aligned_scores = scores_df.loc[weights.index, col]
        exposures[col.replace("_score", "")] = (weights * aligned_scores).sum()
    return pd.Series(exposures)


def full_risk_report(portfolio_value, benchmark_value, periods_per_year=12):
    """Generate a complete risk report comparing strategy vs benchmark."""
    port_returns = to_returns(portfolio_value)
    bench_returns = to_returns(benchmark_value)

    dd_port, dd_series_port = max_drawdown(portfolio_value)
    dd_bench, _ = max_drawdown(benchmark_value)

    beta, alpha = beta_alpha(port_returns, bench_returns, periods_per_year)

    report = {
        "CAGR": annualize_return(port_returns, periods_per_year),
        "Benchmark CAGR": annualize_return(bench_returns, periods_per_year),
        "Annual Volatility": annualize_vol(port_returns, periods_per_year),
        "Sharpe Ratio": sharpe_ratio(port_returns, periods_per_year=periods_per_year),
        "Sortino Ratio": sortino_ratio(port_returns, periods_per_year=periods_per_year),
        "Max Drawdown": dd_port,
        "Benchmark Max Drawdown": dd_bench,
        "VaR (95%, monthly)": value_at_risk(port_returns),
        "CVaR (95%, monthly)": conditional_value_at_risk(port_returns),
        "Beta vs Benchmark": beta,
        "Alpha (annualized)": alpha,
    }
    return report, dd_series_port


if __name__ == "__main__":
    import os
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

    portfolio_value = pd.read_csv(os.path.join(DATA_DIR, "backtest_portfolio.csv"), index_col=0, parse_dates=True).iloc[:, 0]
    benchmark_value = pd.read_csv(os.path.join(DATA_DIR, "backtest_benchmark.csv"), index_col=0, parse_dates=True).iloc[:, 0]

    report, drawdown_series = full_risk_report(portfolio_value, benchmark_value)

    print("=== RISK REPORT ===")
    for k, v in report.items():
        if "Ratio" in k or "Beta" in k:
            print(f"{k:30s}: {v:.2f}")
        else:
            print(f"{k:30s}: {v*100:+.2f}%")
