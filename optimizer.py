"""
Portfolio Optimizer: Takes factor scores + a covariance matrix and solves
for optimal portfolio weights using mean-variance optimization.

Objective: maximize (expected_return - risk_aversion * portfolio_variance)
Constraints:
  - weights sum to 100% (fully invested)
  - no shorting (weights >= 0)
  - max weight per stock (concentration limit)
  - max weight per sector (diversification limit)
"""
import cvxpy as cp
import numpy as np
import pandas as pd


def compute_covariance(prices, tickers, window=252):
    """Annualized covariance matrix of daily returns over the trailing window."""
    returns = prices[tickers].pct_change().tail(window)
    cov = returns.cov() * 252
    return cov


def optimize_portfolio(scores_df, prices, risk_aversion=3.0,
                        max_weight_per_stock=0.08, max_weight_per_sector=0.30,
                        cov_window=252):
    """
    Solve for optimal portfolio weights.

    Parameters
    ----------
    scores_df : DataFrame from factor_model.compute_factor_scores(), must have
                'composite_score' and 'sector' columns, indexed by ticker.
    prices : full price history DataFrame (for covariance estimation)
    risk_aversion : higher = more conservative (penalizes variance more).
                    Roughly: 1=aggressive, 3=balanced, 8=conservative.
    max_weight_per_stock : concentration limit, e.g. 0.08 = no more than 8% in one name
    max_weight_per_sector : diversification limit, e.g. 0.30 = no more than 30% in one sector

    Returns
    -------
    weights : Series indexed by ticker
    """
    tickers = scores_df.index.tolist()
    n = len(tickers)

    # Expected return proxy: composite factor score (higher score = higher
    # expected outperformance). We scale it to a plausible annual return range.
    expected_returns = scores_df["composite_score"].values * 0.05  # scale factor

    cov_matrix = compute_covariance(prices, tickers, window=cov_window)
    cov_matrix = cov_matrix.reindex(index=tickers, columns=tickers).values

    # Regularize covariance slightly to keep the solver numerically stable
    cov_matrix = cov_matrix + np.eye(n) * 1e-6

    w = cp.Variable(n)
    portfolio_return = expected_returns @ w
    portfolio_variance = cp.quad_form(w, cov_matrix)

    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight_per_stock,
    ]

    # Sector constraints
    sectors = scores_df["sector"].unique()
    for sector in sectors:
        sector_mask = (scores_df["sector"] == sector).values.astype(float)
        constraints.append(sector_mask @ w <= max_weight_per_sector)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    if w.value is None:
        raise RuntimeError(f"Optimizer failed to converge. Status: {problem.status}")

    weights = pd.Series(np.clip(w.value, 0, None), index=tickers)
    weights = weights[weights > 0.001]  # drop dust positions
    weights = weights / weights.sum()   # renormalize after dropping dust

    return weights.sort_values(ascending=False)


def portfolio_summary(weights, scores_df):
    """Quick readable summary of a portfolio's composition."""
    df = pd.DataFrame({"weight": weights})
    df["sector"] = scores_df.loc[df.index, "sector"]
    df["composite_score"] = scores_df.loc[df.index, "composite_score"]

    print(f"\n{len(df)} holdings, top 10:")
    print(df.head(10).round(4))

    print("\nSector allocation:")
    print(df.groupby("sector")["weight"].sum().sort_values(ascending=False).round(4))


if __name__ == "__main__":
    import os
    from factor_model import compute_factor_scores

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    prices = pd.read_csv(os.path.join(DATA_DIR, "prices.csv"), index_col=0, parse_dates=True)
    fundamentals = pd.read_csv(os.path.join(DATA_DIR, "fundamentals.csv"), index_col=0)

    scores = compute_factor_scores(prices, fundamentals)

    print("=== BALANCED portfolio (risk_aversion=3) ===")
    weights_balanced = optimize_portfolio(scores, prices, risk_aversion=3.0)
    portfolio_summary(weights_balanced, scores)

    print("\n\n=== CONSERVATIVE portfolio (risk_aversion=8) ===")
    weights_conservative = optimize_portfolio(scores, prices, risk_aversion=8.0)
    portfolio_summary(weights_conservative, scores)
