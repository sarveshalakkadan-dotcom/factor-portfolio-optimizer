# Factor-Based Portfolio Optimizer & Risk Analytics Engine

A quantitative portfolio construction system that scores equities on five classic
academic factors (value, momentum, size, quality, low-volatility), builds
risk-constrained optimal portfolios using convex optimization, and validates
the strategy with a walk-forward backtest and institutional-grade risk analytics.

**[Live Dashboard Screenshot / Demo GIF here]**

## Why this project

Systematic factor investing underpins a large share of AUM at quant funds and
smart-beta ETF providers. This project reproduces that pipeline end-to-end:
data → signal → optimization → validation → risk measurement — the same
workflow used on a real quant research desk, just scaled down.

## What it does

1. **Data Pipeline** — pulls historical prices and fundamentals for a
   ~60-stock, 7-sector universe (`src/data_pipeline.py`, live Yahoo Finance data)
2. **Factor Model** — scores every stock on value (P/E, P/B), momentum
   (12-1 month return), size (market cap), quality (ROE, margins), and
   low-volatility, then combines them into a composite score
3. **Portfolio Optimizer** — solves a mean-variance optimization problem via
   `cvxpy`, subject to no-shorting, per-stock concentration limits, and
   per-sector diversification limits
4. **Walk-Forward Backtest** — rebalances monthly using only information
   available at each point in time (strict no-lookahead-bias design),
   including realistic transaction costs
5. **Risk Analytics** — Sharpe, Sortino, max drawdown, VaR, CVaR, beta/alpha
   vs. benchmark, and portfolio-level factor exposure
6. **Interactive Dashboard** — Streamlit app where you drag a risk-tolerance
   slider and watch the portfolio, backtest, and risk metrics update live

## Architecture

```
factor_portfolio/
├── src/
│   ├── data_pipeline.py     # Live data ingestion (Yahoo Finance)
│   ├── synthetic_data.py    # Realistic synthetic data for dev/testing
│   ├── factor_model.py      # Factor scoring engine
│   ├── optimizer.py         # cvxpy-based mean-variance optimizer
│   ├── backtest.py          # Walk-forward backtest engine
│   └── risk_metrics.py      # Risk analytics library
├── dashboard/
│   └── app.py                # Streamlit interactive dashboard
├── data/                     # Generated price/fundamental/backtest data
└── requirements.txt
```

## Key design decisions worth highlighting in an interview

- **No lookahead bias**: at every rebalance date, the model only sees data
  that would have actually been available at that time. This is the #1
  thing that separates a rigorous backtest from a misleading one.
- **Transaction costs are modeled**: turnover is penalized at 10bps per
  rebalance, so the backtest reflects tradeable, not theoretical, performance.
- **Risk-based, not just return-based, evaluation**: Sharpe/Sortino ratios,
  drawdown, and tail-risk (VaR/CVaR) are reported alongside raw returns,
  because a strategy's risk profile is as important as its return.
- **Constrained optimization, not naive top-N selection**: the optimizer
  balances expected return against portfolio variance and enforces
  diversification limits — closer to how real portfolios are built than
  simply buying the top 10 highest-scoring stocks.

## Running it

```bash
pip install -r requirements.txt

# Get real market data (requires internet access)
python src/data_pipeline.py

# Or generate synthetic data for quick testing
python src/synthetic_data.py

# Run the factor model / optimizer / backtest / risk report individually
python src/factor_model.py
python src/optimizer.py
python src/backtest.py
python src/risk_metrics.py

# Launch the interactive dashboard
streamlit run dashboard/app.py
```

## Note on data

This repo ships with a synthetic data generator (`src/synthetic_data.py`) that
mimics realistic market structure (sector co-movement, factor-linked
fundamentals) so the project runs immediately without an internet connection
or API key. Swap in `src/data_pipeline.py` for live Yahoo Finance data —
every downstream module reads the same `prices.csv` / `fundamentals.csv`
schema, so nothing else changes.

## Possible extensions

- Add Fama-French 5-factor regression to decompose returns into true factor
  loadings rather than composite heuristic scores
- Add a risk-parity or Black-Litterman optimization mode alongside
  mean-variance
- Extend to a long/short market-neutral variant
- Add real-time paper trading via a broker API (Alpaca) to track live performance
