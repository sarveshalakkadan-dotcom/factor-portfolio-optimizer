"""
Interactive Dashboard: Streamlit app that ties together the factor model,
optimizer, backtest, and risk analytics into one live, explorable interface.

Run with: streamlit run dashboard/app.py
"""
import sys
import os
# NOTE: this repo uses a FLAT structure on GitHub/Streamlit Cloud --
# app.py, factor_model.py, forensic_score.py, prices.csv, etc. all sit
# directly in the same root folder, so no sys.path changes are needed;
# Python already looks in the script's own directory for imports.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from factor_model import compute_factor_scores
from optimizer import optimize_portfolio
from backtest import run_backtest
from risk_metrics import full_risk_report, to_returns, factor_exposure
from forensic_score import run_forensic_check
from data_pipeline import fetch_custom_universe

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Factor Portfolio Optimizer", layout="wide")


@st.cache_data
def load_data():
    prices = pd.read_csv(os.path.join(DATA_DIR, "prices.csv"), index_col=0, parse_dates=True)
    fundamentals = pd.read_csv(os.path.join(DATA_DIR, "fundamentals.csv"), index_col=0)
    return prices, fundamentals


@st.cache_data
def get_scores(_prices, _fundamentals):
    return compute_factor_scores(_prices, _fundamentals)


@st.cache_data
def get_optimized_weights(_scores, _prices, risk_aversion, max_stock, max_sector):
    return optimize_portfolio(_scores, _prices, risk_aversion=risk_aversion,
                               max_weight_per_stock=max_stock, max_weight_per_sector=max_sector)


@st.cache_data
def get_backtest(_prices, _fundamentals, risk_aversion):
    return run_backtest(_prices, _fundamentals, risk_aversion=risk_aversion)


@st.cache_data(show_spinner=False)
def get_forensic_scores(tickers):
    """
    Runs the Beneish M-Score forensic accounting screen for each ticker.

    This is a statistical screening tool (Beneish, 1999) that estimates
    whether a company's financials show patterns associated with earnings
    manipulation -- it is NOT a fraud accusation or litigation record.

    Cached so this only re-runs when the actual ticker list changes,
    since each check requires live financial-statement lookups.
    """
    results = {}
    for ticker in tickers:
        check = run_forensic_check(ticker)
        if check["insufficient_data"]:
            results[ticker] = "Insufficient data"
        else:
            results[ticker] = f"{check['risk_tier']} ({check['m_score']:.2f})"
    return results


# ---------------- Sidebar Controls ----------------
st.sidebar.title("Portfolio Controls")
st.sidebar.markdown("Adjust risk tolerance and constraints to see the portfolio rebuild in real time.")

risk_label = st.sidebar.select_slider(
    "Risk Tolerance",
    options=["Aggressive", "Moderate-Aggressive", "Balanced", "Moderate-Conservative", "Conservative"],
    value="Balanced"
)
risk_map = {
    "Aggressive": 1.0, "Moderate-Aggressive": 2.0, "Balanced": 3.0,
    "Moderate-Conservative": 5.0, "Conservative": 8.0
}
risk_aversion = risk_map[risk_label]

max_stock_weight = st.sidebar.slider("Max weight per stock", 0.03, 0.20, 0.08, 0.01)
max_sector_weight = st.sidebar.slider("Max weight per sector", 0.15, 0.50, 0.30, 0.05)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data note: this demo runs on synthetic market data generated to mimic real "
    "equity behavior (correlated returns, sector structure, factor-linked fundamentals). "
    "Swap in `src/data_pipeline.py` output for live Yahoo Finance data — everything "
    "else works unchanged."
)

# ---------------- Load & Compute ----------------
prices, fundamentals = load_data()
scores = get_scores(prices, fundamentals)
weights = get_optimized_weights(scores, prices, risk_aversion, max_stock_weight, max_sector_weight)

# ---------------- Header ----------------
st.title("📊 Factor-Based Portfolio Optimizer")
st.caption("Multi-factor stock scoring → risk-constrained optimization → backtested performance → risk analytics")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Current Portfolio", "Factor Scores", "Backtest Performance", "Risk Analytics", "Build Your Own Portfolio"]
)

# ---------------- Tab 1: Current Portfolio ----------------
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Optimized Holdings")
        holdings_df = pd.DataFrame({"Weight": weights})
        holdings_df["Sector"] = scores.loc[holdings_df.index, "sector"]
        holdings_df["Factor Score"] = scores.loc[holdings_df.index, "composite_score"].round(3)
        holdings_df["Weight"] = (holdings_df["Weight"] * 100).round(2).astype(str) + "%"
        st.dataframe(holdings_df, use_container_width=True, height=400)

    with col2:
        st.subheader("Sector Allocation")
        sector_alloc = pd.DataFrame({"weight": weights})
        sector_alloc["sector"] = scores.loc[sector_alloc.index, "sector"]
        sector_summary = sector_alloc.groupby("sector")["weight"].sum().sort_values(ascending=False)

        fig = go.Figure(data=[go.Pie(labels=sector_summary.index, values=sector_summary.values, hole=0.4)])
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Portfolio Factor Exposure")
    st.caption("How tilted is this portfolio toward each factor, on net? Intentional tilts (not accidental ones) are the goal.")
    exposures = factor_exposure(weights, scores)
    fig2 = go.Figure(data=[go.Bar(x=exposures.index, y=exposures.values,
                                    marker_color=['#2E86AB' if v >= 0 else '#C73E1D' for v in exposures.values])])
    fig2.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Tab 2: Factor Scores ----------------
with tab2:
    st.subheader("All Stocks Ranked by Composite Factor Score")
    display_cols = ["sector", "composite_score", "value_score", "momentum_score",
                     "size_score", "quality_score", "lowvol_score"]

    run_forensic = st.checkbox(
        "Run forensic accounting screen (Beneish M-Score)",
        value=False,
        help="Pulls live financial statement data per stock, so this may take a "
             "moment to load the first time for a larger watchlist. Off by default "
             "so the page loads instantly."
    )

    display_df = scores[display_cols].round(3).copy()

    if run_forensic:
        with st.spinner("Running forensic accounting screen across the watchlist..."):
            forensic_results = get_forensic_scores(tuple(display_df.index))
        display_df["Forensic Risk Score"] = display_df.index.map(forensic_results)

        st.dataframe(
            display_df,
            use_container_width=True,
            height=600,
            column_config={
                "Forensic Risk Score": st.column_config.TextColumn(
                    "Forensic Risk Score",
                    help=(
                        "Based on the Beneish M-Score (Beneish, 1999) — a statistical model "
                        "using 8 financial-statement ratios that estimates whether a company's "
                        "numbers show patterns associated with earnings manipulation. This is a "
                        "screening tool used by analysts and auditors, not an accusation of "
                        "wrongdoing. 'Elevated' means a statistically higher likelihood based on "
                        "this model; it does not mean fraud has been found or confirmed."
                    )
                )
            }
        )
    else:
        st.dataframe(display_df, use_container_width=True, height=600)

# ---------------- Tab 3: Backtest ----------------
with tab3:
    st.subheader("Walk-Forward Backtest (Monthly Rebalance)")
    with st.spinner("Running backtest across full history..."):
        results = get_backtest(prices, fundamentals, risk_aversion)

    port_val = results["portfolio_value"]
    bench_val = results["benchmark_value"]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=port_val.index, y=port_val.values, name="Factor Strategy",
                                line=dict(color="#2E86AB", width=2.5)))
    fig3.add_trace(go.Scatter(x=bench_val.index, y=bench_val.values, name="S&P 500 (Benchmark)",
                                line=dict(color="#888888", width=2, dash="dash")))
    fig3.update_layout(title="Cumulative Growth of $1", height=450,
                        yaxis_title="Portfolio Value", xaxis_title="Date")
    st.plotly_chart(fig3, use_container_width=True)

    total_ret_port = port_val.iloc[-1] / port_val.iloc[0] - 1
    total_ret_bench = bench_val.iloc[-1] / bench_val.iloc[0] - 1
    years = (port_val.index[-1] - port_val.index[0]).days / 365.25
    cagr_port = (port_val.iloc[-1] / port_val.iloc[0]) ** (1/years) - 1
    cagr_bench = (bench_val.iloc[-1] / bench_val.iloc[0]) ** (1/years) - 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategy Total Return", f"{total_ret_port*100:+.1f}%")
    c2.metric("Benchmark Total Return", f"{total_ret_bench*100:+.1f}%")
    c3.metric("Strategy CAGR", f"{cagr_port*100:+.2f}%")
    c4.metric("Avg Monthly Turnover", f"{results['turnover_history'].mean()*100:.1f}%")

# ---------------- Tab 4: Risk Analytics ----------------
with tab4:
    st.subheader("Risk-Adjusted Performance Metrics")
    report, drawdown_series = full_risk_report(port_val, bench_val)

    c1, c2, c3 = st.columns(3)
    c1.metric("Sharpe Ratio", f"{report['Sharpe Ratio']:.2f}")
    c2.metric("Sortino Ratio", f"{report['Sortino Ratio']:.2f}")
    c3.metric("Beta vs Benchmark", f"{report['Beta vs Benchmark']:.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Max Drawdown", f"{report['Max Drawdown']*100:.1f}%",
               delta=f"{(report['Max Drawdown']-report['Benchmark Max Drawdown'])*100:+.1f}% vs benchmark",
               delta_color="normal")
    c5.metric("VaR (95%, monthly)", f"{report['VaR (95%, monthly)']*100:.1f}%")
    c6.metric("CVaR (95%, monthly)", f"{report['CVaR (95%, monthly)']*100:.1f}%")

    st.markdown("---")
    st.subheader("Drawdown Over Time")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series.values * 100,
                                fill='tozeroy', line=dict(color="#C73E1D"), name="Drawdown %"))
    fig4.update_layout(height=350, yaxis_title="Drawdown (%)")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("Monthly Return Distribution")
    port_returns = to_returns(port_val)
    fig5 = go.Figure(data=[go.Histogram(x=port_returns.values * 100, nbinsx=30, marker_color="#2E86AB")])
    fig5.update_layout(height=300, xaxis_title="Monthly Return (%)", yaxis_title="Frequency")
    st.plotly_chart(fig5, use_container_width=True)

# ---------------- Tab 5: Build Your Own Portfolio ----------------
with tab5:
    st.subheader("Build Your Own Factor Portfolio")
    st.caption(
        "Paste any list of tickers, set your own factor tilts, and this tab pulls "
        "live data, scores it, optimizes it, and backtests it — just for you."
    )

    if "custom_prices" not in st.session_state:
        st.session_state.custom_prices = None
        st.session_state.custom_fundamentals = None
        st.session_state.custom_weights = None
        st.session_state.custom_scores = None
        st.session_state.custom_backtest = None

    with st.form("custom_universe_form"):
        ticker_input = st.text_area(
            "Paste tickers (comma, space, or newline separated)",
            placeholder="AAPL, MSFT, NVDA, JPM, XOM, ...",
            height=100,
        )
        fetch_submitted = st.form_submit_button("Fetch Live Data")

    if fetch_submitted:
        raw_tickers = [t.strip() for t in ticker_input.replace(",", " ").replace("\n", " ").split(" ") if t.strip()]
        if len(raw_tickers) < 5:
            st.error(
                "Please enter at least 5 tickers. Fewer than that, combined with the "
                "sector concentration limit, often makes the optimizer infeasible."
            )
        else:
            try:
                with st.spinner(f"Pulling live price + fundamental data for {len(raw_tickers)} tickers... "
                                 f"this can take a minute for a larger watchlist."):
                    custom_prices, custom_fundamentals = fetch_custom_universe(raw_tickers)
                st.session_state.custom_prices = custom_prices
                st.session_state.custom_fundamentals = custom_fundamentals
                # Clear any previously-built portfolio from a prior ticker list
                st.session_state.custom_weights = None
                st.session_state.custom_scores = None
                st.session_state.custom_backtest = None
                st.success(f"Loaded live data for {custom_fundamentals.shape[0]} valid tickers.")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Something went wrong pulling data: {e}")

    if st.session_state.custom_prices is not None:
        custom_prices = st.session_state.custom_prices
        custom_fundamentals = st.session_state.custom_fundamentals

        st.markdown("---")
        st.markdown("**Set your factor tilts**")
        st.caption(
            "Higher = more influence on the composite score. These are relative "
            "weights, not percentages — they don't need to add up to 100."
        )

        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        w_value = fc1.slider("Value", 0, 100, 20, key="custom_w_value")
        w_momentum = fc2.slider("Momentum", 0, 100, 20, key="custom_w_momentum")
        w_size = fc3.slider("Size", 0, 100, 20, key="custom_w_size")
        w_quality = fc4.slider("Quality", 0, 100, 20, key="custom_w_quality")
        w_lowvol = fc5.slider("Low-Vol", 0, 100, 20, key="custom_w_lowvol")

        custom_factor_weights = {
            "value_score": w_value,
            "momentum_score": w_momentum,
            "size_score": w_size,
            "quality_score": w_quality,
            "lowvol_score": w_lowvol,
        }

        custom_risk_label = st.select_slider(
            "Risk Tolerance",
            options=["Aggressive", "Moderate-Aggressive", "Balanced", "Moderate-Conservative", "Conservative"],
            value="Balanced",
            key="custom_risk_label",
        )
        custom_risk_aversion = risk_map[custom_risk_label]

        build_clicked = st.button("Build My Portfolio", type="primary")

        if build_clicked:
            try:
                with st.spinner("Scoring, optimizing, and backtesting your custom portfolio..."):
                    custom_scores = compute_factor_scores(
                        custom_prices, custom_fundamentals, factor_weights=custom_factor_weights
                    )

                    # --- Dynamic concentration caps ---
                    # The main dashboard's fixed caps (8% per stock, 30% per
                    # sector) assume a ~60-stock universe. A user-pasted
                    # watchlist can be much smaller or concentrated in one or
                    # two sectors, which makes those fixed caps mathematically
                    # infeasible (e.g. 5 stocks can never sum to 100% if each
                    # is capped at 8%). Rather than just failing, loosen the
                    # caps just enough to guarantee a feasible solution, with
                    # a comfortable 50% margin over the tight theoretical bound.
                    n_stocks = len(custom_scores)
                    n_sectors = custom_scores["sector"].nunique()
                    dynamic_max_stock = max(0.08, min(0.50, 1.5 / n_stocks))
                    dynamic_max_sector = max(0.30, min(0.80, 1.5 / n_sectors))

                    custom_weights = optimize_portfolio(
                        custom_scores, custom_prices, risk_aversion=custom_risk_aversion,
                        max_weight_per_stock=dynamic_max_stock,
                        max_weight_per_sector=dynamic_max_sector,
                    )
                    custom_backtest = run_backtest(
                        custom_prices, custom_fundamentals,
                        risk_aversion=custom_risk_aversion,
                        factor_weights=custom_factor_weights,
                        max_weight_per_stock=dynamic_max_stock,
                        max_weight_per_sector=dynamic_max_sector,
                    )
                    if dynamic_max_stock > 0.08 or dynamic_max_sector > 0.30:
                        st.info(
                            f"Your watchlist ({n_stocks} stocks across {n_sectors} sectors) is smaller "
                            f"and less diversified than the main 60-stock universe, so concentration caps "
                            f"were loosened to keep the optimizer feasible: max {dynamic_max_stock*100:.0f}% "
                            f"per stock, max {dynamic_max_sector*100:.0f}% per sector."
                        )
                st.session_state.custom_scores = custom_scores
                st.session_state.custom_weights = custom_weights
                st.session_state.custom_backtest = custom_backtest
            except Exception as e:
                st.error(
                    f"Couldn't build a portfolio from this universe: {e}\n\n"
                    "This is often the optimizer failing to find a feasible solution — "
                    "try a larger or more sector-diverse ticker list."
                )

        if st.session_state.custom_weights is not None:
            custom_scores = st.session_state.custom_scores
            custom_weights = st.session_state.custom_weights
            custom_backtest = st.session_state.custom_backtest

            st.markdown("---")
            cc1, cc2 = st.columns([1, 1])

            with cc1:
                st.subheader("Optimized Holdings")
                holdings_df_c = pd.DataFrame({"Weight": custom_weights})
                holdings_df_c["Sector"] = custom_scores.loc[holdings_df_c.index, "sector"]
                holdings_df_c["Factor Score"] = custom_scores.loc[holdings_df_c.index, "composite_score"].round(3)
                holdings_df_c["Weight"] = (holdings_df_c["Weight"] * 100).round(2).astype(str) + "%"
                st.dataframe(holdings_df_c, use_container_width=True, height=350)

            with cc2:
                st.subheader("Sector Allocation")
                sector_alloc_c = pd.DataFrame({"weight": custom_weights})
                sector_alloc_c["sector"] = custom_scores.loc[sector_alloc_c.index, "sector"]
                sector_summary_c = sector_alloc_c.groupby("sector")["weight"].sum().sort_values(ascending=False)
                fig_c = go.Figure(data=[go.Pie(labels=sector_summary_c.index, values=sector_summary_c.values, hole=0.4)])
                fig_c.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
                st.plotly_chart(fig_c, use_container_width=True)

            st.markdown("---")
            st.subheader("Backtest: Your Portfolio vs S&P 500")
            cport = custom_backtest["portfolio_value"]
            cbench = custom_backtest["benchmark_value"]

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=cport.index, y=cport.values, name="Your Strategy",
                                          line=dict(color="#2E86AB", width=2.5)))
            fig_bt.add_trace(go.Scatter(x=cbench.index, y=cbench.values, name="S&P 500 (Benchmark)",
                                          line=dict(color="#888888", width=2, dash="dash")))
            fig_bt.update_layout(title="Cumulative Growth of $1", height=400,
                                   yaxis_title="Portfolio Value", xaxis_title="Date")
            st.plotly_chart(fig_bt, use_container_width=True)

            if len(cport) > 1:
                total_ret_c = cport.iloc[-1] / cport.iloc[0] - 1
                total_ret_cb = cbench.iloc[-1] / cbench.iloc[0] - 1
                years_c = (cport.index[-1] - cport.index[0]).days / 365.25
                if years_c > 0:
                    cagr_c = (cport.iloc[-1] / cport.iloc[0]) ** (1 / years_c) - 1
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Your Strategy Total Return", f"{total_ret_c*100:+.1f}%")
                    m2.metric("Benchmark Total Return", f"{total_ret_cb*100:+.1f}%")
                    m3.metric("Your Strategy CAGR", f"{cagr_c*100:+.2f}%")
