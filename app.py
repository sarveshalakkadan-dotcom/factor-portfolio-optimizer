"""
Interactive Dashboard: Streamlit app that ties together the factor model,
optimizer, backtest, and risk analytics into one live, explorable interface.

Run with: streamlit run dashboard/app.py
"""
import sys
import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from factor_model import compute_factor_scores
from optimizer import optimize_portfolio
from backtest import run_backtest
from risk_metrics import full_risk_report, to_returns, factor_exposure

DATA_DIR = os.path.dirname(__file__)

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
   st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: live equity prices and fundamentals sourced from Yahoo Finance, "
    "covering a 60-stock universe across 7 sectors."
)
)

# ---------------- Load & Compute ----------------
prices, fundamentals = load_data()
scores = get_scores(prices, fundamentals)
weights = get_optimized_weights(scores, prices, risk_aversion, max_stock_weight, max_sector_weight)

# ---------------- Header ----------------
st.title("📊 Factor-Based Portfolio Optimizer")
st.caption("Multi-factor stock scoring → risk-constrained optimization → backtested performance → risk analytics")

tab1, tab2, tab3, tab4 = st.tabs(["Current Portfolio", "Factor Scores", "Backtest Performance", "Risk Analytics"])

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
    st.dataframe(scores[display_cols].round(3), use_container_width=True, height=600)

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
