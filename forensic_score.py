"""
forensic_score.py

Step 2 of the Beneish M-Score module: take the current_year / prior_year
financial data (from forensic_data_fetch.py) and compute the 8 Beneish
ratios, then combine them into the final M-Score and a plain-English
risk tier.

This does NOT accuse any company of fraud. It applies a published,
peer-reviewed statistical formula (Beneish, 1999) that estimates the
likelihood a company's financials show patterns associated with
earnings manipulation. It is a screening tool, not a verdict.
"""

from forensic_data_fetch import get_financial_data

# The standard published M-Score threshold. Companies scoring above
# this are flagged as having a statistically higher likelihood of
# earnings manipulation. This is Beneish's original research threshold,
# not something we chose ourselves.
MSCORE_THRESHOLD = -1.78


def _safe_divide(numerator, denominator):
    """Divide safely, returning None if the denominator is zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def compute_beneish_mscore(financial_data):
    """
    Takes the dict returned by get_financial_data() and computes the
    8 Beneish ratios + final M-Score.

    Returns a dict with:
        - ticker
        - insufficient_data (bool)
        - reason (str, only set if insufficient_data)
        - ratios (dict of the 8 individual ratio values)
        - m_score (float)
        - risk_tier ("Elevated" or "Low/Moderate")
    """
    ticker = financial_data["ticker"]

    result = {
        "ticker": ticker,
        "insufficient_data": False,
        "reason": None,
        "ratios": {},
        "m_score": None,
        "risk_tier": None,
    }

    # If step 1 already flagged this ticker as missing fields, stop here
    if financial_data["insufficient_data"]:
        result["insufficient_data"] = True
        result["reason"] = f"Missing fields: {financial_data['missing_fields']}"
        return result

    cur = financial_data["current_year"]
    pri = financial_data["prior_year"]

    # --- 1. DSRI: Days Sales in Receivables Index ---
    dsri_cur = _safe_divide(cur["receivables"], cur["revenue"])
    dsri_pri = _safe_divide(pri["receivables"], pri["revenue"])
    dsri = _safe_divide(dsri_cur, dsri_pri) if dsri_cur is not None and dsri_pri is not None else None

    # --- 2. GMI: Gross Margin Index ---
    gm_cur = _safe_divide(cur["revenue"] - cur["cogs"], cur["revenue"])
    gm_pri = _safe_divide(pri["revenue"] - pri["cogs"], pri["revenue"])
    gmi = _safe_divide(gm_pri, gm_cur) if gm_cur is not None and gm_pri is not None else None

    # --- 3. AQI: Asset Quality Index ---
    aq_cur = _safe_divide(cur["current_assets"] + cur["ppe"], cur["total_assets"])
    aq_pri = _safe_divide(pri["current_assets"] + pri["ppe"], pri["total_assets"])
    if aq_cur is not None and aq_pri is not None:
        aqi = _safe_divide(1 - aq_cur, 1 - aq_pri)
    else:
        aqi = None

    # --- 4. SGI: Sales Growth Index ---
    sgi = _safe_divide(cur["revenue"], pri["revenue"])

    # --- 5. DEPI: Depreciation Index ---
    dep_rate_cur = _safe_divide(cur["depreciation"], cur["depreciation"] + cur["ppe"])
    dep_rate_pri = _safe_divide(pri["depreciation"], pri["depreciation"] + pri["ppe"])
    depi = _safe_divide(dep_rate_pri, dep_rate_cur) if dep_rate_cur is not None and dep_rate_pri is not None else None

    # --- 6. SGAI: SG&A Expense Index ---
    sga_cur = _safe_divide(cur["sga"], cur["revenue"])
    sga_pri = _safe_divide(pri["sga"], pri["revenue"])
    sgai = _safe_divide(sga_cur, sga_pri) if sga_cur is not None and sga_pri is not None else None

    # --- 7. LVGI: Leverage Index ---
    lvg_cur = _safe_divide(cur["long_term_debt"] + cur["current_liabilities"], cur["total_assets"])
    lvg_pri = _safe_divide(pri["long_term_debt"] + pri["current_liabilities"], pri["total_assets"])
    lvgi = _safe_divide(lvg_cur, lvg_pri) if lvg_cur is not None and lvg_pri is not None else None

    # --- 8. TATA: Total Accruals to Total Assets ---
    tata = _safe_divide(cur["net_income"] - cur["cash_flow_ops"], cur["total_assets"])

    ratios = {
        "DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi,
        "DEPI": depi, "SGAI": sgai, "LVGI": lvgi, "TATA": tata,
    }

    # If any ratio couldn't be computed (e.g. a zero denominator), we
    # can't produce a reliable M-Score, so mark as insufficient data.
    if any(value is None for value in ratios.values()):
        result["insufficient_data"] = True
        result["reason"] = "One or more ratios could not be computed (zero denominator)."
        result["ratios"] = ratios
        return result

    m_score = (
        -4.84
        + 0.92 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )

    result["ratios"] = ratios
    result["m_score"] = m_score
    result["risk_tier"] = "Elevated" if m_score > MSCORE_THRESHOLD else "Low/Moderate"

    return result


def run_forensic_check(ticker_symbol):
    """
    Convenience function: fetches the data AND computes the score in
    one call. This is what the Streamlit app will actually call.
    """
    financial_data = get_financial_data(ticker_symbol)
    return compute_beneish_mscore(financial_data)


if __name__ == "__main__":
    test_tickers = ["AAPL", "MSFT", "TSLA"]

    for symbol in test_tickers:
        score_result = run_forensic_check(symbol)
        print(f"\n--- {symbol} ---")
        if score_result["insufficient_data"]:
            print(f"Insufficient data: {score_result['reason']}")
        else:
            print(f"M-Score: {score_result['m_score']:.2f}")
            print(f"Risk tier: {score_result['risk_tier']}")
            print("Ratios:")
            for name, value in score_result["ratios"].items():
                print(f"  {name}: {value:.3f}")
