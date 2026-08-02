"""
forensic_data_fetch.py

Step 1 of the Beneish M-Score module: pull two years of financial
statement data per ticker from yfinance, and extract the specific
line items needed to compute the 8 Beneish ratios.

If any required line item is missing for a ticker, the function
returns insufficient_data=True instead of crashing, so the app can
display "Insufficient data" for that ticker.
"""

import yfinance as yf

# Each financial statement line item can appear under slightly different
# labels depending on the company / yfinance version. We check each
# possible label in order and use whichever one is found first.
FIELD_ALIASES = {
    "revenue": ["Total Revenue"],
    "receivables": ["Net Receivables", "Receivables", "Accounts Receivable"],
    "cogs": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "ppe": ["Net PPE", "Property Plant And Equipment Net", "Property Plant Equipment"],
    "total_assets": ["Total Assets"],
    "depreciation": ["Reconciled Depreciation", "Depreciation And Amortization In Income Statement", "Depreciation Amortization Depletion"],
    "sga": ["Selling General And Administration", "Selling General Administrative"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "cash_flow_ops": ["Operating Cash Flow", "Total Cash From Operating Activities"],
    "long_term_debt": ["Long Term Debt"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
}


def _find_value(df, aliases, column):
    """
    Look through a DataFrame's row index for the first matching alias,
    and return the value in the given column (a year/date).
    Returns None if none of the aliases are found.
    """
    for alias in aliases:
        if alias in df.index:
            value = df.loc[alias, column]
            if value is not None and not _is_nan(value):
                return float(value)
    return None


def _is_nan(value):
    try:
        return value != value  # NaN != NaN is True
    except TypeError:
        return False


def get_financial_data(ticker_symbol):
    """
    Pull two years of income statement + balance sheet + cash flow data
    for a single ticker, and extract the fields needed for the Beneish
    M-Score.

    Returns a dict with:
        - ticker
        - insufficient_data (bool)
        - missing_fields (list of field names that couldn't be found)
        - current_year: dict of field -> value
        - prior_year: dict of field -> value
    """
    result = {
        "ticker": ticker_symbol,
        "insufficient_data": False,
        "missing_fields": [],
        "current_year": {},
        "prior_year": {},
    }

    try:
        stock = yf.Ticker(ticker_symbol)
        income_stmt = stock.financials          # annual income statement
        balance_sheet = stock.balance_sheet      # annual balance sheet
        cash_flow = stock.cashflow               # annual cash flow statement
    except Exception:
        result["insufficient_data"] = True
        result["missing_fields"] = ["could not fetch statements"]
        return result

    # We need at least 2 years (columns) of data to compute ratios
    if income_stmt.shape[1] < 2 or balance_sheet.shape[1] < 2 or cash_flow.shape[1] < 2:
        result["insufficient_data"] = True
        result["missing_fields"] = ["fewer than 2 years of statements available"]
        return result

    # Columns are sorted most-recent-first by default in yfinance
    current_col = income_stmt.columns[0]
    prior_col = income_stmt.columns[1]

    # Map each field to which statement it lives in
    field_sources = {
        "revenue": income_stmt,
        "receivables": balance_sheet,
        "cogs": income_stmt,
        "current_assets": balance_sheet,
        "ppe": balance_sheet,
        "total_assets": balance_sheet,
        "depreciation": cash_flow,
        "sga": income_stmt,
        "net_income": income_stmt,
        "cash_flow_ops": cash_flow,
        "long_term_debt": balance_sheet,
        "current_liabilities": balance_sheet,
    }

    for field, aliases in FIELD_ALIASES.items():
        source_df = field_sources[field]

        # balance_sheet and cash_flow may have their own current/prior
        # columns that don't perfectly line up with income_stmt's dates,
        # so we grab by position (0 = most recent, 1 = prior year) instead.
        source_current_col = source_df.columns[0]
        source_prior_col = source_df.columns[1]

        current_value = _find_value(source_df, aliases, source_current_col)
        prior_value = _find_value(source_df, aliases, source_prior_col)

        if current_value is None or prior_value is None:
            result["missing_fields"].append(field)
        else:
            result["current_year"][field] = current_value
            result["prior_year"][field] = prior_value

    if result["missing_fields"]:
        result["insufficient_data"] = True

    return result


if __name__ == "__main__":
    # Quick manual test — run this file directly to sanity-check on a
    # couple of tickers before wiring it into the Streamlit app.
    test_tickers = ["AAPL", "MSFT", "TSLA"]

    for symbol in test_tickers:
        data = get_financial_data(symbol)
        print(f"\n--- {symbol} ---")
        if data["insufficient_data"]:
            print(f"Insufficient data. Missing: {data['missing_fields']}")
        else:
            print("All required fields found.")
            print("Current year:", data["current_year"])
            print("Prior year:", data["prior_year"])
