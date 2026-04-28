"""
Data loading module for pairs trading project.

Pulls historical price data from Yahoo Finance and caches it locally
to avoid repeated API calls.
"""

import os
import pandas as pd
import yfinance as yf
from pathlib import Path


# Stock universe organized by sector
UNIVERSE = {
    "Technology": [
        "AAPL", "MSFT", "GOOGL", "META", "NVDA",
        "AMD", "INTC", "ORCL", "CRM", "ADBE"
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "C", "GS",
        "MS", "USB", "PNC", "TFC", "COF"
    ],
    "ConsumerStaples": [
    "KO", "PEP", "PG", "CL", "KMB",
    "GIS", "MKC", "HSY", "MDLZ", "CPB"  
],
    "Energy": [
    "XOM", "CVX", "COP", "SLB", "EOG",
    "PSX", "MPC", "VLO", "OXY", "BKR"  
]
}


def get_all_tickers() -> list[str]:
    """Return a flat list of all tickers across all sectors."""
    return [ticker for sector_tickers in UNIVERSE.values() for ticker in sector_tickers]


def get_ticker_to_sector() -> dict[str, str]:
    """Return a dict mapping each ticker to its sector."""
    return {
        ticker: sector
        for sector, tickers in UNIVERSE.items()
        for ticker in tickers
    }


def download_prices(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
    cache_path: str | Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download.
    start : str
        Start date in YYYY-MM-DD format.
    end : str or None
        End date in YYYY-MM-DD format. If None, uses today.
    cache_path : str or Path or None
        Where to cache the data. If file exists and force_refresh is False,
        loads from cache instead of redownloading.
    force_refresh : bool
        If True, redownload even if cache exists.

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and tickers as columns.
        Values are adjusted close prices.
    """
    cache_path = Path(cache_path) if cache_path else None

    # Try loading from cache first
    if cache_path and cache_path.exists() and not force_refresh:
        print(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    # Otherwise download fresh
    print(f"Downloading {len(tickers)} tickers from Yahoo Finance...")
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,  # use adjusted close (handles splits and dividends)
        progress=False,
    )

    # yfinance returns a multi-level column DataFrame.
    # Extract just the 'Close' prices (which are adjusted because auto_adjust=True).
    if len(tickers) == 1:
        # Single ticker: columns are flat
        prices = raw[["Close"]].copy()
        prices.columns = tickers
    else:
        # Multiple tickers: columns are MultiIndex (field, ticker)
        prices = raw["Close"].copy()

    # Drop any rows where ALL tickers are NaN 
    prices = prices.dropna(how="all")

    print(f"Downloaded {len(prices)} rows for {len(prices.columns)} tickers.")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")

    # Save to cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)
        print(f"Cached to {cache_path}")

    return prices


if __name__ == "__main__":
    # Quick test when running this file directly
    tickers = get_all_tickers()
    prices = download_prices(
        tickers=tickers,
        start="2018-01-01",
        cache_path="data/prices.csv",
    )
    print("\nFirst few rows:")
    print(prices.head())
    print("\nShape:", prices.shape)
    print("Missing data per ticker:")
    print(prices.isna().sum().sort_values(ascending=False).head(10))