"""
Cointegration testing module for pairs trading.

Implements the Engle-Granger two-step cointegration test:
1. OLS regression: A_t = alpha + beta * B_t + epsilon_t
2. ADF test on residuals epsilon_t

Also computes half-life of mean reversion for cointegrated pairs.
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


@dataclass
class CointegrationResult:
    """Container for cointegration test results between two assets."""
    ticker_a: str
    ticker_b: str
    alpha: float
    beta: float
    pvalue: float
    half_life: float | None  # None if not mean-reverting (lambda >= 0)
    is_cointegrated: bool   # True if pvalue < threshold AND half_life is reasonable


def engle_granger(
    series_a: pd.Series,
    series_b: pd.Series,
    pvalue_threshold: float = 0.05,
) -> CointegrationResult:
    """
    Run the Engle-Granger cointegration test between two price series.

    Parameters
    ----------
    series_a, series_b : pd.Series
        Price series with the same DatetimeIndex.
    pvalue_threshold : float
        ADF p-value below which we declare cointegration.

    Returns
    -------
    CointegrationResult
        Test results including alpha, beta, p-value, half-life, and cointegration flag.
    """
    # Align series and drop NaNs
    df = pd.concat([series_a, series_b], axis=1).dropna()
    df.columns = ["A", "B"]

    if len(df) < 100:
        # Not enough data for a meaningful test
        return CointegrationResult(
            ticker_a=series_a.name, ticker_b=series_b.name,
            alpha=np.nan, beta=np.nan, pvalue=1.0,
            half_life=None, is_cointegrated=False
        )

    # Step A: OLS regression A = alpha + beta * B + epsilon
    X = sm.add_constant(df["B"])
    model = sm.OLS(df["A"], X).fit()
    alpha = model.params["const"]
    beta = model.params["B"]
    residuals = model.resid

    # Step B: ADF test on residuals
    adf_result = adfuller(residuals, autolag="AIC")
    pvalue = adf_result[1]

    # Compute half-life of mean reversion if pair passes p-value threshold
    half_life = None
    if pvalue < pvalue_threshold:
        half_life = compute_half_life(residuals)

    # Determine if this is a tradeable cointegrated pair
    is_cointegrated = (
        pvalue < pvalue_threshold
        and half_life is not None
        and 1 < half_life < 252  # between 1 day and 1 year
    )

    return CointegrationResult(
        ticker_a=series_a.name,
        ticker_b=series_b.name,
        alpha=alpha,
        beta=beta,
        pvalue=pvalue,
        half_life=half_life,
        is_cointegrated=is_cointegrated,
    )


def compute_half_life(spread: pd.Series) -> float | None:
    """
    Estimate half-life of mean reversion from a spread series.

    Fits delta_X_t = c + lambda * X_{t-1} + epsilon_t via OLS.
    Half-life = ln(2) / -lambda.
    Returns None if lambda >= 0 (not mean-reverting).
    """
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()

    # Align after shifts
    aligned = pd.concat([spread_diff, spread_lag], axis=1).dropna()
    aligned.columns = ["delta", "lag"]

    X = sm.add_constant(aligned["lag"])
    model = sm.OLS(aligned["delta"], X).fit()
    lam = model.params["lag"]

    if lam >= 0:
        # Not mean-reverting — process is explosive or random walk
        return None

    return float(np.log(2) / -lam)


def test_all_pairs(
    prices: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run Engle-Granger on every pair of columns in `prices`.

    Returns a DataFrame with columns:
        ticker_a, ticker_b, alpha, beta, pvalue, half_life, is_cointegrated
    sorted by pvalue ascending.
    """
    tickers = list(prices.columns)
    n_pairs = len(tickers) * (len(tickers) - 1) // 2
    if verbose:
        print(f"Testing {n_pairs} pairs across {len(tickers)} tickers...")

    results = []
    for i, (ta, tb) in enumerate(combinations(tickers, 2)):
        result = engle_granger(
            prices[ta],
            prices[tb],
            pvalue_threshold=pvalue_threshold,
        )
        results.append(result)

        if verbose and (i + 1) % 100 == 0:
            print(f"  ...tested {i + 1}/{n_pairs} pairs")

    df = pd.DataFrame([
        {
            "ticker_a": r.ticker_a,
            "ticker_b": r.ticker_b,
            "alpha": r.alpha,
            "beta": r.beta,
            "pvalue": r.pvalue,
            "half_life": r.half_life,
            "is_cointegrated": r.is_cointegrated,
        }
        for r in results
    ])

    df = df.sort_values("pvalue").reset_index(drop=True)
    return df


if __name__ == "__main__":
    from src.data_loader import download_prices, get_all_tickers, get_ticker_to_sector

    # Load data
    prices = download_prices(
        tickers=get_all_tickers(),
        cache_path="data/prices.csv",
    )

    # In-sample / out-of-sample split
    in_sample_end = "2022-12-31"
    in_sample = prices.loc[:in_sample_end]
    print(f"In-sample period: {in_sample.index.min().date()} to {in_sample.index.max().date()}")
    print(f"In-sample rows: {len(in_sample)}")
    print()

    # Sanity check on the four pairs we discussed
    print("=" * 60)
    print("Sanity check: testing the 4 pairs from explore_prices.py")
    print("=" * 60)
    for ta, tb in [("KO", "PEP"), ("JPM", "BAC"), ("XOM", "CVX"), ("AAPL", "XOM")]:
        result = engle_granger(in_sample[ta], in_sample[tb])
        hl_str = f"{result.half_life:.1f} days" if result.half_life else "N/A"
        coint_str = "✓" if result.is_cointegrated else "✗"
        print(f"  {ta:5s} / {tb:5s}: p = {result.pvalue:.4f}, "
              f"β = {result.beta:.3f}, half-life = {hl_str}, "
              f"cointegrated: {coint_str}")
    print()

    # Full pair sweep
    print("=" * 60)
    print("Running full pair sweep on in-sample data...")
    print("=" * 60)
    results_df = test_all_pairs(in_sample, pvalue_threshold=0.05)

    # Summary stats
    n_cointegrated = results_df["is_cointegrated"].sum()
    n_passed_pvalue = (results_df["pvalue"] < 0.05).sum()
    print(f"\nTotal pairs tested: {len(results_df)}")
    print(f"Pairs with p < 0.05: {n_passed_pvalue}")
    print(f"Pairs with p < 0.05 AND tradeable half-life: {n_cointegrated}")

    # Annotate with sector info
    sectors = get_ticker_to_sector()
    results_df["sector_a"] = results_df["ticker_a"].map(sectors)
    results_df["sector_b"] = results_df["ticker_b"].map(sectors)
    results_df["same_sector"] = results_df["sector_a"] == results_df["sector_b"]

    # Save full results
    out_path = Path("data/cointegration_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved full results to {out_path}")

    # Show top tradeable pairs
    tradeable = results_df[results_df["is_cointegrated"]].head(20)
    print("\nTop 20 tradeable cointegrated pairs (by p-value):")
    print(tradeable[["ticker_a", "ticker_b", "pvalue", "beta", "half_life", "same_sector"]].to_string(index=False))

    # Show what fraction of tradeable pairs are within-sector
    if n_cointegrated > 0:
        same_sector_frac = results_df[results_df["is_cointegrated"]]["same_sector"].mean()
        print(f"\nFraction of tradeable pairs within the same sector: {same_sector_frac:.1%}")