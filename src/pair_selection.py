"""
Filtering and selection of cointegrated pairs.

Takes raw cointegration test results and applies economic + statistical
filters to produce a tradeable universe of pairs.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import download_prices, get_all_tickers, get_ticker_to_sector


def filter_pairs(
    results_df: pd.DataFrame,
    pvalue_max: float = 0.05,
    half_life_min: float = 5,
    half_life_max: float = 60,
    require_same_sector: bool = True,
    beta_min: float = 0.1,
    beta_max: float = 10.0,
) -> pd.DataFrame:
    """
    Apply filters to the raw cointegration results.

    Returns a filtered DataFrame ranked by p-value.
    """
    n_total = len(results_df)
    df = results_df.copy()

    # 1. Statistical
    df = df[df["pvalue"] < pvalue_max]
    n_after_pvalue = len(df)

    # 2. Half-life
    df = df[(df["half_life"] >= half_life_min) & (df["half_life"] <= half_life_max)]
    n_after_halflife = len(df)

    # 3. Sector
    if require_same_sector:
        df = df[df["same_sector"]]
    n_after_sector = len(df)

    # 4. Beta sanity (use absolute value to handle negative betas)
    df = df[df["beta"].abs().between(beta_min, beta_max)]
    n_after_beta = len(df)

    # Rank by p-value
    df = df.sort_values("pvalue").reset_index(drop=True)

    print(f"Filtering pipeline:")
    print(f"  Started with: {n_total} pairs")
    print(f"  After p < {pvalue_max}: {n_after_pvalue}")
    print(f"  After half-life in [{half_life_min}, {half_life_max}]: {n_after_halflife}")
    print(f"  After same-sector: {n_after_sector}")
    print(f"  After beta in [{beta_min}, {beta_max}]: {n_after_beta}")

    return df


def plot_spread(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    alpha: float,
    beta: float,
    save_dir: Path,
):
    """
    Plot the cointegration spread (residuals from the regression) over time.

    Spread_t = A_t - alpha - beta * B_t

    A cointegrated pair should show this spread oscillating around zero.
    """
    df = pd.concat([prices[ticker_a], prices[ticker_b]], axis=1).dropna()
    df.columns = ["A", "B"]
    spread = df["A"] - alpha - beta * df["B"]

    # Compute z-score for visualization (we'll formalize this in Step 5)
    z = (spread - spread.mean()) / spread.std()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top: raw spread
    axes[0].plot(spread.index, spread, color="navy", linewidth=1)
    axes[0].axhline(spread.mean(), color="black", linestyle="--", alpha=0.5, label=f"Mean = {spread.mean():.2f}")
    axes[0].axhline(spread.mean() + spread.std(), color="red", linestyle=":", alpha=0.5, label="±1 std")
    axes[0].axhline(spread.mean() - spread.std(), color="red", linestyle=":", alpha=0.5)
    axes[0].set_title(f"Spread: {ticker_a} - {alpha:.2f} - {beta:.3f} × {ticker_b}")
    axes[0].set_ylabel("Spread (USD)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: z-score
    axes[1].plot(z.index, z, color="purple", linewidth=1)
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[1].axhline(2, color="red", linestyle=":", alpha=0.5, label="±2 (entry threshold)")
    axes[1].axhline(-2, color="red", linestyle=":", alpha=0.5)
    axes[1].set_title(f"Z-score of spread")
    axes[1].set_ylabel("Z-score")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    save_path = save_dir / f"spread_{ticker_a}_{ticker_b}.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {save_path}")


if __name__ == "__main__":
    # Load raw cointegration results
    raw_path = Path("data/cointegration_results.csv")
    if not raw_path.exists():
        print(f"Error: {raw_path} not found. Run src/cointegration.py first.")
        raise SystemExit(1)

    raw = pd.read_csv(raw_path)
    print(f"Loaded {len(raw)} raw test results from {raw_path}\n")

    # Apply filters
    selected = filter_pairs(
        raw,
        pvalue_max=0.05,
        half_life_min=5,
        half_life_max=60,
        require_same_sector=True,
        beta_min=0.1,
        beta_max=10.0,
    )

    print(f"\nSelected {len(selected)} tradeable pairs:")
    print(selected[["ticker_a", "ticker_b", "pvalue", "beta", "half_life", "sector_a"]].to_string(index=False))

    # Save selection
    out_path = Path("data/selected_pairs.csv")
    selected.to_csv(out_path, index=False)
    print(f"\nSaved selected pairs to {out_path}")

    # Visualize the top 5 selected pairs' spreads
    print("\nGenerating spread plots for top 5 pairs...")
    save_dir = Path("plots")
    save_dir.mkdir(exist_ok=True)

    # Reload in-sample prices
    prices = download_prices(get_all_tickers(), cache_path="data/prices.csv")
    in_sample = prices.loc[:"2022-12-31"]

    for _, row in selected.head(5).iterrows():
        plot_spread(
            in_sample,
            row["ticker_a"],
            row["ticker_b"],
            row["alpha"],
            row["beta"],
            save_dir,
        )

    print("\nDone.")