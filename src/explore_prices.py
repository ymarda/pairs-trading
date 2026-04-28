"""
Visual exploration of price data.

Generates plots that build intuition for what stock prices look like
and motivate why pairs trading works.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.data_loader import download_prices, get_all_tickers, UNIVERSE


def plot_sector(prices: pd.DataFrame, sector: str, save_dir: Path):
    """Plot all stocks in a single sector on one chart, normalized to 100 at start."""
    tickers = UNIVERSE[sector]
    sector_prices = prices[tickers].dropna()

    # Normalize each series to start at 100 so we can compare on the same axis
    normalized = (sector_prices / sector_prices.iloc[0]) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in tickers:
        ax.plot(normalized.index, normalized[ticker], label=ticker, linewidth=1)

    ax.set_title(f"{sector} sector — prices normalized to 100 at start")
    ax.set_ylabel("Price (normalized)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    save_path = save_dir / f"sector_{sector.lower()}.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_pair_example(prices: pd.DataFrame, ticker_a: str, ticker_b: str, save_dir: Path):
    """Plot two stocks side-by-side and their ratio, to build intuition for pair trading."""
    pair = prices[[ticker_a, ticker_b]].dropna()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: both prices, normalized
    norm = (pair / pair.iloc[0]) * 100
    axes[0].plot(norm.index, norm[ticker_a], label=ticker_a, linewidth=1.2)
    axes[0].plot(norm.index, norm[ticker_b], label=ticker_b, linewidth=1.2)
    axes[0].set_title(f"{ticker_a} vs {ticker_b} — normalized prices")
    axes[0].set_ylabel("Price (normalized to 100)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: their ratio
    ratio = pair[ticker_a] / pair[ticker_b]
    axes[1].plot(ratio.index, ratio, color="purple", linewidth=1.2)
    axes[1].axhline(ratio.mean(), color="black", linestyle="--", alpha=0.5, label=f"Mean = {ratio.mean():.2f}")
    axes[1].set_title(f"Ratio: {ticker_a} / {ticker_b}")
    axes[1].set_ylabel("Ratio")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    save_path = save_dir / f"pair_{ticker_a}_{ticker_b}.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    save_dir = Path("plots")
    save_dir.mkdir(exist_ok=True)

    # Load cached prices
    tickers = get_all_tickers()
    prices = download_prices(tickers=tickers, cache_path="data/prices.csv")

    # Plot each sector
    for sector in UNIVERSE.keys():
        plot_sector(prices, sector, save_dir)

    # Plot a few example pairs that might be cointegrated
    plot_pair_example(prices, "KO", "PEP", save_dir)      
    plot_pair_example(prices, "JPM", "BAC", save_dir)     
    plot_pair_example(prices, "XOM", "CVX", save_dir)     
    plot_pair_example(prices, "AAPL", "XOM", save_dir)    