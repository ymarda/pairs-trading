"""
Robustness analysis for the pairs trading strategy.

Includes:
1. Per-pair return attribution
2. Parameter sensitivity (z_entry, z_stop, rolling_window)
3. Borrow cost impact
4. Drop-top-N analysis
"""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import download_prices, get_all_tickers
from src.signals import (
    SignalParams,
    compute_spread,
    compute_rolling_zscore,
    generate_signals,
)
from src.backtest import (
    compute_pair_pnl,
    compute_portfolio_pnl,
    compute_metrics,
)


# ---------------------------------------------------------------------
# 1. Per-pair return attribution
# ---------------------------------------------------------------------

def per_pair_attribution(
    prices: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    signal_params: SignalParams,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Compute total return, Sharpe, and other metrics for each pair separately.
    """
    rows = []
    for _, row in selected_pairs.iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        alpha, beta = row["alpha"], row["beta"]

        spread = compute_spread(prices[a], prices[b], alpha, beta)
        z = compute_rolling_zscore(spread, window=signal_params.rolling_window)
        signals = generate_signals(z, signal_params)
        pnl_df = compute_pair_pnl(
            prices[a], prices[b], alpha, beta, signals, transaction_cost_bps
        )

        metrics = compute_metrics(pnl_df["return"])
        metrics["pair"] = f"{a}/{b}"
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("pair")
    df = df.sort_values("sharpe", ascending=False)
    return df


# ---------------------------------------------------------------------
# 2. Parameter sensitivity
# ---------------------------------------------------------------------

def parameter_sensitivity(
    prices: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    z_entries: list[float],
    z_stops: list[float],
    rolling_windows: list[int],
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Sweep over parameter combinations and report Sharpe for each.
    """
    rows = []
    for z_entry, z_stop, window in product(z_entries, z_stops, rolling_windows):
        params = SignalParams(
            z_entry=z_entry,
            z_exit=0.0,
            z_stop=z_stop,
            rolling_window=window,
        )
        returns = compute_portfolio_pnl(
            prices, selected_pairs, params, transaction_cost_bps
        )
        metrics = compute_metrics(returns["portfolio_return"])
        rows.append({
            "z_entry": z_entry,
            "z_stop": z_stop,
            "rolling_window": window,
            "sharpe": metrics["sharpe"],
            "annual_return": metrics["annual_return"],
            "max_drawdown": metrics["max_drawdown"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. Borrow cost adjustment
# ---------------------------------------------------------------------

def apply_borrow_cost(
    portfolio_returns: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    annual_borrow_cost: float = 0.005,
) -> pd.DataFrame:
    """
    Apply a daily borrow cost on short positions.

    A pairs trade always has one short leg, so when in any position we
    pay borrow on roughly half the gross notional. Daily cost = annual / 252.
    Simplification: subtract a flat daily borrow cost for any day a pair is in position.

    Returns the modified portfolio_returns DataFrame.
    """
    daily_borrow = annual_borrow_cost / 252.0  # daily cost on $1 of short notional
    half_borrow = daily_borrow / 2.0  # half because dollar-neutral $2 notional, $1 short

    df = portfolio_returns.copy()
    pair_columns = [c for c in df.columns if c not in ["portfolio_return", "portfolio_cumret"]]

    # For each pair, when return != 0 (i.e., in position), subtract daily borrow
    for col in pair_columns:
        in_position = df[col] != 0
        df.loc[in_position, col] = df.loc[in_position, col] - half_borrow

    df["portfolio_return"] = df[pair_columns].mean(axis=1)
    df["portfolio_cumret"] = (1 + df["portfolio_return"]).cumprod() - 1
    return df


# ---------------------------------------------------------------------
# 4. Drop-top-N analysis
# ---------------------------------------------------------------------

def drop_top_n_analysis(
    prices: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    signal_params: SignalParams,
    transaction_cost_bps: float = 5.0,
    drop_counts: list[int] = [0, 1, 3, 5, 10],
) -> pd.DataFrame:
    """
    Compute Sharpe after dropping the top-N best pairs.

    If the strategy depends on a few outliers, dropping them will collapse Sharpe.
    A robust strategy degrades smoothly.
    """
    # Get per-pair attribution to rank pairs
    attribution = per_pair_attribution(
        prices, selected_pairs, signal_params, transaction_cost_bps
    )
    ranked_pairs = attribution.index.tolist()  # already sorted by Sharpe descending

    rows = []
    for drop_n in drop_counts:
        if drop_n >= len(ranked_pairs):
            continue
        kept_pairs_labels = ranked_pairs[drop_n:]
        kept_df = selected_pairs[
            selected_pairs.apply(
                lambda r: f"{r['ticker_a']}/{r['ticker_b']}" in kept_pairs_labels,
                axis=1,
            )
        ]
        returns = compute_portfolio_pnl(
            prices, kept_df, signal_params, transaction_cost_bps
        )
        metrics = compute_metrics(returns["portfolio_return"])
        rows.append({
            "drop_top_n": drop_n,
            "n_pairs": len(kept_df),
            "sharpe": metrics["sharpe"],
            "annual_return": metrics["annual_return"],
            "max_drawdown": metrics["max_drawdown"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    prices = download_prices(get_all_tickers(), cache_path="data/prices.csv")
    in_sample = prices.loc[:"2022-12-31"]
    selected = pd.read_csv("data/selected_pairs.csv")

    base_params = SignalParams(z_entry=2.0, z_exit=0.0, z_stop=3.0, rolling_window=60)

    print("=" * 60)
    print("ROBUSTNESS ANALYSIS — IN-SAMPLE (2018-2022)")
    print("=" * 60)

    # ---- 1. Per-pair attribution ----
    print("\n[1] Per-pair return attribution")
    print("-" * 60)
    attribution = per_pair_attribution(in_sample, selected, base_params)
    print(attribution[["sharpe", "annual_return", "max_drawdown", "hit_rate"]].round(3).to_string())

    # Show how concentrated returns are
    total_ret = attribution["annual_return"].sum()
    top3_share = attribution["annual_return"].head(3).sum() / total_ret if total_ret > 0 else 0
    top5_share = attribution["annual_return"].head(5).sum() / total_ret if total_ret > 0 else 0
    n_negative = (attribution["sharpe"] < 0).sum()
    print(f"\nTop 3 pairs contribute {top3_share:.1%} of total return")
    print(f"Top 5 pairs contribute {top5_share:.1%} of total return")
    print(f"Pairs with negative Sharpe: {n_negative} of {len(attribution)}")

    # ---- 2. Parameter sensitivity ----
    print("\n[2] Parameter sensitivity (varying z_entry, z_stop, rolling_window)")
    print("-" * 60)
    sensitivity = parameter_sensitivity(
        in_sample,
        selected,
        z_entries=[1.5, 2.0, 2.5],
        z_stops=[3.0, 4.0],
        rolling_windows=[40, 60, 90],
    )
    print(sensitivity.round(3).to_string(index=False))

    # ---- 3. Borrow cost impact ----
    print("\n[3] Borrow cost impact")
    print("-" * 60)
    base_returns = compute_portfolio_pnl(in_sample, selected, base_params, 5.0)
    base_metrics = compute_metrics(base_returns["portfolio_return"])

    for borrow_bps in [0, 25, 50, 100]:
        adjusted = apply_borrow_cost(base_returns, selected, annual_borrow_cost=borrow_bps / 10000)
        m = compute_metrics(adjusted["portfolio_return"])
        print(f"  Borrow cost = {borrow_bps:>3} bps annual: "
              f"Sharpe = {m['sharpe']:.2f}, Annual return = {m['annual_return']:.2%}")

    # ---- 4. Drop-top-N analysis ----
    print("\n[4] Drop top-N robustness")
    print("-" * 60)
    drop_results = drop_top_n_analysis(in_sample, selected, base_params)
    print(drop_results.round(3).to_string(index=False))

    # ---- Save attribution table ----
    out_path = Path("data/per_pair_attribution.csv")
    attribution.to_csv(out_path)
    print(f"\nSaved per-pair attribution to {out_path}")

    # ---- Plot attribution chart ----
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["green" if s > 0 else "red" for s in attribution["sharpe"]]
    ax.barh(range(len(attribution)), attribution["sharpe"], color=colors, alpha=0.7)
    ax.set_yticks(range(len(attribution)))
    ax.set_yticklabels(attribution.index, fontsize=9)
    ax.axvline(0, color="black", linestyle="-", alpha=0.4)
    ax.set_xlabel("Sharpe ratio (in-sample)")
    ax.set_title("Per-Pair Sharpe Ratios")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig("plots/per_pair_sharpe.png", dpi=120)
    plt.close(fig)
    print("Saved plots/per_pair_sharpe.png")