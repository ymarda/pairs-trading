"""
Out-of-sample test of the pairs trading strategy.

Applies the strategy unchanged from in-sample to 2023-today.
This is the honest test of whether the strategy actually works.
"""

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
    compute_metrics,
)


def run_strategy_full_period(
    prices: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    signal_params: SignalParams,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Run the strategy over the full price history (in-sample + out-of-sample).

    The α, β are taken from selected_pairs (estimated on in-sample only).
    The rolling z-score uses past data only.
    Returns DataFrame with daily returns and cumulative return.
    """
    pair_returns = {}

    for _, row in selected_pairs.iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        alpha, beta = row["alpha"], row["beta"]
        pair_label = f"{a}/{b}"

        spread = compute_spread(prices[a], prices[b], alpha, beta)
        z = compute_rolling_zscore(spread, window=signal_params.rolling_window)
        signals = generate_signals(z, signal_params)
        pnl_df = compute_pair_pnl(
            prices[a], prices[b], alpha, beta, signals, transaction_cost_bps
        )

        pair_returns[pair_label] = pnl_df["return"]

    returns_df = pd.DataFrame(pair_returns).fillna(0.0)
    returns_df["portfolio_return"] = returns_df.mean(axis=1)
    return returns_df


def plot_combined_equity_curve(
    full_returns: pd.DataFrame,
    in_sample_end: str,
    save_path: Path,
):
    """
    Plot equity curve over full period, marking in-sample / out-of-sample boundary.
    """
    cumret = (1 + full_returns["portfolio_return"]).cumprod() - 1
    in_sample_mask = full_returns.index <= in_sample_end
    out_sample_mask = full_returns.index > in_sample_end

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: equity curve
    axes[0].plot(cumret.index[in_sample_mask], cumret[in_sample_mask],
                 color="navy", linewidth=1.2, label="In-sample (2018-2022)")
    axes[0].plot(cumret.index[out_sample_mask], cumret[out_sample_mask],
                 color="darkorange", linewidth=1.4, label="Out-of-sample (2023+)")
    axes[0].axvline(pd.Timestamp(in_sample_end), color="red", linestyle="--",
                    alpha=0.6, label="In-sample / out-of-sample boundary")
    axes[0].axhline(0, color="black", linestyle="-", alpha=0.3)
    axes[0].set_title("Pairs Trading Strategy — Full Equity Curve (with 5bps costs)")
    axes[0].set_ylabel("Cumulative return")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Bottom: drawdown over full period
    running_max = (1 + full_returns["portfolio_return"]).cumprod().cummax()
    cumret_total = (1 + full_returns["portfolio_return"]).cumprod()
    drawdown = (cumret_total - running_max) / running_max
    axes[1].fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
    axes[1].plot(drawdown.index, drawdown, color="darkred", linewidth=0.8)
    axes[1].axvline(pd.Timestamp(in_sample_end), color="red", linestyle="--", alpha=0.6)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")


def per_pair_oos_attribution(
    prices: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    in_sample_end: str,
    signal_params: SignalParams,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Compare each pair's in-sample vs out-of-sample Sharpe.

    Helps identify which pairs survived the regime shift and which broke down.
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

        in_sample_returns = pnl_df.loc[:in_sample_end, "return"]
        out_sample_returns = pnl_df.loc[pd.Timestamp(in_sample_end) + pd.Timedelta(days=1):, "return"]

        is_metrics = compute_metrics(in_sample_returns)
        oos_metrics = compute_metrics(out_sample_returns)

        rows.append({
            "pair": f"{a}/{b}",
            "is_sharpe": is_metrics["sharpe"],
            "oos_sharpe": oos_metrics["sharpe"],
            "is_return": is_metrics["annual_return"],
            "oos_return": oos_metrics["annual_return"],
        })

    df = pd.DataFrame(rows).set_index("pair")
    df = df.sort_values("oos_sharpe", ascending=False)
    return df


if __name__ == "__main__":
    # Load full data
    prices = download_prices(get_all_tickers(), cache_path="data/prices.csv")
    selected = pd.read_csv("data/selected_pairs.csv")

    in_sample_end = "2022-12-31"
    signal_params = SignalParams(z_entry=2.0, z_exit=0.0, z_stop=3.0, rolling_window=60)

    print("Running strategy on full period (in-sample + out-of-sample)...")
    full_returns = run_strategy_full_period(
        prices, selected, signal_params, transaction_cost_bps=5.0
    )

    # Split
    is_returns = full_returns.loc[:in_sample_end, "portfolio_return"]
    oos_start = pd.Timestamp(in_sample_end) + pd.Timedelta(days=1)
    oos_returns = full_returns.loc[oos_start:, "portfolio_return"]

    is_metrics = compute_metrics(is_returns)
    oos_metrics = compute_metrics(oos_returns)

    # Print comparison
    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE TEST RESULTS")
    print("=" * 70)
    print(f"In-sample period:    {is_returns.index.min().date()} to {is_returns.index.max().date()}")
    print(f"Out-of-sample period: {oos_returns.index.min().date()} to {oos_returns.index.max().date()}")
    print(f"In-sample days: {len(is_returns)}, Out-of-sample days: {len(oos_returns)}")
    print()
    print(f"{'Metric':<22} {'In-sample':>15} {'Out-of-sample':>17}")
    print("-" * 70)
    print(f"{'Annual return':<22} {is_metrics['annual_return']:>14.2%} {oos_metrics['annual_return']:>16.2%}")
    print(f"{'Annual vol':<22} {is_metrics['annual_vol']:>14.2%} {oos_metrics['annual_vol']:>16.2%}")
    print(f"{'Sharpe ratio':<22} {is_metrics['sharpe']:>15.2f} {oos_metrics['sharpe']:>17.2f}")
    print(f"{'Max drawdown':<22} {is_metrics['max_drawdown']:>14.2%} {oos_metrics['max_drawdown']:>16.2%}")
    print(f"{'Daily hit rate':<22} {is_metrics['hit_rate']:>14.2%} {oos_metrics['hit_rate']:>16.2%}")
    print(f"{'Profit factor':<22} {is_metrics['profit_factor']:>15.2f} {oos_metrics['profit_factor']:>17.2f}")
    print()

    # Verdict
    sharpe_decay = (is_metrics["sharpe"] - oos_metrics["sharpe"]) / is_metrics["sharpe"]
    print(f"Sharpe decay: {sharpe_decay:.1%}")
    if oos_metrics["sharpe"] > 1.0:
        verdict = "STRONG: Out-of-sample Sharpe > 1.0 — strategy holds up well."
    elif oos_metrics["sharpe"] > 0.5:
        verdict = "REASONABLE: Out-of-sample Sharpe 0.5-1.0 — strategy works but degraded."
    elif oos_metrics["sharpe"] > 0:
        verdict = "WEAK: Out-of-sample Sharpe < 0.5 — strategy barely positive."
    else:
        verdict = "FAILED: Out-of-sample Sharpe negative — strategy did not generalize."
    print(f"Verdict: {verdict}")

    # Plot
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plot_combined_equity_curve(full_returns, in_sample_end, plots_dir / "equity_curve_full.png")

    # Per-pair OOS attribution
    print("\nPer-pair in-sample vs out-of-sample comparison:")
    print("-" * 70)
    pair_comparison = per_pair_oos_attribution(
        prices, selected, in_sample_end, signal_params, transaction_cost_bps=5.0
    )
    print(pair_comparison.round(3).to_string())

    # Stats: how many pairs had positive OOS Sharpe?
    oos_positive = (pair_comparison["oos_sharpe"] > 0).sum()
    oos_strong = (pair_comparison["oos_sharpe"] > 1.0).sum()
    is_to_oos_correlation = pair_comparison["is_sharpe"].corr(pair_comparison["oos_sharpe"])

    print(f"\nPairs with positive OOS Sharpe: {oos_positive} of {len(pair_comparison)}")
    print(f"Pairs with OOS Sharpe > 1.0: {oos_strong}")
    print(f"Correlation between IS and OOS Sharpe: {is_to_oos_correlation:.3f}")
    print()
    if is_to_oos_correlation > 0.3:
        print("Positive IS→OOS correlation: in-sample Sharpe is somewhat predictive of OOS Sharpe.")
    elif is_to_oos_correlation > 0:
        print("Weak IS→OOS correlation: in-sample performance only weakly predicts OOS.")
    else:
        print("Negative IS→OOS correlation: best in-sample pairs are NOT the best out-of-sample.")
        print("This suggests in-sample selection was overfitting to noise.")

    # Save comparison
    pair_comparison.to_csv("data/oos_per_pair.csv")
    print(f"\nSaved per-pair OOS comparison to data/oos_per_pair.csv")

    # Save full returns
    full_returns.to_csv("data/full_period_returns.csv")
    print(f"Saved full period returns to data/full_period_returns.csv")