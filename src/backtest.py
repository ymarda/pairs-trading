"""
Backtesting module for pairs trading.

Takes signals from signals.py and computes daily P&L, aggregates across
pairs, and produces performance metrics (Sharpe, drawdown, hit rate, etc.).
"""

from dataclasses import dataclass
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


@dataclass
class BacktestParams:
    """Parameters controlling the backtest."""
    transaction_cost_bps: float = 5.0  # Round-trip cost per trade in basis points (1 bp = 0.01%)
    initial_capital: float = 1.0       # Dollar capital per pair 


def compute_pair_pnl(
    series_a: pd.Series,
    series_b: pd.Series,
    alpha: float,
    beta: float,
    signals: pd.DataFrame,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Compute daily returns for a single pair given trading signals.

    Uses dollar-neutral pairs trading:
        long_spread  → +$1 in A, -$1 in B → daily return = r_A - r_B
        short_spread → -$1 in A, +$1 in B → daily return = r_B - r_A

    Capital deployed per trade = $2 (one dollar long, one dollar short).
    Returns are computed as profit on the $2 of gross notional.
    """
    df = pd.concat([series_a, series_b], axis=1).dropna()
    df.columns = ["A", "B"]
    df = df.join(signals[["position", "signal"]], how="inner")

    # Daily log returns of each leg
    df["ret_A"] = np.log(df["A"] / df["A"].shift(1))
    df["ret_B"] = np.log(df["B"] / df["B"].shift(1))

    # Position held *yesterday* earns today's return (no look-ahead)
    prev_position = df["position"].shift(1)

    # Dollar-neutral: long A short B (long_spread), or short A long B (short_spread)
    # Return is on $2 of gross notional, so divide by 2 to get return on capital
    df["gross_return"] = 0.0
    df.loc[prev_position == "long_spread", "gross_return"] = (df["ret_A"] - df["ret_B"]) / 2
    df.loc[prev_position == "short_spread", "gross_return"] = (df["ret_B"] - df["ret_A"]) / 2

    # Transaction costs apply when a trade closes (round-trip cost on $2 notional)
    df["cost"] = 0.0
    cost_pct = (transaction_cost_bps / 10000.0)  # bps to decimal, applied to gross capital
    closing_signals = df["signal"].isin(["exit", "stop_loss"])
    df.loc[closing_signals, "cost"] = cost_pct

    # Net return after costs
    df["return"] = df["gross_return"] - df["cost"]

    # Cumulative compounded return
    df["cumulative_return"] = (1 + df["return"]).cumprod() - 1

    return df


def compute_portfolio_pnl(
    prices: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    signal_params: SignalParams,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Run signal generation + P&L computation for all selected pairs and aggregate.

    Returns a DataFrame indexed by date with:
        per-pair return columns (one per pair)
        portfolio_return : equal-weighted average across all pairs
        portfolio_cumret : compounded portfolio return
    """
    pair_returns = {}

    for _, row in selected_pairs.iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        alpha, beta = row["alpha"], row["beta"]
        pair_label = f"{a}/{b}"

        # Generate signals
        spread = compute_spread(prices[a], prices[b], alpha, beta)
        z = compute_rolling_zscore(spread, window=signal_params.rolling_window)
        signals = generate_signals(z, signal_params)

        # Compute pair P&L
        pnl_df = compute_pair_pnl(
            prices[a], prices[b], alpha, beta, signals, transaction_cost_bps
        )

        pair_returns[pair_label] = pnl_df["return"]

    returns_df = pd.DataFrame(pair_returns)
    returns_df = returns_df.fillna(0.0)

    # Equal-weighted portfolio return
    returns_df["portfolio_return"] = returns_df.mean(axis=1)
    returns_df["portfolio_cumret"] = (1 + returns_df["portfolio_return"]).cumprod() - 1

    return returns_df


def compute_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Compute key performance metrics from a daily return series."""
    r = returns.dropna()

    # Annualized return and volatility
    mean_ret = r.mean()
    std_ret = r.std()
    annual_ret = mean_ret * periods_per_year
    annual_vol = std_ret * np.sqrt(periods_per_year)

    # Sharpe (assuming risk-free rate = 0)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0.0

    # Total return and CAGR
    cumret = (1 + r).cumprod()
    total_return = cumret.iloc[-1] - 1
    n_years = len(r) / periods_per_year
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Max drawdown
    running_max = cumret.cummax()
    drawdown = (cumret - running_max) / running_max
    max_dd = drawdown.min()

    # Hit rate (% of days with positive return)
    nonzero_days = r[r != 0]
    hit_rate = (nonzero_days > 0).mean() if len(nonzero_days) > 0 else 0

    # Profit factor
    gross_profit = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "annual_return": annual_ret,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "cagr": cagr,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
    }


def plot_equity_curve(
    returns_no_costs: pd.DataFrame,
    returns_with_costs: pd.DataFrame,
    save_path: Path,
):
    """Plot equity curves (cumulative returns) with and without costs."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: equity curves
    axes[0].plot(returns_no_costs.index, (1 + returns_no_costs["portfolio_return"]).cumprod() - 1,
                 label="No transaction costs", color="green", linewidth=1.2)
    axes[0].plot(returns_with_costs.index, (1 + returns_with_costs["portfolio_return"]).cumprod() - 1,
                 label="With 5 bps round-trip costs", color="navy", linewidth=1.2)
    axes[0].axhline(0, color="black", linestyle="--", alpha=0.4)
    axes[0].set_title("Pairs Trading Strategy — In-Sample Equity Curve")
    axes[0].set_ylabel("Cumulative return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: drawdown
    cumret_costs = (1 + returns_with_costs["portfolio_return"]).cumprod()
    drawdown = (cumret_costs - cumret_costs.cummax()) / cumret_costs.cummax()
    axes[1].fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
    axes[1].plot(drawdown.index, drawdown, color="darkred", linewidth=0.8)
    axes[1].set_title("Drawdown (with costs)")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    # Load data
    prices = download_prices(get_all_tickers(), cache_path="data/prices.csv")
    in_sample = prices.loc[:"2022-12-31"]
    selected = pd.read_csv("data/selected_pairs.csv")

    signal_params = SignalParams(z_entry=2.0, z_exit=0.0, z_stop=3.0, rolling_window=60)

    print(f"Backtesting strategy on {len(selected)} pairs over {len(in_sample)} days")
    print(f"In-sample period: {in_sample.index.min().date()} to {in_sample.index.max().date()}\n")

    # Run two versions: no costs and with realistic costs
    print("Running without transaction costs...")
    returns_no_costs = compute_portfolio_pnl(
        in_sample, selected, signal_params, transaction_cost_bps=0.0
    )
    metrics_no_costs = compute_metrics(returns_no_costs["portfolio_return"])

    print("Running with 5 bps round-trip transaction costs...")
    returns_with_costs = compute_portfolio_pnl(
        in_sample, selected, signal_params, transaction_cost_bps=5.0
    )
    metrics_with_costs = compute_metrics(returns_with_costs["portfolio_return"])

    # Print comparison
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS — IN-SAMPLE (2018-2022)")
    print("=" * 60)
    print(f"{'Metric':<22} {'No Costs':>15} {'With Costs':>15}")
    print("-" * 60)
    print(f"{'Annual return':<22} {metrics_no_costs['annual_return']:>14.2%} {metrics_with_costs['annual_return']:>14.2%}")
    print(f"{'Annual vol':<22} {metrics_no_costs['annual_vol']:>14.2%} {metrics_with_costs['annual_vol']:>14.2%}")
    print(f"{'Sharpe ratio':<22} {metrics_no_costs['sharpe']:>15.2f} {metrics_with_costs['sharpe']:>15.2f}")
    print(f"{'CAGR':<22} {metrics_no_costs['cagr']:>14.2%} {metrics_with_costs['cagr']:>14.2%}")
    print(f"{'Total return':<22} {metrics_no_costs['total_return']:>14.2%} {metrics_with_costs['total_return']:>14.2%}")
    print(f"{'Max drawdown':<22} {metrics_no_costs['max_drawdown']:>14.2%} {metrics_with_costs['max_drawdown']:>14.2%}")
    print(f"{'Daily hit rate':<22} {metrics_no_costs['hit_rate']:>14.2%} {metrics_with_costs['hit_rate']:>14.2%}")
    print(f"{'Profit factor':<22} {metrics_no_costs['profit_factor']:>15.2f} {metrics_with_costs['profit_factor']:>15.2f}")
    print()

    # Plot equity curve
    save_dir = Path("plots")
    save_dir.mkdir(exist_ok=True)
    plot_equity_curve(returns_no_costs, returns_with_costs, save_dir / "equity_curve_in_sample.png")

    # Save daily returns for later analysis
    out_path = Path("data/backtest_returns_in_sample.csv")
    returns_with_costs.to_csv(out_path)
    print(f"Saved daily returns to {out_path}")