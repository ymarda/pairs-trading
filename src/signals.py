"""
Signal generation for pairs trading.

Converts cointegration spreads into trading signals (long/short/flat)
using rolling z-score and a state machine for position management.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import download_prices, get_all_tickers


@dataclass
class SignalParams:
    """Parameters controlling signal generation."""
    z_entry: float = 2.0       # Enter position when |z| > z_entry
    z_exit: float = 0.0        # Exit when |z| crosses below z_exit
    z_stop: float = 3.0        # Stop-loss when |z| > z_stop
    rolling_window: int = 60   # Lookback for rolling mean/std


def compute_spread(
    series_a: pd.Series,
    series_b: pd.Series,
    alpha: float,
    beta: float,
) -> pd.Series:
    """Compute the cointegration spread S_t = A_t - alpha - beta * B_t."""
    df = pd.concat([series_a, series_b], axis=1).dropna()
    df.columns = ["A", "B"]
    return df["A"] - alpha - beta * df["B"]


def compute_rolling_zscore(
    spread: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Compute z-score using a rolling window — uses only past information.

    Z_t = (S_t - rolling_mean(S, window)_t) / rolling_std(S, window)_t

    The first `window` values will be NaN since we need that much history.
    """
    rolling_mean = spread.rolling(window=window, min_periods=window).mean()
    rolling_std = spread.rolling(window=window, min_periods=window).std()
    z = (spread - rolling_mean) / rolling_std
    return z


def generate_signals(
    z_score: pd.Series,
    params: SignalParams,
) -> pd.DataFrame:
    """
    Generate trading signals from a z-score series using the state machine.

    Returns a DataFrame with columns:
        zscore   : the input z-score
        position : the position state ('flat', 'long_spread', 'short_spread')
        signal   : trade events ('enter_long', 'enter_short', 'exit', 'stop_loss', None)
    """
    df = pd.DataFrame({"zscore": z_score})
    df["position"] = "flat"
    df["signal"] = None

    current_position = "flat"

    for i in range(len(df)):
        z = df["zscore"].iloc[i]

        # Skip if z-score is NaN (within rolling window warm-up)
        if pd.isna(z):
            df.iloc[i, df.columns.get_loc("position")] = "flat"
            continue

        if current_position == "flat":
            if z < -params.z_entry:
                current_position = "long_spread"
                df.iloc[i, df.columns.get_loc("signal")] = "enter_long"
            elif z > params.z_entry:
                current_position = "short_spread"
                df.iloc[i, df.columns.get_loc("signal")] = "enter_short"

        elif current_position == "long_spread":
            # Stop-loss: spread blew out further
            if z < -params.z_stop:
                current_position = "flat"
                df.iloc[i, df.columns.get_loc("signal")] = "stop_loss"
            # Exit: spread reverted to mean
            elif z >= -params.z_exit:
                current_position = "flat"
                df.iloc[i, df.columns.get_loc("signal")] = "exit"

        elif current_position == "short_spread":
            # Stop-loss
            if z > params.z_stop:
                current_position = "flat"
                df.iloc[i, df.columns.get_loc("signal")] = "stop_loss"
            # Exit
            elif z <= params.z_exit:
                current_position = "flat"
                df.iloc[i, df.columns.get_loc("signal")] = "exit"

        df.iloc[i, df.columns.get_loc("position")] = current_position

    return df


def plot_signals(
    pair_a: str,
    pair_b: str,
    z_score: pd.Series,
    signals: pd.DataFrame,
    params: SignalParams,
    save_dir: Path,
):
    """
    Visualize z-score, thresholds, and trade entries/exits for a pair.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot z-score
    ax.plot(signals.index, signals["zscore"], color="purple", linewidth=0.8, label="Z-score")

    # Plot thresholds
    ax.axhline(0, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(params.z_entry, color="red", linestyle=":", alpha=0.6, label=f"±{params.z_entry} entry")
    ax.axhline(-params.z_entry, color="red", linestyle=":", alpha=0.6)
    ax.axhline(params.z_stop, color="darkred", linestyle="--", alpha=0.4, label=f"±{params.z_stop} stop")
    ax.axhline(-params.z_stop, color="darkred", linestyle="--", alpha=0.4)

    # Mark trade events
    enter_long = signals[signals["signal"] == "enter_long"]
    enter_short = signals[signals["signal"] == "enter_short"]
    exits = signals[signals["signal"] == "exit"]
    stops = signals[signals["signal"] == "stop_loss"]

    ax.scatter(enter_long.index, enter_long["zscore"],
               color="green", marker="^", s=80, zorder=5, label="Long entry")
    ax.scatter(enter_short.index, enter_short["zscore"],
               color="red", marker="v", s=80, zorder=5, label="Short entry")
    ax.scatter(exits.index, exits["zscore"],
               color="blue", marker="o", s=40, zorder=5, label="Exit")
    ax.scatter(stops.index, stops["zscore"],
               color="black", marker="x", s=80, zorder=5, label="Stop loss")

    ax.set_title(f"{pair_a} / {pair_b} — trading signals (rolling z-score, window={params.rolling_window})")
    ax.set_ylabel("Z-score")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    save_path = save_dir / f"signals_{pair_a}_{pair_b}.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {save_path}")


def summarize_trades(signals: pd.DataFrame) -> dict:
    """Compute summary statistics from a signal DataFrame."""
    n_long = (signals["signal"] == "enter_long").sum()
    n_short = (signals["signal"] == "enter_short").sum()
    n_exits = (signals["signal"] == "exit").sum()
    n_stops = (signals["signal"] == "stop_loss").sum()

    days_in_position = (signals["position"] != "flat").sum()
    total_days = len(signals.dropna(subset=["zscore"]))
    pct_in_position = days_in_position / total_days if total_days > 0 else 0

    return {
        "long_entries": int(n_long),
        "short_entries": int(n_short),
        "total_entries": int(n_long + n_short),
        "exits": int(n_exits),
        "stop_losses": int(n_stops),
        "days_in_position": int(days_in_position),
        "pct_time_in_position": float(pct_in_position),
    }


if __name__ == "__main__":
    # Load data and selected pairs
    selected = pd.read_csv("data/selected_pairs.csv")
    prices = download_prices(get_all_tickers(), cache_path="data/prices.csv")
    in_sample = prices.loc[:"2022-12-31"]

    params = SignalParams(z_entry=2.0, z_exit=0.0, z_stop=3.0, rolling_window=60)

    save_dir = Path("plots")
    save_dir.mkdir(exist_ok=True)

    print(f"Generating signals for {len(selected)} pairs with parameters:")
    print(f"  z_entry = {params.z_entry}, z_exit = {params.z_exit}, z_stop = {params.z_stop}")
    print(f"  rolling_window = {params.rolling_window}")
    print()

    summary_rows = []

    for _, row in selected.iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        alpha, beta = row["alpha"], row["beta"]

        spread = compute_spread(in_sample[a], in_sample[b], alpha, beta)
        z = compute_rolling_zscore(spread, window=params.rolling_window)
        signals = generate_signals(z, params)

        stats = summarize_trades(signals)
        stats["pair"] = f"{a}/{b}"
        summary_rows.append(stats)

        # Plot the top 5
        if len(summary_rows) <= 5:
            plot_signals(a, b, z, signals, params, save_dir)

    # Print summary table
    summary_df = pd.DataFrame(summary_rows).set_index("pair")
    summary_df = summary_df[["total_entries", "long_entries", "short_entries",
                              "exits", "stop_losses", "days_in_position", "pct_time_in_position"]]
    print("\nSignal generation summary (in-sample):")
    print(summary_df.to_string())

    # Aggregate stats
    total_trades = summary_df["total_entries"].sum()
    total_stops = summary_df["stop_losses"].sum()
    avg_pct_in_position = summary_df["pct_time_in_position"].mean()

    years = len(in_sample) / 252
    print(f"\nAggregate across {len(selected)} pairs over {years:.1f} years:")
    print(f"  Total trade entries: {total_trades} ({total_trades / years:.1f} per year)")
    print(f"  Total stop-losses: {total_stops}")
    print(f"  Average % of time pairs are in position: {avg_pct_in_position:.1%}")

    # Save summary
    summary_df.to_csv("data/signal_summary.csv")
    print("\nSaved summary to data/signal_summary.csv")