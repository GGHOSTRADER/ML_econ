"""
backtesting.py

Sanity-check backtest:
- Use your preprocessed dataframe (Datetime index, OHLC, Volume, Returns, features).
- Strategy:
    * Buy 1 contract at 09:30 (at the OPEN price of the 09:30 bar)
    * Exit (flatten to 0) at 15:00 (at the OPEN price of the 15:00 bar)
    * No other trades
- Use only the last 3 calendar months of data.
- Save:
    * Trade-based equity curve (cumulative PnL) with date on x-axis into ./backtest
    * Trade summary CSV with entry/exit info and per-trade net PnL
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Config: path to your preprocessed data and backtest dir
# =============================================================================

BACKTEST_DF_PATH = "full_df_processed.pkl"  # change if needed
BACKTEST_DIR = "backtest"


# =============================================================================
# Transaction cost model
# =============================================================================


def calculate_realistic_costs(
    position_change: int, current_price: float, bar_volume: float
) -> float:
    """
    Model slippage + commissions for ES-like futures.

    position_change : change in number of contracts (e.g., +2, -1)
    current_price   : bar's trade/OPEN price
    bar_volume      : volume on that bar (used as liquidity proxy)

    Returns total cost in currency units.
    """
    position_change = int(position_change)
    if position_change == 0:
        return 0.0

    # Fixed commissions: $2.50 per contract per side
    commission = 2.50 * abs(position_change)

    # Simple market impact: square-root style (very crude)
    daily_volume = bar_volume * 390.0  # approximate daily volume from 1-min bars
    if daily_volume <= 0:
        return commission

    participation_rate = abs(position_change) / daily_volume
    slippage_bps = 2.0 * np.sqrt(participation_rate * 100.0)  # basis points
    slippage_price = current_price * (slippage_bps / 10000.0)

    # ES notional move = slippage_price * contracts * 50
    slippage = slippage_price * abs(position_change) * 50.0

    return commission + slippage


# =============================================================================
# Core backtester
# =============================================================================


class Backtester:
    """
    Simple backtesting engine on a preprocessed DataFrame.

    The DataFrame must have at least:
      - 'Open'
      - 'Volume'

    We use:
      - OPEN prices for:
          * trade execution (entry and exit)
          * PnL path (open-to-open)
      - Volume for liquidity / slippage

    Trading logic is delegated to a user-defined policy:

        policy_fn(df, i, current_position) -> desired_position (int)

    After run(), you get:
        - summary dict
        - trades_df with round-trip trades
    """

    def __init__(
        self,
        df: pd.DataFrame,
        price_col: str = "Open",  # IMPORTANT: we work on OPEN prices
        volume_col: str = "Volume",
        contract_multiplier: float = 50.0,
        initial_capital: float = 100_000.0,
        max_position: int = 10,
    ):
        self.df = df.sort_index()
        self.price_col = price_col
        self.volume_col = volume_col
        self.multiplier = contract_multiplier
        self.initial_capital = float(initial_capital)
        self.max_position = int(max_position)

        self.equity_curve: pd.Series | None = None
        self.positions: pd.Series | None = None
        self.cash_series: pd.Series | None = None
        self.pnl_series: pd.Series | None = None
        self.trades: list[dict] = []
        self.trades_df: pd.DataFrame | None = None

    def run(self, policy_fn, start_idx: int, end_idx: int | None = None):
        """
        Run a single backtest.

        policy_fn(df, i, current_pos) -> desired_position (int)

        start_idx : integer location index in df (0-based)
        end_idx   : last index (exclusive) to trade; if None, run to df end-1.

        Returns:
            summary_dict, trades_df
        """
        n = len(self.df)
        if n < 2:
            raise ValueError("DataFrame too short for backtest (need at least 2 rows).")

        if end_idx is None or end_idx > n - 1:
            end_idx = n - 1  # we look at i and i+1 in the loop

        if start_idx < 0 or start_idx >= end_idx:
            raise ValueError("Invalid start_idx / end_idx for backtest.")

        cash = self.initial_capital
        pos = 0  # current number of contracts
        equity = []

        positions = np.zeros(n, dtype=np.int32)
        cash_series = np.zeros(n, dtype=np.float64)
        pnl_series = np.zeros(n, dtype=np.float64)

        self.trades.clear()

        for i in range(start_idx, end_idx):
            row = self.df.iloc[i]
            price_now = float(row[self.price_col])  # OPEN price of bar i
            bar_vol = float(row[self.volume_col])

            # 1) Policy decides target position at the OPEN of this bar
            desired_pos = policy_fn(self.df, i, pos)
            desired_pos = int(
                np.clip(desired_pos, -self.max_position, self.max_position)
            )

            # 2) Trade to that desired position at this bar's OPEN
            position_change = desired_pos - pos
            trade_cost = 0.0
            if position_change != 0:
                trade_cost = calculate_realistic_costs(
                    position_change, price_now, bar_vol
                )
                self.trades.append(
                    {
                        "time": self.df.index[i],
                        "price": price_now,  # OPEN price of bar i
                        "position_change": position_change,
                        "new_position": desired_pos,
                        "cost": trade_cost,
                    }
                )

            pos = desired_pos

            # 3) Realized PnL from OPEN(i) -> OPEN(i+1)
            next_price = float(self.df.iloc[i + 1][self.price_col])  # OPEN of next bar
            price_change = next_price - price_now
            pnl = pos * price_change * self.multiplier

            cash = cash + pnl - trade_cost
            equity.append(cash)

            positions[i + 1] = pos
            cash_series[i + 1] = cash
            pnl_series[i + 1] = pnl - trade_cost

        # Build output series (still kept, but plotting will be trade-based)
        self.equity_curve = pd.Series(
            equity, index=self.df.index[start_idx + 1 : end_idx + 1]
        )
        self.positions = pd.Series(positions, index=self.df.index)
        self.cash_series = pd.Series(cash_series, index=self.df.index)
        self.pnl_series = pd.Series(pnl_series, index=self.df.index)

        summary = self._summary()
        trades_df = self._build_round_trip_trades()
        self.trades_df = trades_df

        return summary, trades_df

    def _summary(self) -> dict:
        """
        Compute basic performance stats from bar-level equity path:
        - total return
        - annualized return & vol (assuming 1-min bars, ~390 bars/day)
        - Sharpe
        - max drawdown
        - net_profit (final - initial)
        """
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return {}
        eq = self.equity_curve.values
        if len(eq) < 2:
            return {}
        initial_capital = float(self.initial_capital)
        final_equity = float(eq[-1])
        net_profit = final_equity - initial_capital
        rets = np.diff(eq) / eq[:-1]
        total_return = (final_equity / eq[0]) - 1.0
        # Annualization factor: 1-min bars, ~390 bars/day, 252 trading days/year
        ann_factor = 252 * 390
        if len(rets) > 0:
            ann_return = (1.0 + total_return) ** (ann_factor / len(rets)) - 1.0
        else:
            ann_return = 0.0
        ann_vol = np.std(rets) * np.sqrt(ann_factor) if len(rets) > 1 else 0.0
        sharpe = ann_return / (ann_vol + 1e-12)
        max_eq = np.maximum.accumulate(eq)
        drawdowns = (eq - max_eq) / max_eq
        max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0
        return {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "net_profit": net_profit,  # <--- added
            "total_return": float(total_return),
            "annualized_return": float(ann_return),
            "annualized_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": max_dd,
            "n_trades": len(self.trades),
        }

    def _build_round_trip_trades(self) -> pd.DataFrame:
        """
        Build a round-trip trade summary DataFrame.

        Assumes:
            - A "trade" starts when position goes 0 -> non-zero
            - A "trade" ends when position goes non-zero -> 0
        This matches the open-close policy (1 trade per day).
        """
        if not self.trades or self.pnl_series is None or self.cash_series is None:
            return pd.DataFrame(
                columns=[
                    "entry_time",
                    "entry_price",
                    "exit_time",
                    "exit_price",
                    "direction",
                    "contracts",
                    "trade_pnl",
                    "entry_equity",
                    "exit_equity",
                ]
            )

        rows = []
        current = None
        idx_map = {ts: i for i, ts in enumerate(self.df.index)}

        for tr in self.trades:
            t = tr["time"]
            price = tr["price"]
            new_pos = tr["new_position"]

            idx = idx_map[t]

            if current is None:
                # start trade only when we go from flat to non-zero
                if new_pos != 0:
                    current = {
                        "entry_idx": idx,
                        "entry_time": t,
                        "entry_price": price,  # OPEN at entry bar
                        "direction": int(np.sign(new_pos)),
                        "contracts": abs(int(new_pos)),
                    }
            else:
                # close when we go back to flat
                if new_pos == 0:
                    entry_idx = current["entry_idx"]
                    exit_idx = idx

                    # per-bar PnL is stored at index of the *next* bar:
                    #   pnl_series[i+1] = pnl for interval [i, i+1]
                    # after entry at entry_idx, first pnl is at entry_idx+1
                    # last bar with non-zero pos is exit_idx-1,
                    # last pnl index is exit_idx
                    pnl_slice = self.pnl_series.iloc[entry_idx + 1 : exit_idx + 1]
                    trade_pnl = float(pnl_slice.sum())

                    entry_equity = float(self.cash_series.iloc[entry_idx])
                    exit_equity = float(self.cash_series.iloc[exit_idx])

                    row = {
                        "entry_time": current["entry_time"],
                        "entry_price": current["entry_price"],
                        "exit_time": t,
                        "exit_price": price,  # OPEN at exit bar
                        "direction": current["direction"],
                        "contracts": current["contracts"],
                        "trade_pnl": trade_pnl,  # net, after costs
                        "entry_equity": entry_equity,
                        "exit_equity": exit_equity,
                    }
                    rows.append(row)
                    current = None
                else:
                    # scaling in/out not handled; ignore for now (not needed here)
                    pass

        trades_df = pd.DataFrame(rows)
        trades_df = trades_df[
            [
                "entry_time",
                "entry_price",
                "exit_time",
                "exit_price",
                "direction",
                "contracts",
                "trade_pnl",
                "entry_equity",
                "exit_equity",
            ]
        ]
        return trades_df


# =============================================================================
# Policy: buy at 09:30 OPEN, flat at 15:00 OPEN
# =============================================================================


def open_close_policy(df: pd.DataFrame, i: int, current_pos: int) -> int:
    """
    Deterministic intraday policy:

    - At 09:30 -> target position = +1 contract (buy at that bar's OPEN)
    - At 15:00 -> target position = 0 (sell at that bar's OPEN)
    - Other times -> keep current position

    Assumes df.index is a DatetimeIndex.
    """
    ts = df.index[i]
    t = ts.time()

    # 09:30 -> open position
    if t.hour == 9 and t.minute == 30:
        return 1

    # 15:00 -> exit
    if t.hour == 15 and t.minute == 0:
        return 0

    # Otherwise, hold whatever you currently have
    return current_pos


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    os.makedirs(BACKTEST_DIR, exist_ok=True)

    print(f"Loading preprocessed data from: {BACKTEST_DF_PATH}")
    df = pd.read_pickle(BACKTEST_DF_PATH)

    # Basic sanity: required columns
    required_cols = ["Open", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "DataFrame index must be a DatetimeIndex for time-based policy."
        )

    df = df.sort_index()

    # -------------------------------------------------------------------------
    # Restrict to last 3 calendar months
    # -------------------------------------------------------------------------
    last_ts = df.index.max()
    cutoff_ts = last_ts - pd.DateOffset(months=3)
    df_recent = df[df.index >= cutoff_ts].copy()

    print(f"Original rows: {len(df)}, last 3 months rows: {len(df_recent)}")
    print(f"Date range used: {df_recent.index.min()} -> {df_recent.index.max()}")

    if len(df_recent) < 2:
        raise ValueError("Not enough data in last 3 months to run backtest.")

    # -------------------------------------------------------------------------
    # Run backtest on last 3 months
    # -------------------------------------------------------------------------
    bt = Backtester(
        df_recent,
        price_col="Open",  # IMPORTANT: open-to-open logic
        volume_col="Volume",
        contract_multiplier=50.0,  # ES multiplier
        initial_capital=100_000.0,
        max_position=1,  # we only ever use 0 or +1 here
    )

    start_idx = 0
    end_idx = None  # run to the end

    summary, trades_df = bt.run(open_close_policy, start_idx=start_idx, end_idx=end_idx)

    print("\nBacktest summary (last 3 months, 1 contract, 09:30-open -> 15:00-open):")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # -------------------------------------------------------------------------
    # Save trade-based equity curve plot (cumulative PnL)
    # -------------------------------------------------------------------------
    if trades_df is not None and len(trades_df) > 0:
        # Sort by exit_time (when the trade is realized)
        trades_sorted = trades_df.sort_values("exit_time")
        cum_pnl = trades_sorted["trade_pnl"].cumsum()

        plt.figure(figsize=(10, 4))
        plt.plot(
            trades_sorted["exit_time"], cum_pnl, linewidth=1, label="Cumulative PnL"
        )
        plt.axhline(0.0, linestyle="--", linewidth=1)  # horizontal line at 0
        plt.xlabel("Date")
        plt.ylabel("Cumulative PnL ($)")
        plt.title(
            "Trade-based Equity Curve - Last 3 Months (09:30 OPEN Long, 15:00 OPEN Flat)"
        )
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        eq_path = os.path.join(BACKTEST_DIR, "equity_curve_last3m.png")
        plt.savefig(eq_path)
        plt.close()
        print(f"\nSaved trade-based equity curve plot to: {eq_path}")
    else:
        print("\nNo trades to plot (something went off).")

    # -------------------------------------------------------------------------
    # Save trades summary
    # -------------------------------------------------------------------------
    if trades_df is not None and len(trades_df) > 0:
        trades_path = os.path.join(BACKTEST_DIR, "trades_summary_last3m.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"Saved trades summary to: {trades_path}")

        print("\nTrades summary (head):")
        print(trades_df.head())
    else:
        print("\nNo round-trip trades detected (check policy / data).")

    print("\nFirst 10 raw trade events (rebalancings):")
    for tr in bt.trades[:10]:
        print(tr)
