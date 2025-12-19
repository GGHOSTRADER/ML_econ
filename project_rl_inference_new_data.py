import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------- Paths ----------

DATA_PATH = "transformer_rv15_predictions_new_data.pkl"  # change if needed
ACTOR_PATH = "actor_sac_daily.pt"

EQUITY_PLOT_PATH = "rl_equity_curve.png"
TRADES_CSV_PATH = "rl_trades.csv"
EQUITY_COMPARISON_PLOT_PATH = "rl_vs_buyhold_equity.png"

# ---------- Actor definition (must match training) ----------


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        logits = self.net(state)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def sample_action(self, state_np, device, greedy: bool = True) -> int:
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
        logits, probs = self.forward(state)
        if greedy:
            action = torch.argmax(probs, dim=-1).item()
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().item()
        return action


# ---------- Cost model (same as training) ----------


def calculate_realistic_costs(
    position_change: int, current_price: float, bar_volume: float
) -> float:
    """
    Simple cost model:
      - Commission per contract
      - Slippage as sqrt(participation_rate) bps
    """
    position_change = int(position_change)
    if position_change == 0:
        return 0.0

    # Commission: $2.50 per contract
    commission = 2.50 * abs(position_change)

    # Approximate daily volume from 1-min bar volume
    daily_volume = bar_volume * 390.0
    if daily_volume <= 0:
        return commission

    participation_rate = abs(position_change) / daily_volume
    slippage_bps = 2.0 * np.sqrt(participation_rate * 100.0)
    slippage_price = current_price * (slippage_bps / 10000.0)

    # ES multiplier
    multiplier = 50.0
    slippage = slippage_price * abs(position_change) * multiplier

    return commission + slippage


# ---------- Load data (ENTIRE unseen set) ----------

print(f"Loading {DATA_PATH} ...")
df_all = pd.read_pickle(DATA_PATH)

if not isinstance(df_all.index, pd.DatetimeIndex):
    raise TypeError("DataFrame index must be a DatetimeIndex.")

df_all = df_all.sort_index()

if df_all.empty:
    raise ValueError("DataFrame is empty after loading and sorting.")

print("Data shape (full unseen set):", df_all.shape)

# ---------- Features ----------

feature_cols = [
    "parkinson_vol_5",
    "parkinson_vol_15",
    "parkinson_vol_30",
    "ofi_5",
    "ofi_15",
    "ofi_30",
    "volume_percentile",
    "volume_momentum",
    "amihud_illiquidity",
    "vwap_distance",
    "minutes_since_open",
    "is_first_last_30min",
    "pred_vol_15m_fwd",  # transformer forecast
]

cols_needed = ["Open", "Volume"]
missing_basic = [c for c in cols_needed if c not in df_all.columns]
if missing_basic:
    raise ValueError(f"Missing basic columns in df_all: {missing_basic}")

missing_features = [c for c in feature_cols if c not in df_all.columns]
if missing_features:
    raise ValueError(f"Missing RL feature columns in df_all: {missing_features}")

print(f"Using {len(feature_cols)} features:")
print(feature_cols)

# ---------- Load actor ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

state_dim = len(feature_cols) + 1  # + position
action_dim = 3  # 0: flat, 1: long, 2: short

actor = Actor(state_dim, action_dim).to(device)
state_dict = torch.load(ACTOR_PATH, map_location=device)
actor.load_state_dict(state_dict)
actor.eval()
print(f"Loaded actor from {ACTOR_PATH}")

# ---------- Run greedy evaluation over ALL days ----------

df = df_all.sort_index()
days = np.array(sorted(df.index.normalize().unique()))

position = 0
cash = 0.0
equity_curve = []
equity_time = []

trades = []  # timestamp, day, action, position, pnl_step, cost, equity
multiplier = 50.0  # ES

for day in days:
    day_mask = df.index.normalize() == day
    idxs = np.where(day_mask)[0]
    if len(idxs) < 2:
        continue

    # Start each day flat (same assumption as training)
    position = 0

    for k in range(len(idxs) - 1):
        i = idxs[k]
        j = idxs[k + 1]

        row_now = df.iloc[i]
        row_next = df.iloc[j]

        # Build state = [features..., current_position]
        features = row_now[feature_cols].values.astype(np.float32)
        state = np.concatenate([features, np.array([position], dtype=np.float32)])

        # Actor picks action
        action = actor.sample_action(state, device, greedy=True)

        if action == 0:
            desired_pos = 0
        elif action == 1:
            desired_pos = 1
        elif action == 2:
            desired_pos = -1
        else:
            raise ValueError(f"Bad action {action}")

        desired_pos = int(np.clip(desired_pos, -1, 1))

        price_now = float(row_now["Open"])
        price_next = float(row_next["Open"])
        bar_vol = float(row_now["Volume"])

        pos_change = desired_pos - position

        cost = calculate_realistic_costs(pos_change, price_now, bar_vol)
        pnl_step = desired_pos * (price_next - price_now) * multiplier

        position = desired_pos
        cash += pnl_step - cost

        ts = df.index[i]
        equity_curve.append(cash)
        equity_time.append(ts)

        trades.append(
            {
                "timestamp": ts,
                "day": day,
                "action": action,
                "position": position,
                "pnl_step": pnl_step,
                "cost": cost,
                "equity": cash,
            }
        )

# ---------- Save trades + plot equity + compare vs buy & hold ----------

trades_df = pd.DataFrame(trades)
trades_df.to_csv(TRADES_CSV_PATH, index=False)
print(f"Saved trades to {TRADES_CSV_PATH}")

if len(equity_curve) > 0:
    equity_series = pd.Series(equity_curve, index=pd.DatetimeIndex(equity_time))

    # ----- Buy & Hold benchmark (1 contract) -----
    # Use the same timestamps as the RL equity for a fair comparison
    bh_prices = df_all["Open"].loc[equity_series.index]
    start_price = bh_prices.iloc[0]
    end_price = bh_prices.iloc[-1]

    equity_bh = (bh_prices - start_price) * multiplier  # 1 contract buy & hold
    buy_hold_pnl = (end_price - start_price) * multiplier

    # ----- Plot RL vs Buy & Hold -----
    plt.figure(figsize=(10, 4))
    plt.plot(equity_series.index, equity_series.values, label="RL Policy (1 contract)")
    plt.plot(equity_bh.index, equity_bh.values, label="Buy & Hold (1 contract)")
    plt.axhline(0.0, linestyle="--", alpha=0.7)
    plt.title("RL Policy vs Buy & Hold (full unseen set, 1 contract)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EQUITY_COMPARISON_PLOT_PATH)
    plt.close()
    print(f"Saved equity comparison plot to {EQUITY_COMPARISON_PLOT_PATH}")

    # ----- RL-only equity plot (original) -----
    plt.figure(figsize=(10, 4))
    plt.plot(equity_series.index, equity_series.values)
    plt.axhline(0.0, linestyle="--", alpha=0.7)
    plt.title("RL Policy Equity Curve (greedy, full unseen set, 1 contract)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EQUITY_PLOT_PATH)
    plt.close()
    print(f"Saved RL-only equity curve plot to {EQUITY_PLOT_PATH}")

    # ----- Stats -----
    final_pnl = equity_series.iloc[-1]
    max_dd = (equity_series.cummax() - equity_series).max()
else:
    final_pnl = 0.0
    max_dd = 0.0
    buy_hold_pnl = 0.0
    print("No equity points generated (no trades).")

print("\nEval summary:")
print(f"Final PnL (RL): {final_pnl:,.2f}")
print(f"Max drawdown (RL): {max_dd:,.2f}")
print(f"Buy & Hold PnL (1 contract): {buy_hold_pnl:,.2f}")
