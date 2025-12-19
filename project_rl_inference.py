import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DATA_FULL_PATH = "full_df_processed.pkl"
DATA_COMPRESSED_PATH = "compressed_df.pkl"
ACTOR_PATH = "actor_sac_daily.pt"

EQUITY_PLOT_PATH = "rl_equity_curve.png"
TRADES_CSV_PATH = "rl_trades.csv"


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
    position_change = int(position_change)
    if position_change == 0:
        return 0.0

    commission = 2.50 * abs(position_change)
    daily_volume = bar_volume * 390.0
    if daily_volume <= 0:
        return commission

    participation_rate = abs(position_change) / daily_volume
    slippage_bps = 2.0 * np.sqrt(participation_rate * 100.0)
    slippage_price = current_price * (slippage_bps / 10000.0)
    slippage = slippage_price * abs(position_change) * 50.0
    return commission + slippage


# ---------- Load data & build features as in RL training ----------

print(f"Loading {DATA_FULL_PATH} and {DATA_COMPRESSED_PATH} ...")
df_full = pd.read_pickle(DATA_FULL_PATH)
df_comp = pd.read_pickle(DATA_COMPRESSED_PATH)

if not isinstance(df_full.index, pd.DatetimeIndex):
    raise TypeError("full_df index must be DatetimeIndex.")
if not isinstance(df_comp.index, pd.DatetimeIndex):
    raise TypeError("compressed_df index must be DatetimeIndex.")

df_full = df_full.sort_index()
df_comp = df_comp.sort_index()

# last 3 months, same as training
last_ts = df_full.index.max()
cutoff_ts = last_ts - pd.DateOffset(months=3)
df_full_3m = df_full[df_full.index >= cutoff_ts].copy()
df_comp_3m = df_comp[df_comp.index >= cutoff_ts].copy()

cols_needed = ["Open", "Volume"]
for c in cols_needed:
    if c not in df_full_3m.columns:
        raise ValueError(f"{c} not in full_df_3m.")

drop_cols = [c for c in cols_needed if c in df_comp_3m.columns]
if drop_cols:
    print("Dropping overlapping columns from compressed_df:", drop_cols)
    df_comp_3m = df_comp_3m.drop(columns=drop_cols)

df_join = df_full_3m[cols_needed].join(df_comp_3m, how="inner")
if df_join.empty:
    raise ValueError("Joined df is empty")

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
    "pred_vol_15m_fwd",
]
missing = [c for c in feature_cols if c not in df_join.columns]
if missing:
    raise ValueError(f"Missing RL features in df_join: {missing}")

print(f"Eval using {len(feature_cols)} features:", feature_cols)

# ---------- Load actor ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

state_dim = len(feature_cols) + 1  # + position
action_dim = 3  # flat, long, short

actor = Actor(state_dim, action_dim).to(device)
state_dict = torch.load(ACTOR_PATH, map_location=device)
actor.load_state_dict(state_dict)
actor.eval()
print(f"Loaded actor from {ACTOR_PATH}")

# ---------- Run greedy evaluation over all days ----------

df = df_join  # shorthand
df = df.sort_index()
days = np.array(sorted(df.index.normalize().unique()))

position = 0
cash = 0.0
equity_curve = []
equity_time = []

trades = []  # one row per bar: timestamp, day, action, position, pnl_step, cost, cash

multiplier = 50.0

for day in days:
    day_mask = df.index.normalize() == day
    idxs = np.where(day_mask)[0]
    if len(idxs) < 2:
        continue

    # start day flat
    position = 0

    for k in range(len(idxs) - 1):
        i = idxs[k]
        j = idxs[k + 1]

        row_now = df.iloc[i]
        row_next = df.iloc[j]

        features = row_now[feature_cols].values.astype(np.float32)
        state = np.concatenate([features, np.array([position], dtype=np.float32)])

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

# ---------- Save trades + plot equity ----------

trades_df = pd.DataFrame(trades)
trades_df.to_csv(TRADES_CSV_PATH, index=False)
print(f"Saved trades to {TRADES_CSV_PATH}")

equity_series = pd.Series(equity_curve, index=pd.DatetimeIndex(equity_time))

plt.figure(figsize=(10, 4))
plt.plot(equity_series.index, equity_series.values)
plt.axhline(0.0, linestyle="--")
plt.title("RL Policy Equity Curve (greedy, last 3 months, 1 contract)")
plt.xlabel("Time")
plt.ylabel("Cumulative PnL ($)")
plt.grid(True)
plt.tight_layout()
plt.savefig(EQUITY_PLOT_PATH)
plt.close()
print(f"Saved equity curve plot to {EQUITY_PLOT_PATH}")

# basic stats
final_pnl = equity_series.iloc[-1] if len(equity_series) > 0 else 0.0
max_dd = (
    (equity_series.cummax() - equity_series).max() if len(equity_series) > 0 else 0.0
)
print("\nEval summary:")
print(f"Final PnL: {final_pnl:,.2f}")
print(f"Max drawdown: {max_dd:,.2f}")
