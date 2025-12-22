"""
project_rl.py

- Load:
    full_df_processed.pkl  (OHLC, Volume, Returns, targets, features)
    compressed_df.pkl      (12 microstructure features + 15m vol forecast)

- Build RL environment:
    * State: 13 compressed features (12 microstructure + pred_vol_15m_fwd)
             + current position (1 dim)
    * Actions: 0=flat, 1=long 1, 2=short 1
    * Reward: per-bar (pnl - cost) scaled by reward_scale
    * Episode: one trading day (sampled from last 3 months)

- Train a simple discrete SAC agent and log per-episode stats:
    episode, day, steps, ep_reward, ep_pnl
"""

import os
import numpy as np
import pandas as pd
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


DATA_FULL_PATH = "full_df_processed_splits_REPORT_full_data.pkl"
# ------------------------------------------------------------------
# GOT TO CHANGE FULL PATH AND COMPRESSED PATH TO REPORT VERSIONS FOR USING TRANSFORMER OR LIGHTGBM FORECASTS
DATA_COMPRESSED_PATH = "compressed_df_REPORT_lgb.pkl"  # HERE <----
TRAIN_LOG_PATH = "rl_training_log_REPORT_lgb.csv"
# DATA_COMPRESSED_PATH = "compressed_df_REPORT_transformer.pkl"  # HERE <----
# TRAIN_LOG_PATH = "rl_training_log_REPORT_transformer.csv"

# ------------------------------------------------------------------


# =============================================================================
# Costs (same logic as backtester)
# =============================================================================


def calculate_realistic_costs(
    position_change: int, current_price: float, bar_volume: float
) -> float:
    """
    Slippage + commissions for ES-like futures.
    """
    position_change = int(position_change)
    if position_change == 0:
        return 0.0

    # Fixed commissions
    commission = 2.50 * abs(position_change)

    # Crude square-root impact model
    daily_volume = bar_volume * 390.0
    if daily_volume <= 0:
        return commission

    participation_rate = abs(position_change) / daily_volume
    slippage_bps = 2.0 * np.sqrt(participation_rate * 100.0)
    slippage_price = current_price * (slippage_bps / 10000.0)

    # ES notional
    slippage = slippage_price * abs(position_change) * 50.0
    return commission + slippage


# =============================================================================
# Daily trading environment
# =============================================================================


class TradingEnvDaily:
    """
    One episode = one day (all 1-min bars in that day's slice).

    State:
        [compressed_features_at_t (13 dims), current_position (1 dim)]

    Actions:
        0 = flat
        1 = long 1
        2 = short 1

    Reward:
        (pnl - cost) / reward_scale, where
        pnl is from OPEN(t) -> OPEN(t+1) using NEW position.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols,
        price_col: str = "Open",
        volume_col: str = "Volume",
        contract_multiplier: float = 50.0,
        reward_scale: float = 100.0,
        max_position: int = 1,
    ):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")

        self.df = df.sort_index().copy()
        self.feature_cols = list(feature_cols)
        self.price_col = price_col
        self.volume_col = volume_col
        self.multiplier = contract_multiplier
        self.reward_scale = reward_scale
        self.max_position = max_position

        # Precompute available days
        self.days = np.array(sorted(self.df.index.normalize().unique()))
        if len(self.days) == 0:
            raise ValueError("No days available in df for TradingEnvDaily.")

        # Episode state
        self.current_day = None
        self.idx = None
        self.idx_start = None
        self.idx_end = None  # last usable index (we use idx and idx+1)
        self.position = 0
        self.cash = 0.0

        # RL dimensions
        self.state_dim = len(self.feature_cols) + 1  # + position
        self.action_dim = 3  # flat, long, short

    def _get_state(self):
        row = self.df.iloc[self.idx]
        features = row[self.feature_cols].values.astype(np.float32)
        pos = np.array([self.position], dtype=np.float32)
        return np.concatenate([features, pos], axis=0)

    def reset(self, day: pd.Timestamp | None = None):
        """
        Start a new episode on a given day (if provided) or a random day.
        """
        if day is None:
            self.current_day = np.random.choice(self.days)
        else:
            self.current_day = pd.Timestamp(day).normalize()
            if self.current_day not in self.days:
                raise ValueError(f"Day {self.current_day} not in env days.")

        # Get indices for that day
        day_mask = self.df.index.normalize() == self.current_day
        idxs = np.where(day_mask)[0]
        if len(idxs) < 2:
            # Not enough bars in this day, sample another
            return self.reset(day=None)

        self.idx_start = idxs[0]
        # We need idx and idx+1 inside bounds, so last usable idx is idxs[-2]
        self.idx_end = idxs[-2]

        self.idx = self.idx_start
        self.position = 0
        self.cash = 0.0

        return self._get_state()

    def step(self, action: int):
        if self.idx is None:
            raise RuntimeError("Call reset() before step().")

        # Map discrete action -> target position
        if action == 0:
            desired_pos = 0
        elif action == 1:
            desired_pos = +1
        elif action == 2:
            desired_pos = -1
        else:
            raise ValueError(f"Invalid action {action}")

        desired_pos = int(np.clip(desired_pos, -self.max_position, self.max_position))

        row = self.df.iloc[self.idx]
        price_now = float(row[self.price_col])  # OPEN
        bar_vol = float(row[self.volume_col])

        # Transaction cost at current bar's OPEN
        position_change = desired_pos - self.position
        cost = calculate_realistic_costs(position_change, price_now, bar_vol)

        # PnL from OPEN(t) -> OPEN(t+1)
        next_price = float(self.df.iloc[self.idx + 1][self.price_col])
        price_change = next_price - price_now
        pnl = desired_pos * price_change * self.multiplier

        self.position = desired_pos
        self.cash += pnl - cost

        reward = (pnl - cost) / self.reward_scale

        self.idx += 1
        done = self.idx >= self.idx_end

        next_state = self._get_state()
        info = {
            "day": self.current_day,
            "pnl": pnl,
            "cost": cost,
            "cash": self.cash,
            "position": self.position,
        }
        return next_state, reward, done, info


# =============================================================================
# SAC components (discrete)
# =============================================================================


class Actor(nn.Module):
    """Policy: state -> logits/probs over discrete actions."""

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

    def sample_action(self, state_np, device, greedy: bool = False) -> int:
        """
        state_np: np.array of shape (state_dim,)
        """
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
        logits, probs = self.forward(state)
        if greedy:
            action = torch.argmax(probs, dim=-1).item()
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().item()
        return action


class Critic(nn.Module):
    """Twin Q-networks: (state, one-hot(action)) -> Q-value."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        input_dim = state_dim + action_dim
        self.action_dim = action_dim

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action_onehot):
        sa = torch.cat([state, action_onehot], dim=-1)
        return self.q1(sa), self.q2(sa)


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# SAC training tools
# =============================================================================


def one_hot(actions, action_dim):
    return F.one_hot(actions.long(), num_classes=action_dim).float()


def update_sac(
    buffer,
    batch_size,
    actor,
    critic,
    critic_target,
    actor_opt,
    critic_opt,
    gamma,
    alpha,
    tau,
    device,
    action_dim,
    state_dim,
):
    if len(buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    states = torch.from_numpy(states).float().to(device)
    next_states = torch.from_numpy(next_states).float().to(device)
    actions = torch.from_numpy(actions).long().to(device)
    rewards = torch.from_numpy(rewards).float().to(device).unsqueeze(-1)
    dones = torch.from_numpy(dones).float().to(device).unsqueeze(-1)

    batch = states.size(0)

    # ---- Critic update ----
    with torch.no_grad():
        next_logits, next_probs = actor(next_states)
        log_next_probs = torch.log(next_probs + 1e-8)

        # Q(s', a') for all actions
        all_actions = (
            torch.eye(action_dim, device=device).unsqueeze(0).repeat(batch, 1, 1)
        )
        next_states_expanded = (
            next_states.unsqueeze(1).repeat(1, action_dim, 1).reshape(-1, state_dim)
        )
        all_actions_flat = all_actions.view(-1, action_dim)

        q1_next, q2_next = critic_target(next_states_expanded, all_actions_flat)
        q1_next = q1_next.view(batch, action_dim)
        q2_next = q2_next.view(batch, action_dim)
        q_next = torch.min(q1_next, q2_next)

        v_next = (next_probs * (q_next - alpha * log_next_probs)).sum(
            dim=1, keepdim=True
        )
        target_q = rewards + (1.0 - dones) * gamma * v_next

    action_oh = one_hot(actions, action_dim)
    q1, q2 = critic(states, action_oh)
    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    # ---- Actor update ----
    logits, probs = actor(states)
    log_probs = torch.log(probs + 1e-8)

    all_actions = torch.eye(action_dim, device=device).unsqueeze(0).repeat(batch, 1, 1)
    states_expanded = (
        states.unsqueeze(1).repeat(1, action_dim, 1).reshape(-1, state_dim)
    )
    all_actions_flat = all_actions.view(-1, action_dim)

    q1_pi, q2_pi = critic(states_expanded, all_actions_flat)
    q1_pi = q1_pi.view(batch, action_dim)
    q2_pi = q2_pi.view(batch, action_dim)
    q_pi = torch.min(q1_pi, q2_pi)

    # J_pi = E_s [ sum_a pi(a|s) (alpha log pi(a|s) - Q(s, a)) ]
    actor_loss = (probs * (alpha * log_probs - q_pi)).sum(dim=1).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    # ---- Target soft update ----
    with torch.no_grad():
        for p, tp in zip(critic.parameters(), critic_target.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)


# =============================================================================
# Main training
# =============================================================================

if __name__ == "__main__":
    # -----------------------------
    # Load and align data
    # -----------------------------
    print(f"Loading {DATA_FULL_PATH} and {DATA_COMPRESSED_PATH} ...")
    df_full = pd.read_pickle(DATA_FULL_PATH)
    df_comp = pd.read_pickle(DATA_COMPRESSED_PATH)

    if not isinstance(df_full.index, pd.DatetimeIndex):
        raise TypeError("full_df index must be DatetimeIndex.")
    if not isinstance(df_comp.index, pd.DatetimeIndex):
        raise TypeError("compressed_df index must be DatetimeIndex.")

    df_full = df_full.sort_index()
    df_comp = df_comp.sort_index()

    # Last 3 months slice based on full_df index
    last_ts = df_full.index.max()
    cutoff_ts = last_ts - pd.DateOffset(months=3)
    df_full_3m = df_full[df_full.index >= cutoff_ts].copy()
    df_comp_3m = df_comp[df_comp.index >= cutoff_ts].copy()

    # We only want Open/Volume from full_df, and features from compressed_df
    cols_needed = ["Open", "Volume"]
    for c in cols_needed:
        if c not in df_full_3m.columns:
            raise ValueError(f"{c} not found in full_df_3m.")

    # Drop any overlapping raw columns from compressed_df (e.g. Open/Volume)
    drop_cols = [c for c in cols_needed if c in df_comp_3m.columns]
    if drop_cols:
        print("Dropping overlapping columns from compressed_df:", drop_cols)
        df_comp_3m = df_comp_3m.drop(columns=drop_cols)

    # Join on DatetimeIndex: left has Open/Volume, right has compressed features
    df_join = df_full_3m[cols_needed].join(df_comp_3m, how="inner")
    if df_join.empty:
        raise ValueError(
            "Joined df is empty; index mismatch between full and compressed."
        )

    # Explicit RL features: 12 microstructure + 15m forecast
    desired_feature_cols = [
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
        "pred_vol_15m_fwd",  # transformer 15-min vol forecast
    ]

    missing = [c for c in desired_feature_cols if c not in df_join.columns]
    if missing:
        raise ValueError(f"Missing expected RL feature columns in df_join: {missing}")

    feature_cols = desired_feature_cols
    print(f"Using {len(feature_cols)} features for RL (compressed df):")
    print(feature_cols)

    # -----------------------------
    # Build env
    # -----------------------------
    env = TradingEnvDaily(
        df_join,
        feature_cols=feature_cols,
        price_col="Open",
        volume_col="Volume",
        contract_multiplier=50.0,
        reward_scale=100.0,
        max_position=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    state_dim = env.state_dim
    action_dim = env.action_dim

    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    critic_target = Critic(state_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)

    buffer = ReplayBuffer(capacity=100_000)

    gamma = 0.99
    alpha = 0.01
    tau = 0.005
    batch_size = 256

    num_episodes = 200  # number of daily episodes to train

    # optional: log to csv
    if os.path.exists(TRAIN_LOG_PATH):
        os.remove(TRAIN_LOG_PATH)
    with open(TRAIN_LOG_PATH, "w") as f:
        f.write("episode,day,steps,ep_reward,ep_pnl\n")

    for ep in range(num_episodes):
        state = env.reset()
        current_day = env.current_day
        done = False
        ep_reward = 0.0
        ep_pnl = 0.0
        steps = 0

        while not done:
            action = actor.sample_action(state, device, greedy=False)

            next_state, reward, done, info = env.step(action)

            pnl_step = info["pnl"] - info["cost"]
            ep_pnl += pnl_step

            buffer.push(
                state,
                action,
                reward,
                next_state,
                float(done),
            )

            state = next_state
            ep_reward += reward
            steps += 1

            update_sac(
                buffer,
                batch_size,
                actor,
                critic,
                critic_target,
                actor_opt,
                critic_opt,
                gamma,
                alpha,
                tau,
                device,
                action_dim,
                state_dim,
            )

        print(
            f"Episode {ep:03d} | day {current_day.date()} | "
            f"steps: {steps:4d} | ep_reward: {ep_reward:8.3f} | ep_pnl: {ep_pnl:8.2f}"
        )

        with open(TRAIN_LOG_PATH, "a") as f:
            f.write(f"{ep},{current_day.date()},{steps},{ep_reward:.6f},{ep_pnl:.6f}\n")

    print(f"\nTraining finished. Log saved to {TRAIN_LOG_PATH}")

    # Optionally save models (LGB)
    torch.save(actor.state_dict(), "actor_sac_daily_REPORT_lgb.pt")
    torch.save(critic.state_dict(), "critic_sac_daily_REPORT_lgb.pt")
    print("Saved actor_sac_daily_REPORT.pt and critic_sac_daily_REPORT_lgb.pt")

    # Optionally save models (TRASFORMER)
    # torch.save(actor.state_dict(), "actor_sac_daily_REPORT_trans.pt")
    # torch.save(critic.state_dict(), "critic_sac_daily_REPORT_trans.pt")
    # print("Saved actor_sac_daily_REPORT.pt and critic_sac_daily_REPORT_trans.pt")
