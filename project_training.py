import math
import os
import pandas as pd
import numpy as np
import talib as ta

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# =========================
# Helper functions
# =========================


def qlike_loss(
    preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    QLIKE loss for volatility forecasts.

    preds, targets: forecast and realized *volatility* (sigma).
    We convert to variance inside and apply the standard QLIKE form.
    """
    pred_var = torch.clamp(preds**2, min=eps)
    true_var = torch.clamp(targets**2, min=eps)

    loss = torch.log(pred_var) + true_var / pred_var
    return loss.mean()


def parkinson_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Parkinson volatility estimator using High/Low.

    sigma^2 = (1 / (4 * ln 2)) * E[(ln(High/Low))^2] over a rolling window
    """
    hl = np.log(df["High"] / df["Low"])
    sigma2 = (hl**2).rolling(window).mean() / (4 * np.log(2))
    return np.sqrt(sigma2)


def order_flow_imbalance(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Simple order flow imbalance using Up (buy) and Down (sell) volume.

    OFI(window) = rolling sum of (Up - Down) over the last `window` bars.
    """
    net_of = df["Up"] - df["Down"]
    return net_of.rolling(window).sum()


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile of the *latest* value within its rolling window.
    Returns in [0, 1], not 0–100.
    """

    def last_rank_pct(x: pd.Series) -> float:
        if len(x) <= 1:
            return np.nan
        rank = x.rank().iloc[-1]  # 1..len(x)
        return (rank - 1) / (len(x) - 1)

    return series.rolling(window).apply(last_rank_pct, raw=False)


def get_minutes_since_open(
    index: pd.DatetimeIndex,
    open_time: str = "09:30",
) -> pd.Series:
    """
    Minutes since the (fixed) session open each day.

    Bars before open get 0 (clamped).
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex for get_minutes_since_open")

    open_t = pd.to_datetime(open_time).time()

    mins = []
    for ts in index:
        session_open = pd.Timestamp.combine(ts.date(), open_t)
        delta_min = (ts - session_open).total_seconds() / 60.0
        mins.append(max(delta_min, 0.0))

    return pd.Series(mins, index=index)


def get_session_flag(
    index: pd.DatetimeIndex,
    first_last: int = 30,
    open_time: str = "09:30",
) -> pd.Series:
    """
    Flag bars where:

      - 0  < minutes_since_open < 31   (first 30 minutes)
      - 359 < minutes_since_open < 390 (last ~30 minutes)
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex for get_session_flag")

    mins = get_minutes_since_open(index, open_time=open_time)

    first_mask = (mins > 0) & (mins < 31)
    last_mask = (mins > 359) & (mins < 390)

    flag = (first_mask | last_mask).astype(int)
    return flag


# =========================
# Main feature engineering
# =========================


def engineer_features(df: pd.DataFrame, window_sizes=[5, 15, 30]) -> pd.DataFrame:
    """
    Engineer microstructure features given a bar dataframe with:

        'Open', 'High', 'Low', 'Close', 'Up', 'Down', 'VWAP',
        'Volume', 'Returns', 'ATR'
    """
    features = {}

    # 1. Volatility estimators (Parkinson)
    for w in window_sizes:
        features[f"parkinson_vol_{w}"] = parkinson_volatility(df, w)

    # 2. Order flow imbalance
    for w in window_sizes:
        features[f"ofi_{w}"] = order_flow_imbalance(df, w)

    # 3. Volume dynamics
    features["volume_percentile"] = rolling_percentile(df["Volume"], 60)
    features["volume_momentum"] = df["Volume"].pct_change(5)

    # 4. Price impact / liquidity
    vol_safe = df["Volume"].replace(0, np.nan)
    features["amihud_illiquidity"] = np.abs(df["Returns"]) / vol_safe
    features["vwap_distance"] = (df["Close"] - df["VWAP"]) / df["ATR"]

    # 5. Temporal features
    features["minutes_since_open"] = get_minutes_since_open(df.index)
    features["is_first_last_30min"] = get_session_flag(df.index, first_last=30)

    return pd.DataFrame(features, index=df.index)


# =========================
# Target construction
# =========================


def add_realized_vol_target(df: pd.DataFrame, horizon: int = 15) -> pd.DataFrame:
    """
    Add forward-looking realized volatility target over the next `horizon` 1-min bars.

        rv_horizon_fwd(t) = sqrt( sum_{k=1..horizon} r_{t+k}^2 )
    """
    if "Returns" not in df.columns:
        raise ValueError("Returns column missing; run load_and_preprocess first.")

    sq = df["Returns"] ** 2
    cumsum = sq.cumsum()

    # Sum_{k=t+1}^{t+h} r_k^2 = S_{t+h} - S_t
    forward_sum = cumsum.shift(-horizon) - cumsum
    rv_future = np.sqrt(forward_sum.clip(lower=0))
    df[f"rv_{horizon}m_fwd"] = rv_future
    return df


# =========================
# Preprocessing
# =========================


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Build datetime index from Date + Time
    if "Datetime" not in df.columns:
        df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.set_index("Datetime").sort_index()

    # Volume from Up/Down
    if "Volume" not in df.columns:
        df["Volume"] = df["Up"] + df["Down"]

    # Returns (log returns)
    if "Returns" not in df.columns:
        df["Returns"] = np.log(df["Close"]).diff()

    # ATR from TA-Lib (14-period by default)
    if "ATR" not in df.columns:
        df["ATR"] = ta.ATR(
            df["High"].values,
            df["Low"].values,
            df["Close"].values,
            timeperiod=14,
        )

    return df


# =========================
# Feature transformation + normalization
# =========================


def normalize_features(features_df: pd.DataFrame):
    """
    Per-feature transforms + RobustScaler.
    """
    df = features_df.copy()

    # 1. Log-transform heavy positive stuff
    log_cols = [
        "parkinson_vol_5",
        "parkinson_vol_15",
        "parkinson_vol_30",
        "amihud_illiquidity",
    ]
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # 2. Clip extremes
    clip_cols = ["vwap_distance", "volume_momentum"]
    for col in clip_cols:
        if col in df.columns:
            lower = df[col].quantile(0.005)
            upper = df[col].quantile(0.995)
            df[col] = df[col].clip(lower=lower, upper=upper)

    # 3. Scale minutes_since_open to [0, 1]
    if "minutes_since_open" in df.columns:
        max_minutes = df["minutes_since_open"].max()
        if max_minutes > 0:
            df["minutes_since_open"] = df["minutes_since_open"] / max_minutes

    # Binary columns: do NOT scale
    binary_cols = ["is_first_last_30min"]
    cont_cols = [c for c in df.columns if c not in binary_cols]

    scaler = RobustScaler()
    scaled = scaler.fit_transform(df[cont_cols])
    df_scaled = pd.DataFrame(scaled, columns=cont_cols, index=df.index)

    for col in binary_cols:
        if col in df.columns:
            df_scaled[col] = df[col]

    # Preserve original column order
    df_scaled = df_scaled[features_df.columns]

    return df_scaled, scaler


# =========================
# Positional encoding + Transformer model
# =========================


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class VolatilityForecaster(nn.Module):
    def __init__(self, n_features=12, d_model=128, n_heads=4, n_layers=3):
        super().__init__()

        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        out = self.fc_out(x)
        return out.squeeze(-1)


# =========================
# Dataset for sliding windows
# =========================


class VolDataset(Dataset):
    def __init__(self, df, feature_cols, target_col="rv_15m_fwd", seq_len=120):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len

        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)

        self.X = X
        self.y = y
        self.n_seq = len(df) - seq_len + 1

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_len

        x_seq = self.X[start:end]
        y_t = self.y[end - 1]

        x_tensor = torch.from_numpy(x_seq)
        y_tensor = torch.tensor(y_t, dtype=torch.float32)

        return x_tensor, y_tensor


# =========================
# Training / evaluation helpers
# =========================


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x)
        loss = criterion(preds, y)
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def get_predictions_and_targets(model, loader, device):
    model.eval()
    preds_all = []
    trues_all = []

    for x, y in loader:
        x = x.to(device)
        out = model(x)
        preds_all.append(out.cpu().numpy())
        trues_all.append(y.numpy())

    preds = np.concatenate(preds_all)
    trues = np.concatenate(trues_all)
    return preds, trues


def windowed_correlation(y_true, y_pred, window_size=5000, min_fraction=0.5):
    corrs = []
    n = len(y_true)

    for start in range(0, n, window_size):
        end = start + window_size
        if end - start < window_size * min_fraction:
            break
        yt = y_true[start:end]
        yp = y_pred[start:end]

        if np.std(yt) == 0 or np.std(yp) == 0:
            corrs.append(np.nan)
        else:
            c = np.corrcoef(yt, yp)[0, 1]
            corrs.append(c)

    return np.array(corrs)


def plot_window_series(
    window_idx,
    window_size,
    y_true,
    y_pred,
    time_index,
    corr_value=None,
    plots_dir="plots",
):
    """
    Plot realized vs predicted vol for a given window using a pre-aligned time_index.
    """
    os.makedirs(plots_dir, exist_ok=True)

    n = len(y_true)
    start = window_idx * window_size
    end = min((window_idx + 1) * window_size, n)

    y_t = y_true[start:end]
    y_p = y_pred[start:end]
    ts = time_index[start:end]

    plt.figure(figsize=(10, 4))
    plt.plot(ts, y_t, label="Realized vol", linewidth=1)
    plt.plot(ts, y_p, label="Predicted vol", linewidth=1)
    title = f"Window {window_idx} vol series"
    if corr_value is not None:
        title += f" (corr={corr_value:.3f})"
    plt.title(title)
    plt.xticks(rotation=45)
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fname = os.path.join(plots_dir, f"window_{window_idx:02d}_series.png")
    plt.savefig(fname)
    plt.close()


# =========================
# Main
# =========================

if __name__ == "__main__":
    data_path = "data_ml_final.txt"
    DATA_CACHE_PATH = "full_df_processed.pkl"
    MODEL_CHECKPOINT_PATH = "vol_forecaster_checkpoint.pt"
    PLOTS_DIR = "plots"

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Load + preprocess
    df = load_and_preprocess(data_path)
    print("Raw df with technical columns:")
    print(df.head())

    # 1b. Add 15-min realized vol target
    df = add_realized_vol_target(df, horizon=15)
    print("\nSample of rv_15m_fwd:")
    print(df[["Returns", "rv_15m_fwd"]].head(25))

    # 2. Engineer features
    features_df = engineer_features(df)
    print("\nEngineered features:")
    print(features_df.head())

    # 3. Merge back with price data + target
    full_df = pd.concat([df, features_df], axis=1)
    print("\nFull df with prices + features + target (before dropna):")
    print(full_df.head())
    print("\nColumns in full_df:")
    print(full_df.columns)

    # 4. Drop empty lines
    full_df = full_df.dropna()

    # 5. Normalize engineered features
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
    ]

    scaled_features, scaler = normalize_features(full_df[feature_cols])
    full_df[feature_cols] = scaled_features

    print("\nFull df after normalization of engineered features:")
    print(full_df[feature_cols].describe().round(3))

    # 5b. Scale target
    target_col = "rv_15m_fwd"
    target_scale = full_df[target_col].median()
    full_df[target_col] = full_df[target_col] / target_scale

    # chronological order
    full_df = full_df.sort_index()

    # cache processed dataframe
    full_df.to_pickle(DATA_CACHE_PATH)
    print(f"\nSaved processed dataframe to {DATA_CACHE_PATH}")

    # =========================
    # Build datasets and loaders
    # =========================

    seq_len = 120

    n = len(full_df)
    split_idx = int(n * 0.8)

    train_df = full_df.iloc[:split_idx]
    val_df = full_df.iloc[split_idx - seq_len + 1 :]

    train_ds = VolDataset(train_df, feature_cols, target_col, seq_len=seq_len)
    val_ds = VolDataset(val_df, feature_cols, target_col, seq_len=seq_len)

    batch_size = 256
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # =========================
    # Model, optimizer, training
    # =========================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)

    model = VolatilityForecaster(
        n_features=len(feature_cols), d_model=128, n_heads=4, n_layers=3
    )
    model.to(device)

    criterion = qlike_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    n_epochs = 6

    train_loss_hist = []
    val_loss_hist = []
    val_mae_hist = []
    val_corr_hist = []

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        val_preds_scaled, val_targets_scaled = get_predictions_and_targets(
            model, val_loader, device
        )

        val_preds = val_preds_scaled * target_scale
        val_targets = val_targets_scaled * target_scale

        val_mae = mean_absolute_error(val_targets, val_preds)
        if np.std(val_targets) > 0 and np.std(val_preds) > 0:
            val_corr = np.corrcoef(val_targets, val_preds)[0, 1]
        else:
            val_corr = np.nan

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        val_mae_hist.append(val_mae)
        val_corr_hist.append(val_corr)

        print(
            f"Epoch {epoch:02d} | "
            f"train QLIKE: {train_loss:.6e} | "
            f"val QLIKE: {val_loss:.6e} | "
            f"val MAE (true scale): {val_mae:.6e} | "
            f"val Corr: {val_corr:.4f}"
        )

    # save model checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "target_scale": float(target_scale),
        "seq_len": seq_len,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 3,
    }
    torch.save(checkpoint, MODEL_CHECKPOINT_PATH)
    print(f"\nSaved model checkpoint to {MODEL_CHECKPOINT_PATH}")

    # =========================
    # Final evaluation on validation (true scale)
    # =========================

    val_preds_scaled, val_targets_scaled = get_predictions_and_targets(
        model, val_loader, device
    )
    val_preds = val_preds_scaled * target_scale
    val_targets = val_targets_scaled * target_scale

    n_val = len(val_targets)
    offset = seq_len - 1  # first target corresponds to val_df.index[offset]
    time_for_targets = val_df.index[offset : offset + n_val]
    assert len(time_for_targets) == n_val

    # Windowed correlations
    window_size = 5000
    corr_windows = windowed_correlation(val_targets, val_preds, window_size=window_size)
    print(f"\nWindowed correlations (window_size={window_size}):")
    print(corr_windows)

    print("\nWindow | Corr   | Start Time                -> End Time")
    window_times = []
    for i, corr in enumerate(corr_windows):
        start = i * window_size
        end = min((i + 1) * window_size, n_val)

        ts_start = time_for_targets[start]
        ts_end = time_for_targets[end - 1]

        window_times.append((i, corr, ts_start, ts_end))
        print(f"{i:02d}     | {corr:0.3f} | {ts_start} -> {ts_end}")

    # Identify bad / good windows by correlation threshold
    bad_threshold = 0.70
    bad_idxs = [i for i, c in enumerate(corr_windows) if c < bad_threshold]
    good_idxs = [i for i, c in enumerate(corr_windows) if c >= bad_threshold]

    print(f"\nBad windows (corr < {bad_threshold}): {bad_idxs}")
    print(f"Good windows (corr >= {bad_threshold}): {good_idxs}")

    # Save per-window series plots
    for w in bad_idxs[:3]:
        plot_window_series(
            w,
            window_size,
            val_targets,
            val_preds,
            time_for_targets,
            corr_value=corr_windows[w],
            plots_dir=PLOTS_DIR,
        )

    for w in good_idxs[:3]:
        plot_window_series(
            w,
            window_size,
            val_targets,
            val_preds,
            time_for_targets,
            corr_value=corr_windows[w],
            plots_dir=PLOTS_DIR,
        )

    # =========================
    # Plots (aggregate)
    # =========================

    n_plot = min(5000, len(val_targets))
    plt.figure(figsize=(10, 4))
    plt.plot(val_targets[-n_plot:], label="Realized vol (val)", linewidth=1)
    plt.plot(val_preds[-n_plot:], label="Predicted vol (val)", linewidth=1)
    plt.title("Realized vs Predicted 15-min Vol (Validation Tail)")
    plt.xlabel("Time steps (last segment of validation)")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "val_tail_vol.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, n_epochs + 1), val_corr_hist, marker="o")
    plt.title("Validation Correlation vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Pearson correlation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "val_corr_epochs.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, n_epochs + 1), train_loss_hist, marker="o", label="Train QLIKE")
    plt.plot(range(1, n_epochs + 1), val_loss_hist, marker="o", label="Val QLIKE")
    plt.yscale("log")
    plt.title("QLIKE Loss vs Epoch (log scale)")
    plt.xlabel("Epoch")
    plt.ylabel("QLIKE loss (log)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "val_qlike_epochs.png"))
    plt.close()

    # =========================
    # Example: inference on last window
    # =========================

    last_window = full_df[feature_cols].values.astype(np.float32)[-seq_len:]
    x_last = torch.from_numpy(last_window).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_vol_scaled = model(x_last).item()

    pred_vol = pred_vol_scaled * target_scale
    print("\nPredicted 15-min forward vol on last window:", pred_vol)
