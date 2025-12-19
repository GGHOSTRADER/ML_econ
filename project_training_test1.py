import os
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


# =========================
# Model + Dataset (same as training)
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

        # free a bit of GPU mem as we go
        del x, y, out

    preds = np.concatenate(preds_all)
    trues = np.concatenate(trues_all)
    return preds, trues


# =========================
# Metrics + lag diagnostics
# =========================


def qlike_eval(true_sigma, pred_sigma, eps=1e-12):
    true_var = np.maximum(np.asarray(true_sigma) ** 2, eps)
    pred_var = np.maximum(np.asarray(pred_sigma) ** 2, eps)
    return np.mean(np.log(pred_var) + true_var / pred_var)


def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.mean((a - b) ** 2))


def best_lag_corr(y_true, y_pred, max_lag=30):
    """
    Find lag (in bars) that maximizes correlation between y_true and y_pred.
    Positive lag => predictions need to be shifted *forward* to align with truth
    (i.e., model is lagging).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    lags = range(-max_lag, max_lag + 1)
    best_lag = 0
    best_corr = np.nan

    for k in lags:
        if k < 0:
            a = y_true[-k:]
            b = y_pred[: len(a)]
        elif k > 0:
            a = y_true[:-k]
            b = y_pred[k:]
        else:
            a = y_true
            b = y_pred

        if len(a) < 2:
            continue

        c = np.corrcoef(a, b)[0, 1]
        if np.isnan(best_corr) or c > best_corr:
            best_corr = c
            best_lag = k

    corr0 = np.corrcoef(y_true, y_pred)[0, 1]
    return best_lag, best_corr, corr0


def best_shift_rmse(y_true, y_pred, max_lag=30):
    """
    Compare RMSE at lag 0 vs best possible RMSE over lags.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    lags = range(-max_lag, max_lag + 1)
    best_lag = 0
    best_rmse = None

    for k in lags:
        if k < 0:
            a = y_true[-k:]
            b = y_pred[: len(a)]
        elif k > 0:
            a = y_true[:-k]
            b = y_pred[k:]
        else:
            a = y_true
            b = y_pred

        if len(a) < 1:
            continue

        r = rmse(a, b)
        if best_rmse is None or r < best_rmse:
            best_rmse = r
            best_lag = k

    rmse0 = rmse(y_true, y_pred)
    return best_lag, best_rmse, rmse0


# =========================
# Inference + plotting
# =========================

if __name__ == "__main__":
    DATA_CACHE_PATH = "full_df_processed.pkl"
    MODEL_CHECKPOINT_PATH = "vol_forecaster_checkpoint.pt"
    PLOTS_DIR = "plots_infer"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1) Load processed dataframe (already normalized & with target)
    full_df = pd.read_pickle(DATA_CACHE_PATH)
    full_df = full_df.sort_index()

    # 2) Load checkpoint on CPU (not "gpu")
    ckpt = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu")

    feature_cols = ckpt["feature_cols"]
    target_scale = ckpt["target_scale"]
    seq_len = ckpt["seq_len"]
    d_model = ckpt["d_model"]
    n_heads = ckpt["n_heads"]
    n_layers = ckpt["n_layers"]
    target_col = "rv_15m_fwd"

    # 3) Rebuild train/val split exactly as in training (80/20, chronological)
    n = len(full_df)
    split_idx = int(n * 0.8)

    train_df = full_df.iloc[:split_idx]
    val_df = full_df.iloc[split_idx - seq_len + 1 :]

    # 4) Build validation dataset/loader
    val_ds = VolDataset(val_df, feature_cols, target_col, seq_len=seq_len)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, drop_last=False)

    # 5) Recreate model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = VolatilityForecaster(
        n_features=len(feature_cols),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 6) Inference on validation set
    val_preds_scaled, val_targets_scaled = get_predictions_and_targets(
        model, val_loader, device
    )
    val_preds = val_preds_scaled * target_scale
    val_targets = val_targets_scaled * target_scale

    # ---------- METRICS ON VALIDATION ----------
    mae = mean_absolute_error(val_targets, val_preds)
    mse = mean_squared_error(val_targets, val_preds)
    rmse_val = math.sqrt(mse)
    ql = qlike_eval(val_targets, val_preds)

    try:
        corr_val, _ = pearsonr(val_targets, val_preds)
    except Exception:
        corr_val = float("nan")

    print("\n=== TRANSFORMER VALIDATION METRICS ===")
    print(f"MAE   : {mae:.6f}")
    print(f"RMSE  : {rmse_val:.6f}")
    print(f"QLIKE : {ql:.6f}")
    print(f"Corr  : {corr_val:.4f}")

    max_lag = 30
    lag_corr_best, corr_best, corr0 = best_lag_corr(val_targets, val_preds, max_lag)
    lag_rmse_best, rmse_best, rmse0 = best_shift_rmse(val_targets, val_preds, max_lag)

    print("\n--- Lag diagnostics on VALIDATION ---")
    print(f"Max lag searched      : ±{max_lag} bars")
    print(f"Correlation at lag 0  : {corr0:.4f}")
    print(f"Best lag (corr)       : {lag_corr_best} bars")
    print(f"Corr at best lag      : {corr_best:.4f}")
    print(f"RMSE at lag 0         : {rmse0:.6f}")
    print(f"Best lag (RMSE)       : {lag_rmse_best} bars")
    print(f"RMSE at best lag      : {rmse_best:.6f}")

    # ---------- CONTINUE WITH PLOTTING ----------
    n_val = len(val_targets)

    # Align timestamps: first target corresponds to val_df.index[seq_len-1]
    offset = seq_len - 1
    time_for_targets = val_df.index[offset : offset + n_val]
    assert len(time_for_targets) == n_val

    # =========================
    # Plot 1: ALL points, x = step index
    # =========================
    steps_all = np.arange(n_val)

    plt.figure(figsize=(12, 4))
    plt.plot(steps_all, val_targets, label="Realized vol (val)", linewidth=1)
    plt.plot(steps_all, val_preds, label="Predicted vol (val)", linewidth=1)
    plt.title("Realized vs Predicted 15-min Vol (Validation, all points - steps)")
    plt.xlabel("Step index")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "val_all_steps.png"))
    plt.close()

    # =========================
    # Plot 2: ALL points, compressed time axis with date labels
    # =========================
    days_all = np.array([ts.date() for ts in time_for_targets])
    _, first_idx_all = np.unique(days_all, return_index=True)

    max_ticks = 10
    step_ticks_all = max(1, len(first_idx_all) // max_ticks)
    tick_positions_all = first_idx_all[::step_ticks_all]
    tick_labels_all = [
        time_for_targets[i].strftime("%Y-%m-%d") for i in tick_positions_all
    ]

    plt.figure(figsize=(12, 4))
    plt.plot(steps_all, val_targets, label="Realized vol (val)", linewidth=1)
    plt.plot(steps_all, val_preds, label="Predicted vol (val)", linewidth=1)
    plt.xticks(tick_positions_all, tick_labels_all, rotation=45)
    plt.title(
        "Realized vs Predicted 15-min Vol "
        "(Validation, all points - compressed dates)"
    )
    plt.xlabel("Compressed time (trading steps)")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "val_all_compressed_dates.png"))
    plt.close()

    # =========================
    # Tail selection (last 5000 or all if shorter)
    # =========================
    tail_size = min(5000, n_val)
    tail_slice = slice(n_val - tail_size, n_val)

    tail_targets = val_targets[tail_slice]
    tail_preds = val_preds[tail_slice]
    tail_times = time_for_targets[tail_slice]

    steps_tail = np.arange(tail_size)

    # =========================
    # Plot 3: tail (last 5000), x = step index
    # =========================
    plt.figure(figsize=(12, 4))
    plt.plot(steps_tail, tail_targets, label="Realized vol (val)", linewidth=1)
    plt.plot(steps_tail, tail_preds, label="Predicted vol (val)", linewidth=1)
    plt.title(
        f"Realized vs Predicted 15-min Vol "
        f"(Validation tail: last {tail_size} points - steps)"
    )
    plt.xlabel("Step index (tail)")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "val_tail_steps.png"))
    plt.close()

    # =========================
    # Plot 4: tail (last 5000), compressed time axis with date labels
    # =========================
    days_tail = np.array([ts.date() for ts in tail_times])
    _, first_idx_tail = np.unique(days_tail, return_index=True)

    max_ticks_tail = 10
    step_ticks_tail = max(1, len(first_idx_tail) // max_ticks_tail)
    tick_positions_tail = first_idx_tail[::step_ticks_tail]
    tick_labels_tail = [tail_times[i].strftime("%Y-%m-%d") for i in tick_positions_tail]

    plt.figure(figsize=(12, 4))
    plt.plot(steps_tail, tail_targets, label="Realized vol (val)", linewidth=1)
    plt.plot(steps_tail, tail_preds, label="Predicted vol (val)", linewidth=1)
    plt.xticks(tick_positions_tail, tick_labels_tail, rotation=45)
    plt.title(
        f"Realized vs Predicted 15-min Vol "
        f"(Validation tail: last {tail_size} points - compressed dates)"
    )
    plt.xlabel("Compressed time (trading steps, tail)")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "val_tail_compressed_dates.png"))
    plt.close()

    print(f"Done. Plots saved in: {PLOTS_DIR}")
