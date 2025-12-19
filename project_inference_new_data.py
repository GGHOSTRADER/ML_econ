import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


# =========================================
# Model + Dataset (same as training)
# =========================================


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

        # Only the 12 features + target are used.
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

        del x, y, out

    preds = np.concatenate(preds_all)
    trues = np.concatenate(trues_all)
    return preds, trues


# =========================================
# Metrics, Lag, Spike
# =========================================


def qlike_eval(true_sigma, pred_sigma, eps=1e-12):
    true_var = np.maximum(np.asarray(true_sigma) ** 2, eps)
    pred_var = np.maximum(np.asarray(pred_sigma) ** 2, eps)
    return np.mean(np.log(pred_var) + true_var / pred_var)


def eval_block(name, y_true, y_hat):
    mae = mean_absolute_error(y_true, y_hat)
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    ql = qlike_eval(y_true, y_hat)

    try:
        corr, _ = pearsonr(y_true, y_hat)
    except Exception:
        corr = np.nan

    print(f"\n=== {name} ===")
    print(f"MAE   : {mae:.6f}")
    print(f"RMSE  : {rmse:.6f}")
    print(f"QLIKE : {ql:.6f}")
    print(f"Corr  : {corr:.4f}")


def best_lag_corr(y_true, y_pred, max_lag=30):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    lags = range(-max_lag, max_lag + 1)
    best_lag = 0
    best_corr = -9.0

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
        if c > best_corr:
            best_corr = c
            best_lag = k

    corr0 = np.corrcoef(y_true, y_pred)[0, 1]
    return best_lag, best_corr, corr0


def best_shift_rmse(y_true, y_pred, max_lag=30):
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

        r = np.sqrt(np.mean((a - b) ** 2))
        if best_rmse is None or r < best_rmse:
            best_rmse = r
            best_lag = k

    rmse0 = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return best_lag, best_rmse, rmse0


def spike_metrics(y_true, y_pred, quantile=0.95, max_lead=10):
    """
    Spike definition:
      - spike = top q of y_true
      - predicted spike = y_pred >= same threshold
      - capture window: last max_lead bars up to spike bar
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    thresh = np.quantile(y_true, quantile)
    true_spike = y_true >= thresh
    pred_spike = y_pred >= thresh

    n_true = true_spike.sum()
    n_pred = pred_spike.sum()
    tp = np.sum(true_spike & pred_spike)

    recall = tp / n_true if n_true > 0 else np.nan
    precision = tp / n_pred if n_pred > 0 else np.nan

    spike_indices = np.where(true_spike)[0]
    leads = []

    for i in spike_indices:
        j0 = max(0, i - max_lead)
        window_idx = np.arange(j0, i + 1)
        window_pred = pred_spike[window_idx]
        if window_pred.any():
            j_first = window_idx[window_pred.argmax()]
            lead = i - j_first
            leads.append(lead)

    n_captured = len(leads)
    frac_captured = n_captured / n_true if n_true > 0 else np.nan

    avg_lead = float(np.mean(leads)) if n_captured > 0 else np.nan
    min_lead = float(np.min(leads)) if n_captured > 0 else np.nan
    max_lead_val = float(np.max(leads)) if n_captured > 0 else np.nan

    print("\n--- Spike metrics (TEST, Transformer) ---")
    print(f"Spike threshold (q={quantile:.2f}) : {thresh:.6f}")
    print(f"Total spikes (true)               : {n_true}")
    print(f"Total predicted spikes            : {n_pred}")
    print(f"TP (spike & predicted spike)      : {tp}")
    print(f"Spike recall                      : {recall:.4f}")
    print(f"Spike precision                   : {precision:.4f}")
    print(f"Frac spikes captured (<= {max_lead} bars lead) : {frac_captured:.4f}")
    if n_captured > 0:
        print(
            f"Lead time (bars) avg/min/max      : "
            f"{avg_lead:.2f} / {min_lead:.0f} / {max_lead_val:.0f}"
        )
    else:
        print("Lead time stats                   : no spikes captured in window")

    return {
        "threshold": thresh,
        "n_true": int(n_true),
        "n_pred": int(n_pred),
        "tp": int(tp),
        "recall": recall,
        "precision": precision,
        "frac_captured": frac_captured,
        "avg_lead": avg_lead,
        "min_lead": min_lead,
        "max_lead": max_lead_val,
    }


# =========================================
# Inference + Evaluation + Plots
# =========================================

if __name__ == "__main__":
    DATA_CACHE_PATH = "new_data_full_df_processed.pkl"  # <<< input DF
    MODEL_CHECKPOINT_PATH = "vol_forecaster_checkpoint.pt"
    PLOTS_DIR = "plots_infer"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1) Load processed dataframe and keep only rows with target
    full_df = pd.read_pickle(DATA_CACHE_PATH).sort_index()
    target_col = "rv_15m_fwd"

    if target_col not in full_df.columns:
        raise KeyError(f"DataFrame must contain '{target_col}' as target.")

    # Only rows where target is present; OHLC columns (if any) are ignored for training
    full_df = full_df[full_df[target_col].notna()].copy()

    # 2) Load checkpoint (CPU is fine, we'll move model to device)
    ckpt = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu")

    feature_cols = ckpt["feature_cols"]  # these should be your 12 features
    target_scale = ckpt["target_scale"]
    seq_len = ckpt["seq_len"]
    d_model = ckpt["d_model"]
    n_heads = ckpt["n_heads"]
    n_layers = ckpt["n_layers"]

    # 3) Split indexes: 80/10/10 on the cleaned df
    n = len(full_df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_df = full_df.iloc[:train_end]
    val_df = full_df.iloc[train_end - seq_len + 1 : val_end]
    test_df = full_df.iloc[val_end - seq_len + 1 :]

    # 4) Build datasets/loaders
    train_ds = VolDataset(train_df, feature_cols, target_col, seq_len=seq_len)
    val_ds = VolDataset(val_df, feature_cols, target_col, seq_len=seq_len)
    test_ds = VolDataset(test_df, feature_cols, target_col, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

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

    # 6) Inference for each split (in scaled space)
    train_preds_s, train_targets_s = get_predictions_and_targets(
        model, train_loader, device
    )
    val_preds_s, val_targets_s = get_predictions_and_targets(model, val_loader, device)
    test_preds_s, test_targets_s = get_predictions_and_targets(
        model, test_loader, device
    )

    # 7) Rescale back to original volatility units
    train_preds = train_preds_s * target_scale
    train_targets = train_targets_s * target_scale

    val_preds = val_preds_s * target_scale
    val_targets = val_targets_s * target_scale

    test_preds = test_preds_s * target_scale
    test_targets = test_targets_s * target_scale

    # 8) Align timestamps for each split
    n_train_seq = len(train_ds)
    time_train = full_df.index[seq_len - 1 : seq_len - 1 + n_train_seq]

    n_val_seq = len(val_ds)
    time_val = full_df.index[train_end : train_end + n_val_seq]

    n_test_seq = len(test_ds)
    time_test = full_df.index[val_end : val_end + n_test_seq]

    assert len(time_train) == len(train_targets)
    assert len(time_val) == len(val_targets)
    assert len(time_test) == len(test_targets)

    # 9) Metrics per split
    eval_block("TRAIN", train_targets, train_preds)
    eval_block("VAL", val_targets, val_preds)
    eval_block("TEST", test_targets, test_preds)

    # 10) Lag diagnostics on TEST
    max_lag = 30
    lag_corr_best, corr_best, corr0 = best_lag_corr(
        test_targets, test_preds, max_lag=max_lag
    )
    lag_rmse_best, rmse_best, rmse0 = best_shift_rmse(
        test_targets, test_preds, max_lag=max_lag
    )

    print("\n--- Lag diagnostics on TEST (Transformer) ---")
    print(f"Max lag searched      : ±{max_lag} bars")
    print(f"Correlation at lag 0  : {corr0:.4f}")
    print(f"Best lag (corr)       : {lag_corr_best} bars")
    print(f"Corr at best lag      : {corr_best:.4f}")
    print(f"RMSE at lag 0         : {rmse0:.6f}")
    print(f"Best lag (RMSE)       : {lag_rmse_best} bars")
    print(f"RMSE at best lag      : {rmse_best:.6f}")

    # 11) Spike metrics on TEST
    _ = spike_metrics(test_targets, test_preds, quantile=0.95, max_lead=10)

    # 12) Save predictions + ALL features + OHLC into one pickle

    # Start from full_df so we keep all engineered features + OHLCV, flags, etc.
    pred_df = full_df.copy()

    # Store true-scale realized volatility in a separate column
    # (full_df[target_col] is assumed to be in scaled units)
    true_col = target_col + "_true"
    pred_df[true_col] = full_df[target_col] * target_scale

    # Column name for the forecast that RL will use as a feature
    pred_col = "pred_vol_15m_fwd"

    # Initialize prediction column with NaN, then fill train/val/test regions
    pred_df[pred_col] = np.nan
    pred_df.loc[time_train, pred_col] = train_preds
    pred_df.loc[time_val, pred_col] = val_preds
    pred_df.loc[time_test, pred_col] = test_preds

    out_path = "transformer_rv15_predictions_new_data.pkl"
    pred_df.to_pickle(out_path)
    print(f"\nSaved full feature + prediction DataFrame to '{out_path}'.")

    # 13) Plots on TEST set
    steps = np.arange(len(test_targets))
    dates = time_test.to_pydatetime()

    plt.figure(figsize=(12, 5))
    plt.plot(steps, test_targets, label="Realized RV 15m (test)", linewidth=1)
    plt.plot(steps, test_preds, label="Transformer forecast", linewidth=1)
    plt.title("Transformer vs Realized 15m Volatility (Test)")
    plt.xlabel("Step")
    plt.ylabel("Volatility")
    plt.legend()

    n_ticks = 10
    tick_pos = np.linspace(0, len(steps) - 1, n_ticks, dtype=int)
    tick_labels = [dates[i].strftime("%Y-%m-%d") for i in tick_pos]
    plt.xticks(tick_pos, tick_labels, rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "transformer_test_full_new.png"))
    plt.close()

    zoom_n = 5000
    start = max(0, len(test_targets) - zoom_n)

    steps_zoom = np.arange(len(test_targets) - start)
    test_targets_zoom = test_targets[start:]
    test_preds_zoom = test_preds[start:]
    dates_zoom = time_test.to_pydatetime()[start:]

    plt.figure(figsize=(12, 5))
    plt.plot(
        steps_zoom, test_targets_zoom, label="Realized RV 15m (test, tail)", linewidth=1
    )
    plt.plot(
        steps_zoom, test_preds_zoom, label="Transformer forecast (tail)", linewidth=1
    )
    plt.title(
        f"Transformer vs Realized 15m Volatility "
        f"(Last {len(steps_zoom)} Test Steps)"
    )
    plt.xlabel("Step")
    plt.ylabel("Volatility")
    plt.legend()

    n_ticks_zoom = 8
    tick_pos_zoom = np.linspace(0, len(steps_zoom) - 1, n_ticks_zoom, dtype=int)
    tick_labels_zoom = [dates_zoom[i].strftime("%Y-%m-%d") for i in tick_pos_zoom]
    plt.xticks(tick_pos_zoom, tick_labels_zoom, rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "transformer_test_tail_new.png"))
    plt.close()

    print(f"\nDone. Plots saved in: {PLOTS_DIR}")
