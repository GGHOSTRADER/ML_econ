import os
import math
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error


# =========================
# QLIKE loss (sigma-space)
# =========================


def qlike_loss(
    preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    QLIKE loss for volatility forecasts (sigma). Computed in the SAME scale as targets.
    If your dataset target is scaled by target_scale, this is "scaled-space" QLIKE.
    """
    pred_var = torch.clamp(preds**2, min=eps)
    true_var = torch.clamp(targets**2, min=eps)
    return (torch.log(pred_var) + true_var / pred_var).mean()


def qlike_eval_sigma_np(y_true_sigma, y_pred_sigma, eps=1e-12) -> float:
    """
    QLIKE evaluation in sigma-space for numpy arrays.
    """
    y_true_sigma = np.asarray(y_true_sigma)
    y_pred_sigma = np.asarray(y_pred_sigma)
    true_var = np.maximum(y_true_sigma**2, eps)
    pred_var = np.maximum(y_pred_sigma**2, eps)
    return float(np.mean(np.log(pred_var) + true_var / pred_var))


# =========================
# Transformer model
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
            nn.Softplus(),  # enforce positive sigma
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        out = self.fc_out(x)
        return out.squeeze(-1)


# =========================
# Dataset: sequences ending on split rows
# =========================


class SplitEndSeqDataset(Dataset):
    """
    Builds sequences of length seq_len ending at specified row indices (end_indices).
    The last row in the sequence is the target row.
    """

    def __init__(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        end_indices: np.ndarray,
        seq_len: int,
    ):
        self.X_all = X_all
        self.y_all = y_all
        self.end_indices = np.asarray(end_indices, dtype=np.int64)
        self.seq_len = int(seq_len)

        if self.seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        # keep only endpoints that can form a full sequence
        self.end_indices = self.end_indices[self.end_indices >= (self.seq_len - 1)]
        if len(self.end_indices) == 0:
            raise ValueError("No valid sequences: all endpoints are < seq_len-1.")

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, i):
        end = int(self.end_indices[i])
        start = end - self.seq_len + 1

        x_seq = self.X_all[start : end + 1]  # (seq_len, n_features)
        y_t = self.y_all[end]  # scalar

        return torch.from_numpy(x_seq), torch.tensor(y_t, dtype=torch.float32)


@torch.no_grad()
def get_predictions_and_targets(model, loader, device):
    model.eval()
    preds_all, trues_all = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        preds_all.append(out.cpu().numpy())
        trues_all.append(y.numpy())
    return np.concatenate(preds_all), np.concatenate(trues_all)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    n_obs = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total += loss.item() * bs
        n_obs += bs
    return total / max(1, n_obs)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n_obs = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x)
        loss = criterion(preds, y)

        bs = x.size(0)
        total += loss.item() * bs
        n_obs += bs
    return total / max(1, n_obs)


# =========================
# Main training (NO feature engineering)
# =========================

if __name__ == "__main__":
    # ---------------------
    # Inputs (already processed)
    # ---------------------
    PROCESSED_PATH = "full_df_processed_splits_REPORT_full_data.pkl"  # dataframe with engineered+transformed features + scaled target + split
    BUNDLE_PATH = "feature_transform_bundle.joblib"  # contains target_scale and feature_cols, etc.
    MODEL_OUT = "vol_forecaster_checkpoint_NEW_REPORT.pt"
    PLOTS_DIR = "plots_train_from_processed_trueqlike"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---------------------
    # Hyperparams
    # ---------------------
    seq_len = 120
    batch_size = 256
    n_epochs = 6
    lr = 1e-4
    weight_decay = 1e-5

    d_model = 128
    n_heads = 4
    n_layers = 3

    # ---------------------
    # Load bundle (source of truth)
    # ---------------------
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(f"Missing bundle file: {BUNDLE_PATH}")

    bundle = joblib.load(BUNDLE_PATH)

    # These should exist because you saved them in build_processed_dataset()
    target_scale = float(bundle["target_scale"])
    target_col = str(bundle.get("target_col", "rv_15m_fwd"))
    feature_cols = list(bundle.get("feature_cols", []))

    if not feature_cols:
        # fallback if bundle doesn't have feature_cols for some reason
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

    # ---------------------
    # Load processed dataframe
    # ---------------------
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"Missing processed dataframe: {PROCESSED_PATH}")

    full_df = pd.read_pickle(PROCESSED_PATH).sort_index()

    if "split" not in full_df.columns:
        raise KeyError(
            "Processed dataframe must include a 'split' column with values train/val/test."
        )

    # integrity checks
    missing_f = [c for c in feature_cols if c not in full_df.columns]
    if missing_f:
        raise KeyError(f"Missing feature columns in processed df: {missing_f}")

    if target_col not in full_df.columns:
        raise KeyError(f"Missing target column '{target_col}' in processed df")

    # drop rows with missing required data
    full_df = full_df.dropna(subset=feature_cols + [target_col, "split"]).copy()

    # Global arrays (chronological)
    X_all = full_df[feature_cols].values.astype(np.float32)
    y_all = full_df[target_col].values.astype(np.float32)  # NOTE: this is scaled target
    split_arr = full_df["split"].astype(str).values
    idx_all = np.arange(len(full_df))

    train_end_idx = idx_all[split_arr == "train"]
    val_end_idx = idx_all[split_arr == "val"]
    test_end_idx = idx_all[split_arr == "test"]

    if len(train_end_idx) == 0 or len(val_end_idx) == 0 or len(test_end_idx) == 0:
        raise ValueError(
            "One or more splits are empty. Check 'split' values in the processed dataset."
        )

    # Datasets/loaders: sequences end at indices in each split
    train_ds = SplitEndSeqDataset(X_all, y_all, train_end_idx, seq_len=seq_len)
    val_ds = SplitEndSeqDataset(X_all, y_all, val_end_idx, seq_len=seq_len)
    test_ds = SplitEndSeqDataset(X_all, y_all, test_end_idx, seq_len=seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # ---------------------
    # Model / optimizer
    # ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = VolatilityForecaster(
        n_features=len(feature_cols),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)

    criterion = qlike_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---------------------
    # Train with BOTH:
    # - scaled-space QLIKE (optimization)
    # - true-space QLIKE (comparison)
    # ---------------------
    train_qlike_scaled_hist = []
    val_qlike_scaled_hist = []

    train_qlike_true_hist = []
    val_qlike_true_hist = []

    val_mae_true_hist = []
    val_corr_true_hist = []

    for epoch in range(1, n_epochs + 1):
        train_loss_scaled = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss_scaled = eval_one_epoch(model, val_loader, criterion, device)

        # Record scaled-space QLIKE (loss)
        train_qlike_scaled_hist.append(train_loss_scaled)
        val_qlike_scaled_hist.append(val_loss_scaled)

        # Compute true-space QLIKE on full val set (for fair comparison vs LightGBM)
        val_preds_s, val_targets_s = get_predictions_and_targets(
            model, val_loader, device
        )

        val_preds_true = val_preds_s * target_scale
        val_targets_true = val_targets_s * target_scale

        val_qlike_true = qlike_eval_sigma_np(val_targets_true, val_preds_true)
        val_mae_true = mean_absolute_error(val_targets_true, val_preds_true)

        if np.std(val_targets_true) > 0 and np.std(val_preds_true) > 0:
            val_corr_true = float(np.corrcoef(val_targets_true, val_preds_true)[0, 1])
        else:
            val_corr_true = float("nan")

        # For train true-QLIKE curve, compute occasionally or every epoch (costly but ok)
        train_preds_s, train_targets_s = get_predictions_and_targets(
            model, train_loader, device
        )
        train_preds_true = train_preds_s * target_scale
        train_targets_true = train_targets_s * target_scale
        train_qlike_true = qlike_eval_sigma_np(train_targets_true, train_preds_true)

        train_qlike_true_hist.append(train_qlike_true)
        val_qlike_true_hist.append(val_qlike_true)

        val_mae_true_hist.append(val_mae_true)
        val_corr_true_hist.append(val_corr_true)

        print(
            f"Epoch {epoch:02d} | "
            f"train QLIKE (scaled): {train_loss_scaled:.6e} | "
            f"val QLIKE (scaled): {val_loss_scaled:.6e} || "
            f"train QLIKE (TRUE): {train_qlike_true:.6e} | "
            f"val QLIKE (TRUE): {val_qlike_true:.6e} | "
            f"val MAE (TRUE): {val_mae_true:.6e} | "
            f"val Corr (TRUE): {val_corr_true:.4f}"
        )

    # ---------------------
    # Save checkpoint (include target_scale!)
    # ---------------------
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "target_col": target_col,
        "target_scale": float(target_scale),
        "seq_len": int(seq_len),
        "d_model": int(d_model),
        "n_heads": int(n_heads),
        "n_layers": int(n_layers),
    }
    torch.save(checkpoint, MODEL_OUT)
    print(f"\nSaved model checkpoint to {MODEL_OUT}")

    # ---------------------
    # Plot: QLIKE vs epoch (TRUE SCALE)  ✅ what you asked for
    # ---------------------
    plt.figure(figsize=(6, 4))
    plt.plot(
        range(1, n_epochs + 1),
        train_qlike_true_hist,
        marker="o",
        label="Train QLIKE (true units)",
    )
    plt.plot(
        range(1, n_epochs + 1),
        val_qlike_true_hist,
        marker="o",
        label="Val QLIKE (true units)",
    )

    plt.title("QLIKE vs Epoch (TRUE units, log y)")
    plt.xlabel("Epoch")
    plt.ylabel("QLIKE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "qlike_epochs_true_units_new_REPORT.png"))
    plt.close()

    # (Optional) also save the scaled-loss plot so you can debug optimization
    plt.figure(figsize=(6, 4))
    plt.plot(
        range(1, n_epochs + 1),
        train_qlike_scaled_hist,
        marker="o",
        label="Train QLIKE (scaled loss)",
    )
    plt.plot(
        range(1, n_epochs + 1),
        val_qlike_scaled_hist,
        marker="o",
        label="Val QLIKE (scaled loss)",
    )
    plt.yscale("log")
    plt.title("QLIKE vs Epoch (scaled loss, log y)")
    plt.xlabel("Epoch")
    plt.ylabel("QLIKE loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "qlike_epochs_scaled_loss_new_REPORT.png"))
    plt.close()

    # Plot val correlation (true units) vs epoch
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, n_epochs + 1), val_corr_true_hist, marker="o")
    plt.title("Validation Correlation vs Epoch (TRUE units)")
    plt.xlabel("Epoch")
    plt.ylabel("Pearson correlation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "val_corr_epochs_true_new_REPORT.png"))
    plt.close()

    # ---------------------
    # Final TEST evaluation (TRUE units)
    # ---------------------
    test_preds_s, test_targets_s = get_predictions_and_targets(
        model, test_loader, device
    )
    test_preds_true = test_preds_s * target_scale
    test_targets_true = test_targets_s * target_scale

    test_qlike_true = qlike_eval_sigma_np(test_targets_true, test_preds_true)
    test_mae_true = mean_absolute_error(test_targets_true, test_preds_true)
    if np.std(test_targets_true) > 0 and np.std(test_preds_true) > 0:
        test_corr_true = float(np.corrcoef(test_targets_true, test_preds_true)[0, 1])
    else:
        test_corr_true = float("nan")

    print("\n=== TEST (TRUE units) ===")
    print(f"MAE   : {test_mae_true:.6e}")
    print(f"QLIKE : {test_qlike_true:.6e}")
    print(f"Corr  : {test_corr_true:.4f}")

    print(f"\nDone. Plots saved in: {PLOTS_DIR}")
