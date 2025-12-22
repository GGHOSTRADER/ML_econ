"""
make_rl_test_report.py

Creates: rl_test_report.pkl

- Loads full processed dataframe
- Runs transformer inference to produce pred_vol_15m_fwd (TRUE sigma)
- Adds pred_vol_15m_fwd column to the full dataframe
- Drops warmup rows (first seq_len-1 valid rows) so RL never sees NaNs
- Saves the result as rl_test_report.pkl

This output is intended to be fed into your RL evaluation script, which needs:
  - Open, Volume
  - 12 microstructure features
  - pred_vol_15m_fwd
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Paths / config
# =============================================================================

PROCESSED_PATH = "live_processed_REPORT.pkl"
BUNDLE_PATH = "feature_transform_bundle.joblib"
CKPT_PATH = "vol_forecaster_checkpoint_NEW_REPORT.pt"

OUT_PATH = "rl_test_report_trans.pkl"

BATCH_SIZE = 512
NUM_WORKERS = 0  # Windows-safe


# =============================================================================
# Model (must match training)
# =============================================================================


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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


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
            nn.Softplus(),  # positive sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        out = self.fc_out(x)
        return out.squeeze(-1)


# =============================================================================
# Dataset: sequences ending at specified endpoints
# =============================================================================


class EndSeqDataset(Dataset):
    """
    Builds sequences of length seq_len ending at integer endpoints.

    Returns:
      x_seq: (seq_len, n_features) float32
      end_idx: int64 endpoint index in df_valid (NOT in df_full)
    """

    def __init__(self, X_all: np.ndarray, end_indices: np.ndarray, seq_len: int):
        self.X_all = X_all
        self.end_indices = np.asarray(end_indices, dtype=np.int64)
        self.seq_len = int(seq_len)

        self.end_indices = self.end_indices[self.end_indices >= (self.seq_len - 1)]
        if len(self.end_indices) == 0:
            raise ValueError("No valid endpoints: all < seq_len-1")

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, i):
        end = int(self.end_indices[i])
        start = end - self.seq_len + 1
        x_seq = self.X_all[start : end + 1]
        return torch.from_numpy(x_seq), np.int64(end)


@torch.no_grad()
def predict_all_endpoints(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds = []
    ends = []
    for x_seq, end_idx in loader:
        x_seq = x_seq.to(device)
        out = model(x_seq)
        preds.append(out.detach().cpu().numpy())
        ends.append(end_idx.numpy())
    return np.concatenate(preds), np.concatenate(ends)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # ---- load bundle ----
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(f"Missing bundle file: {BUNDLE_PATH}")
    bundle = joblib.load(BUNDLE_PATH)

    target_scale = float(bundle["target_scale"])
    feature_cols = list(bundle.get("feature_cols", []))

    # fallback to known 12
    if not feature_cols:
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

    # ---- load processed df ----
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"Missing processed dataframe: {PROCESSED_PATH}")

    df_full = pd.read_pickle(PROCESSED_PATH).sort_index()

    if not isinstance(df_full.index, pd.DatetimeIndex):
        raise TypeError("Processed dataframe index must be a DatetimeIndex.")
    if df_full.index.has_duplicates:
        raise ValueError("Processed dataframe index has duplicates.")

    # RL eval script requires these
    required_cols = ["Open", "Volume"] + feature_cols
    missing_req = [c for c in required_cols if c not in df_full.columns]
    if missing_req:
        raise KeyError(f"Missing required columns in processed df: {missing_req}")

    # ---- create df_valid (rows where all features exist) for inference ----
    valid_mask = df_full[feature_cols].notna().all(axis=1)
    df_valid = df_full.loc[valid_mask].copy()
    if df_valid.empty:
        raise ValueError("No rows left after filtering NaNs in feature columns.")

    # mapping: df_valid iloc index -> df_full iloc index
    valid_positions_in_full = np.flatnonzero(valid_mask.values)

    # ---- load checkpoint ----
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Missing checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    seq_len = int(ckpt.get("seq_len", 120))
    d_model = int(ckpt.get("d_model", 128))
    n_heads = int(ckpt.get("n_heads", 4))
    n_layers = int(ckpt.get("n_layers", 3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = VolatilityForecaster(
        n_features=len(feature_cols),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # ---- build arrays & run inference ----
    X_all = df_valid[feature_cols].values.astype(np.float32)
    end_idx_all = np.arange(len(df_valid), dtype=np.int64)

    ds = EndSeqDataset(X_all=X_all, end_indices=end_idx_all, seq_len=seq_len)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Running inference for {len(ds):,} endpoints (seq_len={seq_len}) ...")
    preds_scaled, ends_valid = predict_all_endpoints(model, loader, device=device)
    preds_true = preds_scaled * target_scale

    # ---- write pred_vol_15m_fwd back into FULL df ----
    df_out = df_full.copy()
    df_out["pred_vol_15m_fwd"] = np.nan

    # ends_valid are positions in df_valid; map to df_full iloc positions
    ends_full = valid_positions_in_full[ends_valid]
    df_out.iloc[ends_full, df_out.columns.get_loc("pred_vol_15m_fwd")] = preds_true

    # ---- drop warmup NaNs so RL never sees NaNs ----
    # Warmup corresponds to the first (seq_len-1) rows in df_valid, but those may be scattered in df_full.
    # The clean rule for RL: keep only rows where pred_vol_15m_fwd is present and finite.
    df_out = df_out[df_out["pred_vol_15m_fwd"].notna()].copy()
    df_out = df_out[np.isfinite(df_out["pred_vol_15m_fwd"].values)].copy()

    # enforce positivity (should hold due to Softplus)
    if (df_out["pred_vol_15m_fwd"] <= 0).any():
        bad = df_out[df_out["pred_vol_15m_fwd"] <= 0].head(5)
        raise ValueError(f"Non-positive pred_vol_15m_fwd found. Example:\n{bad}")

    df_out.to_pickle(OUT_PATH)
    print(f"Saved: {OUT_PATH}")
    print("Shape:", df_out.shape)
