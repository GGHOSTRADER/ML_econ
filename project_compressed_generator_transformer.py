"""
make_compressed_df_with_forecast.py

Glue script that:
1) Loads your processed dataframe (engineered features + scaled target + split).
2) Loads the transformer checkpoint + feature bundle.
3) Rebuilds sequences (length seq_len) ending at each timestamp t (t >= seq_len-1).
4) Runs inference to produce pred_vol_15m_fwd (TRUE units, sigma).
5) Saves compressed_df.pkl containing:
   - the 12 microstructure features
   - pred_vol_15m_fwd
   - (optional) split column for debugging

This is the artifact your DRL script expects at DATA_COMPRESSED_PATH.

USAGE:
  python make_compressed_df_with_forecast.py

Edit the PATHS section if your filenames differ.
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

PROCESSED_PATH = "full_df_processed_splits_REPORT_full_data.pkl"
BUNDLE_PATH = "feature_transform_bundle.joblib"
CKPT_PATH = "vol_forecaster_checkpoint_NEW_REPORT.pt"

OUT_PATH = "compressed_df_REPORT_transformer.pkl"  # what RL loads
OUT_PATH_FULL = "compressed_df_with_open_volume.pkl"  # optional convenience artifact

BATCH_SIZE = 512
NUM_WORKERS = 0  # keep 0 for Windows safety


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
    Builds sequences of length seq_len ending at specified integer endpoints.

    Returns:
        x_seq: (seq_len, n_features) float32
        end_idx: int64 endpoint index in the original dataframe
    """

    def __init__(self, X_all: np.ndarray, end_indices: np.ndarray, seq_len: int):
        self.X_all = X_all
        self.end_indices = np.asarray(end_indices, dtype=np.int64)
        self.seq_len = int(seq_len)

        if self.seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        self.end_indices = self.end_indices[self.end_indices >= (self.seq_len - 1)]
        if len(self.end_indices) == 0:
            raise ValueError("No valid endpoints: all < seq_len-1")

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, i):
        end = int(self.end_indices[i])
        start = end - self.seq_len + 1
        x_seq = self.X_all[start : end + 1]  # (seq_len, n_features)
        return torch.from_numpy(x_seq), np.int64(end)


@torch.no_grad()
def predict_all_endpoints(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds = []
    ends = []
    for x_seq, end_idx in loader:
        x_seq = x_seq.to(device)
        out = model(x_seq)  # (batch,)
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
    target_col = str(bundle.get("target_col", "rv_15m_fwd"))
    feature_cols = list(bundle.get("feature_cols", []))

    # fallback to your known 12 features if bundle doesn't have them
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

    df = pd.read_pickle(PROCESSED_PATH).sort_index()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Processed dataframe index must be a DatetimeIndex.")
    if df.index.has_duplicates:
        raise ValueError(
            "Processed dataframe index has duplicates. Fix before proceeding."
        )

    # integrity
    missing_f = [c for c in feature_cols if c not in df.columns]
    if missing_f:
        raise KeyError(f"Missing feature columns in processed df: {missing_f}")
    if target_col not in df.columns:
        print(f"NOTE: target column '{target_col}' not found (ok for inference).")

    # drop rows with missing features
    df = df.dropna(subset=feature_cols).copy()
    if df.empty:
        raise ValueError("No rows left after dropping NaNs in feature columns.")

    # ---- load checkpoint ----
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Missing checkpoint: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    # Pull hyperparams from checkpoint (source of truth)
    seq_len = int(ckpt.get("seq_len", 120))
    d_model = int(ckpt.get("d_model", 128))
    n_heads = int(ckpt.get("n_heads", 4))
    n_layers = int(ckpt.get("n_layers", 3))

    # sanity: features count must match
    if len(feature_cols) != int(ckpt.get("feature_cols_len", len(feature_cols))):
        # we didn't store feature_cols_len; just do a practical check:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = VolatilityForecaster(
        n_features=len(feature_cols),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # ---- build global arrays ----
    X_all = df[feature_cols].values.astype(np.float32)
    end_idx_all = np.arange(len(df), dtype=np.int64)

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
    preds_scaled, ends = predict_all_endpoints(model, loader, device=device)

    # preds are in the same scale as training targets (scaled sigma)
    preds_true = preds_scaled * target_scale

    # ---- write predictions into df aligned by endpoint index ----
    df_out = df.copy()
    df_out["pred_vol_15m_fwd"] = np.nan
    df_out["pred_vol_15m_fwd_scaled"] = np.nan

    # ends are integer positions into df_out
    df_out.iloc[ends, df_out.columns.get_loc("pred_vol_15m_fwd")] = preds_true
    df_out.iloc[ends, df_out.columns.get_loc("pred_vol_15m_fwd_scaled")] = preds_scaled

    # first seq_len-1 rows should remain NaN by construction
    n_nan = int(df_out["pred_vol_15m_fwd"].isna().sum())
    print(f"NaNs in pred_vol_15m_fwd (expected ~{seq_len-1}): {n_nan}")

    # ---- build compressed df ----
    keep_cols = feature_cols + ["pred_vol_15m_fwd"]
    # optional: keep split if present for debugging
    if "split" in df_out.columns:
        keep_cols.append("split")

    compressed_df = df_out[keep_cols].copy()

    # hard check: after the warmup period, forecasts should exist
    after_warmup = compressed_df.iloc[seq_len - 1 :]
    if after_warmup["pred_vol_15m_fwd"].isna().any():
        bad = after_warmup[after_warmup["pred_vol_15m_fwd"].isna()]
        raise ValueError(
            f"Forecast contains NaNs after warmup window. Example indices:\n{bad.head(5).index}"
        )

    # also enforce positive (should be, due to Softplus)
    if (after_warmup["pred_vol_15m_fwd"] <= 0).any():
        raise ValueError(
            "pred_vol_15m_fwd has non-positive values. Something is wrong."
        )

    compressed_df.to_pickle(OUT_PATH)
    print(f"Saved: {OUT_PATH}")
    print("compressed_df shape:", compressed_df.shape)

    # ---- optional: make RL simpler by saving Open/Volume alongside features ----
    if "Open" in df_out.columns and "Volume" in df_out.columns:
        out_full = df_out[["Open", "Volume"] + keep_cols].copy()
        out_full.to_pickle(OUT_PATH_FULL)
        print(f"Saved optional combined file: {OUT_PATH_FULL}")

    print("\nDone.")
