"""
make_rl_test_report_lgb.py

Creates: rl_test_report_lgb.pkl

- Loads full processed dataframe (with DatetimeIndex)
- Loads trained LightGBM model (saved booster)
- Predicts sigma (vol forecast) for every row with valid features
- Writes pred_vol_15m_fwd column back into full dataframe aligned by timestamp
- Drops rows where pred_vol_15m_fwd is missing/invalid so RL never sees NaNs
- Saves result as rl_test_report_lgb.pkl

Expected columns in processed df:
  - Open, Volume
  - 12 microstructure features
Output is intended for RL evaluation script.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

# =============================================================================
# Paths / config
# =============================================================================

# This should be the "full processed" dataframe you want RL to run on.
# If you truly want to use the same dataset as your LGBM training script,
# keep it as below. If you have a separate "live_processed_REPORT.pkl",
# change PROCESSED_PATH accordingly.
PROCESSED_PATH = "live_processed_REPORT.pkl"

MODEL_PATH = os.path.join("light_plots", "lgb_model.txt")
FEATURES_PATH = os.path.join("light_plots", "lgb_feature_cols.json")

OUT_PATH = "rl_test_report_lgb.pkl"

PRED_COL = "pred_vol_15m_fwd"

# Optional safety floor (QLIKE-friendly; avoids exactly 0)
SIGMA_FLOOR = 1e-6

# =============================================================================
# Helpers
# =============================================================================


def load_feature_cols() -> list[str]:
    """Load feature cols from FEATURES_PATH if present, else fallback to known 12."""
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            cols = json.load(f)
        if not isinstance(cols, list) or len(cols) == 0:
            raise ValueError(
                f"{FEATURES_PATH} exists but does not contain a valid list."
            )
        return cols

    # Fallback
    return [
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


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # ---- sanity: files exist ----
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"Missing processed dataframe: {PROCESSED_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing LightGBM model file: {MODEL_PATH}")

    # ---- load feature list ----
    feature_cols = load_feature_cols()

    # ---- load df ----
    df_full = pd.read_pickle(PROCESSED_PATH).sort_index()

    if not isinstance(df_full.index, pd.DatetimeIndex):
        raise TypeError("Processed dataframe index must be a DatetimeIndex.")
    if df_full.index.has_duplicates:
        raise ValueError("Processed dataframe index has duplicates.")

    # RL eval script requires these (based on your transformer bridge)
    required_cols = ["Open", "Volume"] + feature_cols
    missing = [c for c in required_cols if c not in df_full.columns]
    if missing:
        raise KeyError(f"Missing required columns in processed df: {missing}")

    # ---- build df_valid for prediction (rows where all features exist) ----
    valid_mask = df_full[feature_cols].notna().all(axis=1)
    df_valid = df_full.loc[valid_mask].copy()
    if df_valid.empty:
        raise ValueError("No rows left after filtering NaNs in feature columns.")

    # ---- load model ----
    booster = lgb.Booster(model_file=MODEL_PATH)

    # ---- predict ----
    X = df_valid[feature_cols].values
    preds = booster.predict(X)

    preds = np.asarray(preds, dtype=float)

    # Safety: floor to keep positive sigma
    preds = np.maximum(preds, SIGMA_FLOOR)

    # ---- write back into full df aligned on index ----
    df_out = df_full.copy()
    df_out[PRED_COL] = np.nan
    df_out.loc[df_valid.index, PRED_COL] = preds

    # ---- drop rows where prediction missing/invalid so RL never sees NaNs ----
    df_out = df_out[df_out[PRED_COL].notna()].copy()
    df_out = df_out[np.isfinite(df_out[PRED_COL].values)].copy()

    # enforce strictly positive (should hold due to floor)
    if (df_out[PRED_COL] <= 0).any():
        bad = df_out[df_out[PRED_COL] <= 0].head(5)
        raise ValueError(f"Non-positive {PRED_COL} found. Example:\n{bad}")

    # ---- save ----
    df_out.to_pickle(OUT_PATH)
    print(f"Saved: {OUT_PATH}")
    print("Shape:", df_out.shape)
    print("Pred col stats:")
    print(df_out[PRED_COL].describe())
