"""
make_compressed_df_with_lgb_forecast.py

Glue script that produces the SAME artifact your DRL expects (compressed_df.pkl),
but with LightGBM forecasts instead of Transformer forecasts.

Output:
  - compressed_df_lgb.pkl   (12 microstructure features + pred_vol_15m_fwd)
  - optional: compressed_df_lgb_with_open_volume.pkl (Open/Volume + features + forecast)

Key points (no silent foot-guns):
- Uses YOUR trained LightGBM model (lgb_model.txt) + feature list json.
- Generates predictions for ALL rows where the 12 features are present.
- Writes an index-aligned forecast column:
    pred_vol_15m_fwd  = predicted sigma in TRUE units (if target_scale available)
- Also stores:
    pred_vol_15m_fwd_scaled (the raw model output in scaled sigma units)
- Enforces a small floor so QLIKE / RL never sees 0 or negative sigma.

USAGE:
  python make_compressed_df_with_lgb_forecast.py

Then in project_rl.py, point DATA_COMPRESSED_PATH to the output file
(or rename it to compressed_df.pkl).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb


# =============================================================================
# Paths / config
# =============================================================================

# Source data (must contain the 12 engineered features)
PROCESSED_PATH = "full_df_processed_splits_REPORT_full_data.pkl"

# Trained LightGBM model + feature list created by your training script
PLOTS_DIR = "light_plots"
MODEL_PATH = os.path.join(PLOTS_DIR, "lgb_model.txt")
FEATURES_JSON = os.path.join(PLOTS_DIR, "lgb_feature_cols.json")

# OPTIONAL: bundle provides target_scale to convert scaled sigma -> true sigma
# If missing, we'll still output scaled predictions (and set pred_vol_15m_fwd == scaled)
BUNDLE_PATH = "feature_transform_bundle.joblib"

# Outputs (pick one to use in RL)
OUT_COMPRESSED = "compressed_df_REPORT_lgb.pkl"
OUT_COMPRESSED_WITH_OHLC = "compressed_df_lgb_with_open_volume.pkl"  # optional

# Safety floor for sigma
SIGMA_FLOOR = 1e-6

# The RL expects this column name
FORECAST_COL = "pred_vol_15m_fwd"


# =============================================================================
# Helpers
# =============================================================================


def require(path: str, msg: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{msg}: {path}")


def load_feature_cols(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cols = json.load(f)
    if not isinstance(cols, list) or not cols:
        raise ValueError(f"Feature list in {path} is empty or invalid.")
    return cols


def get_target_scale(bundle_path: str) -> float | None:
    if not os.path.exists(bundle_path):
        return None
    bundle = joblib.load(bundle_path)
    if "target_scale" not in bundle:
        return None
    try:
        return float(bundle["target_scale"])
    except Exception:
        return None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    require(PROCESSED_PATH, "Missing processed dataframe")
    require(MODEL_PATH, "Missing LightGBM model file")
    require(FEATURES_JSON, "Missing LightGBM feature list json")

    # ---- load processed df ----
    df = pd.read_pickle(PROCESSED_PATH).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Processed dataframe index must be a DatetimeIndex.")
    if df.index.has_duplicates:
        raise ValueError(
            "Processed dataframe index has duplicates. Fix before proceeding."
        )

    # ---- load features + model ----
    feature_cols = load_feature_cols(FEATURES_JSON)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Processed df is missing required feature columns: {missing}")

    model = lgb.Booster(model_file=MODEL_PATH)

    # ---- optional scale conversion ----
    target_scale = get_target_scale(BUNDLE_PATH)
    if target_scale is None:
        print(f"WARNING: {BUNDLE_PATH} missing or has no target_scale.")
        print("         Will output forecasts in *scaled* units only.")
        target_scale = 1.0

    # ---- build X (drop rows with any NaNs in features) ----
    X = df[feature_cols]
    valid_mask = X.notna().all(axis=1)
    if valid_mask.sum() == 0:
        raise ValueError("No valid rows: all feature rows contain NaNs.")

    X_valid = X.loc[valid_mask]

    print("Total rows:", len(df))
    print("Valid feature rows:", len(X_valid))

    # ---- predict (scaled sigma) ----
    pred_scaled = model.predict(
        X_valid, num_iteration=model.best_iteration or model.current_iteration()
    )
    pred_scaled = np.asarray(pred_scaled, dtype=np.float64)

    # enforce positive sigma (RL + QLIKE stability)
    pred_scaled = np.maximum(pred_scaled, SIGMA_FLOOR)

    # convert to TRUE sigma units if target_scale != 1
    pred_true = pred_scaled * float(target_scale)
    pred_true = np.maximum(pred_true, SIGMA_FLOOR)

    # ---- write back aligned to original index ----
    df_out = df.copy()
    df_out[f"{FORECAST_COL}_scaled"] = np.nan
    df_out[FORECAST_COL] = np.nan

    df_out.loc[X_valid.index, f"{FORECAST_COL}_scaled"] = pred_scaled
    df_out.loc[X_valid.index, FORECAST_COL] = pred_true

    # sanity
    after = df_out.loc[X_valid.index, FORECAST_COL]
    if (after <= 0).any():
        raise ValueError(
            "Forecast column has non-positive values after flooring (should not happen)."
        )

    # ---- build compressed df (what RL expects) ----
    keep_cols = feature_cols + [FORECAST_COL]
    # optionally keep split for debugging if present
    if "split" in df_out.columns:
        keep_cols.append("split")

    compressed_df = df_out[keep_cols].copy()

    # hard check: no NaNs in the compressed feature set where we have valid features
    if compressed_df.loc[X_valid.index, keep_cols].isna().any().any():
        bad_counts = (
            compressed_df.loc[X_valid.index, keep_cols]
            .isna()
            .sum()
            .sort_values(ascending=False)
        )
        raise ValueError(
            f"NaNs detected in compressed_df on valid rows:\n{bad_counts.head(20)}"
        )

    compressed_df.to_pickle(OUT_COMPRESSED)
    print(f"Saved -> {OUT_COMPRESSED}")
    print("compressed_df shape:", compressed_df.shape)

    # ---- optional: save Open/Volume alongside (lets you drop df_full join in RL) ----
    if "Open" in df_out.columns and "Volume" in df_out.columns:
        out_full = df_out[["Open", "Volume"] + keep_cols].copy()
        out_full.to_pickle(OUT_COMPRESSED_WITH_OHLC)
        print(f"Saved optional combined file -> {OUT_COMPRESSED_WITH_OHLC}")
    else:
        print("NOTE: Open/Volume not found in processed df; skipping combined file.")

    # ---- quick stats ----
    s = compressed_df.loc[X_valid.index, FORECAST_COL]
    print("\nForecast stats (TRUE units):")
    print(s.describe())

    print("\nDone.")
