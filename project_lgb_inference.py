import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

# =========================================
# Config
# =========================================
DATA_PATH = "full_df_processed_splits_REPORT_full_data.pkl"

MODEL_PATH = "light_plots/lgb_model.txt"
FEATURES_PATH = "light_plots/lgb_feature_cols.json"

OUT_PRED_PATH = "light_plots/lgb_rv15_predictions_scaled_INFERENCE.pkl"

TARGET_COL = "rv_15m_fwd"  # optional (used only if present)
SPLIT_COL = "split"  # optional

SIGMA_FLOOR = 1e-6


# =========================================
# Load model
# =========================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

model = lgb.Booster(model_file=MODEL_PATH)
print(f"Loaded LightGBM model from {MODEL_PATH}")


# =========================================
# Load feature list (CRITICAL)
# =========================================
if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(f"Missing feature file: {FEATURES_PATH}")

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    feature_cols = json.load(f)

print(f"Loaded {len(feature_cols)} features")


# =========================================
# Load data
# =========================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Missing data file: {DATA_PATH}")

df = pd.read_pickle(DATA_PATH).sort_index()

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing required feature columns in data: {missing}")

X = df[feature_cols].copy()

# Drop rows with NaNs in features
valid_mask = X.notna().all(axis=1)
X_valid = X.loc[valid_mask]

print(f"Running inference on {len(X_valid)} rows")


# =========================================
# Predict (scaled sigma)
# =========================================
y_pred = model.predict(X_valid)

# Safety floor (same logic as training)
y_pred = np.maximum(y_pred, SIGMA_FLOOR)


# =========================================
# Assemble output
# =========================================
out_df = df.copy()
out_df["lgb_pred_sigma_scaled"] = np.nan
out_df.loc[X_valid.index, "lgb_pred_sigma_scaled"] = y_pred

# Optional: keep truth if available
if TARGET_COL in out_df.columns:
    out_df["rv_15m_fwd_scaled_true"] = out_df[TARGET_COL]

# Optional: keep split info if available
if SPLIT_COL in out_df.columns:
    out_df[SPLIT_COL] = out_df[SPLIT_COL]

out_df.to_pickle(OUT_PRED_PATH)
print(f"Saved inference predictions -> {OUT_PRED_PATH}")
