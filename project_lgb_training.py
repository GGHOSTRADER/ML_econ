import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# =========================================
# Config
# =========================================
FILE_PATH = "full_df_processed_splits_REPORT_full_data.pkl"
TARGET_COL = "rv_15m_fwd"  # already SCALED in your pipeline
SPLIT_COL = "split"

PLOTS_DIR = "light_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_OUT = os.path.join(PLOTS_DIR, "lgb_model.txt")
FEATURES_OUT = os.path.join(PLOTS_DIR, "lgb_feature_cols.json")
PRED_OUT = os.path.join(PLOTS_DIR, "lgb_rv15_predictions_scaled_REPORT.pkl")

# Safety floor for sigma predictions inside QLIKE
SIGMA_FLOOR = 1e-6

# Train limit (hard stop)
MAX_BOOST_ROUNDS = 200


# =========================================
# Load data
# =========================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Could not find {FILE_PATH}")

df = pd.read_pickle(FILE_PATH).sort_index()

for c in [TARGET_COL, SPLIT_COL]:
    if c not in df.columns:
        raise KeyError(f"Missing required column '{c}' in dataframe.")

df = df[df[SPLIT_COL].isin(["train", "val", "test"])].copy()
df = df[df[TARGET_COL].notna()].copy()

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
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing engineered feature columns: {missing}")

# Split by tag
train_df = df[df[SPLIT_COL] == "train"].copy()
val_df = df[df[SPLIT_COL] == "val"].copy()
test_df = df[df[SPLIT_COL] == "test"].copy()


def clean_xy(d: pd.DataFrame):
    X = d[feature_cols]
    y = d[TARGET_COL].astype(float)
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask]


X_train, y_train = clean_xy(train_df)
X_val, y_val = clean_xy(val_df)
X_test, y_test = clean_xy(test_df)

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)

# Save feature list for inference reproducibility
with open(FEATURES_OUT, "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, indent=2)
print(f"Saved feature columns -> {FEATURES_OUT}")


# =========================================
# QLIKE (stable)
# =========================================
def qlike_eval(true_sigma, pred_sigma, eps=1e-12, sigma_floor=SIGMA_FLOOR):
    true_sigma = np.asarray(true_sigma)
    pred_sigma = np.asarray(pred_sigma)

    # QLIKE requires strictly positive variance; enforce floor on sigma
    pred_sigma = np.maximum(pred_sigma, sigma_floor)

    true_var = np.maximum(true_sigma**2, eps)
    pred_var = np.maximum(pred_sigma**2, eps)
    return float(np.mean(np.log(pred_var) + true_var / pred_var))


def qlike_metric(y_pred, dataset, eps=1e-12):
    y_true = dataset.get_label()
    loss = qlike_eval(y_true, y_pred, eps=eps, sigma_floor=SIGMA_FLOOR)
    return "qlike", loss, False  # lower is better


# =========================================
# Train (hard stop at 200)
# =========================================
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "None",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}

evals_result = {}
callbacks = [
    # You can keep early stopping, but training will never exceed 200 anyway
    lgb.early_stopping(stopping_rounds=100, first_metric_only=True),
    lgb.log_evaluation(period=50),
    lgb.record_evaluation(evals_result),
]

print(f"Training LightGBM (max {MAX_BOOST_ROUNDS} rounds) on val QLIKE...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    valid_names=["train", "val"],
    num_boost_round=MAX_BOOST_ROUNDS,  # <<< HARD STOP
    feval=qlike_metric,
    callbacks=callbacks,
)

best_iter = (
    model.best_iteration if model.best_iteration is not None else MAX_BOOST_ROUNDS
)
print("Best iteration:", best_iter)

# Save model weights (save best iteration if available)
model.save_model(MODEL_OUT, num_iteration=best_iter)
print(f"Saved model -> {MODEL_OUT}")


# =========================================
# Plot train vs val QLIKE (NO y-cap)
# =========================================
train_qlike = evals_result["train"]["qlike"]
val_qlike = evals_result["val"]["qlike"]
iters = np.arange(1, len(train_qlike) + 1)

plt.figure(figsize=(9, 5))
plt.plot(iters, train_qlike, label="Train QLIKE (scaled)")
plt.plot(iters, val_qlike, label="Val QLIKE (scaled)")
plt.axvline(best_iter, linestyle="--", label="Best iteration")
plt.xlabel("Iteration")
plt.ylabel("QLIKE (scaled)")
plt.title("LightGBM Training vs Validation QLIKE")
plt.legend()
plt.tight_layout()
qlike_plot_path = os.path.join(PLOTS_DIR, "lgb_qlike_curve_scaled.png")
plt.savefig(qlike_plot_path)
plt.close()
print(f"Saved plot -> {qlike_plot_path}")


# =========================================
# Predictions + evaluation (scaled)
# =========================================
def eval_block(name, y_true, y_hat):
    mae = mean_absolute_error(y_true, y_hat)
    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
    ql = qlike_eval(y_true, y_hat)
    try:
        corr, _ = pearsonr(y_true, y_hat)
    except Exception:
        corr = np.nan

    print(f"\n=== {name} (SCALED) ===")
    print(f"MAE   : {mae:.6f}")
    print(f"RMSE  : {rmse:.6f}")
    print(f"QLIKE : {ql:.6f}")
    print(f"Corr  : {corr:.4f}")


y_train_pred = model.predict(X_train, num_iteration=best_iter)
y_val_pred = model.predict(X_val, num_iteration=best_iter)
y_test_pred = model.predict(X_test, num_iteration=best_iter)

eval_block("TRAIN", y_train.values, y_train_pred)
eval_block("VAL", y_val.values, y_val_pred)
eval_block("TEST", y_test.values, y_test_pred)

# Save predictions dataframe
pred_df = df[[SPLIT_COL]].copy()
pred_df["rv_15m_fwd_scaled_true"] = df[TARGET_COL]
pred_df["lgb_pred_sigma_scaled"] = np.nan
pred_df.loc[X_train.index, "lgb_pred_sigma_scaled"] = y_train_pred
pred_df.loc[X_val.index, "lgb_pred_sigma_scaled"] = y_val_pred
pred_df.loc[X_test.index, "lgb_pred_sigma_scaled"] = y_test_pred

pred_df.to_pickle(PRED_OUT)
print(f"\nSaved predictions -> {PRED_OUT}")


# =========================================
# Plot TEST series (scaled) — from Nov 19 onward
# =========================================
test_year = pd.Timestamp(y_test.index.max()).year
START_DATE = pd.Timestamp(year=test_year, month=11, day=19)

test_plot_mask = (pred_df[SPLIT_COL] == "test") & (pred_df.index >= START_DATE)
plot_df = pred_df.loc[
    test_plot_mask, ["rv_15m_fwd_scaled_true", "lgb_pred_sigma_scaled"]
].dropna()

if plot_df.empty:
    raise ValueError(f"No TEST rows found on/after {START_DATE.date()} to plot.")

y_true_plot = plot_df["rv_15m_fwd_scaled_true"].values
y_pred_plot = plot_df["lgb_pred_sigma_scaled"].values
dates = plot_df.index.to_pydatetime()
steps = np.arange(len(plot_df))

plt.figure(figsize=(12, 5))
plt.plot(steps, y_true_plot, label="Realized RV 15m (scaled)", linewidth=1)
plt.plot(steps, y_pred_plot, label="LightGBM forecast (scaled)", linewidth=1)
plt.title(
    f"LightGBM vs Realized 15m Volatility (TEST, scaled) — from {START_DATE.date()}"
)
plt.xlabel("Step")
plt.ylabel("Volatility (scaled)")
plt.legend()

n_ticks = 10
tick_pos = np.linspace(0, len(steps) - 1, min(n_ticks, len(steps)), dtype=int)
tick_labels = [dates[i].strftime("%Y-%m-%d") for i in tick_pos]
plt.xticks(tick_pos, tick_labels, rotation=45)

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "lgb_test_series_scaled_from_nov19.png")
plt.savefig(out_path)
plt.close()
print(f"Saved plot -> {out_path}")
