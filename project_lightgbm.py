import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# =========================================
# 0. Load your processed data
# =========================================
file_path = "full_df_processed.pkl"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find {file_path}")

df = pd.read_pickle(file_path)
df = df.sort_index()

TARGET_COL = "rv_15m_fwd"
if TARGET_COL not in df.columns:
    raise KeyError(f"DataFrame must contain '{TARGET_COL}' as the volatility target.")

df = df[df[TARGET_COL].notna()].copy()

# =========================================
# 1. Build feature matrix X and target y (sigma)
# =========================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

exclude_cols = [
    TARGET_COL,
    # e.g. "garch_rv15_forecast",
]

feature_cols = [c for c in numeric_cols if c not in exclude_cols]

print("Number of feature columns:", len(feature_cols))
print("Feature columns sample:", feature_cols[:15])

X = df[feature_cols].copy()
y = df[TARGET_COL].astype(float).copy()  # sigma (vol)

mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]
time_index = df.index[mask]

print("Final dataset shape:", X.shape, y.shape)

# =========================================
# 2. Time-based train / val / test split
# =========================================
n = len(X)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("Train size:", len(X_train))
print("Val size  :", len(X_val))
print("Test size :", len(X_test))

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)


# =========================================
# 3. Custom QLIKE metric (on sigma)
# =========================================
def qlike_metric(y_pred, train_data, eps=1e-12):
    """
    Custom QLIKE evaluation for LightGBM.
    y_pred: predicted sigma (vol)
    """
    y_true = train_data.get_label()
    true_var = np.maximum(y_true**2, eps)
    pred_var = np.maximum(y_pred**2, eps)
    loss = np.log(pred_var) + true_var / pred_var
    return "qlike", float(loss.mean()), False  # lower is better


# =========================================
# 4. LightGBM params
# =========================================
params = {
    "objective": "regression",  # standard L2 regression
    "metric": "None",  # we only use custom qlike metric
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}

print("Training LightGBM model with QLIKE eval metric...")

evals_result = {}
callbacks = [
    lgb.early_stopping(stopping_rounds=100),
    lgb.log_evaluation(period=100),
    lgb.record_evaluation(evals_result),
]

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    valid_names=["train", "val"],
    num_boost_round=2000,
    feval=qlike_metric,  # evaluate using QLIKE
    callbacks=callbacks,
)

print("Best iteration:", model.best_iteration)

# =========================================
# 5. Plot training vs validation QLIKE
# =========================================
train_qlike = evals_result["train"]["qlike"]
val_qlike = evals_result["val"]["qlike"]
iters = np.arange(1, len(train_qlike) + 1)

plt.figure(figsize=(8, 5))
plt.plot(iters, train_qlike, label="Train QLIKE")
plt.plot(iters, val_qlike, label="Val QLIKE")
plt.axvline(model.best_iteration, linestyle="--", label="Best iteration")
plt.xlabel("Iteration")
plt.ylabel("QLIKE")
plt.title("LightGBM Training vs Validation QLIKE")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================
# 6. Predictions: train / val / test (sigma)
# =========================================
y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)


# =========================================
# 7. Evaluation metrics
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


eval_block("TRAIN", y_train.values, y_train_pred)
eval_block("VAL", y_val.values, y_val_pred)
eval_block("TEST", y_test.values, y_test_pred)


# =========================================
# 7b. Lag diagnostics on TEST
# =========================================
def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.mean((a - b) ** 2))


def best_lag_corr(y_true, y_pred, max_lag=30):
    """
    Find lag (in bars) that maximizes correlation between y_true and y_pred.
    Positive lag means y_pred is shifted forward (pred lags true).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    lags = range(-max_lag, max_lag + 1)
    best_lag = 0
    best_corr = np.nan

    for k in lags:
        if k < 0:
            a = y_true[-k:]  # drop first |k| from true
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

    # corr at lag 0 for reference
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


max_lag = 30  # bars to search in each direction

lag_corr_best, corr_best, corr0 = best_lag_corr(y_test.values, y_test_pred, max_lag)
lag_rmse_best, rmse_best, rmse0 = best_shift_rmse(y_test.values, y_test_pred, max_lag)

print("\n--- Lag diagnostics on TEST ---")
print(f"Max lag searched      : ±{max_lag} bars")
print(f"Correlation at lag 0  : {corr0:.4f}")
print(f"Best lag (corr)       : {lag_corr_best} bars")
print(f"Corr at best lag      : {corr_best:.4f}")
print(f"RMSE at lag 0         : {rmse0:.6f}")
print(f"Best lag (RMSE)       : {lag_rmse_best} bars")
print(f"RMSE at best lag      : {rmse_best:.6f}")


# =========================================
# 7c. SPIKE CAPTURE METRICS on TEST
# =========================================
def spike_metrics(y_true, y_pred, quantile=0.95, max_lead=10):
    """
    Evaluate how well the model captures volatility spikes.

    - Spike defined as y_true >= q-quantile of y_true
    - A predicted spike is y_pred >= same threshold
    - We compute:
        * Spike recall
        * Spike precision
        * Fraction of spikes with at least one predicted spike
          in the last `max_lead` bars (including the spike bar)
        * Lead time stats (in bars) for captured spikes
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

    # Lead-time based capture: did model signal within last max_lead bars?
    spike_indices = np.where(true_spike)[0]
    leads = []

    for i in spike_indices:
        j0 = max(0, i - max_lead)
        window_idx = np.arange(j0, i + 1)
        window_pred_spike = pred_spike[window_idx]

        if window_pred_spike.any():
            # earliest predicted spike in the lookback window
            j_first = window_idx[window_pred_spike.argmax()]
            lead = i - j_first  # how many bars before the spike
            leads.append(lead)

    n_captured = len(leads)
    frac_captured = n_captured / n_true if n_true > 0 else np.nan

    avg_lead = float(np.mean(leads)) if n_captured > 0 else np.nan
    min_lead = float(np.min(leads)) if n_captured > 0 else np.nan
    max_lead_val = float(np.max(leads)) if n_captured > 0 else np.nan

    print("\n--- Spike metrics on TEST ---")
    print(f"Spike threshold (q={quantile:.2f}) : {thresh:.6f}")
    print(f"Total spikes (true)               : {n_true}")
    print(f"Total predicted spikes            : {n_pred}")
    print(f"TP (spike & predicted spike)      : {tp}")
    print(f"Spike recall                      : {recall:.4f}")
    print(f"Spike precision                   : {precision:.4f}")
    print(f"Frac spikes captured (<= {max_lead} bars lead) : {frac_captured:.4f}")
    if n_captured > 0:
        print(
            f"Lead time (bars) avg/min/max      : {avg_lead:.2f} / {min_lead:.0f} / {max_lead_val:.0f}"
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


_ = spike_metrics(y_test.values, y_test_pred, quantile=0.95, max_lead=10)

# =========================================
# 8. Save predictions with timestamps
# =========================================
pred_df = pd.DataFrame(
    {
        "Datetime": time_index,
        "rv_15m_fwd": y.values,
    }
).set_index("Datetime")

pred_df.loc[X_train.index, "lgb_pred_sigma"] = y_train_pred
pred_df.loc[X_val.index, "lgb_pred_sigma"] = y_val_pred
pred_df.loc[X_test.index, "lgb_pred_sigma"] = y_test_pred

pred_df.to_pickle("lgb_rv15_predictions_qlike.pkl")
print("\nSaved LightGBM QLIKE-trained predictions to 'lgb_rv15_predictions_qlike.pkl'.")

# =========================================
# 9. Plot test set with step index + date labels
# =========================================
steps = np.arange(len(y_test))
dates = y_test.index.to_pydatetime()

plt.figure(figsize=(12, 5))
plt.plot(steps, y_test.values, label="Realized RV 15m (test)", linewidth=1)
plt.plot(steps, y_test_pred, label="LightGBM QLIKE forecast", linewidth=1)
plt.title("LightGBM (QLIKE eval) vs Realized 15m Volatility (Test)")
plt.xlabel("Step")
plt.ylabel("Volatility")
plt.legend()

n_ticks = 10
tick_pos = np.linspace(0, len(steps) - 1, n_ticks, dtype=int)
tick_labels = [dates[i].strftime("%Y-%m-%d") for i in tick_pos]
plt.xticks(tick_pos, tick_labels, rotation=45)

plt.tight_layout()
plt.show()

# =========================================
# 10. Zoomed plot: last 5000 test steps
# =========================================
zoom_n = 5000
start = max(0, len(y_test) - zoom_n)

steps_zoom = np.arange(len(y_test) - start)
y_test_zoom = y_test.values[start:]
y_test_pred_zoom = y_test_pred[start:]
dates_zoom = y_test.index.to_pydatetime()[start:]

plt.figure(figsize=(12, 5))
plt.plot(steps_zoom, y_test_zoom, label="Realized RV 15m (test, tail)", linewidth=1)
plt.plot(
    steps_zoom, y_test_pred_zoom, label="LightGBM QLIKE forecast (tail)", linewidth=1
)
plt.title(
    f"LightGBM (QLIKE eval) vs Realized 15m Volatility (Last {len(steps_zoom)} Test Steps)"
)
plt.xlabel("Step")
plt.ylabel("Volatility")
plt.legend()

n_ticks_zoom = 8
tick_pos_zoom = np.linspace(0, len(steps_zoom) - 1, n_ticks_zoom, dtype=int)
tick_labels_zoom = [dates_zoom[i].strftime("%Y-%m-%d") for i in tick_pos_zoom]
plt.xticks(tick_pos_zoom, tick_labels_zoom, rotation=45)

plt.tight_layout()
plt.show()
