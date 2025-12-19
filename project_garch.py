import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================================
# 0. Load your data
# =========================================
file_path = "full_df_processed.pkl"

try:
    df = pd.read_pickle(file_path)
    print("DataFrame successfully unpickled and loaded:")
    print(df.head())
except FileNotFoundError:
    raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    raise RuntimeError(f"An error occurred during unpickling: {e}")

# Make sure it's time-sorted and index is datetime if possible
df = df.sort_index()

# =========================================
# 1. Build 1-minute log returns
# =========================================
if "Close" not in df.columns:
    raise KeyError("DataFrame must contain a 'Close' column.")

# 1-min log returns (scaled to percent)
df["log_ret"] = np.log(df["Close"]).diff()
df["r"] = df["log_ret"] * 100.0

# Drop the first NaN and any other NaNs
returns = df["r"].dropna()

print("Returns series:")
print(returns.head())

# =========================================
# 2. Fit GARCH(1,1) on 1-min returns
# =========================================
am = arch_model(
    returns,
    vol="Garch",
    p=1,
    q=1,
    mean="Constant",
    dist="normal",
)

res = am.fit(disp="off")
print(res.summary())

# =========================================
# 3. 15-step-ahead (15-min) RV forecast series
# =========================================
h = 15  # 15 x 1-minute steps

# choose a burn-in before starting forecasts
start_idx = int(len(returns) * 0.5)
start_date = returns.index[start_idx]

fc = res.forecast(horizon=h, start=start_date)

# fc.variance: rows = forecast origin times, columns = steps 1...h
var_paths = fc.variance  # (T_forecast x h)

# sum over the horizon to get 15-min variance, then sqrt → 15-min sigma
var_15m_series = var_paths.sum(axis=1)
rv_15m_series = np.sqrt(var_15m_series)
rv_15m_series.name = "garch_rv15_forecast"

print("Sample of 15-min RV forecasts from GARCH:")
print(rv_15m_series.head())

# =========================================
# 4. Merge forecast back into df
# =========================================
# This assumes df.index is the same time axis as returns.index
df["garch_rv15_forecast"] = rv_15m_series

# =========================================
# 5. Align with your realized 15-min RV target
# =========================================
# Assumes you already have df["rv_15m_fwd"] computed in your pipeline
if "rv_15m_fwd" not in df.columns:
    raise KeyError(
        "DataFrame must contain 'rv_15m_fwd' as the realized 15-min RV target."
    )

mask = df["rv_15m_fwd"].notna() & df["garch_rv15_forecast"].notna()

realized = df.loc[mask, "rv_15m_fwd"]
pred = df.loc[mask, "garch_rv15_forecast"]

print("Aligned realized vs forecast shapes:", realized.shape, pred.shape)

# =========================================
# 6. Plot realized vs GARCH forecast
# =========================================
plt.figure(figsize=(12, 5))
plt.plot(realized.index, realized.values, label="Realized RV 15m", linewidth=1)
plt.plot(pred.index, pred.values, label="GARCH RV 15m forecast", linewidth=1)
plt.legend()
plt.title("15-min Realized Vol vs GARCH Forecast")
plt.xlabel("Time")
plt.ylabel("Volatility (same units as returns)")
plt.tight_layout()
plt.show()

# Optional: zoom into a subset
sample = slice(0, 2000)  # first 2000 points
plt.figure(figsize=(12, 5))
plt.plot(
    realized.index[sample],
    realized.values[sample],
    label="Realized RV 15m",
    linewidth=1,
)
plt.plot(
    pred.index[sample], pred.values[sample], label="GARCH RV 15m forecast", linewidth=1
)
plt.legend()
plt.title("15-min RV vs GARCH (zoomed)")
plt.tight_layout()
plt.show()

# =========================================
# 7. Error metrics (MAE, RMSE, QLIKE)
# =========================================
mae = mean_absolute_error(realized, pred)
rmse = mean_squared_error(realized, pred, squared=False)

print("MAE :", mae)
print("RMSE:", rmse)

# QLIKE on sigma (convert to variance)
eps = 1e-12
pred_var = np.maximum(pred**2, eps)
true_var = np.maximum(realized**2, eps)
qlike = np.mean(np.log(pred_var) + true_var / pred_var)

print("QLIKE:", qlike)
