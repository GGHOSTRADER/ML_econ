import os
import pandas as pd
import numpy as np
import talib as ta

from sklearn.preprocessing import RobustScaler


# =========================
# Helper functions
# =========================


def parkinson_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Parkinson volatility estimator using High/Low.

    sigma^2 = (1 / (4 * ln 2)) * E[(ln(High/Low))^2] over a rolling window
    """
    hl = np.log(df["High"] / df["Low"])
    sigma2 = (hl**2).rolling(window).mean() / (4 * np.log(2))
    return np.sqrt(sigma2)


def order_flow_imbalance(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Simple order flow imbalance using Up (buy) and Down (sell) volume.

    OFI(window) = rolling sum of (Up - Down) over the last `window` bars.
    """
    net_of = df["Up"] - df["Down"]
    return net_of.rolling(window).sum()


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile of the *latest* value within its rolling window.
    Returns in [0, 1], not 0–100.
    """

    def last_rank_pct(x: pd.Series) -> float:
        if len(x) <= 1:
            return np.nan
        rank = x.rank().iloc[-1]  # 1..len(x)
        return (rank - 1) / (len(x) - 1)

    return series.rolling(window).apply(last_rank_pct, raw=False)


def get_minutes_since_open(
    index: pd.DatetimeIndex,
    open_time: str = "09:30",
) -> pd.Series:
    """
    Minutes since the (fixed) session open each day.

    Bars before open get 0 (clamped).
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex for get_minutes_since_open")

    open_t = pd.to_datetime(open_time).time()

    mins = []
    for ts in index:
        session_open = pd.Timestamp.combine(ts.date(), open_t)
        delta_min = (ts - session_open).total_seconds() / 60.0
        mins.append(max(delta_min, 0.0))

    return pd.Series(mins, index=index)


def get_session_flag(
    index: pd.DatetimeIndex,
    first_last: int = 30,
    open_time: str = "09:30",
) -> pd.Series:
    """
    Flag bars where:

      - 0  < minutes_since_open < 31   (first 30 minutes)
      - 359 < minutes_since_open < 390 (last ~30 minutes)
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex for get_session_flag")

    mins = get_minutes_since_open(index, open_time=open_time)

    first_mask = (mins > 0) & (mins < 31)
    last_mask = (mins > 359) & (mins < 390)

    flag = (first_mask | last_mask).astype(int)
    return flag


# =========================
# Main feature engineering
# =========================


def engineer_features(df: pd.DataFrame, window_sizes=[5, 15, 30]) -> pd.DataFrame:
    """
    Engineer microstructure features given a bar dataframe with:

        'Open', 'High', 'Low', 'Close', 'Up', 'Down', 'VWAP',
        'Volume', 'Returns', 'ATR'
    """
    features = {}

    # 1. Volatility estimators (Parkinson)
    for w in window_sizes:
        features[f"parkinson_vol_{w}"] = parkinson_volatility(df, w)

    # 2. Order flow imbalance
    for w in window_sizes:
        features[f"ofi_{w}"] = order_flow_imbalance(df, w)

    # 3. Volume dynamics
    features["volume_percentile"] = rolling_percentile(df["Volume"], 60)
    features["volume_momentum"] = df["Volume"].pct_change(5)

    # 4. Price impact / liquidity
    vol_safe = df["Volume"].replace(0, np.nan)
    features["amihud_illiquidity"] = np.abs(df["Returns"]) / vol_safe
    features["vwap_distance"] = (df["Close"] - df["VWAP"]) / df["ATR"]

    # 5. Temporal features
    features["minutes_since_open"] = get_minutes_since_open(df.index)
    features["is_first_last_30min"] = get_session_flag(df.index, first_last=30)

    return pd.DataFrame(features, index=df.index)


# =========================
# Target construction
# =========================


def add_realized_vol_target(df: pd.DataFrame, horizon: int = 15) -> pd.DataFrame:
    """
    Add forward-looking realized volatility target over the next `horizon` 1-min bars.

        rv_horizon_fwd(t) = sqrt( sum_{k=1..horizon} r_{t+k}^2 )
    """
    if "Returns" not in df.columns:
        raise ValueError("Returns column missing; run load_and_preprocess first.")

    sq = df["Returns"] ** 2
    cumsum = sq.cumsum()

    # Sum_{k=t+1}^{t+h} r_k^2 = S_{t+h} - S_t
    forward_sum = cumsum.shift(-horizon) - cumsum
    rv_future = np.sqrt(forward_sum.clip(lower=0))
    df[f"rv_{horizon}m_fwd"] = rv_future
    return df


# =========================
# Preprocessing
# =========================


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Build datetime index from Date + Time
    if "Datetime" not in df.columns:
        df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.set_index("Datetime").sort_index()

    # Volume from Up/Down
    if "Volume" not in df.columns:
        df["Volume"] = df["Up"] + df["Down"]

    # Returns (log returns)
    if "Returns" not in df.columns:
        df["Returns"] = np.log(df["Close"]).diff()

    # ATR from TA-Lib (14-period by default)
    if "ATR" not in df.columns:
        df["ATR"] = ta.ATR(
            df["High"].values,
            df["Low"].values,
            df["Close"].values,
            timeperiod=14,
        )

    return df


# =========================
# Feature transformation + normalization
# =========================


def normalize_features(features_df: pd.DataFrame):
    """
    Per-feature transforms + RobustScaler.
    """
    df = features_df.copy()

    # 1. Log-transform heavy positive stuff
    log_cols = [
        "parkinson_vol_5",
        "parkinson_vol_15",
        "parkinson_vol_30",
        "amihud_illiquidity",
    ]
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # 2. Clip extremes
    clip_cols = ["vwap_distance", "volume_momentum"]
    for col in clip_cols:
        if col in df.columns:
            lower = df[col].quantile(0.005)
            upper = df[col].quantile(0.995)
            df[col] = df[col].clip(lower=lower, upper=upper)

    # 3. Scale minutes_since_open to [0, 1]
    if "minutes_since_open" in df.columns:
        max_minutes = df["minutes_since_open"].max()
        if max_minutes > 0:
            df["minutes_since_open"] = df["minutes_since_open"] / max_minutes

    # Binary columns: do NOT scale
    binary_cols = ["is_first_last_30min"]
    cont_cols = [c for c in df.columns if c not in binary_cols]

    scaler = RobustScaler()
    scaled = scaler.fit_transform(df[cont_cols])
    df_scaled = pd.DataFrame(scaled, columns=cont_cols, index=df.index)

    for col in binary_cols:
        if col in df.columns:
            df_scaled[col] = df[col]

    # Preserve original column order
    df_scaled = df_scaled[features_df.columns]

    return df_scaled, scaler


# =========================
# Main
# =========================

if __name__ == "__main__":
    data_path = "data_ml_inference.txt"
    DATA_CACHE_PATH = "new_data_full_df_processed.pkl"

    # 1. Load + preprocess
    df = load_and_preprocess(data_path)
    print("Raw df with technical columns:")
    print(df.head())

    # 1b. Add 15-min realized vol target
    df = add_realized_vol_target(df, horizon=15)
    print("\nSample of rv_15m_fwd:")
    print(df[["Returns", "rv_15m_fwd"]].head(25))

    # 2. Engineer features
    features_df = engineer_features(df)
    print("\nEngineered features:")
    print(features_df.head())

    # 3. Merge back with price data + target
    full_df = pd.concat([df, features_df], axis=1)
    print("\nFull df with prices + features + target (before dropna):")
    print(full_df.head())
    print("\nColumns in full_df:")
    print(full_df.columns)

    # 4. Drop empty lines
    full_df = full_df.dropna()

    # 5. Normalize engineered features
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

    scaled_features, scaler = normalize_features(full_df[feature_cols])
    full_df[feature_cols] = scaled_features

    print("\nFull df after normalization of engineered features:")
    print(full_df[feature_cols].describe().round(3))

    # 5b. Scale target (same logic as training)
    target_col = "rv_15m_fwd"
    target_scale = full_df[target_col].median()
    full_df[target_col] = full_df[target_col] / target_scale

    print(f"\nTarget scale (median rv_15m_fwd): {target_scale}")

    # chronological order
    full_df = full_df.sort_index()

    # cache processed dataframe
    full_df.to_pickle(DATA_CACHE_PATH)
    print(f"\nSaved processed dataframe to {DATA_CACHE_PATH}")
