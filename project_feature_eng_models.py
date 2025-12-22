import os
import math
import joblib
import numpy as np
import pandas as pd
import talib as ta

from sklearn.preprocessing import RobustScaler


# =========================
# Helper functions
# =========================


def parkinson_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    hl = np.log(df["High"] / df["Low"])
    sigma2 = (hl**2).rolling(window).mean() / (4 * np.log(2))
    return np.sqrt(sigma2)


def order_flow_imbalance(df: pd.DataFrame, window: int) -> pd.Series:
    net_of = df["Up"] - df["Down"]
    return net_of.rolling(window).sum()


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def last_rank_pct(x: pd.Series) -> float:
        if len(x) <= 1:
            return np.nan
        rank = x.rank().iloc[-1]  # 1..len(x)
        return (rank - 1) / (len(x) - 1)

    return series.rolling(window).apply(last_rank_pct, raw=False)


def get_minutes_since_open(
    index: pd.DatetimeIndex, open_time: str = "09:30"
) -> pd.Series:
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
    index: pd.DatetimeIndex, first_last: int = 30, open_time: str = "09:30"
) -> pd.Series:
    mins = get_minutes_since_open(index, open_time=open_time)
    first_mask = (mins > 0) & (mins < (first_last + 1))
    last_mask = (mins > 359) & (mins < 390)  # ~last 30 min of regular session
    return (first_mask | last_mask).astype(int)


# =========================
# Feature engineering
# =========================


def engineer_features(df: pd.DataFrame, window_sizes=(5, 15, 30)) -> pd.DataFrame:
    features = {}

    for w in window_sizes:
        features[f"parkinson_vol_{w}"] = parkinson_volatility(df, w)

    for w in window_sizes:
        features[f"ofi_{w}"] = order_flow_imbalance(df, w)

    features["volume_percentile"] = rolling_percentile(df["Volume"], 60)
    features["volume_momentum"] = df["Volume"].pct_change(5)

    vol_safe = df["Volume"].replace(0, np.nan)
    features["amihud_illiquidity"] = np.abs(df["Returns"]) / vol_safe
    features["vwap_distance"] = (df["Close"] - df["VWAP"]) / df["ATR"]

    features["minutes_since_open"] = get_minutes_since_open(df.index)
    features["is_first_last_30min"] = get_session_flag(df.index, first_last=30)

    return pd.DataFrame(features, index=df.index)


# =========================
# Target
# =========================


def add_realized_vol_target(df: pd.DataFrame, horizon: int = 15) -> pd.DataFrame:
    if "Returns" not in df.columns:
        raise ValueError("Returns column missing; run load_and_preprocess first.")

    sq = df["Returns"] ** 2
    cumsum = sq.cumsum()
    forward_sum = cumsum.shift(-horizon) - cumsum
    rv_future = np.sqrt(forward_sum.clip(lower=0))
    df[f"rv_{horizon}m_fwd"] = rv_future
    return df


# =========================
# Preprocessing
# =========================


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Datetime index
    if "Datetime" not in df.columns:
        df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.set_index("Datetime").sort_index()

    # Volume
    if "Volume" not in df.columns:
        df["Volume"] = df["Up"] + df["Down"]

    # Returns (log)
    if "Returns" not in df.columns:
        df["Returns"] = np.log(df["Close"]).diff()

    # ATR
    if "ATR" not in df.columns:
        df["ATR"] = ta.ATR(
            df["High"].values,
            df["Low"].values,
            df["Close"].values,
            timeperiod=14,
        )

    return df


# =========================
# Train-fit transforms (NO leakage)
# =========================


def fit_feature_transform(
    train_features: pd.DataFrame,
    feature_cols,
    binary_cols=("is_first_last_30min",),
    log_cols=(
        "parkinson_vol_5",
        "parkinson_vol_15",
        "parkinson_vol_30",
        "amihud_illiquidity",
    ),
    clip_cols=("vwap_distance", "volume_momentum"),
    clip_q=(0.005, 0.995),
):
    """
    Fit transformation parameters ONLY on training features.
    Returns a bundle: scaler + clip bounds + max_minutes + config.
    """
    df = train_features.copy()

    # 1) log1p on heavy positive
    for c in log_cols:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))

    # 2) clip bounds computed from TRAIN only
    clip_bounds = {}
    q_lo, q_hi = clip_q
    for c in clip_cols:
        if c in df.columns:
            lo = df[c].quantile(q_lo)
            hi = df[c].quantile(q_hi)
            clip_bounds[c] = (float(lo), float(hi))
            df[c] = df[c].clip(lower=lo, upper=hi)

    # 3) minutes_since_open scaled by TRAIN max only
    max_minutes = None
    if "minutes_since_open" in df.columns:
        max_minutes = float(df["minutes_since_open"].max())
        if max_minutes and max_minutes > 0:
            df["minutes_since_open"] = df["minutes_since_open"] / max_minutes

    # 4) robust scaler fit on TRAIN only (continuous cols only)
    binary_cols = tuple(binary_cols)
    cont_cols = [c for c in feature_cols if c not in binary_cols]

    scaler = RobustScaler()
    scaler.fit(df[cont_cols])

    bundle = {
        "feature_cols": list(feature_cols),
        "binary_cols": list(binary_cols),
        "cont_cols": list(cont_cols),
        "log_cols": list(log_cols),
        "clip_cols": list(clip_cols),
        "clip_q": list(clip_q),
        "clip_bounds": clip_bounds,
        "max_minutes": max_minutes,
        "scaler": scaler,
    }
    return bundle


def apply_feature_transform(features_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    Apply TRAIN-fitted transforms to any split (val/test/live).
    """
    df = features_df.copy()

    # log1p
    for c in bundle["log_cols"]:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))

    # clip using TRAIN bounds
    for c, (lo, hi) in bundle["clip_bounds"].items():
        if c in df.columns:
            df[c] = df[c].clip(lower=lo, upper=hi)

    # minutes_since_open using TRAIN max
    if "minutes_since_open" in df.columns and bundle.get("max_minutes"):
        mm = bundle["max_minutes"]
        if mm and mm > 0:
            df["minutes_since_open"] = df["minutes_since_open"] / mm

    # scale continuous cols using TRAIN scaler
    scaler: RobustScaler = bundle["scaler"]
    cont_cols = bundle["cont_cols"]
    binary_cols = set(bundle["binary_cols"])

    scaled = scaler.transform(df[cont_cols])
    out = pd.DataFrame(scaled, columns=cont_cols, index=df.index)

    # put binary cols back unchanged
    for c in bundle["feature_cols"]:
        if c in binary_cols and c in df.columns:
            out[c] = df[c]

    # enforce original order
    out = out[bundle["feature_cols"]]
    return out


# =========================
# Target scaling (fit on TRAIN only)
# =========================


def fit_target_scale(train_target: pd.Series) -> float:
    # Median scaling is robust; fit ONLY on train to prevent leakage
    s = float(train_target.median())
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"Bad target_scale computed from train median: {s}")
    return s


# =========================
# Main pipeline
# =========================


def build_processed_dataset(
    raw_path: str,
    out_df_path: str = "full_df_processed_splits_REPORT.pkl",
    out_bundle_path: str = "feature_transform_bundle.joblib",
    horizon: int = 15,
    split_fracs=(0.8, 0.1, 0.1),
):
    # 1) load + preprocess + target + features
    df = load_and_preprocess(raw_path)
    df = add_realized_vol_target(df, horizon=horizon)

    features_df = engineer_features(df)
    full_df = pd.concat([df, features_df], axis=1).dropna().sort_index()

    target_col = f"rv_{horizon}m_fwd"
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

    # 2) split rows chronologically
    n = len(full_df)
    n_train = int(n * split_fracs[0])
    n_val = int(n * split_fracs[1])
    # remainder goes to test
    train_end = n_train
    val_end = n_train + n_val

    train_idx = full_df.index[:train_end]
    val_idx = full_df.index[train_end:val_end]
    test_idx = full_df.index[val_end:]

    # 3) fit feature transform ONLY on train
    train_features_raw = full_df.loc[train_idx, feature_cols]
    bundle = fit_feature_transform(train_features_raw, feature_cols=feature_cols)

    # 4) apply transform to all rows using train-fitted params
    full_df.loc[:, feature_cols] = apply_feature_transform(
        full_df[feature_cols], bundle
    )

    # 5) fit target scale ONLY on train, apply to all
    target_scale = fit_target_scale(full_df.loc[train_idx, target_col])
    full_df[target_col] = full_df[target_col] / target_scale

    # 6) add split column
    full_df["split"] = "unknown"
    full_df.loc[train_idx, "split"] = "train"
    full_df.loc[val_idx, "split"] = "val"
    full_df.loc[test_idx, "split"] = "test"

    # 7) save outputs
    # Save dataframe
    full_df.to_pickle(out_df_path)
    print(f"Saved processed dataset with split column -> {out_df_path}")

    # Save bundle + target_scale (so live data can be transformed identically)
    bundle_out = dict(bundle)
    bundle_out.update(
        {
            "target_col": target_col,
            "target_scale": float(target_scale),
            "horizon": int(horizon),
        }
    )
    joblib.dump(bundle_out, out_bundle_path)
    print(f"Saved feature transform bundle -> {out_bundle_path}")

    # sanity print
    print("\nSplit counts:")
    print(full_df["split"].value_counts(dropna=False))
    print("\nTrain target median (unscaled):", target_scale)

    return full_df, bundle_out


# =========================
# Live / new data application
# =========================


def apply_bundle_to_new_data(
    raw_path_new: str,
    bundle_path: str = "feature_transform_bundle.joblib",
    out_path: str = "live_processed.pkl",
):
    """
    For "new test only" live trading data:
      - compute same technicals/features/target
      - apply EXACT same transforms from training bundle
      - scale target using training target_scale
    """
    bundle = joblib.load(bundle_path)

    horizon = int(bundle["horizon"])
    target_col = bundle["target_col"]
    feature_cols = bundle["feature_cols"]
    target_scale = float(bundle["target_scale"])

    df = load_and_preprocess(raw_path_new)
    df = add_realized_vol_target(df, horizon=horizon)

    features_df = engineer_features(df)
    full_df = pd.concat([df, features_df], axis=1).dropna().sort_index()

    # Apply train-fitted feature transforms
    full_df.loc[:, feature_cols] = apply_feature_transform(
        full_df[feature_cols], bundle
    )

    # Apply train-fitted target scaling
    full_df[target_col] = full_df[target_col] / target_scale

    # Mark as live/test
    full_df["split"] = "live"

    full_df.to_pickle(out_path)
    print(f"Saved live-processed dataset -> {out_path}")
    return full_df


if __name__ == "__main__":
    # Example usage:
    # RAW_PATH = "data_ml_final.txt"
    # build_processed_dataset(
    #    raw_path=RAW_PATH,
    #    out_df_path="full_df_processed_splits_REPORT_full_data.pkl",
    #    out_bundle_path="feature_transform_bundle.joblib",
    #    horizon=15,
    #    split_fracs=(0.8, 0.1, 0.1),
    # )

    # Later, for live data:
    apply_bundle_to_new_data(
        raw_path_new="data_ml_inference.txt",
        bundle_path="feature_transform_bundle.joblib",
        out_path="live_processed_REPORT.pkl",
    )
