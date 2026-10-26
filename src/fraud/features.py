"""Feature builder. Every card-history feature uses only rows before the current one."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "processed" / "transactions.parquet"
CACHE = ROOT / "data" / "processed" / "features.parquet"

NUMERIC = [
    "amt", "amt_log", "hour", "weekday", "is_weekend", "is_night", "city_pop",
    "tx_count_prior", "amt_prior_mean", "amt_ratio_mean", "secs_since_prev",
    "secs_since_same_merchant", "card_age_days", "new_merchant", "new_category",
    "tx_count_1h", "tx_count_24h", "tx_count_7d",
]
CATEGORICAL = ["category", "merchant", "state"]
BASELINE_CAT = ["category", "state"]
KEYS = ["ts", "cc_num", "trans_num", "is_fraud", "split"]


def _window_count(df: pd.DataFrame, seconds: int) -> np.ndarray:
    ts = df["ts"].astype("datetime64[s]").astype("int64").to_numpy()
    out = np.zeros(len(df), dtype=np.int32)
    for idx in df.groupby("cc_num", sort=False).indices.values():
        t = ts[idx]
        left = np.searchsorted(t, t - seconds, side="left")
        out[idx] = np.arange(len(t)) - left
    return out


def split_name(ts: pd.Series) -> pd.Series:
    out = pd.Series("stream", index=ts.index, dtype="object")
    out[ts < pd.Timestamp("2020-12-01")] = "holdout"
    out[ts < pd.Timestamp("2020-11-01")] = "val"
    out[ts < pd.Timestamp("2020-10-01")] = "train"
    return out


def build() -> pd.DataFrame:
    df = pd.read_parquet(RAW)
    df["ts"] = pd.to_datetime(df["ts"])
    df["is_fraud"] = df["is_fraud"].astype("int8")
    df = df.sort_values(["ts", "trans_num"]).reset_index(drop=True)

    df["hour"] = df["ts"].dt.hour.astype("int8")
    df["weekday"] = df["ts"].dt.day_of_week.astype("int8")
    df["is_weekend"] = (df["weekday"] >= 5).astype("int8")
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype("int8")
    df["amt_log"] = np.log1p(df["amt"]).astype("float32")

    df = df.sort_values(["cc_num", "ts"], kind="mergesort").reset_index(drop=True)
    df["tx_count_prior"] = df.groupby("cc_num", sort=False).cumcount().astype("int32")
    prior_sum = df.groupby("cc_num", sort=False)["amt"].cumsum() - df["amt"]
    prior_mean = prior_sum / df["tx_count_prior"].clip(lower=1)
    prior_mean = prior_mean.where(df["tx_count_prior"] > 0)
    df["amt_prior_mean"] = prior_mean.astype("float32")
    df["amt_ratio_mean"] = (df["amt"] / (prior_mean + 1.0)).fillna(1.0).astype("float32")
    df["secs_since_prev"] = (
        df.groupby("cc_num", sort=False)["ts"].diff().dt.total_seconds().fillna(-1).astype("float32")
    )
    first_ts = df.groupby("cc_num", sort=False)["ts"].transform("min")
    df["card_age_days"] = ((df["ts"] - first_ts).dt.total_seconds() / 86400).astype("float32")
    df["new_merchant"] = (~df.duplicated(["cc_num", "merchant"])).astype("int8")
    df["new_category"] = (~df.duplicated(["cc_num", "category"])).astype("int8")
    prev_same = df.groupby(["cc_num", "merchant"], sort=False)["ts"].shift(1)
    df["secs_since_same_merchant"] = (
        (df["ts"] - prev_same).dt.total_seconds().fillna(-1).astype("float32")
    )
    df["tx_count_1h"] = _window_count(df, 3600)
    df["tx_count_24h"] = _window_count(df, 86400)
    df["tx_count_7d"] = _window_count(df, 604800)

    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    df["split"] = split_name(df["ts"])
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df[KEYS + NUMERIC + CATEGORICAL]


def load(rebuild: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    df = build()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


if __name__ == "__main__":
    out = load(rebuild=True)
    print(out.groupby("split", observed=True)["is_fraud"].agg(["size", "sum", "mean"]))
    print(out.dtypes)
