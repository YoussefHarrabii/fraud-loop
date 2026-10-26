"""Concatenate Sparkov chunks, save one parquet, and audit label/leakage properties.

Run: .venv\\Scripts\\python.exe scripts\\prepare_sparkov.py
Reads data/generated/*.csv, writes data/processed/transactions.parquet and DOCS/dataset-audit.md.
"""

from pathlib import Path
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "generated"
DST = ROOT / "data" / "processed" / "transactions.parquet"
OUT = ROOT / "DOCS" / "dataset-audit.md"


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load() -> pd.DataFrame:
    files = sorted(glob.glob(str(SRC / "*.csv")))
    files = [f for f in files if Path(f).name != "customers.csv"]
    if not files:
        sys.exit(f"no chunks in {SRC}; run the Sparkov generator first")
    frames = [pd.read_csv(f, sep="|") for f in files]
    tx = pd.concat(frames, ignore_index=True)
    for col in ["lat", "long", "merch_lat", "merch_long", "amt", "unix_time", "is_fraud"]:
        tx[col] = pd.to_numeric(tx[col], errors="coerce")
    tx["ts"] = pd.to_datetime(tx.trans_date + " " + tx.trans_time)
    tx = tx.sort_values("ts").reset_index(drop=True)
    return tx


def section_volume(tx: pd.DataFrame, lines: list[str]) -> None:
    lines.append("## 1. Volume and label\n")
    lines.append(f"- rows: {len(tx):,}")
    lines.append(f"- columns: {tx.shape[1]:,}")
    lines.append(f"- date span: {tx.ts.min()} to {tx.ts.max()}")
    fraud = int(tx.is_fraud.sum())
    lines.append(f"- fraud rows: {fraud:,} ({fraud / len(tx):.3%})")
    lines.append(f"- duplicate trans_num: {int(tx.trans_num.duplicated().sum()):,}")
    per = tx.groupby("cc_num")["is_fraud"].agg(["size", "sum"])
    any_fraud = per["sum"] > 0
    lines.append(f"- cards: {len(per):,}; cards with at least one fraud row: {int(any_fraud.sum()):,}")
    all_fraud = per["sum"] == per["size"]
    lines.append(f"- cards where every row is fraud: {int(all_fraud.sum()):,}, holding {int(per.loc[all_fraud, 'size'].sum()):,} rows ({per.loc[all_fraud, 'size'].sum() / len(tx):.2%} of all rows)")
    lines.append(f"- fraud rows per fraud card: median {per.loc[any_fraud, 'sum'].median():.0f}, mean {per.loc[any_fraud, 'sum'].mean():.1f}, max {int(per.loc[any_fraud, 'sum'].max()):,}")
    lines.append(f"- rows per card: median {per['size'].median():.0f}, mean {per['size'].mean():.1f}, max {int(per['size'].max()):,}")
    lines.append("")


def section_month(tx: pd.DataFrame, lines: list[str]) -> None:
    lines.append("## 2. Fraud by month\n")
    m = tx.groupby(tx.ts.dt.to_period("M"))["is_fraud"].agg(["size", "sum", "mean"])
    lines.append("| month | rows | frauds | rate |")
    lines.append("|---|---|---|---|")
    for idx, row in m.iterrows():
        lines.append(f"| {idx} | {int(row['size']):,} | {int(row['sum']):,} | {row['mean']:.3%} |")
    lines.append("")


def section_leaks(tx: pd.DataFrame, lines: list[str]) -> None:
    lines.append("## 3. Leakage candidates\n")
    y = tx.is_fraud.to_numpy()
    dist = haversine(tx.lat, tx.long, tx.merch_lat, tx.merch_long)
    tx = tx.assign(distance_km=dist)
    auc = roc_auc_score(y, dist)
    ap = average_precision_score(y, dist)
    lines.append(f"- distance cardholder to merchant, single feature: ROC-AUC {auc:.3f}, PR-AUC {ap:.3f}")
    far = dist >= dist.quantile(0.95)
    lines.append(f"- fraud rate in the farthest 5% of transactions: {tx.loc[far, 'is_fraud'].mean():.3%}")
    lines.append(f"- fraud rate elsewhere: {tx.loc[~far, 'is_fraud'].mean():.3%}")
    lines.append("")

    mer = tx.groupby("merchant")["is_fraud"].agg(["size", "sum", "mean"])
    only = mer[(mer["sum"] > 0) & (mer["sum"] == mer["size"])]
    hot = mer[mer["mean"] > 0.5]
    lines.append(f"- merchants that only ever appear in fraud: {len(only):,} of {len(mer):,}")
    lines.append(f"- merchants with fraud rate over 50%: {len(hot):,}")
    if len(hot):
        lines.append(f"- share of all rows from those merchants: {tx.merchant.isin(hot.index).mean():.2%}")
    lines.append("")

    cat = tx.groupby("category")["is_fraud"].agg(["size", "mean"]).sort_values("mean", ascending=False)
    lines.append("Fraud rate by category, top 5:")
    lines.append("")
    lines.append("| category | rows | rate |")
    lines.append("|---|---|---|")
    for idx, row in cat.head(5).iterrows():
        lines.append(f"| {idx} | {int(row['size']):,} | {row['mean']:.2%} |")
    lines.append("")

    prof = tx.groupby("profile")["is_fraud"].agg(["size", "sum", "mean"]).sort_values("mean", ascending=False)
    lines.append("Fraud rate by generator profile, top 5:")
    lines.append("")
    lines.append("| profile | rows | rate |")
    lines.append("|---|---|---|")
    for idx, row in prof.head(5).iterrows():
        lines.append(f"| {idx} | {int(row['size']):,} | {row['mean']:.2%} |")
    lines.append("")

    stamp = pd.to_datetime(tx.unix_time, unit="s")
    offset = ((tx.ts - stamp).dt.total_seconds() / 3600).round(0).astype(int)
    lines.append("- unix_time is UTC, trans_date + trans_time are local Europe/Berlin. Offset:")
    for value, count in offset.value_counts().sort_index().items():
        lines.append(f"  - {value:+d}h: {count:,} rows")
    lines.append("- 1h in winter and 2h in summer is DST, not corruption. Drop unix_time to avoid mixing clocks.")
    lines.append("")


def section_amounts(tx: pd.DataFrame, lines: list[str]) -> None:
    lines.append("## 4. Amounts\n")
    lines.append("| class | rows | mean | median | p95 | max |")
    lines.append("|---|---|---|---|---|---|")
    for label, name in [(0, "legit"), (1, "fraud")]:
        s = tx.loc[tx.is_fraud == label, "amt"]
        lines.append(f"| {name} | {len(s):,} | {s.mean():.2f} | {s.median():.2f} | {s.quantile(.95):.2f} | {s.max():.2f} |")
    lines.append("")
    y = tx.is_fraud.to_numpy()
    auc = roc_auc_score(y, tx.amt)
    ap = average_precision_score(y, tx.amt)
    lines.append(f"- amount alone: ROC-AUC {auc:.3f}, PR-AUC {ap:.3f}")
    lines.append(f"- fraud rate above the fraud median amount (457): {tx.loc[tx.amt > 457, 'is_fraud'].mean():.3%}")
    lines.append(f"- fraud rate below it: {tx.loc[tx.amt <= 457, 'is_fraud'].mean():.3%}")
    lines.append("")
    hour = tx.groupby(tx.ts.dt.hour)["is_fraud"].agg(["size", "mean"])
    lines.append("Fraud rate by hour, top 3:")
    lines.append("")
    for idx, row in hour.sort_values("mean", ascending=False).head(3).iterrows():
        lines.append(f"- {idx:02d}:00, {int(row['size']):,} rows, {row['mean']:.2%}")
    lines.append("")


def main() -> None:
    tx = load()
    DST.parent.mkdir(parents=True, exist_ok=True)
    tx.to_parquet(DST, index=False)
    lines = ["# Sparkov dataset audit", ""]
    section_volume(tx, lines)
    section_month(tx, lines)
    section_leaks(tx, lines)
    section_amounts(tx, lines)
    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {DST} and {OUT}")


if __name__ == "__main__":
    main()
