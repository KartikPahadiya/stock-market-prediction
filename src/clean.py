"""Clean and validate raw data.
Paths and parameters now come from config.yaml via config_loader.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

from config_loader import data_path, cfg

# ======================================================
# Paths (from config)
# ======================================================

RAW_DIR = data_path("raw")
PROCESSED_DIR = data_path("processed_stocks").parent  # data/processed

STOCK_RAW = RAW_DIR / "stocks"
BENCHMARK_RAW = RAW_DIR / "benchmark"
MACRO_RAW = RAW_DIR / "macro"
NEWS_RAW = RAW_DIR / "news"

STOCK_PROCESSED = PROCESSED_DIR / "stocks"
BENCHMARK_PROCESSED = PROCESSED_DIR / "benchmark"
MACRO_PROCESSED = PROCESSED_DIR / "macro"
NEWS_PROCESSED = PROCESSED_DIR / "news"

for d in (STOCK_PROCESSED, BENCHMARK_PROCESSED, MACRO_PROCESSED, NEWS_PROCESSED):
    d.mkdir(parents=True, exist_ok=True)

FFILL_LIMIT = cfg.clean.get("ffill_limit", 3)
OUTLIER_Z = cfg.clean.get("outlier_z_threshold", 4.0)


# ======================================================
# Inspect
# ======================================================

def inspect_dataframe(df, name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())


# ======================================================
# STOCK DATA
# ======================================================

quality_log = []
stock_folder = STOCK_RAW

for file in os.listdir(stock_folder):
    if not file.endswith(".csv"):
        continue

    path = os.path.join(stock_folder, file)
    df = pd.read_csv(path)
    before_rows = len(df)

    inspect_dataframe(df, f"Stock - {file}")

    df["Date"] = pd.to_datetime(df["Date"])

    if "Adj Close" in df.columns:
        df.drop(columns=["Adj Close"], inplace=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df = df.sort_values("Date")
    df = df.drop_duplicates()

    # Calendar reindexing using REAL NYSE trading calendar
    # pd.bdate_range injects phantom holidays (Thanksgiving, Good Friday, etc.)
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=min_date.strftime("%Y-%m-%d"),
        end_date=max_date.strftime("%Y-%m-%d")
    )
    trading_cal = pd.DatetimeIndex(schedule.index, name="Date")
    df = df.set_index("Date").reindex(trading_cal).reset_index()

    # Capped forward-fill only -- NEVER backfill
    df = df.ffill(limit=FFILL_LIMIT)

    # OHLC invariant check
    bad_rows = (
        (df["Low"] > df[["Open", "Close", "High"]].min(axis=1)) |
        (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
    )
    if bad_rows.any():
        print(f"WARNING: {bad_rows.sum()} rows violate OHLC invariant -- quarantining")
        df.loc[bad_rows, ["Open", "High", "Low", "Close"]] = np.nan

    # Outlier flagging (don't delete)
    r = np.log(df["Close"]).diff()
    r_mean = r.rolling(63, min_periods=10).mean()
    r_std = r.rolling(63, min_periods=10).std()
    z = (r - r_mean) / r_std
    df["is_outlier"] = z.abs() > OUTLIER_Z

    df = df.reset_index(drop=True)

    print("\nRemaining Missing Values:")
    print(df.isnull().sum())

    df.to_csv(STOCK_PROCESSED / file, index=False)
    print(f"Saved cleaned {file}")

    quality_log.append({
        "file": file,
        "rows_in": before_rows,
        "rows_out": len(df),
        "gaps_remaining": int(df.isnull().sum().sum()),
        "outliers_flagged": int(df["is_outlier"].sum()),
        "ohlc_violations": int(bad_rows.sum())
    })


# ======================================================
# BENCHMARK DATA
# ======================================================

benchmark_folder = BENCHMARK_RAW

for file in os.listdir(benchmark_folder):
    if not file.endswith(".csv"):
        continue

    path = os.path.join(benchmark_folder, file)
    df = pd.read_csv(path)
    inspect_dataframe(df, f"Benchmark - {file}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Close"]]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.sort_values("Date").drop_duplicates()
    df = df.ffill(limit=FFILL_LIMIT)
    df = df.reset_index(drop=True)

    print("\nRemaining Missing Values:")
    print(df.isnull().sum())
    df.to_csv(BENCHMARK_PROCESSED / file, index=False)
    print(f"Saved cleaned {file}")


# ======================================================
# MACRO DATA
# ======================================================

macro = pd.read_csv(MACRO_RAW / "macro.csv")
inspect_dataframe(macro, "Macro Data")

macro.rename(columns={"DATE": "Date"}, inplace=True)
macro["Date"] = pd.to_datetime(macro["Date"])

numeric_cols = ["DGS10", "DGS3MO", "CPIAUCSL", "UNRATE"]
macro[numeric_cols] = macro[numeric_cols].apply(pd.to_numeric, errors="coerce")
macro = macro.sort_values("Date").drop_duplicates()
macro = macro.ffill()
macro = macro.dropna(subset=numeric_cols)
macro = macro.reset_index(drop=True)

print("\nRemaining Missing Values:")
print(macro.isnull().sum())
macro.to_csv(MACRO_PROCESSED / "macro.csv", index=False)
print("Saved cleaned Macro")


# ======================================================
# NEWS DATA
# ======================================================

news = pd.read_csv(NEWS_RAW / "news.csv")
inspect_dataframe(news, "News Data")

news["publishedAt"] = pd.to_datetime(news["publishedAt"])
news.drop(columns=["image", "url"], inplace=True, errors="ignore")
news = news.drop_duplicates().sort_values("publishedAt")
news = news.dropna(subset=["headline"])
news["summary"] = news["summary"].fillna("")
news["source"] = news["source"].fillna("Unknown")
news["category"] = news["category"].fillna("Unknown")
news = news.reset_index(drop=True)

print("\nRemaining Missing Values:")
print(news.isnull().sum())
news.to_csv(NEWS_PROCESSED / "news.csv", index=False)
print("Saved cleaned News")


# ======================================================
# Data Quality Report
# ======================================================

pd.DataFrame(quality_log).to_csv(PROCESSED_DIR / "data_quality_report.csv", index=False)
print("\nSaved data quality report to data/processed/data_quality_report.csv")
print("\nAll datasets cleaned successfully!")
