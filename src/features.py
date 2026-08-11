"""Feature engineering: scale-invariant ratios, momentum, volatility, macro.
Parameters now come from config.yaml via config_loader.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from config_loader import data_path

# ==========================================================
# Paths (from config)
# ==========================================================

PROCESSED_DIR = data_path("processed_stocks").parent
STOCK_DIR = data_path("processed_stocks")
BENCHMARK_DIR = data_path("processed_benchmark")
MACRO_DIR = PROCESSED_DIR / "macro"
SENTIMENT_DIR = PROCESSED_DIR / "sentiment"

FEATURE_DIR = data_path("processed_features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load auxiliary data
# ==========================================================

sp500 = pd.read_csv(BENCHMARK_DIR / "SP500.csv")
vix = pd.read_csv(BENCHMARK_DIR / "VIX.csv")
macro = pd.read_csv(MACRO_DIR / "macro.csv")
sentiment = pd.read_csv(SENTIMENT_DIR / "daily_sentiment_unified.csv")

sp500["Date"] = pd.to_datetime(sp500["Date"])
vix["Date"] = pd.to_datetime(vix["Date"])
macro["Date"] = pd.to_datetime(macro["Date"])
sentiment["Date"] = pd.to_datetime(sentiment["Date"])

sp500.rename(columns={"Close": "SP500_Close"}, inplace=True)
vix.rename(columns={"Close": "VIX_Close"}, inplace=True)


# ==========================================================
# Feature Engineering Function
# ==========================================================

def engineer_features(stock_file):
    print("=" * 70)
    print(f"Processing {stock_file.name}")
    print("=" * 70)

    stock = pd.read_csv(stock_file)
    stock["Date"] = pd.to_datetime(stock["Date"])
    stock = stock.sort_values("Date")

    # Merge auxiliary data
    df = stock.merge(sp500, on="Date", how="left")
    df = df.merge(vix, on="Date", how="left")
    df = df.merge(macro, on="Date", how="left")

    ticker = stock_file.stem
    print(f"Ticker : {ticker}")
    stock_sentiment = sentiment[sentiment["ticker"] == ticker].copy()
    df = df.merge(stock_sentiment, on="Date", how="left")

    # Fill macro
    macro_cols = ["DGS10", "DGS3MO", "CPIAUCSL", "UNRATE"]
    df[macro_cols] = df[macro_cols].ffill()
    df.dropna(subset=macro_cols, inplace=True)

    # Fill sentiment defaults
    sentiment_defaults = {
        # REAL VADER (lexicon-based)
        "VADER_Mean": 0.0, "VADER_Std": 0.0,
        "VADER_Pos_Mean": 0.0, "VADER_Neg_Mean": 0.0, "VADER_Neu_Mean": 1.0,
        "Article_Count": 0, "Positive_Count": 0, "Negative_Count": 0, "Neutral_Count": 0,
        "Has_News": 0,
        # FinBERT (domain-tuned transformer)
        "FinBERT_Sentiment_Score": 0.0,
        "FinBERT_Positive_Prob": 0.0, "FinBERT_Negative_Prob": 0.0, "FinBERT_Neutral_Prob": 1.0,
        "FinBERT_Article_Count": 0,
        "FinBERT_Positive_Count": 0, "FinBERT_Negative_Count": 0, "FinBERT_Neutral_Count": 0,
        "FinBERT_Has_News": 0
    }
    df.fillna(sentiment_defaults, inplace=True)
    print(f"Rows with News    : {(df['Has_News'] == 1).sum()}")
    print(f"Rows without News : {(df['Has_News'] == 0).sum()}")
    print(f"Merged Shape : {df.shape}")

    # ------------------------------------------------------
    # Basic Price Features
    # ------------------------------------------------------
    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Range_Percent"] = (df["High"] - df["Low"]) / df["Close"]
    print("[OK] Basic Price Features Created")

    # ------------------------------------------------------
    # Trend Indicators (ratios, not raw levels)
    # ------------------------------------------------------
    for window in [10, 20, 50]:
        sma = SMAIndicator(close=df["Close"], window=window).sma_indicator()
        ema = EMAIndicator(close=df["Close"], window=window).ema_indicator()
        df[f"Price_to_SMA_{window}"] = df["Close"] / sma
        df[f"Price_to_EMA_{window}"] = df["Close"] / ema
    print("[OK] Price-to-SMA/EMA Ratios Created")

    # ------------------------------------------------------
    # RSI
    # ------------------------------------------------------
    df["RSI_14"] = RSIIndicator(close=df["Close"], window=14).rsi()
    print("[OK] RSI Created")

    # ------------------------------------------------------
    # MACD
    # ------------------------------------------------------
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    print("[OK] MACD Created")

    # ------------------------------------------------------
    # Bollinger Bands (normalized)
    # ------------------------------------------------------
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()
    bb_middle = bb.bollinger_mavg()
    df["BB_Width"] = (bb_high - bb_low) / bb_middle
    df["BB_Position"] = (df["Close"] - bb_low) / (bb_high - bb_low)
    print("[OK] Bollinger Bands Created")

    # ------------------------------------------------------
    # ATR (as % of price) — plain rolling mean, NOT Wilder EWM
    # ------------------------------------------------------
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR_pct"] = tr.rolling(14, min_periods=14).mean() / df["Close"]
    print("[OK] ATR Created")

    # ------------------------------------------------------
    # Rolling Statistics (ratios)
    # ------------------------------------------------------
    rolling_mean_7 = df["Close"].rolling(7).mean()
    rolling_mean_30 = df["Close"].rolling(30).mean()
    df["Price_to_RollingMean_7"] = df["Close"] / rolling_mean_7
    df["Price_to_RollingMean_30"] = df["Close"] / rolling_mean_30
    df["RollingSTD_7_pct"] = df["Close"].rolling(7).std() / df["Close"]
    df["RollingSTD_30_pct"] = df["Close"].rolling(30).std() / df["Close"]
    print("[OK] Rolling Statistics Created")

    # ------------------------------------------------------
    # Volume Features (ratios)
    # ------------------------------------------------------
    df["Volume_Change"] = df["Volume"].pct_change()
    volume_ma_20 = df["Volume"].rolling(20).mean()
    df["Volume_to_MA20"] = df["Volume"] / volume_ma_20
    print("[OK] Volume Features Created")

    # ------------------------------------------------------
    # Calendar Features
    # ------------------------------------------------------
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    print("[OK] Calendar Features Created")

    # ------------------------------------------------------
    # Lag Features (returns only)
    # ------------------------------------------------------
    df["Return_Lag1"] = df["Daily_Return"].shift(1)
    df["Return_Lag2"] = df["Daily_Return"].shift(2)
    df["Return_Lag3"] = df["Daily_Return"].shift(3)
    print("[OK] Lag Features Created")

    # ------------------------------------------------------
    # Market Features
    # ------------------------------------------------------
    df["SP500_Return"] = df["SP500_Close"].pct_change()
    df["VIX_Return"] = df["VIX_Close"].pct_change()
    df["Relative_Return"] = df["Daily_Return"] - df["SP500_Return"]
    df["Market_Volatility"] = df["SP500_Return"].rolling(10).std()
    print("[OK] Market Features Created")

    # ------------------------------------------------------
    # Macro Features
    # ------------------------------------------------------
    df["Yield_Curve"] = df["DGS10"] - df["DGS3MO"]
    df["Inflation_Change"] = df["CPIAUCSL"].pct_change()
    df["Unemployment_Change"] = df["UNRATE"].diff()
    df["InterestRate_Change"] = df["DGS10"].diff()
    print("[OK] Macro Features Created")

    # ------------------------------------------------------
    # Target Variables
    # ------------------------------------------------------
    df["Target_Close"] = df["Close"].shift(-1)
    df["Target_Return"] = np.log(df["Close"] / df["Close"].shift(1)).shift(-1)
    df["Target_Direction"] = (df["Target_Return"] > 0).astype(int)
    print("[OK] Target Variables Created")

    # ------------------------------------------------------
    # Remove NaNs
    # ------------------------------------------------------
    before_rows = len(df)
    all_check_cols = [c for c in df.columns if c not in ["Date", "ticker"]]
    nan_report = df[all_check_cols].isnull().sum()
    nan_report = nan_report[nan_report > 0]
    if len(nan_report) > 0:
        print(f"\nWARNING: NaNs found before drop, {stock_file.stem}:")
        print(nan_report.to_string())

    df.dropna(subset=all_check_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    after_rows = len(df)
    print(f"Rows before cleaning : {before_rows}")
    print(f"Rows after cleaning  : {after_rows}")
    print(f"Removed rows         : {before_rows - after_rows}")

    # ------------------------------------------------------
    # Drop raw price level columns
    # ------------------------------------------------------
    raw_level_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_10", "SMA_20", "SMA_50", "EMA_10", "EMA_20", "EMA_50",
        "BB_High", "BB_Low", "BB_Middle",
        "Rolling_Mean_7", "Rolling_Mean_30",
        "Rolling_STD_7", "Rolling_STD_30",
        "ATR_14", "Volume_MA_10", "Volume_MA_20",
        "Close_Lag1", "Close_Lag2", "Close_Lag3",
        "Close_Lag5", "Close_Lag10",
        "Open_Close_Diff", "High_Low_Range",
        "SP500_Close", "VIX_Close",
        "DGS10", "DGS3MO", "CPIAUCSL", "UNRATE",
    ]
    cols_to_drop = [c for c in raw_level_cols if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    print(f"[OK] Dropped {len(cols_to_drop)} raw level columns")

    # ------------------------------------------------------
    # Final column order
    # ------------------------------------------------------
    feature_columns = [col for col in df.columns if col not in ["Target_Close", "Target_Return", "Target_Direction"]]
    df = df[feature_columns + ["Target_Close", "Target_Return", "Target_Direction"]]
    print(f"Final Shape : {df.shape}")

    output_path = FEATURE_DIR / stock_file.name
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return df


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    stock_files = sorted(STOCK_DIR.glob("*.csv"))
    print(f"\nFound {len(stock_files)} stock files.\n")
    for stock_file in stock_files:
        engineer_features(stock_file)
    print("\nAll feature files generated successfully!")
