"""
==============================================================
EDA & Diagnostics
PRD Section 6 — Exploratory Data Analysis
==============================================================

Generates and saves the following diagnostic outputs to reports/eda/:
  1. Returns distribution (histogram + KDE) per ticker + skew/kurtosis
  2. Correlation heatmap of key features
  3. Rolling 30-day volatility vs VIX
  4a. Seasonal heatmap (avg returns by month x year) — PRD §6.2
  4b. Seasonal heatmap (avg returns by month x day-of-week) — bonus
  5. Autocorrelation function (ACF) plots for returns
  6. Cumulative log-returns comparison across tickers
  7. ADF stationarity test — BOTH returns & price levels (PRD §6.3)
  8. Feature–target correlation bar chart
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from scipy import stats

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features"
BENCH_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark"
CLEAN_DIR = PROJECT_ROOT / "data" / "processed" / "stocks"
REPORT_DIR = PROJECT_ROOT / "reports" / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Style
# ==========================================================

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150

# ==========================================================
# Load Data
# ==========================================================

def load_all_features():
    files = sorted(FEATURE_DIR.glob("*.csv"))
    data = {}
    for f in files:
        ticker = f.stem
        df = pd.read_csv(f, parse_dates=["Date"])
        data[ticker] = df.sort_values("Date").reset_index(drop=True)
    return data


def load_cleaned_prices():
    files = sorted(CLEAN_DIR.glob("*.csv"))
    prices = {}
    for f in files:
        ticker = f.stem
        df = pd.read_csv(f, parse_dates=["Date"])
        prices[ticker] = df.sort_values("Date").reset_index(drop=True)
    return prices


def load_benchmarks():
    sp500 = pd.read_csv(BENCH_DIR / "SP500.csv", parse_dates=["Date"])
    vix = pd.read_csv(BENCH_DIR / "VIX.csv", parse_dates=["Date"])
    return sp500, vix


# ==========================================================
# 1. Returns Distribution
# ==========================================================

def plot_returns_distributions(data):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    skew_kurt_rows = []
    for ax, (ticker, df) in zip(axes, data.items()):
        returns = df["Daily_Return"].dropna()
        sns.histplot(returns, bins=60, kde=True, ax=ax, stat="density",
                     color="steelblue", edgecolor="none")
        ax.axvline(returns.mean(), color="red", linestyle="--", linewidth=1,
                   label=f"mu={returns.mean():.4f}")
        ax.set_title(ticker, fontsize=11)
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        skew_kurt_rows.append({
            "Ticker": ticker,
            "Skewness": stats.skew(returns),
            "Kurtosis": stats.kurtosis(returns, fisher=True),
        })

    plt.suptitle("Daily Return Distributions (with KDE)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "01_returns_distributions.png", bbox_inches="tight")
    plt.close(fig)
    print("[1/8] Saved: 01_returns_distributions.png")

    sk_df = pd.DataFrame(skew_kurt_rows)
    sk_df.to_csv(REPORT_DIR / "01b_skew_kurtosis.csv", index=False)
    print("[1b/8] Saved: 01b_skew_kurtosis.csv")
    return sk_df


# ==========================================================
# 2. Correlation Heatmap
# ==========================================================

def plot_correlation_heatmap(data):
    pooled = pd.concat(data.values(), ignore_index=True)

    cols = [
        "Daily_Return", "Log_Return", "RSI_14", "MACD", "MACD_Hist",
        "BB_Width", "BB_Position", "ATR_pct", "Volume_to_MA20",
        "SP500_Return", "VIX_Return", "Relative_Return",
        "Market_Volatility", "Yield_Curve", "Sentiment_Score",
        "Target_Return"
    ]
    cols = [c for c in cols if c in pooled.columns]

    corr = pooled[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Feature Correlation Heatmap (pooled, lower triangle)", fontsize=13)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "02_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("[2/8] Saved: 02_correlation_heatmap.png")


# ==========================================================
# 3. Rolling Volatility vs VIX
# ==========================================================

def plot_rolling_vol_vs_vix(data, vix):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    for ax, (ticker, df) in zip(axes, data.items()):
        df = df.copy()
        df["RollingVol_30"] = df["Daily_Return"].rolling(30).std() * np.sqrt(252)
        merged = pd.merge(df[["Date", "RollingVol_30"]], vix, on="Date", how="inner")

        ax.plot(merged["Date"], merged["RollingVol_30"],
                label="Stock Vol (30d, ann.)", color="steelblue", alpha=0.8)
        ax.plot(merged["Date"], merged["Close"],
                label="VIX", color="crimson", alpha=0.8)
        ax.set_title(ticker, fontsize=11)
        ax.set_ylabel("Volatility / VIX")
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    plt.suptitle("Rolling 30-Day Volatility vs VIX", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "03_rolling_vol_vs_vix.png", bbox_inches="tight")
    plt.close(fig)
    print("[3/8] Saved: 03_rolling_vol_vs_vix.png")


# ==========================================================
# 4. Seasonal Heatmaps
# ==========================================================

def plot_seasonal_heatmap(data):
    pooled = pd.concat(data.values(), ignore_index=True)
    pooled["Year"] = pd.to_datetime(pooled["Date"]).dt.year
    pooled["Month"] = pd.to_datetime(pooled["Date"]).dt.month

    # --- Main deliverable: month x year ---
    seasonal = pooled.groupby(["Month", "Year"])["Daily_Return"].mean().unstack()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(seasonal, annot=True, fmt=".4f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Average Daily Return by Month x Year (pooled) -- Regime Drift",
                 fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Month")
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "04a_seasonal_heatmap_month_year.png",
                bbox_inches="tight")
    plt.close(fig)
    print("[4a/8] Saved: 04a_seasonal_heatmap_month_year.png")

    # --- Bonus: month x day-of-week ---
    pooled["DayOfWeek"] = pd.to_datetime(pooled["Date"]).dt.dayofweek
    seasonal_dow = pooled.groupby(["Month", "DayOfWeek"])["Daily_Return"].mean().unstack()
    seasonal_dow.columns = ["Mon", "Tue", "Wed", "Thu", "Fri"]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(seasonal_dow, annot=True, fmt=".4f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Average Daily Return by Month x Day-of-Week (bonus)", fontsize=13)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Month")
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "04b_seasonal_heatmap_dow.png", bbox_inches="tight")
    plt.close(fig)
    print("[4b/8] Saved: 04b_seasonal_heatmap_dow.png")


# ==========================================================
# 5. Autocorrelation (ACF)
# ==========================================================

def plot_acf(data):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    for ax, (ticker, df) in zip(axes, data.items()):
        returns = df["Daily_Return"].dropna()
        acf_vals = [returns.autocorr(lag=i) for i in range(1, 21)]
        lags = np.arange(1, 21)

        ax.bar(lags, acf_vals, color="steelblue", edgecolor="none")
        ax.axhline(0, color="black", linewidth=0.5)
        conf = 1.96 / np.sqrt(len(returns))
        ax.axhline(conf, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(-conf, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(ticker, fontsize=11)
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("ACF")
        ax.set_ylim(-0.15, 0.15)

    plt.suptitle("Autocorrelation of Daily Returns (Lags 1-20)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "05_acf_returns.png", bbox_inches="tight")
    plt.close(fig)
    print("[5/8] Saved: 05_acf_returns.png")


# ==========================================================
# 6. Cumulative Returns
# ==========================================================

def plot_cumulative_returns(data):
    fig, ax = plt.subplots(figsize=(12, 6))

    for ticker, df in data.items():
        df = df.copy()
        df["CumLogReturn"] = df["Log_Return"].cumsum()
        ax.plot(df["Date"], df["CumLogReturn"], label=ticker, linewidth=1.2)

    ax.set_title("Cumulative Log-Returns", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log-Return")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "06_cumulative_returns.png", bbox_inches="tight")
    plt.close(fig)
    print("[6/8] Saved: 06_cumulative_returns.png")


# ==========================================================
# 7. ADF Stationarity Test (Returns + Price Levels)
# ==========================================================

def adf_stationarity_test(data, prices):
    results = []
    for ticker, df in data.items():
        # Returns
        series = df["Daily_Return"].dropna()
        stat, pvalue, _, _, crit, _ = adfuller(series, autolag="AIC")
        results.append({
            "Ticker": ticker,
            "Series": "Returns",
            "ADF_Statistic": stat,
            "p_value": pvalue,
            "1%_Critical": crit["1%"],
            "5%_Critical": crit["5%"],
            "Stationary_5%": pvalue < 0.05
        })

        # Price level (PRD §6.3)
        if ticker in prices and "Close" in prices[ticker].columns:
            price_series = prices[ticker]["Close"].dropna()
            stat_p, pvalue_p, _, _, crit_p, _ = adfuller(price_series, autolag="AIC")
            results.append({
                "Ticker": ticker,
                "Series": "Price (Close)",
                "ADF_Statistic": stat_p,
                "p_value": pvalue_p,
                "1%_Critical": crit_p["1%"],
                "5%_Critical": crit_p["5%"],
                "Stationary_5%": pvalue_p < 0.05
            })
        else:
            print(f"  WARNING: {ticker} price data missing or no Close column")

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORT_DIR / "07_adf_stationarity.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("tight")
    ax.axis("off")
    table_data = results_df.round(4).values.tolist()
    table = ax.table(
        cellText=table_data,
        colLabels=results_df.columns,
        cellLoc="center",
        loc="center",
        colColours=["#4472C4"] * len(results_df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("ADF Stationarity Test -- Returns vs Price Levels", fontsize=13, pad=20)
    fig.savefig(REPORT_DIR / "07_adf_stationarity_table.png", bbox_inches="tight")
    plt.close(fig)

    print("[7/8] Saved: 07_adf_stationarity.csv + 07_adf_stationarity_table.png")
    print("\nADF Results:")
    print(results_df.to_string(index=False))
    return results_df


# ==========================================================
# 8. Feature–Target Correlation
# ==========================================================

def plot_feature_target_corr(data):
    pooled = pd.concat(data.values(), ignore_index=True)

    drop_cols = ["Date", "ticker", "Target_Close", "Target_Direction",
                 "is_outlier", "Year", "Month", "Quarter", "Week",
                 "Day", "DayOfWeek", "IsMonthStart", "IsMonthEnd"]
    feature_cols = [c for c in pooled.columns
                    if c not in drop_cols and c != "Target_Return"]

    corrs = pooled[feature_cols + ["Target_Return"]].corr()["Target_Return"].drop("Target_Return")
    corrs = corrs.dropna().sort_values(key=abs, ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["green" if v > 0 else "crimson" for v in corrs.values]
    corrs.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
    ax.set_title("Top 20 Features by Absolute Correlation with Target_Return", fontsize=13)
    ax.set_xlabel("Pearson Correlation")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "08_feature_target_correlation.png", bbox_inches="tight")
    plt.close(fig)
    print("[8/8] Saved: 08_feature_target_correlation.png")


# ==========================================================
# Main
# ==========================================================

def main():
    print("=" * 70)
    print("EDA & Diagnostics -- PRD Section 6")
    print("=" * 70)

    print("\nLoading data...")
    data = load_all_features()
    prices = load_cleaned_prices()
    _, vix = load_benchmarks()
    print(f"Loaded {len(data)} feature tickers + VIX + {len(prices)} price tickers")

    print("\nGenerating plots...")
    plot_returns_distributions(data)
    plot_correlation_heatmap(data)
    plot_rolling_vol_vs_vix(data, vix)
    plot_seasonal_heatmap(data)
    plot_acf(data)
    plot_cumulative_returns(data)
    adf_stationarity_test(data, prices)
    plot_feature_target_corr(data)

    print(f"\n{'='*70}")
    print(f"All EDA outputs saved to: {REPORT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
