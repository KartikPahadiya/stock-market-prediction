"""Ingest raw data from Yahoo Finance, FRED, and Finnhub.
All parameters now come from config.yaml via config_loader.
"""
import os
import time
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web

from config_loader import cfg, data_path, tickers, START_DATE, END_DATE, BENCHMARK

# ==========================================================
# Project Paths (from config)
# ==========================================================

RAW_DIR = data_path("raw")
STOCK_DIR = RAW_DIR / "stocks"
BENCHMARK_DIR = RAW_DIR / "benchmark"
MACRO_DIR = RAW_DIR / "macro"
NEWS_DIR = RAW_DIR / "news"

for d in (STOCK_DIR, BENCHMARK_DIR, MACRO_DIR, NEWS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load .env for API keys
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise ValueError("Set FINNHUB_API_KEY in .env file or environment variable")

# ==========================================================
# Download Stock Data
# ==========================================================

TICKERS = tickers()
START = START_DATE
END = END_DATE
MIN_ROWS = cfg.ingest.get("min_rows_per_ticker", 100)


def download_stocks():
    print("\nDownloading Stock Data...\n")
    for ticker in TICKERS:
        print(f"Downloading {ticker}")
        df = yf.download(
            ticker,
            start=START,
            end=END,
            auto_adjust=cfg.ingest.get("auto_adjust", True),
            multi_level_index=False
        )
        df.reset_index(inplace=True)

        if len(df) < MIN_ROWS:
            print(f"WARNING: {ticker} only returned {len(df)} rows -- re-downloading")
            time.sleep(5)
            df = yf.download(ticker, start=START, end=END, auto_adjust=True, multi_level_index=False)
            df.reset_index(inplace=True)

        df.to_csv(STOCK_DIR / f"{ticker}.csv", index=False)
        print(f"{ticker}: {len(df)} rows saved")
    print("Stock data downloaded.\n")


# ==========================================================
# Download Benchmark Data
# ==========================================================

def download_benchmark():
    print("\nDownloading Benchmark Data...\n")
    symbols = {
        "SP500": "^GSPC",
        "VIX": "^VIX"
    }
    for name, symbol in symbols.items():
        print(f"Downloading {name}")
        df = yf.download(symbol, start=START, end=END, auto_adjust=True, multi_level_index=False)
        df.reset_index(inplace=True)
        df.to_csv(BENCHMARK_DIR / f"{name}.csv", index=False)
    print("Benchmark data downloaded.\n")


# ==========================================================
# Download Macroeconomic Data
# ==========================================================

def download_macro():
    print("\nDownloading Macro Data...\n")
    macro = web.DataReader(
        ["DGS10", "DGS3MO", "CPIAUCSL", "UNRATE"],
        "fred",
        START
    )
    macro.reset_index(inplace=True)
    macro.to_csv(MACRO_DIR / "macro.csv", index=False)
    print("Macro data downloaded.\n")


# ==========================================================
# Download News Data
# ==========================================================

def download_news():
    print("\nDownloading News...\n")
    # Use last 12 months of news to stay within API limits
    from_date = (datetime.today() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    to_date = datetime.today().strftime("%Y-%m-%d")
    all_news = []

    for ticker in TICKERS:
        print(f"Fetching {ticker}")
        url = (
            "https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}"
            f"&from={from_date}"
            f"&to={to_date}"
            f"&token={FINNHUB_API_KEY}"
        )
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                print(data)
                continue
            for article in data:
                all_news.append({
                    "ticker": ticker,
                    "publishedAt": pd.to_datetime(article.get("datetime"), unit="s", errors="coerce"),
                    "headline": article.get("headline"),
                    "summary": article.get("summary"),
                    "source": article.get("source"),
                    "category": article.get("category"),
                    "image": article.get("image"),
                    "url": article.get("url")
                })
        except Exception as e:
            print(f"{ticker}: {e}")
        time.sleep(1)

    news = pd.DataFrame(all_news)
    news.drop_duplicates(subset=["ticker", "headline", "publishedAt"], inplace=True)
    news.sort_values("publishedAt", inplace=True)
    news.to_csv(NEWS_DIR / "news.csv", index=False)
    print(f"\nDownloaded {len(news)} articles.\n")


# ==========================================================
# Main
# ==========================================================

def main():
    download_stocks()
    download_benchmark()
    download_macro()
    download_news()
    print("=" * 60)
    print("RAW DATA INGESTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
