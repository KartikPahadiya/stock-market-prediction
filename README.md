# AI-Powered Stock Price Prediction & Portfolio Optimization

**Northgate Quantitative Research — Capstone Project**

An end-to-end quantitative platform combining time-series forecasting, deep learning, financial-news sentiment analysis (VADER + FinBERT), and mean-variance portfolio optimization.

---

## Quick Start

### 1. Clone and setup environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set API key (optional — only for news sentiment)

Create a `.env` file in the project root:

```
FINNHUB_API_KEY=your_key_here
```

> Stock prices and macro data require no API key. Only Finnhub news sentiment needs a free API key.

### 3. One-command rebuild

Run the full pipeline from raw data to final report:

```bash
# Windows
.venv\Scripts\python run_pipeline.py
```

For a faster rebuild that uses cached FinBERT and deep-learning artifacts:

```bash
.venv\Scripts\python run_pipeline.py --fast
```

> **Note**: `models_dl.py` takes ~10-15 minutes. `sentiment_finbert.py` downloads ~400MB on first run.

### 4. Launch dashboard

```bash
.venv\Scripts\streamlit run src\dashboard.py
```

---

## GitHub + Streamlit Cloud Deployment

This repo is ready for Streamlit Community Cloud. Use:

- **Main file path**: `src/dashboard.py`
- **Python dependencies**: `requirements.txt`
- **Streamlit theme/config**: `.streamlit/config.toml`

### Daily refresh automation

The workflow `.github/workflows/daily-pipeline.yml` runs the full pipeline every weekday after the U.S. market close and commits refreshed `data/`, `models/`, and `reports/` artifacts back to the repo. Streamlit then serves the latest committed outputs quickly without retraining inside the web app.

Before enabling it, add this GitHub repository secret:

```
FINNHUB_API_KEY=<your_finnhub_key>
```

Path in GitHub:

```
Repo Settings -> Secrets and variables -> Actions -> New repository secret
```

You can also run the workflow manually from the GitHub **Actions** tab. Choose:

- `full` to run the entire PRD pipeline
- `fast` to skip the slow FinBERT and deep-learning stages and reuse cached outputs

> Important: Streamlit Cloud should not run the training pipeline itself on every page load. The dashboard is intentionally read-only and loads cached artifacts produced by GitHub Actions.

---

## Repository Structure

```
stock-ai-system/
├── .github/workflows/daily-pipeline.yml # Daily GitHub Actions refresh
├── .streamlit/config.toml        # Streamlit Cloud theme/server config
├── config.yaml                  # Central config — universe, dates, splits, seeds
├── run_pipeline.py              # End-to-end PRD pipeline runner
├── requirements.txt             # Pinned dependencies
├── src/
│   ├── config_loader.py         # YAML loader with typed accessors
│   ├── ingest.py                # Download stock, benchmark, macro, news data
│   ├── clean.py                 # Data cleaning with real NYSE calendar
│   ├── sentiment_vader.py       # Real VADER lexicon sentiment scoring
│   ├── sentiment.py             # FinBERT sentiment scoring
│   ├── sentiment_finbert.py     # Compatibility entrypoint for FinBERT
│   ├── merge_sentiment.py       # Unified VADER + FinBERT daily sentiment file
│   ├── features.py              # Scale-invariant feature engineering
│   ├── eda.py                   # Exploratory data analysis (8 diagnostics)
│   ├── math_from_scratch.py     # Manual GD, Sharpe, Sortino, Beta, Cov, MaxDD
│   ├── models_ml.py             # Full ML grid search (Ridge, RF, XGBoost, SVR)
│   ├── models_ml_quick.py       # Documented entrypoint for the 4-model ML pipeline
│   ├── models_dl.py             # Pooled DL (LSTM, GRU, BiLSTM, Transformer)
│   ├── comparison_table.py      # 8-model comparison table
│   ├── walkforward_cv.py        # Time-series cross-validation
│   ├── ablation_vader_finbert.py # Unified no/VADER/FinBERT/both ablation
│   ├── ablation_sentiment.py    # Compatibility entrypoint for ablation
│   ├── ablation_finbert.py      # Compatibility entrypoint for ablation
│   ├── portfolio_opt.py         # MPT with Ledoit-Wolf + backtest
│   ├── recommendations.py       # 5-signal composite recommendations
│   ├── svr_attempt.py           # SVR-specific summary
│   └── dashboard.py             # Streamlit interactive dashboard (7 panels)
├── data/
│   ├── raw/                     # Downloaded, immutable
│   └── processed/               # Cleaned panel + features
├── models/                      # Saved artifacts (.pkl, .pth)
├── reports/                     # Figures + written report (final_report.md)
└── README.md                    # This file
```

---

## Key Results

| Model | RMSE | DirAcc | Rank |
|-------|------|--------|------|
| **Transformer** | **0.01794** | **52.2%** | 1st |
| LSTM | 0.01803 | 50.4% | 2nd |
| GRU | 0.01808 | 49.9% | 3rd |
| BiLSTM | 0.01822 | 49.3% | 4th |
| RandomForest | 0.01936 | 53.0% | 5th |
| Ridge | 0.01969 | 49.0% | 6th |
| XGBoost | 0.01982 | 51.3% | 7th |
| SVR | 0.02778 | 49.2% | 8th |

**Portfolio**: Sharpe-Max returned 54.84% OOS vs. 54.14% for equal-weight and 38.24% for SP500, but equal-weight had the higher OOS Sharpe (1.7314 vs. 1.5781).

**Sentiment**: VADER and FinBERT were tested separately and together. The best sentiment configuration improved RMSE only marginally (0.019324 to 0.019311), so sentiment did not materially improve predictive accuracy.

---

## System Requirements

- Python 3.12+
- 8GB RAM minimum (16GB recommended for DL models)
- ~2GB disk space (including FinBERT model download)
- CPU sufficient (GPU optional, speeds up DL training)

---

## Disclaimer

This output is **educational research**, not financial advice. These results are generated by a student capstone project for analytical demonstration only. Past performance does not guarantee future results. Do not use these signals for actual trading or investment decisions without independent professional advice.
