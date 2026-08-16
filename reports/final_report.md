# AI-Based Stock Market Prediction and Portfolio Optimization System

Northgate Quantitative Research - Capstone Final Report

## Executive Summary

This project implements an end-to-end quantitative research platform for a ten-stock large-cap equity universe: AAPL, AMZN, GOOGL, JNJ, JPM, META, MSFT, NVDA, PG, and XOM. The system downloads public market, macroeconomic, and news data; cleans and aligns it to a trading calendar; engineers leakage-safe return, trend, volatility, volume, macro, and sentiment features; trains classical machine-learning and deep-learning forecasters; optimizes a constrained long-only portfolio; and exposes the outputs through a Streamlit dashboard.

The strongest cached forecasting result is the Transformer model with RMSE 0.01794 and directional accuracy 52.22% on the pooled held-out window. Among classical models, RandomForest is strongest with RMSE 0.01936 and directional accuracy 53.03%. The sentiment ablation shows only marginal movement in RMSE and directional accuracy, so the project treats sentiment as a measured but weak auxiliary signal rather than as a decisive alpha source.

The Sharpe-max portfolio produced the highest in-sample Sharpe and returned 54.84% in the out-of-sample backtest versus 54.14% for equal-weight and 38.24% for SP500 buy-and-hold. However, equal-weight had the better out-of-sample Sharpe ratio: 1.7314 versus 1.5781 for the optimized portfolio. This is an important limitation and is reported honestly.

## Data and Ingestion

The ingestion layer uses Yahoo Finance via yfinance for daily OHLCV data, FRED through pandas-datareader for macro series, and Finnhub for company-news headlines and summaries. The project reads the Finnhub key from `.env` and keeps that file ignored by Git. The PRD's requirement to distinguish document instructions from the user request was handled by treating the DOCX as product requirements only; no text inside it was treated as an instruction to the assistant.

The universe spans ten liquid U.S. equities from 2017 through the configured end date. The pipeline also ingests S&P 500 and VIX benchmark data plus DGS10, DGS3MO, CPIAUCSL, and UNRATE macro series. With `auto_adjust: true`, Yahoo's adjusted price series is used for model features and target-return construction.

## Cleaning and Feature Engineering

The cleaning stage creates processed stock, benchmark, macro, and news files, then emits a data-quality report. It aligns data chronologically, handles limited forward fills, preserves the no-backfill rule for prices, checks OHLC consistency, and prepares a model-ready panel.

Feature engineering uses only values available at or before time `t`. The target is the next-day log return, created with a forward shift only for the label. Features include daily and lagged returns, moving averages, RSI, MACD-style trend signals, Bollinger-band width, ATR-style volatility, volume ratios, benchmark and macro context, and merged VADER/FinBERT sentiment fields. Train, validation, and test splits are chronological.

## EDA and Mathematical Validation

EDA outputs include return distributions, correlation heatmaps, rolling volatility versus VIX, seasonal heatmaps, ADF stationarity checks, cumulative returns, and feature-target correlations. These artifacts support the modelling choice to forecast returns rather than raw prices.

The from-scratch math module validates manual implementations of gradient-descent linear regression, Sharpe ratio, Sortino ratio, Beta, maximum drawdown, covariance, and portfolio variance against library or vectorized equivalents. The cached report shows the manual gradient-descent regression matches sklearn OLS within numerical tolerance and the financial calculations pass their validation checks.

## Forecasting Results

The eight-model comparison table is populated on the held-out pooled test window:

| Rank | Family | Model | RMSE | MAE | R2 | Directional Accuracy |
|---:|---|---|---:|---:|---:|---:|
| 1 | DL | Transformer | 0.01794 | 0.01271 | -0.0012 | 52.22% |
| 2 | DL | LSTM | 0.01803 | 0.01281 | -0.0118 | 50.44% |
| 3 | DL | GRU | 0.01808 | 0.01287 | -0.0173 | 49.93% |
| 4 | DL | BiLSTM | 0.01822 | 0.01296 | -0.0331 | 49.25% |
| 5 | ML | RandomForest | 0.01936 | 0.01336 | 0.0248 | 53.03% |
| 6 | ML | Ridge | 0.01969 | 0.01361 | -0.0081 | 48.95% |
| 7 | ML | XGBoost | 0.01982 | 0.01368 | -0.0219 | 51.30% |
| 8 | ML | SVR | 0.02778 | 0.02170 | -1.0066 | 49.18% |

The results are directionally plausible for noisy daily equity returns: small differences in RMSE, low R2, and directional accuracy clustered near 50-53%. RandomForest is the most reliable classical model, while the Transformer gives the best pooled RMSE.

Walk-forward validation with three splits reports test RMSE 0.01941, MAE 0.01337, and directional accuracy 53.7%, supporting the chronological-validation discipline required by the PRD.

## Sentiment Analysis

The sentiment module includes both a real VADER lexicon baseline and a FinBERT finance-domain transformer. News timestamps are mapped to trading sessions so after-close and weekend articles roll forward rather than leaking into same-session close predictions.

The ablation results are:

| Configuration | RMSE | MAE | R2 | Directional Accuracy |
|---|---:|---:|---:|---:|
| No sentiment | 0.019324 | 0.013347 | 0.02896 | 53.20% |
| VADER only | 0.019315 | 0.013340 | 0.02986 | 53.28% |
| FinBERT only | 0.019323 | 0.013349 | 0.02909 | 53.25% |
| Both VADER + FinBERT | 0.019311 | 0.013342 | 0.03027 | 53.25% |

The measured effect is tiny. Sentiment is retained for interpretability and dashboard context, but the evidence does not support a claim that sentiment materially improves daily-return forecasting on this dataset.

## Portfolio Optimization

The portfolio optimizer uses historical train-window returns, Ledoit-Wolf shrinkage covariance, long-only weights, and a 30% per-asset cap. It computes equal-weight, minimum-variance, Sharpe-maximizing, risk-parity, and SP500 benchmark strategies, then freezes the selected train-window weights for out-of-sample testing.

Out-of-sample results:

| Strategy | Total Return | Annual Volatility | Sharpe | Sortino | Beta | Max Drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Sharpe-Max | 54.84% | 19.27% | 1.5781 | 2.1713 | 0.9858 | -12.28% |
| Equal-Weight | 54.14% | 17.37% | 1.7314 | 2.3307 | 0.9168 | -11.04% |
| SP500 Buy-and-Hold | 38.24% | 17.11% | 1.2860 | 1.6152 | 1.0000 | -12.14% |

The optimized portfolio beats SP500 on both return and Sharpe and narrowly beats equal-weight on total return, but it does not beat equal-weight on Sharpe. That shortfall is a central limitation of the final result.

## Recommendations and Dashboard

The recommendation layer produces transparent BUY/HOLD/SELL outputs from a composite of optimizer weight, historical risk-adjusted return, model reliability, recent sentiment, and momentum. The cached recommendation report currently gives 3 BUY, 4 HOLD, and 3 SELL decisions, with a rebalancing table from equal-weight current holdings to target Sharpe-max weights.

The Streamlit dashboard reads cached artifacts rather than retraining on load. It includes overview, price/prediction, model comparison, portfolio analytics, risk, sentiment, and recommendations panels, plus an educational-use disclaimer.

## Reproducibility

The project now includes `run_pipeline.py` as the single orchestration command:

```bash
python run_pipeline.py
```

For faster local iteration using cached FinBERT and deep-learning outputs:

```bash
python run_pipeline.py --fast
```

Individual modules remain independently runnable from `src/`. Compatibility entrypoints were added for README/PRD script names: `sentiment_finbert.py`, `models_ml_quick.py`, `ablation_sentiment.py`, `ablation_finbert.py`, and `svr_attempt.py`.

## Limitations

Daily equity returns are noisy, and all models show weak explanatory power. The best directional accuracy is only modestly above random. The sentiment experiment is measured and documented, but its incremental value is economically small. The portfolio optimizer improves over SP500 but does not dominate equal-weight on out-of-sample Sharpe, which means the MPT allocation should be interpreted as a research artifact, not a production allocation.

## Disclaimer

This system is educational research and a decision-support demonstration. It is not financial advice, does not guarantee future returns, and should not be used for live trading or investment decisions without independent professional review.
