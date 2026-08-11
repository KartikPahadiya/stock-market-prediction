"""
Merge REAL VADER + FinBERT sentiment into unified daily sentiment file.

This corrects the historical bug where sentiment.py claimed to output "VADER"
but actually used FinBERT. The unified file provides:
  - VADER_Mean, VADER_Pos_Mean, VADER_Neg_Mean, VADER_Neu_Mean (REAL VADER)
  - FinBERT_Sentiment_Score, FinBERT_Positive_Prob, etc. (from sentiment.py)

Uses only files that exist — no hard dependency on sentiment_finbert.py.
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENTIMENT_DIR = PROJECT_ROOT / "data" / "processed" / "sentiment"

# --- Load VADER ---
vader_path = SENTIMENT_DIR / "daily_vader_sentiment.csv"
if not vader_path.exists():
    raise FileNotFoundError(f"{vader_path} missing — run sentiment_vader.py first")
vader = pd.read_csv(vader_path)
vader["Date"] = pd.to_datetime(vader["Date"])

# --- Load FinBERT (from sentiment.py) ---
finbert_path = SENTIMENT_DIR / "daily_sentiment.csv"
if not finbert_path.exists():
    raise FileNotFoundError(f"{finbert_path} missing — run sentiment.py first")
finbert = pd.read_csv(finbert_path)
finbert["Date"] = pd.to_datetime(finbert["Date"])

# Rename to avoid collision
finbert = finbert.rename(columns={
    "Sentiment_Score": "FinBERT_Sentiment_Score",
    "Positive_Prob": "FinBERT_Positive_Prob",
    "Negative_Prob": "FinBERT_Negative_Prob",
    "Neutral_Prob": "FinBERT_Neutral_Prob",
    "Article_Count": "FinBERT_Article_Count",
    "Positive_Count": "FinBERT_Positive_Count",
    "Negative_Count": "FinBERT_Negative_Count",
    "Neutral_Count": "FinBERT_Neutral_Count",
    "Has_News": "FinBERT_Has_News",
})

# --- Merge ---
merged = pd.merge(
    vader, finbert[["ticker", "Date",
                    "FinBERT_Sentiment_Score", "FinBERT_Positive_Prob",
                    "FinBERT_Negative_Prob", "FinBERT_Neutral_Prob",
                    "FinBERT_Article_Count", "FinBERT_Positive_Count",
                    "FinBERT_Negative_Count", "FinBERT_Neutral_Count",
                    "FinBERT_Has_News"]],
    on=["ticker", "Date"],
    how="outer"
)

# Fill VADER missing
vader_defaults = {
    "VADER_Mean": 0.0, "VADER_Std": 0.0,
    "VADER_Pos_Mean": 0.0, "VADER_Neg_Mean": 0.0, "VADER_Neu_Mean": 1.0,
    "Article_Count": 0, "Positive_Count": 0, "Negative_Count": 0, "Neutral_Count": 0,
    "Has_News": 0
}
for col, default in vader_defaults.items():
    if col in merged.columns:
        merged[col] = merged[col].fillna(default)

# Fill FinBERT missing
finbert_defaults = {
    "FinBERT_Sentiment_Score": 0.0,
    "FinBERT_Positive_Prob": 0.0, "FinBERT_Negative_Prob": 0.0, "FinBERT_Neutral_Prob": 1.0,
    "FinBERT_Article_Count": 0,
    "FinBERT_Positive_Count": 0, "FinBERT_Negative_Count": 0, "FinBERT_Neutral_Count": 0,
    "FinBERT_Has_News": 0
}
for col, default in finbert_defaults.items():
    if col in merged.columns:
        merged[col] = merged[col].fillna(default)

# Save
merged.to_csv(SENTIMENT_DIR / "daily_sentiment_unified.csv", index=False)

# Stats
corr = merged["VADER_Mean"].corr(merged["FinBERT_Sentiment_Score"])
print(f"Unified file: {SENTIMENT_DIR / 'daily_sentiment_unified.csv'}")
print(f"Rows: {len(merged)} | VADER days: {len(vader)} | FinBERT days: {len(finbert)}")
print(f"VADER-FinBERT correlation: {corr:.4f}")
print("Done.")
