"""
VADER Sentiment Scoring (REAL implementation)
Uses NLTK's VADER lexicon — a rule-based, lexicon-driven sentiment analyzer.
This is the lightweight baseline that PRD Section 10 explicitly requires.

FinBERT (in sentiment.py and sentiment_finbert.py) is the domain-tuned model.
This script provides the genuine lexicon-based comparison.
"""
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==========================================================
# Paths
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEWS_PROCESSED = PROJECT_ROOT / "data" / "processed" / "news" / "news.csv"
SENTIMENT_DIR = PROJECT_ROOT / "data" / "processed" / "sentiment"
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load news
# ==========================================================
print("Loading processed news...")
news = pd.read_csv(NEWS_PROCESSED)
news["publishedAt"] = pd.to_datetime(news["publishedAt"])
news = news.sort_values("publishedAt").reset_index(drop=True)
print(f"Total articles: {len(news)}")

# ==========================================================
# Initialize VADER
# ==========================================================
print("\nInitializing VADER (lexicon-based sentiment analyzer)...")
analyzer = SentimentIntensityAnalyzer()
print("VADER loaded.")

# ==========================================================
# Score headlines with VADER
# ==========================================================
print("\nScoring headlines with VADER...")

vader_scores = []
vader_pos = []
vader_neg = []
vader_neu = []

for headline in tqdm(news["headline"].fillna("")):
    vs = analyzer.polarity_scores(str(headline))
    vader_scores.append(vs["compound"])
    vader_pos.append(vs["pos"])
    vader_neg.append(vs["neg"])
    vader_neu.append(vs["neu"])

news["VADER_Compound"] = vader_scores
news["VADER_Pos"] = vader_pos
news["VADER_Neg"] = vader_neg
news["VADER_Neu"] = vader_neu

# ==========================================================
# Date alignment (same rules as FinBERT)
# ==========================================================
print("\nAligning to trading sessions...")

news["publishedAt_local"] = (
    news["publishedAt"]
    .dt.tz_localize("UTC")
    .dt.tz_convert("US/Eastern")
)

def map_to_session(ts):
    if ts.time() >= pd.Timestamp("16:00").time() or ts.weekday() >= 5:
        return (ts + pd.Timedelta(days=1)).date()
    return ts.date()

news["Date"] = news["publishedAt_local"].apply(map_to_session)

# ==========================================================
# Aggregate daily VADER sentiment per ticker
# ==========================================================
print("Aggregating daily VADER sentiment...")

daily_vader = (
    news.groupby(["Date", "ticker"])
    .agg(
        VADER_Mean=("VADER_Compound", "mean"),
        VADER_Std=("VADER_Compound", "std"),
        VADER_Pos_Mean=("VADER_Pos", "mean"),
        VADER_Neg_Mean=("VADER_Neg", "mean"),
        VADER_Neu_Mean=("VADER_Neu", "mean"),
        Article_Count=("VADER_Compound", "count"),
        Positive_Count=("VADER_Compound", lambda x: (x > 0.05).sum()),
        Negative_Count=("VADER_Compound", lambda x: (x < -0.05).sum()),
        Neutral_Count=("VADER_Compound", lambda x: ((x >= -0.05) & (x <= 0.05)).sum()),
    )
    .reset_index()
)
daily_vader["Has_News"] = 1

print(f"VADER: {len(daily_vader)} ticker-days with news")

# ==========================================================
# Save
# ==========================================================
daily_vader.to_csv(SENTIMENT_DIR / "daily_vader_sentiment.csv", index=False)
print(f"\nSaved: {SENTIMENT_DIR / 'daily_vader_sentiment.csv'}")

# Distribution
print("\nVADER Sentiment Distribution (article-level):")
print(news["VADER_Compound"].describe())

print("\nVADER Label Distribution:")
labels = news["VADER_Compound"].apply(lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral"))
print(labels.value_counts())

print("\nVADER Pipeline Completed Successfully.")
