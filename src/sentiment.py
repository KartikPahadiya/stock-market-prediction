"""
==========================================================
sentiment.py

Purpose:
    Generate article-level and daily aggregated sentiment
    using FinBERT.

Input:
    data/cleaned/news/news.csv

Outputs:
    data/processed/sentiment/article_sentiment.csv
    data/processed/sentiment/daily_sentiment.csv
==========================================================
"""

import pandas as pd
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NEWS_DIR = PROCESSED_DIR / "news"

SENTIMENT_DIR = PROCESSED_DIR / "sentiment"

SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load News
# ==========================================================

print("=" * 60)
print("Loading News Dataset...")
print("=" * 60)

news = pd.read_csv(NEWS_DIR / "news.csv")

print(f"Articles Loaded : {len(news)}")

# ==========================================================
# Data Preparation
# ==========================================================

news["publishedAt"] = pd.to_datetime(news["publishedAt"])

# Convert to US/Eastern (market timezone) -- drop tz conversion if already Eastern
news["publishedAt_local"] = (
    news["publishedAt"]
    .dt.tz_localize("UTC")
    .dt.tz_convert("US/Eastern")
)


def map_to_session(ts):
    """
    Map a news timestamp to the correct trading session.
    News after 16:00 or on weekends belongs to the NEXT trading session.
    """
    if ts.time() >= pd.Timestamp("16:00").time() or ts.weekday() >= 5:
        return (ts + pd.Timedelta(days=1)).date()
    return ts.date()


news["Date"] = news["publishedAt_local"].apply(map_to_session)

news["headline"] = news["headline"].fillna("")
news["summary"] = news["summary"].fillna("")

news["text"] = news["headline"] + ". " + news["summary"]

print("News Prepared Successfully.\n")

# ==========================================================
# Load FinBERT
# ==========================================================

MODEL_NAME = "ProsusAI/finbert"

print("=" * 60)
print("Loading FinBERT...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()

print("FinBERT Loaded Successfully.\n")

LABELS = ["positive", "negative", "neutral"]

# ==========================================================
# Prediction Function
# ==========================================================

def predict_sentiment(text):

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoded)

    probs = softmax(outputs.logits.numpy()[0])

    return dict(zip(LABELS, probs))

# ==========================================================
# Predict Sentiment
# ==========================================================

print("=" * 60)
print("Running FinBERT...")
print("=" * 60)

positive_probs = []
negative_probs = []
neutral_probs = []

labels = []
scores = []

for text in tqdm(news["text"]):

    result = predict_sentiment(text)

    positive = result["positive"]
    negative = result["negative"]
    neutral = result["neutral"]

    positive_probs.append(positive)
    negative_probs.append(negative)
    neutral_probs.append(neutral)

    label = max(result, key=result.get)

    labels.append(label)

    score = positive - negative

    scores.append(score)

news["Positive_Prob"] = positive_probs
news["Negative_Prob"] = negative_probs
news["Neutral_Prob"] = neutral_probs

news["Sentiment_Label"] = labels
news["Sentiment_Score"] = scores

print("\nArticle Sentiment Generated Successfully.\n")

# ==========================================================
# Save Article-Level Sentiment
# ==========================================================

article_output = SENTIMENT_DIR / "article_sentiment.csv"

news.to_csv(article_output, index=False)

print(f"Saved : {article_output}")

# ==========================================================
# Aggregate Daily Sentiment
# ==========================================================

print("\nAggregating Daily Sentiment...\n")

daily_sentiment = (

    news.groupby(["Date", "ticker"])

    .agg(

        Positive_Prob=("Positive_Prob", "mean"),
        Negative_Prob=("Negative_Prob", "mean"),
        Neutral_Prob=("Neutral_Prob", "mean"),
        Sentiment_Score=("Sentiment_Score", "mean"),
        Article_Count=("Sentiment_Label", "count"),
        Positive_Count=(
            "Sentiment_Label",
            lambda x: (x == "positive").sum()
        ),
        Negative_Count=(
            "Sentiment_Label",
            lambda x: (x == "negative").sum()
        ),
        Neutral_Count=(
            "Sentiment_Label",
            lambda x: (x == "neutral").sum()
        )

    )

    .reset_index()

)

daily_sentiment["Has_News"] = 1

print(daily_sentiment.head())

# ==========================================================
# Save Daily Sentiment
# ==========================================================

daily_output = SENTIMENT_DIR / "daily_sentiment.csv"

daily_sentiment.to_csv(
    daily_output,
    index=False
)

print("\nSaved :", daily_output)

print("\nDaily Sentiment Shape :", daily_sentiment.shape)

print("\nMissing Values\n")
print(daily_sentiment.isnull().sum())

print("\nSentiment Distribution\n")
print(news["Sentiment_Label"].value_counts())

print("\nSentiment Pipeline Completed Successfully.")
