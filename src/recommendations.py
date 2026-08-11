"""
==============================================================
Investment Recommendations
PRD Section 12 - Required Deliverable
==============================================================

Generates per-ticker Buy / Hold / Sell recommendations by
combining five independent signals:
  1. Portfolio-optimization weight (Sharpe-Max allocation)
  2. Historical risk-adjusted return (Sharpe ratio)
  3. ML/DL model reliability (historical directional accuracy)
     -- NOTE: this measures model trustworthiness, not current
     forecast direction. A genuine live-forecast signal would
     require per-ticker signed predictions from the model
     pipeline, which is not yet exported to file.
  4. Sentiment score (average news sentiment, -1 to +1)
  5. Recent price momentum (Price vs SMA-20)

Each signal is z-scored and combined into a composite score.
Thresholds map scores to actionable recommendations with
individual rationales.

Also produces the PRD-mandated portfolio rebalancing table
(current weight, target weight, suggested action, drift).

Outputs:
  reports/recommendations_report.txt
  reports/recommendations_scores.png

DISCLAIMER: This output is educational research, not financial advice.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


# ==========================================================
# Load per-ticker data from all upstream modules
# ==========================================================

def load_portfolio_weights():
    """Extract Sharpe-Max and Risk-Parity weights from portfolio_opt output."""
    files = sorted(FEATURE_DIR.glob("*.csv"))
    dfs = []
    tickers = []
    for f in files:
        ticker = f.stem.replace("_features", "")
        df = pd.read_csv(f, parse_dates=["Date"])[["Date", "Daily_Return"]].dropna()
        df = df.rename(columns={"Daily_Return": ticker})
        dfs.append(df)
        tickers.append(ticker)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="Date", how="inner")
    merged = merged.sort_values("Date").reset_index(drop=True)

    returns = merged[tickers].values
    n = len(merged)
    split_idx = int(n * 0.85)
    train_returns = returns[:split_idx]

    expected_returns = train_returns.mean(axis=0) * 252
    cov_matrix = np.cov(train_returns, rowvar=False, ddof=1) * 252

    from scipy.optimize import minimize

    def metrics(w, mu, sigma):
        r = np.dot(w, mu)
        v = np.sqrt(w @ sigma @ w)
        s = (r - 0.02) / v if v > 0 else 0
        return r, v, s

    def neg_sharpe(w, mu, sigma):
        return -metrics(w, mu, sigma)[2]

    bounds = [(0.0, 0.30)] * len(tickers)
    w0 = np.ones(len(tickers)) / len(tickers)
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    result = minimize(neg_sharpe, w0, args=(expected_returns, cov_matrix),
                      method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-12})
    sharpe_weights = result.x

    def rp_obj(w, sigma):
        sw = sigma @ w
        pv = w @ sw
        if pv <= 1e-12:
            return 1e6
        rc = w * sw / pv
        return np.sum((rc - 1/len(tickers)) ** 2)

    result_rp = minimize(rp_obj, w0, args=(cov_matrix,),
                         method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-12, "maxiter": 2000})
    rp_weights = result_rp.x

    return tickers, sharpe_weights, rp_weights


def load_risk_metrics(tickers):
    """Compute Sharpe, Beta, Sortino, MaxDD for each ticker from full history."""
    sp500 = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "benchmark" / "SP500.csv",
                        parse_dates=["Date"])
    sp500["SP500_Return"] = np.log(sp500["Close"] / sp500["Close"].shift(1))
    market_df = sp500[["Date", "SP500_Return"]].dropna()

    metrics = {}
    for ticker in tickers:
        df = pd.read_csv(FEATURE_DIR / f"{ticker}.csv", parse_dates=["Date"])
        df = df[["Date", "Daily_Return"]].dropna()
        merged = pd.merge(df, market_df, on="Date", how="inner")
        rets = merged["Daily_Return"].values
        mkt = merged["SP500_Return"].values

        excess = rets - (0.02 / 252)
        sharpe = (np.mean(excess) / np.std(rets, ddof=1)) * np.sqrt(252)

        downside = rets[rets < 0]
        dstd = np.std(downside, ddof=1) if len(downside) > 0 else 0
        sortino = (np.mean(excess) / dstd) * np.sqrt(252) if dstd > 0 else np.inf

        cov = np.cov(rets, mkt, ddof=1)[0, 1]
        beta = cov / np.var(mkt, ddof=1) if np.var(mkt, ddof=1) > 0 else 0

        cum = np.exp(np.cumsum(rets))
        peak = np.maximum.accumulate(cum)
        mdd = ((cum - peak) / peak).min()

        metrics[ticker] = {"sharpe": sharpe, "sortino": sortino, "beta": beta, "mdd": mdd}

    return metrics


def load_model_reliability(tickers):
    """
    Load best pooled ML and DL historical directional accuracy per ticker.
    This is a RELIABILITY metric (how often the model was right in backtest),
    NOT a live forecast direction. Used as a proxy for model trustworthiness.
    """
    ml_df = pd.read_csv(REPORT_DIR / "best_ml_models_pooled.csv")
    dl_df = pd.read_csv(REPORT_DIR / "best_dl_models_pooled.csv")

    acc = {}
    for ticker in tickers:
        ml_row = ml_df[ml_df["Ticker"] == ticker]
        dl_row = dl_df[dl_df["Ticker"] == ticker]
        ml_acc = ml_row["Directional_Accuracy"].values[0] if len(ml_row) > 0 else 50.0
        dl_acc = dl_row["Directional_Accuracy"].values[0] if len(dl_row) > 0 else 50.0
        # CSV stores DirAcc as 0-100 (e.g., 53.95), convert to fraction
        acc[ticker] = ((ml_acc + dl_acc) / 2) / 100.0
    return acc


def load_sentiment(tickers):
    """Compute average Sentiment_Score per ticker from latest 30 days of news."""
    sent = {}
    for ticker in tickers:
        df = pd.read_csv(FEATURE_DIR / f"{ticker}.csv", parse_dates=["Date"])
        if "Sentiment_Score" in df.columns and "Has_News" in df.columns:
            # Only days with actual news
            mask = df["Has_News"] == 1
            recent = df.loc[mask, "Sentiment_Score"].dropna().tail(30)
            sent[ticker] = recent.mean() if len(recent) > 0 else 0.0
        else:
            sent[ticker] = 0.0
    return sent


def load_momentum(tickers):
    """Compute recent momentum as Price_to_SMA_20 from latest feature data."""
    mom = {}
    for ticker in tickers:
        df = pd.read_csv(FEATURE_DIR / f"{ticker}.csv", parse_dates=["Date"])
        if "Price_to_SMA_20" in df.columns:
            latest = df["Price_to_SMA_20"].dropna().iloc[-1]
            mom[ticker] = latest
        else:
            mom[ticker] = 1.0
    return mom


# ==========================================================
# Rebalancing table (PRD Listing 12.1)
# ==========================================================

def rebalance(current_w, target_w, band=0.05):
    """
    Compare current portfolio weights against target weights.
    Returns action: INCREASE, REDUCE, or HOLD based on drift band.
    """
    drift = abs(current_w - target_w)
    if drift > band and target_w > current_w:
        return "INCREASE", drift
    elif drift > band and target_w < current_w:
        return "REDUCE", drift
    else:
        return "HOLD", drift


def build_rebalancing_table(tickers, current_w, target_w):
    """Build the PRD-mandated rebalancing table."""
    rows = []
    for t, cw, tw in zip(tickers, current_w, target_w):
        action, drift = rebalance(cw, tw)
        rows.append({
            "Ticker": t,
            "Current_Weight": cw,
            "Target_Weight": tw,
            "Drift": drift,
            "Action": action
        })
    return pd.DataFrame(rows)


# ==========================================================
# Scoring and recommendation engine
# ==========================================================

def z_score(values):
    """Z-score normalization."""
    arr = np.array(values, dtype=float)
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    return (arr - m) / s if s > 0 else np.zeros_like(arr)


def generate_recommendations(tickers, sharpe_w, rp_w, risk_metrics, model_rel, sentiment, momentum):
    """
    Build composite score from five signals and map to Buy/Hold/Sell.

    Signal weights (equal by design):
      - Portfolio signal (Sharpe-Max weight): 20%
      - Risk signal (Sharpe ratio): 20%
      - Model reliability signal (historical DirAcc): 20%
      - Sentiment signal (avg news sentiment): 20%
      - Momentum signal (Price_to_SMA20): 20%
    """
    sharpe_vals = np.array([risk_metrics[t]["sharpe"] for t in tickers])
    sortino_vals = np.array([risk_metrics[t]["sortino"] for t in tickers])
    beta_vals = np.array([risk_metrics[t]["beta"] for t in tickers])
    mdd_vals = np.array([risk_metrics[t]["mdd"] for t in tickers])
    rel_vals = np.array([model_rel[t] for t in tickers])
    sent_vals = np.array([sentiment[t] for t in tickers])
    mom_vals = np.array([momentum[t] for t in tickers])

    z_portfolio = z_score(sharpe_w)
    z_sharpe = z_score(sharpe_vals)
    z_model = z_score(rel_vals)
    z_sent = z_score(sent_vals)
    z_mom = z_score(mom_vals)

    composite = 0.20 * z_portfolio + 0.20 * z_sharpe + 0.20 * z_model + 0.20 * z_sent + 0.20 * z_mom

    recs = []
    for i, ticker in enumerate(tickers):
        score = composite[i]
        if score >= 0.30:
            rec = "BUY"
        elif score <= -0.30:
            rec = "SELL"
        else:
            rec = "HOLD"
        recs.append({
            "ticker": ticker,
            "composite_score": score,
            "recommendation": rec,
            "sharpe_max_weight": sharpe_w[i],
            "risk_parity_weight": rp_w[i],
            "sharpe_ratio": sharpe_vals[i],
            "sortino_ratio": sortino_vals[i],
            "beta": beta_vals[i],
            "max_dd": mdd_vals[i],
            "model_reliability": rel_vals[i],
            "sentiment": sent_vals[i],
            "momentum": mom_vals[i],
            "z_portfolio": z_portfolio[i],
            "z_sharpe": z_sharpe[i],
            "z_model": z_model[i],
            "z_sentiment": z_sent[i],
            "z_momentum": z_mom[i],
        })

    recs.sort(key=lambda x: x["composite_score"], reverse=True)
    return recs


def build_rationale(rec):
    """Build a human-readable rationale string for a single ticker."""
    lines = []

    # Portfolio weight signal
    if rec["sharpe_max_weight"] >= 0.20:
        lines.append(f"Sharpe-Max portfolio allocates {rec['sharpe_max_weight']*100:.1f}%, indicating strong optimizer conviction.")
    elif rec["sharpe_max_weight"] <= 0.02:
        lines.append(f"Sharpe-Max portfolio allocates only {rec['sharpe_max_weight']*100:.1f}%, suggesting optimizer avoidance.")
    else:
        lines.append(f"Sharpe-Max portfolio allocates {rec['sharpe_max_weight']*100:.1f}%.")

    # Risk signal
    lines.append(f"Historical Sharpe: {rec['sharpe_ratio']:.2f}, Sortino: {rec['sortino_ratio']:.2f}, Beta: {rec['beta']:.2f}.")

    # Model reliability signal
    if rec["model_reliability"] >= 0.54:
        lines.append(f"Pooled ML+DL historical reliability: {rec['model_reliability']*100:.1f}% (above-random trust level).")
    elif rec["model_reliability"] <= 0.51:
        lines.append(f"Pooled ML+DL historical reliability: {rec['model_reliability']*100:.1f}% (weak predictive edge).")
    else:
        lines.append(f"Pooled ML+DL historical reliability: {rec['model_reliability']*100:.1f}%.")

    # Sentiment signal
    if rec["sentiment"] > 0.15:
        lines.append(f"Recent news sentiment: {rec['sentiment']:.3f} (net positive coverage).")
    elif rec["sentiment"] < -0.15:
        lines.append(f"Recent news sentiment: {rec['sentiment']:.3f} (net negative coverage).")
    else:
        lines.append(f"Recent news sentiment: {rec['sentiment']:.3f} (neutral coverage).")

    # Momentum signal
    if rec["momentum"] > 1.05:
        lines.append(f"Price is {rec['momentum']*100:.1f}% of 20-day SMA -- positive short-term momentum.")
    elif rec["momentum"] < 0.95:
        lines.append(f"Price is {rec['momentum']*100:.1f}% of 20-day SMA -- negative short-term momentum.")
    else:
        lines.append(f"Price is near 20-day SMA ({rec['momentum']*100:.1f}%) -- neutral momentum.")

    # Honest limitation note
    lines.append("")
    lines.append("NOTE: The model signal used here measures historical reliability,")
    lines.append("not the current predicted direction. A genuine live-forecast signal")
    lines.append("would require per-ticker signed predictions exported from the model pipeline.")

    lines.append(f"\nComposite z-score: {rec['composite_score']:.3f} -> {rec['recommendation']}.")
    return "\n".join(lines)


# ==========================================================
# Main
# ==========================================================

def main():
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("Investment Recommendations - PRD Section 12")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("MANDATORY DISCLAIMER: This output is educational research,")
    report_lines.append("not financial advice. These recommendations are generated by a")
    report_lines.append("student capstone project for analytical demonstration only.")
    report_lines.append("Do not use them for actual trading or investment decisions.")
    report_lines.append("")
    report_lines.append("Methodology:")
    report_lines.append("  Five signals are z-scored and combined with equal weights (20% each):")
    report_lines.append("    1. Portfolio optimization weight (Sharpe-Max allocation)")
    report_lines.append("    2. Historical risk-adjusted return (annualized Sharpe ratio)")
    report_lines.append("    3. ML/DL model RELIABILITY (historical directional accuracy)")
    report_lines.append("       -- measures how often the model was right, NOT current forecast")
    report_lines.append("    4. Sentiment score (average news sentiment, last 30 days with news)")
    report_lines.append("    5. Recent price momentum (Price / SMA-20)")
    report_lines.append("")
    report_lines.append("  Thresholds:  BUY if composite >= +0.30")
    report_lines.append("              SELL if composite <= -0.30")
    report_lines.append("              HOLD otherwise")
    report_lines.append("")

    # Load data
    tickers, sharpe_w, rp_w = load_portfolio_weights()
    risk_metrics = load_risk_metrics(tickers)
    model_rel = load_model_reliability(tickers)
    sentiment = load_sentiment(tickers)
    momentum = load_momentum(tickers)

    # Generate recommendations
    recs = generate_recommendations(tickers, sharpe_w, rp_w, risk_metrics, model_rel, sentiment, momentum)

    # Summary table
    report_lines.append("-" * 78)
    report_lines.append(f"{'Ticker':<8} {'Rec':<6} {'Score':>8} {'SharpeW':>8} {'Sharpe':>8} {'Rel':>8} {'Sent':>8} {'Mom':>8}")
    report_lines.append("-" * 78)
    for rec in recs:
        report_lines.append(
            f"{rec['ticker']:<8} {rec['recommendation']:<6} {rec['composite_score']:>8.3f} "
            f"{rec['sharpe_max_weight']*100:>7.1f}% {rec['sharpe_ratio']:>8.2f} "
            f"{rec['model_reliability']*100:>7.1f}% {rec['sentiment']:>8.3f} {rec['momentum']:>8.3f}"
        )

    # ----------------------------------------------------------
    # Rebalancing table (PRD Section 12, Listing 12.1)
    # ----------------------------------------------------------
    report_lines.append("\n" + "=" * 70)
    report_lines.append("Portfolio Rebalancing Table")
    report_lines.append("=" * 70)
    report_lines.append("Assumes current holdings = equal-weight (10% each).")
    report_lines.append("Target = Sharpe-Max optimized weights. Band = 5%.")
    report_lines.append("")

    current_w = np.ones(len(tickers)) / len(tickers)
    rebal_df = build_rebalancing_table(tickers, current_w, sharpe_w)

    report_lines.append(f"{'Ticker':<8} {'Current':>10} {'Target':>10} {'Drift':>10} {'Action':>10}")
    report_lines.append("-" * 52)
    for _, row in rebal_df.iterrows():
        report_lines.append(
            f"{row['Ticker']:<8} {row['Current_Weight']*100:>9.2f}% {row['Target_Weight']*100:>9.2f}% "
            f"{row['Drift']*100:>9.2f}% {row['Action']:>10}"
        )

    # Per-ticker detailed rationale
    report_lines.append("\n" + "=" * 70)
    report_lines.append("Detailed Rationales")
    report_lines.append("=" * 70)

    for rec in recs:
        report_lines.append("")
        report_lines.append(f"--- {rec['ticker']} ({rec['recommendation']}) ---")
        report_lines.append(build_rationale(rec))

    # Aggregate counts
    buys = sum(1 for r in recs if r["recommendation"] == "BUY")
    holds = sum(1 for r in recs if r["recommendation"] == "HOLD")
    sells = sum(1 for r in recs if r["recommendation"] == "SELL")
    report_lines.append("\n" + "=" * 70)
    report_lines.append("Summary Counts")
    report_lines.append("=" * 70)
    report_lines.append(f"  BUY : {buys}")
    report_lines.append(f"  HOLD: {holds}")
    report_lines.append(f"  SELL: {sells}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    tickers_sorted = [r["ticker"] for r in recs]
    scores = [r["composite_score"] for r in recs]
    colors = ["green" if r["recommendation"] == "BUY" else "red" if r["recommendation"] == "SELL" else "gray" for r in recs]

    axes[0].barh(tickers_sorted[::-1], scores[::-1], color=colors[::-1])
    axes[0].axvline(0.30, color="green", linestyle="--", linewidth=1.5, label="BUY threshold")
    axes[0].axvline(-0.30, color="red", linestyle="--", linewidth=1.5, label="SELL threshold")
    axes[0].axvline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("Composite Z-Score")
    axes[0].set_title("Per-Ticker Composite Scores")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="x")

    x = np.arange(len(tickers_sorted))
    width = 0.15
    axes[1].bar(x - 2*width, [r["z_portfolio"] for r in recs], width, label="Portfolio", color="steelblue")
    axes[1].bar(x - width, [r["z_sharpe"] for r in recs], width, label="Sharpe", color="orange")
    axes[1].bar(x, [r["z_model"] for r in recs], width, label="Model Rel", color="purple")
    axes[1].bar(x + width, [r["z_sentiment"] for r in recs], width, label="Sentiment", color="teal")
    axes[1].bar(x + 2*width, [r["z_momentum"] for r in recs], width, label="Momentum", color="green")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tickers_sorted, rotation=45, ha="right")
    axes[1].set_ylabel("Z-Score")
    axes[1].set_title("Signal Breakdown by Ticker")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "recommendations_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    report_lines.append(f"\nPlot saved to: {REPORT_DIR / 'recommendations_scores.png'}")

    # Save report
    report_text = "\n".join(report_lines)
    report_path = REPORT_DIR / "recommendations_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n{'='*70}")
    print(f"Report saved to: {report_path}")
    print(f"Plot saved to:   {REPORT_DIR / 'recommendations_scores.png'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
