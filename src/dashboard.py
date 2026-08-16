"""User-friendly Streamlit dashboard for the stock AI capstone.

The app is read-only: it loads cached CSV/PNG/TXT artifacts from the pipeline
and never retrains models on page load.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
EDA_DIR = REPORT_DIR / "eda"
DATA_DIR = ROOT / "data" / "processed"
FEATURE_DIR = DATA_DIR / "features"
STOCK_DIR = DATA_DIR / "stocks"
PRED_DIR = DATA_DIR / "predictions"

TICKERS = ["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "META", "MSFT", "NVDA", "PG", "XOM"]
MODEL_OPTIONS = ["RandomForest", "Ridge", "XGBoost", "SVR"]

PALETTE = {
    "page": "#F6F7F9",
    "panel": "#FFFFFF",
    "ink": "#17202A",
    "muted": "#64748B",
    "line": "#D7DEE8",
    "blue": "#2563EB",
    "teal": "#0F766E",
    "green": "#15803D",
    "amber": "#B7791F",
    "red": "#B91C1C",
    "violet": "#6D28D9",
}

REC_COLORS = {"BUY": PALETTE["green"], "HOLD": PALETTE["amber"], "SELL": PALETTE["red"]}


st.set_page_config(
    page_title="Northgate Stock AI",
    page_icon="NQ",
    layout="wide",
    initial_sidebar_state="expanded",
)

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    PALETTE["blue"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["amber"],
    PALETTE["red"],
    PALETTE["green"],
]

st.markdown(
    f"""
    <style>
    :root {{
        --page: {PALETTE["page"]};
        --panel: {PALETTE["panel"]};
        --ink: {PALETTE["ink"]};
        --muted: {PALETTE["muted"]};
        --line: {PALETTE["line"]};
        --blue: {PALETTE["blue"]};
    }}
    .stApp {{ background: var(--page); color: var(--ink); }}
    [data-testid="stHeader"] {{
        background: rgba(246, 247, 249, 0.92);
        border-bottom: 1px solid var(--line);
    }}
    section[data-testid="stSidebar"] {{
        background: #FFFFFF;
        border-right: 1px solid var(--line);
    }}
    h1, h2, h3, h4, h5, h6, p, label, span {{
        color: var(--ink);
        letter-spacing: 0;
    }}
    h1 {{ font-size: 2rem; line-height: 1.15; margin-bottom: 0.2rem; }}
    h2 {{ font-size: 1.28rem; margin-top: 0.4rem; }}
    h3 {{ font-size: 1.03rem; }}
    div[data-testid="stMetric"] {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        min-height: 104px;
    }}
    [data-testid="stMetricLabel"] p {{ color: var(--muted); font-size: 0.82rem; }}
    [data-testid="stMetricValue"] {{ color: var(--ink); font-size: 1.45rem; }}
    .app-kicker {{
        color: var(--blue);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }}
    .subtle {{
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: 0;
    }}
    .notice {{
        background: #FFF7ED;
        border: 1px solid #FED7AA;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        color: #7C2D12;
        font-size: 0.88rem;
    }}
    .callout {{
        background: #FFFFFF;
        border-left: 4px solid var(--blue);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: var(--ink);
        margin: 0.2rem 0 0.8rem 0;
    }}
    .section-label {{
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin: 0.6rem 0 0.25rem 0;
    }}
    .pill {{
        display: inline-block;
        border-radius: 999px;
        padding: 0.18rem 0.55rem;
        font-size: 0.74rem;
        font-weight: 700;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid var(--line);
        border-radius: 8px;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.2rem;
        border-bottom: 1px solid var(--line);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 0.55rem 0.8rem;
    }}
    .stTabs [aria-selected="true"] {{
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-bottom: 1px solid #FFFFFF;
    }}
    .stButton button, .stDownloadButton button {{
        border-radius: 8px;
        border: 1px solid var(--line);
        background: #FFFFFF;
        color: var(--ink);
    }}
    .stButton button:hover, .stDownloadButton button:hover {{
        border-color: var(--blue);
        color: var(--blue);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def metric_delta(value: float | int | None, digits: int = 1) -> str | None:
    if value is None or pd.isna(value):
        return None
    return f"{value * 100:+.{digits}f}%"


def style_plot(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=48, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["ink"], size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#E8EDF3", zerolinecolor="#E8EDF3")
    fig.update_yaxes(gridcolor="#E8EDF3", zerolinecolor="#E8EDF3")
    return fig


@st.cache_data(show_spinner=False)
def read_csv(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


@st.cache_data(show_spinner=False)
def refresh_marker() -> str:
    marker = read_text(REPORT_DIR / "last_refresh_utc.txt").strip()
    if marker:
        return marker
    candidates = [
        REPORT_DIR / "model_comparison.csv",
        REPORT_DIR / "portfolio_opt_report.txt",
        DATA_DIR / "data_quality_report.csv",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return "Not available"
    latest = max(existing, key=lambda p: p.stat().st_mtime)
    return f"{pd.Timestamp(latest.stat().st_mtime, unit='s', tz='UTC').isoformat()} (file timestamp)"


@st.cache_data(show_spinner=False)
def stock_df(ticker: str) -> pd.DataFrame | None:
    df = read_csv(STOCK_DIR / f"{ticker}.csv")
    if df is None:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")


@st.cache_data(show_spinner=False)
def feature_df(ticker: str) -> pd.DataFrame | None:
    df = read_csv(FEATURE_DIR / f"{ticker}.csv")
    if df is None:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")


@st.cache_data(show_spinner=False)
def model_comparison() -> pd.DataFrame:
    df = read_csv(REPORT_DIR / "model_comparison.csv")
    return pd.DataFrame() if df is None else df


@st.cache_data(show_spinner=False)
def sentiment_ablation() -> pd.DataFrame:
    df = read_csv(REPORT_DIR / "vader_finbert_ablation.csv")
    return pd.DataFrame() if df is None else df


@st.cache_data(show_spinner=False)
def ticker_snapshot() -> pd.DataFrame:
    rows = []
    for ticker in TICKERS:
        price = stock_df(ticker)
        features = feature_df(ticker)
        if price is None or price.empty:
            continue

        last = price.iloc[-1]
        ret_1d = price["Close"].pct_change().iloc[-1]
        ret_30d = price["Close"].pct_change(30).iloc[-1]
        vol_30d = price["Close"].pct_change().rolling(30).std().iloc[-1] * np.sqrt(252)
        rsi = features["RSI_14"].iloc[-1] if features is not None and "RSI_14" in features else np.nan
        sentiment = (
            features["FinBERT_Sentiment_Score"].tail(30).mean()
            if features is not None and "FinBERT_Sentiment_Score" in features
            else np.nan
        )
        rows.append(
            {
                "Ticker": ticker,
                "Date": last["Date"],
                "Close": last["Close"],
                "1D Return": ret_1d,
                "30D Return": ret_30d,
                "30D Vol": vol_30d,
                "RSI": rsi,
                "Sentiment": sentiment,
                "Rows": len(price),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def prediction_df(ticker: str, model: str) -> pd.DataFrame | None:
    pred = read_csv(PRED_DIR / f"pooled_{model}_predictions.csv")
    features = feature_df(ticker)
    if pred is None or features is None:
        return None
    ticker_id = TICKERS.index(ticker)
    pred = pred[pred["ticker_id"] == ticker_id].copy()
    if pred.empty:
        return None
    dates = features["Date"].tail(len(pred)).reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    pred["Date"] = dates
    pred["Direction Match"] = np.sign(pred["actual"]) == np.sign(pred["predicted"])
    return pred


@st.cache_data(show_spinner=False)
def returns_panel() -> pd.DataFrame:
    series = {}
    for ticker in TICKERS:
        price = stock_df(ticker)
        if price is None:
            continue
        series[ticker] = price.set_index("Date")["Close"].pct_change()
    return pd.DataFrame(series).dropna(how="all")


@st.cache_data(show_spinner=False)
def parse_portfolio() -> tuple[pd.DataFrame, pd.DataFrame]:
    text = read_text(REPORT_DIR / "portfolio_opt_report.txt")
    weights = []
    in_block = False
    for line in text.splitlines():
        if "Sharpe-Maximizing Portfolio" in line:
            in_block = True
            continue
        if in_block and "Expected Return" in line:
            break
        if in_block:
            match = re.search(r"\b([A-Z]{2,5})\s*:\s*([0-9.]+)", line)
            if match and match.group(1) in TICKERS:
                weights.append({"Ticker": match.group(1), "Weight": float(match.group(2))})

    backtest = []
    for line in text.splitlines():
        match = re.match(
            r"(Sharpe-Max \(chosen\)|Equal-Weight|SP500 Buy-and-Hold)\s+"
            r"([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(-?[0-9.]+)%",
            line.strip(),
        )
        if match:
            backtest.append(
                {
                    "Strategy": match.group(1),
                    "Total Return": float(match.group(2)) / 100,
                    "Ann. Vol": float(match.group(3)) / 100,
                    "Sharpe": float(match.group(4)),
                    "Sortino": float(match.group(5)),
                    "Beta": float(match.group(6)),
                    "Max Drawdown": float(match.group(7)) / 100,
                }
            )
    return pd.DataFrame(weights), pd.DataFrame(backtest)


@st.cache_data(show_spinner=False)
def parse_recommendations() -> tuple[pd.DataFrame, pd.DataFrame]:
    text = read_text(REPORT_DIR / "recommendations_report.txt")
    rec_rows = []
    reb_rows = []

    for line in text.splitlines():
        rec = re.match(
            r"^([A-Z]{2,5})\s+(BUY|HOLD|SELL)\s+(-?[0-9.]+)\s+([0-9.]+)%\s+"
            r"([0-9.]+)\s+([0-9.]+)%\s+(-?[0-9.]+)\s+([0-9.]+)",
            line.strip(),
        )
        if rec:
            rec_rows.append(
                {
                    "Ticker": rec.group(1),
                    "Recommendation": rec.group(2),
                    "Score": float(rec.group(3)),
                    "Target Weight": float(rec.group(4)) / 100,
                    "Sharpe": float(rec.group(5)),
                    "Reliability": float(rec.group(6)) / 100,
                    "Sentiment": float(rec.group(7)),
                    "Momentum": float(rec.group(8)),
                }
            )

        reb = re.match(
            r"^([A-Z]{2,5})\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+"
            r"(INCREASE|REDUCE|HOLD)",
            line.strip(),
        )
        if reb:
            reb_rows.append(
                {
                    "Ticker": reb.group(1),
                    "Current": float(reb.group(2)) / 100,
                    "Target": float(reb.group(3)) / 100,
                    "Drift": float(reb.group(4)) / 100,
                    "Action": reb.group(5),
                }
            )
    return pd.DataFrame(rec_rows), pd.DataFrame(reb_rows)


def recommendation_badge(label: str) -> str:
    color = REC_COLORS.get(label, PALETTE["muted"])
    return (
        f'<span class="pill" style="background:{color}18;color:{color};'
        f'border:1px solid {color}33;">{label}</span>'
    )


def page_title(title: str, subtitle: str) -> None:
    st.markdown('<div class="app-kicker">Northgate Quantitative Research</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="subtle">{subtitle}</p>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### Northgate Stock AI")
    st.caption("Research dashboard")
    ticker = st.selectbox("Ticker", TICKERS, index=0)
    model = st.selectbox("Forecast model", MODEL_OPTIONS, index=0)
    window = st.slider("Chart lookback", 90, 1000, 365, 30)
    st.button("Refresh cached data", on_click=st.cache_data.clear, width="stretch")
    st.divider()
    st.markdown(
        '<div class="notice"><b>Educational research only.</b><br>'
        'Outputs are not financial advice.</div>',
        unsafe_allow_html=True,
    )


snapshot = ticker_snapshot()
comparison = model_comparison()
weights, backtest = parse_portfolio()
rec_df, rebalance_df = parse_recommendations()
last_refresh = refresh_marker()

page_title(
    "AI Stock Prediction Dashboard",
    "Forecasting, sentiment, portfolio risk, and recommendations from cached project outputs.",
)

if snapshot.empty:
    st.error("No processed stock data found. Run the pipeline before opening the dashboard.")
    st.stop()

tabs = st.tabs(
    [
        "Overview",
        "Ticker",
        "Models",
        "Portfolio",
        "Sentiment",
        "Risk",
        "Recommendations",
        "Reports",
    ]
)


with tabs[0]:
    best_model = comparison.iloc[0] if not comparison.empty else None
    best_rec = rec_df.iloc[0] if not rec_df.empty else None
    chosen = backtest[backtest["Strategy"].str.contains("Sharpe-Max", na=False)] if not backtest.empty else pd.DataFrame()

    latest_market_date = pd.to_datetime(snapshot["Date"]).max()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Universe", f"{len(snapshot)} stocks", f"{int(snapshot['Rows'].min()):,}+ rows")
    c2.metric(
        "Latest data",
        latest_market_date.strftime("%Y-%m-%d"),
        "market close",
    )
    c3.metric(
        "Best model",
        best_model["Model"] if best_model is not None else "N/A",
        f"RMSE {best_model['RMSE']:.4f}" if best_model is not None else None,
    )
    c4.metric(
        "Chosen portfolio",
        pct(chosen["Total Return"].iloc[0]) if not chosen.empty else "N/A",
        f"Sharpe {chosen['Sharpe'].iloc[0]:.2f}" if not chosen.empty else None,
    )
    c5.metric(
        "Top signal",
        best_rec["Ticker"] if best_rec is not None else "N/A",
        best_rec["Recommendation"] if best_rec is not None else None,
    )

    st.markdown(
        '<div class="callout"><b>Read this as a research cockpit.</b> '
        'Use the ticker tab for a single-stock view, models for benchmark quality, '
        'portfolio for allocation, and recommendations for the transparent action layer.</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Pipeline refresh marker: {last_refresh}")

    left, right = st.columns([1.5, 1])
    with left:
        shown = snapshot.sort_values("30D Return", ascending=False).copy()
        fig = px.bar(
            shown,
            x="Ticker",
            y="30D Return",
            color="30D Return",
            color_continuous_scale=["#B91C1C", "#F59E0B", "#15803D"],
            title="Recent 30-day return by ticker",
        )
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(style_plot(fig, 390), width="stretch")

    with right:
        if not rec_df.empty:
            counts = rec_df["Recommendation"].value_counts().reindex(["BUY", "HOLD", "SELL"]).fillna(0)
            fig = px.pie(
                names=counts.index,
                values=counts.values,
                color=counts.index,
                color_discrete_map=REC_COLORS,
                title="Recommendation mix",
                hole=0.58,
            )
            st.plotly_chart(style_plot(fig, 390), width="stretch")

    st.markdown('<div class="section-label">Universe snapshot</div>', unsafe_allow_html=True)
    st.dataframe(
        snapshot.style.format(
            {
                "Close": "${:,.2f}",
                "1D Return": "{:+.2%}",
                "30D Return": "{:+.2%}",
                "30D Vol": "{:.1%}",
                "RSI": "{:.1f}",
                "Sentiment": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )


with tabs[1]:
    price = stock_df(ticker)
    features = feature_df(ticker)
    pred = prediction_df(ticker, model)
    if price is None or price.empty:
        st.warning(f"No processed stock file found for {ticker}.")
    else:
        latest = price.iloc[-1]
        returns = price["Close"].pct_change()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest close", money(latest["Close"]), metric_delta(returns.iloc[-1], 2))
        c2.metric("30-day return", pct(price["Close"].pct_change(30).iloc[-1], 2))
        c3.metric("30-day volatility", pct(returns.rolling(30).std().iloc[-1] * np.sqrt(252), 1))
        if features is not None and "RSI_14" in features:
            c4.metric("RSI 14", f"{features['RSI_14'].iloc[-1]:.1f}")
        else:
            c4.metric("RSI 14", "N/A")

        chart_price = price.tail(window).copy()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=chart_price["Date"],
                y=chart_price["Close"],
                name="Close",
                mode="lines",
                line=dict(color=PALETTE["blue"], width=2),
            )
        )
        for span, color in [(20, PALETTE["teal"]), (50, PALETTE["amber"])]:
            ma = price["Close"].rolling(span).mean().tail(window)
            fig.add_trace(
                go.Scatter(
                    x=chart_price["Date"],
                    y=ma,
                    name=f"{span}D average",
                    mode="lines",
                    line=dict(color=color, width=1.5),
                )
            )
        fig.update_layout(title=f"{ticker} price trend")
        st.plotly_chart(style_plot(fig, 430), width="stretch")

        if pred is not None and not pred.empty:
            p = pred.tail(min(window, len(pred))).copy()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=p["Date"], y=p["actual"], name="Actual return", mode="lines", line=dict(color=PALETTE["blue"]))
            )
            fig.add_trace(
                go.Scatter(
                    x=p["Date"],
                    y=p["predicted"],
                    name=f"{model} prediction",
                    mode="lines",
                    line=dict(color=PALETTE["red"], width=1.7),
                )
            )
            fig.update_layout(title=f"{ticker} actual vs predicted next-day returns")
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(style_plot(fig, 390), width="stretch")

            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction rows", f"{len(pred):,}")
            c2.metric("Directional accuracy", pct(pred["Direction Match"].mean(), 1))
            c3.metric("Mean predicted return", pct(pred["predicted"].mean(), 3))
        else:
            st.info(f"No cached prediction file found for {ticker} using {model}.")


with tabs[2]:
    if comparison.empty:
        st.warning("No model comparison table found.")
    else:
        c1, c2, c3 = st.columns(3)
        winner = comparison.iloc[0]
        c1.metric("RMSE winner", winner["Model"], f"{winner['RMSE']:.5f}")
        dir_winner = comparison.sort_values("Directional_Accuracy", ascending=False).iloc[0]
        c2.metric("Best direction", dir_winner["Model"], f"{dir_winner['Directional_Accuracy']:.1f}%")
        c3.metric("Models compared", f"{len(comparison)}", "4 ML + 4 DL")

        left, right = st.columns(2)
        with left:
            fig = px.bar(comparison, x="Model", y="RMSE", color="Family", title="Lower RMSE is better")
            st.plotly_chart(style_plot(fig, 370), width="stretch")
        with right:
            fig = px.bar(
                comparison,
                x="Model",
                y="Directional_Accuracy",
                color="Family",
                title="Directional accuracy",
            )
            fig.add_hline(y=50, line_dash="dash", line_color=PALETTE["muted"])
            st.plotly_chart(style_plot(fig, 370), width="stretch")

        st.dataframe(
            comparison.style.format(
                {
                    "RMSE": "{:.5f}",
                    "MAE": "{:.5f}",
                    "MAPE": "{:.1f}",
                    "R2": "{:.4f}",
                    "Directional_Accuracy": "{:.1f}%",
                }
            ),
            width="stretch",
            hide_index=True,
        )


with tabs[3]:
    if backtest.empty:
        st.warning("Portfolio report could not be parsed.")
    else:
        chosen = backtest[backtest["Strategy"].str.contains("Sharpe-Max", na=False)].iloc[0]
        equal = backtest[backtest["Strategy"] == "Equal-Weight"].iloc[0]
        spx = backtest[backtest["Strategy"] == "SP500 Buy-and-Hold"].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe-Max return", pct(chosen["Total Return"]), metric_delta(chosen["Total Return"] - equal["Total Return"]))
        c2.metric("Sharpe-Max Sharpe", f"{chosen['Sharpe']:.2f}", f"EW {equal['Sharpe']:.2f}")
        c3.metric("SP500 return", pct(spx["Total Return"]))
        c4.metric("Max drawdown", pct(chosen["Max Drawdown"], 1))

        st.markdown(
            '<div class="callout"><b>Honest read:</b> Sharpe-Max narrowly beat equal-weight on total return, '
            'but equal-weight had the better out-of-sample Sharpe ratio.</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns([1, 1])
        with left:
            if not weights.empty:
                fig = px.bar(
                    weights.sort_values("Weight", ascending=False),
                    x="Ticker",
                    y="Weight",
                    color="Weight",
                    color_continuous_scale=["#CBD5E1", "#2563EB"],
                    title="Sharpe-max allocation",
                )
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(style_plot(fig, 390), width="stretch")
        with right:
            fig = px.bar(
                backtest,
                x="Strategy",
                y="Sharpe",
                color="Strategy",
                title="Out-of-sample Sharpe by strategy",
            )
            st.plotly_chart(style_plot(fig, 390), width="stretch")

        st.dataframe(
            backtest.style.format(
                {
                    "Total Return": "{:.2%}",
                    "Ann. Vol": "{:.2%}",
                    "Sharpe": "{:.4f}",
                    "Sortino": "{:.4f}",
                    "Beta": "{:.4f}",
                    "Max Drawdown": "{:.2%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        image_cols = st.columns(2)
        for col, image, caption in [
            (image_cols[0], REPORT_DIR / "efficient_frontier.png", "Efficient frontier"),
            (image_cols[1], REPORT_DIR / "portfolio_backtest.png", "Backtest curve"),
        ]:
            if image.exists():
                col.image(str(image), caption=caption, width="stretch")


with tabs[4]:
    features = feature_df(ticker)
    ablation = sentiment_ablation()
    if features is None or features.empty:
        st.warning(f"No feature data found for {ticker}.")
    else:
        sentiment_cols = [c for c in ["VADER_Mean", "FinBERT_Sentiment_Score"] if c in features.columns]
        news = features[features[sentiment_cols].abs().sum(axis=1) != 0].copy() if sentiment_cols else pd.DataFrame()

        c1, c2, c3 = st.columns(3)
        c1.metric("News days", f"{len(news):,}")
        c2.metric("Avg FinBERT", f"{features['FinBERT_Sentiment_Score'].tail(60).mean():.3f}")
        c3.metric("Avg VADER", f"{features['VADER_Mean'].tail(60).mean():.3f}")

        if not news.empty:
            plot_df = news.tail(window)[["Date"] + sentiment_cols].melt("Date", var_name="Scorer", value_name="Score")
            fig = px.line(plot_df, x="Date", y="Score", color="Scorer", title=f"{ticker} sentiment timeline")
            fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["muted"])
            st.plotly_chart(style_plot(fig, 400), width="stretch")
        else:
            st.info("No non-neutral sentiment rows found for this ticker in the selected data.")

        if not ablation.empty:
            fig = px.bar(ablation, x="Config", y="RMSE", color="Config", title="Sentiment ablation: RMSE")
            st.plotly_chart(style_plot(fig, 360), width="stretch")
            st.dataframe(
                ablation.style.format({"RMSE": "{:.6f}", "MAE": "{:.6f}", "R2": "{:.4f}", "DirAcc": "{:.2f}%"}),
                width="stretch",
                hide_index=True,
            )


with tabs[5]:
    returns = returns_panel()
    if returns.empty or ticker not in returns:
        st.warning("Return panel not available.")
    else:
        r = returns[ticker].dropna()
        cum = (1 + r).cumprod()
        drawdown = cum / cum.cummax() - 1
        rolling_vol = r.rolling(30).std() * np.sqrt(252)
        var_95 = np.percentile(r, 5)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("95% daily VaR", pct(var_95, 2))
        c2.metric("Worst drawdown", pct(drawdown.min(), 1))
        c3.metric("Current 30D vol", pct(rolling_vol.dropna().iloc[-1], 1))
        c4.metric("Return skew", f"{r.skew():.2f}")

        left, right = st.columns(2)
        with left:
            dd = drawdown.tail(window).reset_index()
            dd.columns = ["Date", "Drawdown"]
            fig = px.area(dd, x="Date", y="Drawdown", title=f"{ticker} drawdown")
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(style_plot(fig, 380), width="stretch")
        with right:
            vol = rolling_vol.tail(window).reset_index()
            vol.columns = ["Date", "Annualized volatility"]
            fig = px.line(vol, x="Date", y="Annualized volatility", title=f"{ticker} 30-day volatility")
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(style_plot(fig, 380), width="stretch")

        left, right = st.columns(2)
        with left:
            fig = px.histogram(r, nbins=90, title=f"{ticker} daily return distribution")
            fig.add_vline(x=var_95, line_dash="dash", line_color=PALETTE["red"])
            fig.update_xaxes(tickformat=".1%")
            st.plotly_chart(style_plot(fig, 380), width="stretch")
        with right:
            corr = returns.corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu_r",
                title="Universe return correlation",
            )
            st.plotly_chart(style_plot(fig, 380), width="stretch")


with tabs[6]:
    if rec_df.empty:
        st.warning("Recommendation report could not be parsed.")
    else:
        selected_rec = rec_df[rec_df["Ticker"] == ticker]
        if not selected_rec.empty:
            row = selected_rec.iloc[0]
            st.markdown(
                f'<div class="callout"><b>{ticker}</b> current recommendation: '
                f'{recommendation_badge(row["Recommendation"])} '
                f'with composite score <b>{row["Score"]:.3f}</b>.</div>',
                unsafe_allow_html=True,
            )

        c1, c2, c3 = st.columns(3)
        counts = rec_df["Recommendation"].value_counts()
        c1.metric("BUY", int(counts.get("BUY", 0)))
        c2.metric("HOLD", int(counts.get("HOLD", 0)))
        c3.metric("SELL", int(counts.get("SELL", 0)))

        fig = px.bar(
            rec_df.sort_values("Score", ascending=False),
            x="Ticker",
            y="Score",
            color="Recommendation",
            color_discrete_map=REC_COLORS,
            title="Composite recommendation score",
        )
        fig.add_hline(y=0.30, line_dash="dash", line_color=PALETTE["green"])
        fig.add_hline(y=-0.30, line_dash="dash", line_color=PALETTE["red"])
        st.plotly_chart(style_plot(fig, 420), width="stretch")

        st.dataframe(
            rec_df.style.format(
                {
                    "Score": "{:.3f}",
                    "Target Weight": "{:.2%}",
                    "Sharpe": "{:.2f}",
                    "Reliability": "{:.1%}",
                    "Sentiment": "{:.3f}",
                    "Momentum": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        if not rebalance_df.empty:
            st.markdown('<div class="section-label">Rebalancing plan</div>', unsafe_allow_html=True)
            st.dataframe(
                rebalance_df.style.format({"Current": "{:.2%}", "Target": "{:.2%}", "Drift": "{:.2%}"}),
                width="stretch",
                hide_index=True,
            )


with tabs[7]:
    st.markdown('<div class="section-label">Project report</div>', unsafe_allow_html=True)
    final_report = REPORT_DIR / "final_report.md"
    if final_report.exists():
        st.download_button(
            "Download final report",
            data=final_report.read_bytes(),
            file_name="final_report.md",
            mime="text/markdown",
        )
        with st.expander("Preview final report", expanded=False):
            st.markdown(read_text(final_report))
    else:
        st.info("final_report.md was not found.")

    st.markdown('<div class="section-label">Generated figures</div>', unsafe_allow_html=True)
    figures = [
        REPORT_DIR / "recommendations_scores.png",
        REPORT_DIR / "math_validation.png",
        REPORT_DIR / "LSTM_loss_curves.png",
        REPORT_DIR / "GRU_loss_curves.png",
        REPORT_DIR / "BiLSTM_loss_curves.png",
        REPORT_DIR / "Transformer_loss_curves.png",
    ]
    cols = st.columns(2)
    for idx, image in enumerate([p for p in figures if p.exists()]):
        cols[idx % 2].image(str(image), caption=image.stem.replace("_", " ").title(), width="stretch")

    eda_images = sorted(EDA_DIR.glob("*.png")) if EDA_DIR.exists() else []
    if eda_images:
        st.markdown('<div class="section-label">EDA gallery</div>', unsafe_allow_html=True)
        choice = st.selectbox("EDA figure", [p.name for p in eda_images])
        st.image(str(EDA_DIR / choice), width="stretch")
