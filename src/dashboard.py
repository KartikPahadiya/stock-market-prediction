"""
Streamlit Dashboard — Northgate Quantitative Research Capstone
PRD Section 13: Fully interactive, Plotly-based, seven-panel dashboard.
"""
from pathlib import Path
import re
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================================
# Paths
# ==========================================================
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
PREDICTIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "predictions"
FEATURES_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "features"
STOCKS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "stocks"
EDA_DIR = REPORT_DIR / "eda"

# ==========================================================
# Page config
# ==========================================================
st.set_page_config(
    page_title="Stock AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Sidebar — Global controls & persistent disclaimer
# ==========================================================
st.sidebar.title("⚙️ Controls")

TICKERS = ["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "META", "MSFT", "NVDA", "PG", "XOM"]
selected_ticker = st.sidebar.selectbox("Select Ticker", TICKERS, index=0)

horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    ["1-day", "5-day", "20-day"],
    index=0,
    help="Only 1-day horizon is actively modeled. 5-day and 20-day are UI placeholders."
)

risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate",
    min_value=0.0,
    max_value=0.10,
    value=0.02,
    step=0.005,
    format="%.3f",
    help="Annual risk-free rate used to dynamically recalculate Sharpe ratios."
)

st.sidebar.markdown("---")
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "💰 Price & Prediction",
        "🤖 Model Comparison",
        "💼 Portfolio Analytics",
        "⚠️ Risk Dashboard",
        "😊 Sentiment",
        "🚦 Recommendations",
        "🔍 EDA Gallery",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size:0.8rem; color:#666;">
    ⚠️ <strong>Educational, not financial advice.</strong><br>
    This dashboard is a student capstone project for analytical demonstration only.
    Past performance does not guarantee future results. Do not use for actual trading.
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("**Northgate Quantitative Research**  \nCapstone Project 2026")


# ==========================================================
# Data loaders (no caching — always read fresh)
# ==========================================================
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None


def load_feature_df(ticker):
    path = FEATURES_DIR / f"{ticker}.csv"
    df = load_csv(path)
    if df is not None:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_stock_df(ticker):
    path = STOCKS_DIR / f"{ticker}.csv"
    df = load_csv(path)
    if df is not None:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_predictions(ticker, model):
    """Reconstruct per-ticker predictions with dates from pooled predictions + feature file."""
    ticker_id = TICKERS.index(ticker) if ticker in TICKERS else None
    if ticker_id is None:
        return None

    pooled_path = PREDICTIONS_DIR / f"pooled_{model}_predictions.csv"
    pooled = load_csv(pooled_path)
    if pooled is None:
        return None

    ticker_mask = pooled["ticker_id"] == ticker_id
    ticker_preds = pooled[ticker_mask].copy()
    if ticker_preds.empty:
        return None

    feat_df = load_feature_df(ticker)
    if feat_df is None or feat_df.empty:
        return None

    n_test = len(ticker_preds)
    test_dates = feat_df["Date"].iloc[-n_test:].values

    result = pd.DataFrame({
        "Date": pd.to_datetime(test_dates),
        "Actual": ticker_preds["actual"].values,
        "Prediction": ticker_preds["predicted"].values
    })
    return result


def compute_universe_metrics():
    """Aggregate metrics across all tickers for the Overview panel."""
    records = []
    for t in TICKERS:
        df = load_stock_df(t)
        if df is not None and not df.empty:
            records.append({
                "Ticker": t,
                "Latest_Close": df["Close"].iloc[-1],
                "Latest_Date": df["Date"].iloc[-1],
                "Trading_Days": len(df),
            })
    return pd.DataFrame(records) if records else None


def compute_market_mood():
    """Average FinBERT sentiment across all tickers for last 30 days with news."""
    sentiments = []
    for t in TICKERS:
        df = load_feature_df(t)
        if df is None or df.empty:
            continue
        mask = (df["FinBERT_Has_News"] == 1) | (df["FinBERT_Sentiment_Score"].notna() & (df["FinBERT_Sentiment_Score"] != 0))
        news_df = df[mask].copy()
        if news_df.empty:
            continue
        last_30 = news_df.tail(30)
        sentiments.extend(last_30["FinBERT_Sentiment_Score"].dropna().tolist())

    if not sentiments:
        return None, 0
    avg = np.mean(sentiments)
    n = len(sentiments)
    return avg, n


def compute_risk_metrics(ticker):
    """Compute drawdown, rolling vol, VaR for a ticker."""
    df = load_feature_df(ticker)
    if df is None or df.empty:
        return None
    df = df.sort_values("Date").copy()
    if "Daily_Return" not in df.columns:
        stock = load_stock_df(ticker)
        if stock is not None:
            stock = stock.sort_values("Date").copy()
            stock["Daily_Return"] = stock["Close"].pct_change()
            df = df.merge(stock[["Date", "Daily_Return"]], on="Date", how="left")
    returns = df["Daily_Return"].dropna()
    if len(returns) < 30:
        return None

    cumret = (1 + returns).cumprod()
    running_max = cumret.cummax()
    drawdown = (cumret - running_max) / running_max
    rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
    var_5 = np.percentile(returns, 5)
    corr_sp500 = None
    if "SP500_Return" in df.columns:
        corr_sp500 = returns.corr(df["SP500_Return"].dropna().reindex(returns.index))

    return {
        "dates": df["Date"].values,
        "returns": returns.values,
        "cumret": cumret.values,
        "drawdown": drawdown.values,
        "rolling_vol": rolling_vol.values,
        "var_5": var_5,
        "corr_sp500": corr_sp500,
    }


def compute_correlation_matrix():
    """Daily returns correlation matrix across all tickers."""
    returns_dict = {}
    for t in TICKERS:
        df = load_stock_df(t)
        if df is not None and not df.empty:
            df = df.sort_values("Date").copy()
            df["Daily_Return"] = df["Close"].pct_change()
            returns_dict[t] = df.set_index("Date")["Daily_Return"]
    if not returns_dict:
        return None
    rets_df = pd.DataFrame(returns_dict)
    return rets_df.corr()


def parse_portfolio_report():
    """Parse portfolio_opt_report.txt for weights and metrics."""
    path = REPORT_DIR / "portfolio_opt_report.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")

    weights = {}
    in_sharpe = False
    for line in text.splitlines():
        if "Sharpe-Maximizing Portfolio" in line:
            in_sharpe = True
            continue
        if in_sharpe and line.strip().startswith("["):
            break
        if in_sharpe and ":" in line and any(t in line for t in TICKERS):
            m = re.search(r"([A-Z]+)\s*:\s*([0-9.]+)", line)
            if m:
                weights[m.group(1)] = float(m.group(2))

    metrics = {}
    m = re.search(r"Sharpe Ratio \(in-sample\):\s+([0-9.]+)", text)
    if m:
        metrics["sharpe_insample"] = float(m.group(1))
    m = re.search(r"Expected Return \(in-sample\):\s+([0-9.]+)", text)
    if m:
        metrics["return_insample"] = float(m.group(1))
    m = re.search(r"Volatility \(in-sample\):\s+([0-9.]+)", text)
    if m:
        metrics["vol_insample"] = float(m.group(1))
    m = re.search(r"Sharpe Ratio:\s+([0-9.]+)", text)
    if m:
        metrics["sharpe_oos"] = float(m.group(1))
    m = re.search(r"Sortino Ratio:\s+([0-9.]+)", text)
    if m:
        metrics["sortino_oos"] = float(m.group(1))
    m = re.search(r"Annualized Volatility:\s+([0-9.]+)", text)
    if m:
        metrics["vol_oos"] = float(m.group(1))
    m = re.search(r"Beta \(vs SP500\):\s+([0-9.]+)", text)
    if m:
        metrics["beta_oos"] = float(m.group(1))
    m = re.search(r"Maximum Drawdown:\s+(-?[0-9.]+%?)", text)
    if m:
        metrics["maxdd_oos"] = m.group(1)

    return {"weights": weights, "metrics": metrics}


def parse_recommendations():
    """Parse recommendations_report.txt into structured tables."""
    path = REPORT_DIR / "recommendations_report.txt"
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8")

    rec_rows = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = re.match(
            r"^([A-Z]+)\s+(BUY|HOLD|SELL)\s+(-?[0-9.]+)\s+([0-9.]+%?)\s+(-?[0-9.]+)\s+([0-9.]+%?)\s+(-?[0-9.]+)\s+(-?[0-9.]+)",
            line
        )
        if m:
            rec_rows.append({
                "Ticker": m.group(1),
                "Rec": m.group(2),
                "Score": float(m.group(3)),
                "SharpeW": m.group(4),
                "Sharpe": float(m.group(5)),
                "Rel": m.group(6),
                "Sent": float(m.group(7)),
                "Mom": float(m.group(8)),
            })

    reb_rows = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = re.match(
            r"^([A-Z]+)\s+([0-9.]+%?)\s+([0-9.]+%?)\s+([0-9.]+%?)\s+(REDUCE|HOLD|INCREASE)",
            line
        )
        if m:
            reb_rows.append({
                "Ticker": m.group(1),
                "Current": m.group(2),
                "Target": m.group(3),
                "Drift": m.group(4),
                "Action": m.group(5),
            })

    return pd.DataFrame(rec_rows), pd.DataFrame(reb_rows)


# ==========================================================
# Helper: build price + prediction chart
# ==========================================================
def build_price_prediction_chart(ticker, model, rmse):
    stock_df = load_stock_df(ticker)
    pred_df = load_predictions(ticker, model)

    if stock_df is None or pred_df is None or pred_df.empty:
        return None

    stock_df = stock_df.sort_values("Date").copy()
    pred_df = pred_df.sort_values("Date").copy()

    merged = pd.merge(stock_df[["Date", "Close"]], pred_df[["Date", "Actual", "Prediction"]], on="Date", how="outer").sort_values("Date")

    pred_mask = merged["Prediction"].notna()
    if not pred_mask.any():
        return None

    first_pred_idx = merged[pred_mask].index[0]
    first_close = merged.loc[first_pred_idx, "Close"]
    if pd.isna(first_close):
        first_close = merged["Close"].iloc[:first_pred_idx+1].dropna().iloc[-1] if first_pred_idx > 0 else None
    if first_close is None or pd.isna(first_close):
        return None

    pred_rows = merged[pred_mask].copy()
    pred_prices = [first_close]
    for i in range(1, len(pred_rows)):
        pred_ret = pred_rows["Prediction"].iloc[i]
        if pd.isna(pred_ret):
            pred_prices.append(None)
        else:
            pred_prices.append(pred_prices[-1] * np.exp(pred_ret) if pred_prices[-1] is not None else None)

    pred_rows["Pred_Price"] = pred_prices

    upper_prices = [first_close]
    lower_prices = [first_close]
    for i in range(1, len(pred_rows)):
        pred_ret = pred_rows["Prediction"].iloc[i]
        if pd.isna(pred_ret) or pred_prices[i] is None:
            upper_prices.append(None)
            lower_prices.append(None)
        else:
            upper_prices.append(pred_prices[i] * np.exp(rmse))
            lower_prices.append(pred_prices[i] * np.exp(-rmse))

    pred_rows["Upper_Price"] = upper_prices
    pred_rows["Lower_Price"] = lower_prices

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stock_df["Date"], y=stock_df["Close"],
        mode="lines", name="Actual Close",
        line=dict(color="steelblue", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Close: $%{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=pred_rows["Date"], y=pred_rows["Pred_Price"],
        mode="lines", name=f"Predicted ({model})",
        line=dict(color="orangered", width=2, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>Predicted: $%{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=pred_rows["Date"], y=pred_rows["Upper_Price"],
        mode="lines", line=dict(width=0), showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=pred_rows["Date"], y=pred_rows["Lower_Price"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(255, 69, 0, 0.15)", name="±1 RMSE Band",
        hoverinfo="skip"
    ))

    fig.update_layout(
        title=f"{ticker} — Price History & {model} Forecast Overlay",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500,
    )
    return fig


# ==========================================================
# Page: Overview
# ==========================================================
if page == "🏠 Overview":
    st.title("Northgate Quantitative Research — Stock Prediction Dashboard")
    st.markdown("""
    **Capstone Project**: 10-ticker US equity prediction using ML & DL models  
    **Universe**: AAPL, AMZN, GOOGL, JNJ, JPM, META, MSFT, NVDA, PG, XOM  
    **Period**: 2015–2026 | **Target**: 1-day ahead log return
    """)

    comp = load_csv(REPORT_DIR / "model_comparison.csv")
    universe = compute_universe_metrics()

    col1, col2, col3, col4 = st.columns(4)

    if comp is not None and not comp.empty:
        best = comp.iloc[0]
        col1.metric("Best Model", best["Model"], f"RMSE {best['RMSE']:.4f}")
        col2.metric("Directional Accuracy", f"{best['Directional_Accuracy']:.1f}%")
        col3.metric("R² (explained variance)", f"{best['R2']:.4f}")
    else:
        col1.info("Run comparison_table.py to generate results")

    mood, mood_n = compute_market_mood()
    if mood is not None:
        mood_label = "Bullish 🟢" if mood > 0.1 else ("Bearish 🔴" if mood < -0.1 else "Neutral 🟡")
        col4.metric("Market Mood (FinBERT)", mood_label, f"{mood:.3f} ({mood_n} obs)")
    else:
        col4.info("No sentiment data")

    st.divider()

    if universe is not None and not universe.empty:
        st.subheader("Universe Snapshot")
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(
                universe, x="Ticker", y="Latest_Close",
                title="Latest Closing Prices",
                labels={"Latest_Close": "Close ($)"},
                template="plotly_white", height=350
            )
            fig.update_traces(marker_color="steelblue")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(
                universe[["Ticker", "Latest_Close", "Trading_Days"]]
                .style.format({"Latest_Close": "${:.2f}"}),
                use_container_width=True, hide_index=True
            )

    st.markdown("""
    **Key Insight**: RandomForest wins across headline metrics, yet all R² values hover near zero —
    consistent with weak-form market efficiency and the near-impossibility of predicting daily returns.
    """)


# ==========================================================
# Page: Price & Prediction
# ==========================================================
elif page == "💰 Price & Prediction":
    st.title("Price History & Forecast Overlay")

    st.info(f"Horizon selected: **{horizon}**. Note: Only 1-day ahead predictions are actively modeled; longer horizons are UI placeholders.")

    available_models = ["RandomForest", "Ridge", "XGBoost", "SVR"]
    model_sel = st.selectbox("Select Model for Forecast", available_models, index=0)

    comp = load_csv(REPORT_DIR / "model_comparison.csv")
    rmse = 0.015  # default fallback
    if comp is not None:
        row = comp[comp["Model"] == model_sel]
        if not row.empty:
            rmse = float(row.iloc[0]["RMSE"])

    fig = build_price_prediction_chart(selected_ticker, model_sel, rmse)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Prediction data not available for {selected_ticker} + {model_sel}.")

    pred_df = load_predictions(selected_ticker, model_sel)
    if pred_df is not None and not pred_df.empty:
        st.subheader("Predicted vs Actual Returns (scatter)")
        fig2 = px.scatter(
            pred_df, x="Actual", y="Prediction",
            opacity=0.5, title=f"{selected_ticker} — {model_sel}",
            labels={"Actual": "Actual Return", "Prediction": "Predicted Return"},
            template="plotly_white", height=400
        )
        min_val = min(pred_df["Actual"].min(), pred_df["Prediction"].min())
        max_val = max(pred_df["Actual"].max(), pred_df["Prediction"].max())
        fig2.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                       line=dict(color="red", dash="dash"), name="Perfect Prediction")
        st.plotly_chart(fig2, use_container_width=True)


# ==========================================================
# Page: Model Comparison
# ==========================================================
elif page == "🤖 Model Comparison":
    st.title("Model Comparison Deep-Dive")

    comp = load_csv(REPORT_DIR / "model_comparison.csv")
    if comp is None:
        st.warning("model_comparison.csv not found.")
        st.stop()

    tab1, tab2 = st.tabs(["📊 Metrics Table", "📈 Charts"])

    with tab1:
        st.dataframe(
            comp.style.format({
                "RMSE": "{:.5f}", "MAE": "{:.5f}", "MAPE": "{:.1f}",
                "R2": "{:.4f}", "Directional_Accuracy": "{:.1f}%"
            }),
            use_container_width=True, hide_index=True
        )

    with tab2:
        c1, c2 = st.columns(2)

        fig_rmse = px.bar(
            comp, x="Model", y="RMSE", color="Model",
            title="RMSE by Model", template="plotly_white", height=350
        )
        fig_rmse.update_layout(showlegend=False)
        c1.plotly_chart(fig_rmse, use_container_width=True)

        fig_mae = px.bar(
            comp, x="Model", y="MAE", color="Model",
            title="MAE by Model", template="plotly_white", height=350
        )
        fig_mae.update_layout(showlegend=False)
        c2.plotly_chart(fig_mae, use_container_width=True)

        fig_r2 = px.bar(
            comp, x="Model", y="R2", color="Model",
            title="R² by Model", template="plotly_white", height=350
        )
        fig_r2.add_hline(y=0, line_dash="dash", line_color="black")
        fig_r2.update_layout(showlegend=False)
        c1.plotly_chart(fig_r2, use_container_width=True)

        fig_da = px.bar(
            comp, x="Model", y="Directional_Accuracy", color="Model",
            title="Directional Accuracy by Model", template="plotly_white", height=350
        )
        fig_da.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Coin-flip (50%)")
        fig_da.update_layout(showlegend=False)
        c2.plotly_chart(fig_da, use_container_width=True)

        st.markdown("**Observation**: All R² ≈ 0. No model explains more than ~3 % of variance. Daily returns are dominated by noise.")


# ==========================================================
# Page: Portfolio Analytics
# ==========================================================
elif page == "💼 Portfolio Analytics":
    st.title("Portfolio Optimization & Analytics")

    parsed = parse_portfolio_report()

    col1, col2 = st.columns([3, 2])

    with col1:
        ef_path = REPORT_DIR / "efficient_frontier.png"
        if ef_path.exists():
            st.image(str(ef_path), caption="Efficient Frontier (Monte Carlo + Optimized)", use_container_width=True)
        else:
            st.info("efficient_frontier.png not found")

        bt_path = REPORT_DIR / "portfolio_backtest.png"
        if bt_path.exists():
            st.image(str(bt_path), caption="Portfolio Backtest (Jan 2025 – Aug 2026)", use_container_width=True)
        else:
            st.info("portfolio_backtest.png not found")

    with col2:
        st.subheader("Sharpe-Max Weights")
        if parsed and parsed["weights"]:
            wdf = pd.DataFrame([
                {"Ticker": k, "Weight": v}
                for k, v in parsed["weights"].items()
            ])
            fig_pie = px.pie(
                wdf, names="Ticker", values="Weight",
                title="Optimized Portfolio Allocation",
                template="plotly_white", height=400
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

            st.dataframe(
                wdf.style.format({"Weight": "{:.2%}"}),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("Portfolio weights not parsed from report.")

        st.subheader("Key Metrics")
        if parsed and parsed["metrics"]:
            m = parsed["metrics"]
            ret = m.get("return_insample", 0)
            vol = m.get("vol_insample", 1)
            dyn_sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0

            k1, k2, k3 = st.columns(3)
            k1.metric("Sharpe (in-sample)", f"{dyn_sharpe:.3f}", delta=f"rf={risk_free_rate:.1%}")
            k2.metric("Sortino (OOS)", f"{m.get('sortino_oos', 'N/A')}")
            k3.metric("Ann. Vol (OOS)", f"{m.get('vol_oos', 'N/A')}")
            k4, k5, k6 = st.columns(3)
            k4.metric("Beta vs SP500", f"{m.get('beta_oos', 'N/A')}")
            k5.metric("Max Drawdown", f"{m.get('maxdd_oos', 'N/A')}")
            k6.metric("Exp. Return (in-sample)", f"{ret:.2%}")
        else:
            st.info("Portfolio metrics not parsed.")

    st.divider()
    st.subheader("Portfolio Report (raw)")
    port_report = REPORT_DIR / "portfolio_opt_report.txt"
    if port_report.exists():
        with open(port_report, "r", encoding="utf-8") as f:
            st.text(f.read())
    else:
        st.info("portfolio_opt_report.txt not found")


# ==========================================================
# Page: Risk Dashboard
# ==========================================================
elif page == "⚠️ Risk Dashboard":
    st.title("Risk Dashboard")

    risk_data = compute_risk_metrics(selected_ticker)
    if risk_data is None:
        st.warning(f"Insufficient data for {selected_ticker} risk analysis.")
        st.stop()

    st.subheader("Drawdown Curve")
    dd_df = pd.DataFrame({
        "Date": risk_data["dates"][:len(risk_data["drawdown"])],
        "Drawdown": risk_data["drawdown"]
    })
    fig_dd = px.area(
        dd_df, x="Date", y="Drawdown",
        title=f"{selected_ticker} Cumulative Drawdown",
        labels={"Drawdown": "Drawdown (%)"},
        template="plotly_white", height=350
    )
    fig_dd.update_traces(line_color="firebrick", fillcolor="rgba(178,34,34,0.3)")
    fig_dd.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Rolling Volatility (30-day)")
    vol_df = pd.DataFrame({
        "Date": risk_data["dates"][:len(risk_data["rolling_vol"])],
        "AnnVol": risk_data["rolling_vol"]
    }).dropna()
    fig_vol = px.line(
        vol_df, x="Date", y="AnnVol",
        title=f"{selected_ticker} 30-Day Rolling Annualized Volatility",
        labels={"AnnVol": "Annualized Volatility"},
        template="plotly_white", height=350
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Cross-Ticker Correlation Heatmap")
    corr = compute_correlation_matrix()
    if corr is not None:
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            title="Daily Returns Correlation Matrix",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            height=500, template="plotly_white"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Could not compute correlation matrix.")

    st.subheader("Value at Risk (Historical Simulation)")
    var_val = risk_data["var_5"]
    st.metric(
        label=f"{selected_ticker} 1-day 95% VaR",
        value=f"{var_val:.2%}",
        help="5th percentile of historical daily returns. Interpretation: on 5% of days, losses exceeded this threshold."
    )

    ret_df = pd.DataFrame({"Daily Return": risk_data["returns"]})
    fig_hist = px.histogram(
        ret_df, x="Daily Return", nbins=100,
        title=f"{selected_ticker} Daily Return Distribution",
        template="plotly_white", height=350,
        marginal="box"
    )
    fig_hist.add_vline(x=var_val, line_dash="dash", line_color="red",
                       annotation_text=f"VaR (5%) = {var_val:.2%}")
    fig_hist.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig_hist, use_container_width=True)


# ==========================================================
# Page: Sentiment
# ==========================================================
elif page == "😊 Sentiment":
    st.title("Sentiment Analysis — VADER & FinBERT")

    df = load_feature_df(selected_ticker)
    if df is None or df.empty:
        st.warning(f"Feature data not found for {selected_ticker}.")
        st.stop()

    st.subheader(f"{selected_ticker} Sentiment Timeline")

    news_mask = (df["Has_News"] == 1) | (df["FinBERT_Has_News"] == 1) | (df["VADER_Mean"].notna() & (df["VADER_Mean"] != 0)) | (df["FinBERT_Sentiment_Score"].notna() & (df["FinBERT_Sentiment_Score"] != 0))
    news_df = df[news_mask].copy()

    if news_df.empty:
        st.info("No news-sentiment records for this ticker.")
    else:
        fig_sent = go.Figure()

        if "VADER_Mean" in news_df.columns:
            fig_sent.add_trace(go.Scatter(
                x=news_df["Date"], y=news_df["VADER_Mean"],
                mode="lines+markers", name="VADER",
                line=dict(color="steelblue"), marker=dict(size=4),
                hovertemplate="%{x|%Y-%m-%d}<br>VADER: %{y:.3f}<extra></extra>"
            ))

        if "FinBERT_Sentiment_Score" in news_df.columns:
            fig_sent.add_trace(go.Scatter(
                x=news_df["Date"], y=news_df["FinBERT_Sentiment_Score"],
                mode="lines+markers", name="FinBERT",
                line=dict(color="forestgreen"), marker=dict(size=4),
                hovertemplate="%{x|%Y-%m-%d}<br>FinBERT: %{y:.3f}<extra></extra>"
            ))

        fig_sent.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_sent.update_layout(
            title=f"{selected_ticker} — News Sentiment Over Time",
            xaxis_title="Date", yaxis_title="Sentiment Score",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white", height=450
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    st.subheader("Market Mood Gauge")
    mood, mood_n = compute_market_mood()
    if mood is not None:
        mood_color = "green" if mood > 0.1 else ("red" if mood < -0.1 else "gray")
        mood_text = "Bullish" if mood > 0.1 else ("Bearish" if mood < -0.1 else "Neutral")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=mood,
            title={"text": f"Avg FinBERT Sentiment<br><span style='font-size:0.8em;color:{mood_color}'>{mood_text}</span>"},
            delta={"reference": 0, "valueformat": ".3f"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": mood_color},
                "steps": [
                    {"range": [-1, -0.33], "color": "rgba(255,0,0,0.2)"},
                    {"range": [-0.33, 0.33], "color": "rgba(128,128,128,0.2)"},
                    {"range": [0.33, 1], "color": "rgba(0,128,0,0.2)"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.75, "value": 0}
            }
        ))
        fig_gauge.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption(f"Based on last 30 days with news across all {len(TICKERS)} tickers ({mood_n} observations).")
    else:
        st.info("Market mood cannot be computed — no sentiment data available.")

    # FinBERT vs VADER comparison report (generated live from data)
    st.subheader("FinBERT vs VADER Report")
    try:
        vader_data = []
        finbert_data = []
        for t in TICKERS:
            df = load_feature_df(t)
            if df is not None and not df.empty:
                if "VADER_Mean" in df.columns:
                    vader_data.extend(df["VADER_Mean"][df["Has_News"] == 1].dropna().tolist())
                if "FinBERT_Sentiment_Score" in df.columns:
                    finbert_data.extend(df["FinBERT_Sentiment_Score"][df["FinBERT_Has_News"] == 1].dropna().tolist())

        if vader_data and finbert_data:
            vader_arr = np.array(vader_data)
            finbert_arr = np.array(finbert_data)

            report_text = f"""FinBERT vs VADER Comparison (Live from Feature Data)
{'=' * 60}
VADER (lexicon-based):
  Articles scored: {len(vader_arr)}
  Mean: {vader_arr.mean():.4f}
  Std:  {vader_arr.std():.4f}
  Positive (>0.05): {(vader_arr > 0.05).sum()} ({(vader_arr > 0.05).mean()*100:.1f}%)
  Negative (<-0.05): {(vader_arr < -0.05).sum()} ({(vader_arr < -0.05).mean()*100:.1f}%)

FinBERT (domain-tuned transformer):
  Articles scored: {len(finbert_arr)}
  Mean: {finbert_arr.mean():.4f}
  Std:  {finbert_arr.std():.4f}
  Positive (>0): {(finbert_arr > 0).sum()} ({(finbert_arr > 0).mean()*100:.1f}%)
  Negative (<0): {(finbert_arr < 0).sum()} ({(finbert_arr < 0).mean()*100:.1f}%)

Key Insight:
  Both scorers process identical headlines from Finnhub.
  Their distributions show genuinely different signals:
  VADER is a rule-based lexicon; FinBERT is a fine-tuned transformer.
  Neither materially improves predictive accuracy on daily returns.
"""
            st.text(report_text)
        else:
            st.info("Sentiment data not yet available in feature files.")
    except Exception as e:
        st.info(f"Could not generate comparison report: {e}")


# ==========================================================
# Page: Recommendations
# ==========================================================
elif page == "🚦 Recommendations":
    st.title("Trading Signals & Recommendations")

    rec_df, reb_df = parse_recommendations()

    if rec_df is not None and not rec_df.empty:
        st.subheader("BUY / HOLD / SELL Table")

        def color_rec(val):
            color = "#90EE90" if val == "BUY" else ("#FFB6C1" if val == "SELL" else "#FFFFE0")
            return f"background-color: {color}"

        styled = rec_df.style.map(color_rec, subset=["Rec"]).format({
            "Score": "{:.3f}", "Sharpe": "{:.2f}", "Sent": "{:.3f}", "Mom": "{:.3f}"
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)

        fig_scores = px.bar(
            rec_df, x="Ticker", y="Score", color="Rec",
            color_discrete_map={"BUY": "green", "HOLD": "gold", "SELL": "crimson"},
            title="Composite Signal Scores",
            template="plotly_white", height=400
        )
        fig_scores.add_hline(y=0.30, line_dash="dash", line_color="green", annotation_text="BUY threshold")
        fig_scores.add_hline(y=-0.30, line_dash="dash", line_color="red", annotation_text="SELL threshold")
        st.plotly_chart(fig_scores, use_container_width=True)
    else:
        st.warning("Could not parse recommendation table.")

    st.divider()

    if reb_df is not None and not reb_df.empty:
        st.subheader("Portfolio Rebalancing Plan")
        st.markdown("Assumes current holdings = equal-weight (10 % each). Target = Sharpe-Max optimized weights. Band = 5 %.")

        def color_action(val):
            color = "#90EE90" if val == "INCREASE" else ("#FFB6C1" if val == "REDUCE" else "#FFFFE0")
            return f"background-color: {color}"

        styled_reb = reb_df.style.map(color_action, subset=["Action"])
        st.dataframe(styled_reb, use_container_width=True, hide_index=True)
    else:
        st.info("Rebalancing table not found in report.")

    st.divider()
    st.subheader("Recommendation Report (raw)")
    rec_report = REPORT_DIR / "recommendations_report.txt"
    if rec_report.exists():
        with open(rec_report, "r", encoding="utf-8") as f:
            st.text(f.read())
    else:
        st.info("recommendations_report.txt not found")


# ==========================================================
# Page: EDA Gallery
# ==========================================================
elif page == "🔍 EDA Gallery":
    st.title("Exploratory Data Analysis Gallery")

    eda_files = sorted(EDA_DIR.glob("*.png")) if EDA_DIR.exists() else []
    if not eda_files:
        st.warning("No EDA plots found in reports/eda/. Run eda.py first.")
        st.stop()

    selected_plot = st.selectbox("Select Plot", [f.name for f in eda_files])
    selected_path = EDA_DIR / selected_plot
    if selected_path.exists():
        st.image(str(selected_path), caption=selected_plot.replace("_", " ").replace(".png", ""), use_container_width=True)

    st.divider()
    st.subheader("All Plots")
    cols = st.columns(3)
    for i, f in enumerate(eda_files):
        with cols[i % 3]:
            st.image(str(f), caption=f.name.replace("_", " ").replace(".png", ""), use_container_width=True)
