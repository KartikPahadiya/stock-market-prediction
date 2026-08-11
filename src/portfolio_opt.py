"""
==============================================================
Portfolio Optimization
PRD Section 11 - Required Deliverable
==============================================================

Builds the efficient frontier, computes Sharpe-maximizing,
minimum-variance, and risk-parity portfolios from historical
returns, using:
  - Ledoit-Wolf shrinkage covariance (Section 11.2) to avoid the
    extreme corner solutions raw sample covariance produces
  - A per-asset weight cap (Section 11.4) to prevent concentration
  - An in-sample / out-of-sample backtest (Section 11.4) against
    equal-weight and SP500 buy-and-hold baselines
  - Full risk reporting (Sharpe, Sortino, vol, Beta, max drawdown)
    for the chosen portfolio, evaluated OUT of sample

Outputs:
  reports/portfolio_opt_report.txt
  reports/efficient_frontier.png
  reports/portfolio_backtest.png
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features"
BENCH_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.02

# Held out for backtesting, never used to estimate mu/Sigma.
# Matches the 15% test-window convention used elsewhere in this project.
BACKTEST_FRACTION = 0.15

# Section 11.4: per-asset cap to prevent concentration
MAX_WEIGHT = 0.30

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ==========================================================
# Load date-aligned returns for all tickers (+ SP500 for Beta/backtest)
# ==========================================================

def load_aligned_returns():
    """
    Load all feature CSVs and merge on Date so every row is a
    genuinely matched trading day across all tickers, plus SP500
    daily return for Beta and the buy-and-hold baseline.
    """
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

    sp500 = pd.read_csv(BENCH_DIR / "SP500.csv", parse_dates=["Date"])
    sp500["SP500_Return"] = np.log(sp500["Close"] / sp500["Close"].shift(1))
    merged = pd.merge(merged, sp500[["Date", "SP500_Return"]].dropna(), on="Date", how="inner")

    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged["Date"].values, tickers, merged[tickers].values, merged["SP500_Return"].values


def split_train_backtest(dates, returns, market_returns, backtest_fraction=BACKTEST_FRACTION):
    """Chronological split -- mu/Sigma estimated ONLY on the train window."""
    n = len(dates)
    split_idx = int(n * (1 - backtest_fraction))
    return (
        dates[:split_idx], returns[:split_idx], market_returns[:split_idx],
        dates[split_idx:], returns[split_idx:], market_returns[split_idx:]
    )


# ==========================================================
# Portfolio metrics
# ==========================================================

def portfolio_metrics(weights, expected_returns, cov_matrix):
    port_return = np.dot(weights, expected_returns)
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    sharpe = (port_return - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe


def negative_sharpe(weights, expected_returns, cov_matrix):
    _, _, s = portfolio_metrics(weights, expected_returns, cov_matrix)
    return -s


def portfolio_variance_obj(weights, cov_matrix):
    return weights @ cov_matrix @ weights


# ==========================================================
# Risk metrics for a realized daily-return series
# (same formulas as math_from_scratch.py, applied here to the
#  portfolio's own realized OUT-OF-SAMPLE return series)
# ==========================================================

def sharpe_ratio_series(returns, rf_annual=RISK_FREE_RATE, trading_days=TRADING_DAYS):
    excess = returns - (rf_annual / trading_days)
    return (np.mean(excess) / np.std(returns, ddof=1)) * np.sqrt(trading_days)


def sortino_ratio_series(returns, rf_annual=RISK_FREE_RATE, trading_days=TRADING_DAYS):
    excess = returns - (rf_annual / trading_days)
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 0 else 0
    if downside_std == 0:
        return np.inf
    return (np.mean(excess) / downside_std) * np.sqrt(trading_days)


def beta_series(port_returns, market_returns):
    cov = np.cov(port_returns, market_returns, ddof=1)[0, 1]
    var_m = np.var(market_returns, ddof=1)
    return cov / var_m if var_m > 0 else 0


def max_drawdown_series(returns):
    cum = np.exp(np.cumsum(returns))  # growth of $1
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return dd.min(), cum


# ==========================================================
# Risk Parity
# ==========================================================

def risk_parity_weights(cov_matrix, max_weight=MAX_WEIGHT):
    n = cov_matrix.shape[0]

    def risk_parity_objective(w):
        sigma_w = cov_matrix @ w
        port_var = w @ sigma_w
        if port_var <= 1e-12:
            return 1e6
        rc = w * sigma_w / port_var
        target = 1.0 / n
        return np.sum((rc - target) ** 2)

    w0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        risk_parity_objective, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000}
    )
    return result.x


# ==========================================================
# Efficient Frontier
# ==========================================================

def build_efficient_frontier(expected_returns, cov_matrix, n_points=80, max_weight=MAX_WEIGHT):
    n = len(expected_returns)
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), n_points)

    efficient_vols, efficient_rets = [], []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, target=target: np.dot(w, expected_returns) - target}
        ]
        bounds = [(0.0, max_weight)] * n
        w0 = np.ones(n) / n

        result = minimize(
            portfolio_variance_obj, w0, args=(cov_matrix,),
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-12}
        )

        # Capped weights make some extreme target returns infeasible --
        # skip those rather than plotting a failed/garbage solve.
        if result.success:
            r, v, _ = portfolio_metrics(result.x, expected_returns, cov_matrix)
            efficient_rets.append(r)
            efficient_vols.append(v)

    return np.array(efficient_rets), np.array(efficient_vols)


def generate_random_portfolios(expected_returns, cov_matrix, n_portfolios=20000):
    """
    Section 11.3: 20,000 random long-only portfolios, uncapped, as an
    independent check that the analytical frontier traces the cloud's
    upper-left edge. (Uncapped deliberately -- the cap applies to the
    portfolios we'd actually hold, not to this validation cloud.)
    """
    n = len(expected_returns)
    rets, vols, sharpes = [], [], []
    for _ in range(n_portfolios):
        w = np.random.random(n)
        w = w / np.sum(w)
        r, v, s = portfolio_metrics(w, expected_returns, cov_matrix)
        rets.append(r)
        vols.append(v)
        sharpes.append(s)
    return np.array(rets), np.array(vols), np.array(sharpes)


# ==========================================================
# Main
# ==========================================================

def main():
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("Portfolio Optimization - PRD Section 11")
    report_lines.append("=" * 70)

    # ----------------------------------------------------------
    # Load + chronological train/backtest split
    # ----------------------------------------------------------
    dates, tickers, returns, market_returns = load_aligned_returns()
    n_tickers = len(tickers)

    train_dates, train_returns, train_market, bt_dates, bt_returns, bt_market = \
        split_train_backtest(dates, returns, market_returns)

    report_lines.append(f"\nDataset:")
    report_lines.append(f"  Tickers: {', '.join(tickers)}")
    report_lines.append(f"  Total date-aligned trading days: {len(dates)}")
    report_lines.append(f"  Train window:    {pd.Timestamp(train_dates[0]).date()} to {pd.Timestamp(train_dates[-1]).date()}  (n={len(train_dates)})")
    report_lines.append(f"  Backtest window: {pd.Timestamp(bt_dates[0]).date()} to {pd.Timestamp(bt_dates[-1]).date()}  (n={len(bt_dates)}, held out -- never used to estimate mu/Sigma)")

    # ----------------------------------------------------------
    # Expected returns & covariance -- TRAIN WINDOW ONLY
    # ----------------------------------------------------------
    # mu: historical mean, annualized. Documented choice (PRD 11.1):
    # we use the simple historical mean here rather than the DL/ML
    # forecasters' single-day-ahead predictions, because those models
    # predict tomorrow's return, not a stable long-run expected annual
    # return -- using them directly would require a separate horizon-
    # extension assumption this project doesn't validate. Historical
    # mean is the simpler, more defensible choice, with the known
    # caveat that it's a noisy estimator (see NVDA below).
    expected_returns = train_returns.mean(axis=0) * TRADING_DAYS

    # Sigma: Ledoit-Wolf shrinkage (PRD 11.2) instead of raw sample
    # covariance -- raw Sigma is what drives mean-variance optimizers
    # into extreme, unstable corner solutions when mu is noisily
    # estimated (exactly the case here: NVDA's real historical mean
    # is huge, but that's a very uncertain estimate of its TRUE future
    # expected return).
    lw = LedoitWolf().fit(train_returns)
    cov_matrix = lw.covariance_ * TRADING_DAYS
    cov_matrix_raw = np.cov(train_returns, rowvar=False, ddof=1) * TRADING_DAYS

    report_lines.append(f"\nAnnualized Expected Returns (historical mean, TRAIN window only):")
    for t, r in zip(tickers, expected_returns):
        report_lines.append(f"  {t:6s}: {r:.4f}  ({r*100:.2f}%)")
    report_lines.append(f"\nCovariance: Ledoit-Wolf shrinkage, shrinkage intensity = {lw.shrinkage_:.4f}")
    report_lines.append(f"  (0 = no shrinkage/raw sample Sigma, 1 = fully shrunk to diagonal target)")
    report_lines.append(f"Per-asset weight cap: {MAX_WEIGHT:.0%}")

    bounds = [(0.0, MAX_WEIGHT)] * n_tickers
    w0 = np.ones(n_tickers) / n_tickers

    # ----------------------------------------------------------
    # 1. Equal-Weight Benchmark
    # ----------------------------------------------------------
    report_lines.append("\n[1] Equal-Weight Portfolio")
    report_lines.append("-" * 50)
    w_eq = np.ones(n_tickers) / n_tickers
    r_eq, v_eq, s_eq = portfolio_metrics(w_eq, expected_returns, cov_matrix)
    report_lines.append(f"  Expected Return:  {r_eq:.4f}  ({r_eq*100:.2f}%)")
    report_lines.append(f"  Volatility:       {v_eq:.4f}  ({v_eq*100:.2f}%)")
    report_lines.append(f"  Sharpe Ratio:     {s_eq:.4f}")

    # ----------------------------------------------------------
    # 2. Minimum-Variance Portfolio
    # ----------------------------------------------------------
    report_lines.append("\n[2] Minimum-Variance Portfolio")
    report_lines.append("-" * 50)
    constraints_mv = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result_mv = minimize(
        portfolio_variance_obj, w0, args=(cov_matrix,),
        method="SLSQP", bounds=bounds, constraints=constraints_mv,
        options={"ftol": 1e-12}
    )
    w_mv = result_mv.x
    r_mv, v_mv, s_mv = portfolio_metrics(w_mv, expected_returns, cov_matrix)
    report_lines.append(f"  Weights:")
    for t, w in zip(tickers, w_mv):
        report_lines.append(f"    {t:6s}: {w:.4f}  ({w*100:.2f}%)")
    report_lines.append(f"  Expected Return:  {r_mv:.4f}  ({r_mv*100:.2f}%)")
    report_lines.append(f"  Volatility:       {v_mv:.4f}  ({v_mv*100:.2f}%)")
    report_lines.append(f"  Sharpe Ratio:     {s_mv:.4f}")

    # ----------------------------------------------------------
    # 3. Sharpe-Maximizing Portfolio (the "chosen" portfolio, Section 11.4)
    # ----------------------------------------------------------
    report_lines.append("\n[3] Sharpe-Maximizing Portfolio  <-- chosen portfolio")
    report_lines.append("-" * 50)
    constraints_sharpe = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    result_sharpe = minimize(
        negative_sharpe, w0, args=(expected_returns, cov_matrix),
        method="SLSQP", bounds=bounds, constraints=constraints_sharpe,
        options={"ftol": 1e-12}
    )
    w_sharpe = result_sharpe.x
    r_sharpe, v_sharpe, s_sharpe = portfolio_metrics(w_sharpe, expected_returns, cov_matrix)
    report_lines.append(f"  Weights:")
    for t, w in zip(tickers, w_sharpe):
        report_lines.append(f"    {t:6s}: {w:.4f}  ({w*100:.2f}%)")
    report_lines.append(f"  Expected Return (in-sample):  {r_sharpe:.4f}  ({r_sharpe*100:.2f}%)")
    report_lines.append(f"  Volatility (in-sample):       {v_sharpe:.4f}  ({v_sharpe*100:.2f}%)")
    report_lines.append(f"  Sharpe Ratio (in-sample):     {s_sharpe:.4f}")

    # ----------------------------------------------------------
    # 4. Risk-Parity Portfolio
    # ----------------------------------------------------------
    report_lines.append("\n[4] Risk-Parity Portfolio")
    report_lines.append("-" * 50)
    w_rp = risk_parity_weights(cov_matrix)
    r_rp, v_rp, s_rp = portfolio_metrics(w_rp, expected_returns, cov_matrix)
    report_lines.append(f"  Weights:")
    for t, w in zip(tickers, w_rp):
        report_lines.append(f"    {t:6s}: {w:.4f}  ({w*100:.2f}%)")
    report_lines.append(f"  Expected Return:  {r_rp:.4f}  ({r_rp*100:.2f}%)")
    report_lines.append(f"  Volatility:       {v_rp:.4f}  ({v_rp*100:.2f}%)")
    report_lines.append(f"  Sharpe Ratio:     {s_rp:.4f}")

    # ----------------------------------------------------------
    # 5. Efficient Frontier + Monte Carlo (Section 11.2 / 11.3)
    # ----------------------------------------------------------
    report_lines.append("\n[5] Efficient Frontier + Monte Carlo Cross-Check")
    report_lines.append("-" * 50)

    rand_rets, rand_vols, rand_sharpes = generate_random_portfolios(
        expected_returns, cov_matrix, n_portfolios=20000
    )
    ef_rets, ef_vols = build_efficient_frontier(expected_returns, cov_matrix, n_points=80)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(rand_vols, rand_rets, c=rand_sharpes, cmap="viridis", s=6, alpha=0.4)
    plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")
    ax.plot(ef_vols, ef_rets, color="crimson", linewidth=2, label=f"Efficient Frontier (cap={MAX_WEIGHT:.0%})")
    ax.scatter(v_eq, r_eq, marker="o", s=120, color="blue", edgecolors="black", zorder=5, label="Equal-Weight")
    ax.scatter(v_mv, r_mv, marker="s", s=120, color="green", edgecolors="black", zorder=5, label="Min-Variance")
    ax.scatter(v_sharpe, r_sharpe, marker="*", s=200, color="gold", edgecolors="black", zorder=5, label="Sharpe-Max (chosen)")
    ax.scatter(v_rp, r_rp, marker="D", s=120, color="purple", edgecolors="black", zorder=5, label="Risk-Parity")
    cal_x = np.linspace(0, max(ef_vols.max(), rand_vols.max()) * 1.1, 100)
    cal_y = RISK_FREE_RATE + s_sharpe * cal_x
    ax.plot(cal_x, cal_y, color="orange", linestyle="--", linewidth=1.5, alpha=0.7, label="CAL (Sharpe-Max)")
    ax.set_xlabel("Annualized Volatility", fontsize=12)
    ax.set_ylabel("Annualized Expected Return", fontsize=12)
    ax.set_title("Efficient Frontier - 10-Stock Portfolio (train window, shrunk Sigma, capped weights)", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "efficient_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    report_lines.append(f"  Monte Carlo portfolios simulated: 20,000")
    report_lines.append(f"  Plot saved to: {REPORT_DIR / 'efficient_frontier.png'}")

    # ----------------------------------------------------------
    # 6. Backtest: chosen portfolio vs Equal-Weight vs SP500 (Section 11.4)
    # ----------------------------------------------------------
    report_lines.append("\n[6] Out-of-Sample Backtest")
    report_lines.append("-" * 50)
    report_lines.append("Weights are FROZEN at their train-window values (no rebalancing,")
    report_lines.append("no re-estimation) and applied to realized backtest-window returns.")

    # Fixed-weight daily portfolio returns over the backtest window
    port_returns_bt = bt_returns @ w_sharpe
    eq_returns_bt = bt_returns @ w_eq
    sp500_returns_bt = bt_market  # buy-and-hold SP500

    def summarize_backtest(name, rets, market_rets):
        total_return = np.exp(np.sum(rets)) - 1
        ann_vol = np.std(rets, ddof=1) * np.sqrt(TRADING_DAYS)
        sharpe = sharpe_ratio_series(rets)
        sortino = sortino_ratio_series(rets)
        beta = beta_series(rets, market_rets)
        mdd, growth = max_drawdown_series(rets)
        return {
            "name": name, "total_return": total_return, "ann_vol": ann_vol,
            "sharpe": sharpe, "sortino": sortino, "beta": beta, "mdd": mdd,
            "growth": growth
        }

    bt_sharpe_max = summarize_backtest("Sharpe-Max (chosen)", port_returns_bt, bt_market)
    bt_equal = summarize_backtest("Equal-Weight", eq_returns_bt, bt_market)
    bt_spx = summarize_backtest("SP500 Buy-and-Hold", sp500_returns_bt, bt_market)

    report_lines.append(f"\n{'Strategy':<22} {'TotalRet':>10} {'AnnVol':>9} {'Sharpe':>8} {'Sortino':>9} {'Beta':>7} {'MaxDD':>9}")
    report_lines.append("-" * 78)
    for r in (bt_sharpe_max, bt_equal, bt_spx):
        report_lines.append(
            f"{r['name']:<22} {r['total_return']*100:>9.2f}% {r['ann_vol']*100:>8.2f}% "
            f"{r['sharpe']:>8.4f} {r['sortino']:>9.4f} {r['beta']:>7.4f} {r['mdd']*100:>8.2f}%"
        )

    report_lines.append(f"\nSection 11.4 risk report -- chosen (Sharpe-Max) portfolio, OUT OF SAMPLE:")
    report_lines.append(f"  Sharpe Ratio:          {bt_sharpe_max['sharpe']:.4f}")
    report_lines.append(f"  Sortino Ratio:         {bt_sharpe_max['sortino']:.4f}")
    report_lines.append(f"  Annualized Volatility: {bt_sharpe_max['ann_vol']:.4f}  ({bt_sharpe_max['ann_vol']*100:.2f}%)")
    report_lines.append(f"  Beta (vs SP500):       {bt_sharpe_max['beta']:.4f}")
    report_lines.append(f"  Maximum Drawdown:      {bt_sharpe_max['mdd']*100:.2f}%")

    beat_equal = bt_sharpe_max['total_return'] > bt_equal['total_return']
    beat_spx = bt_sharpe_max['total_return'] > bt_spx['total_return']
    report_lines.append(f"\n  Honest assessment: chosen portfolio {'BEAT' if beat_equal else 'DID NOT beat'} "
                         f"equal-weight out of sample, and {'BEAT' if beat_spx else 'DID NOT beat'} SP500 buy-and-hold.")

    # Backtest growth curve plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bt_dates, bt_sharpe_max["growth"], label="Sharpe-Max (chosen)", linewidth=1.8, color="crimson")
    ax.plot(bt_dates, bt_equal["growth"], label="Equal-Weight", linewidth=1.5, color="steelblue")
    ax.plot(bt_dates, bt_spx["growth"], label="SP500 Buy-and-Hold", linewidth=1.5, color="gray", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Out-of-Sample Backtest: Growth of $1 (weights frozen from train window)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "portfolio_backtest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    report_lines.append(f"\n  Backtest plot saved to: {REPORT_DIR / 'portfolio_backtest.png'}")

    # ----------------------------------------------------------
    # Summary comparison table (in-sample optimizer outputs)
    # ----------------------------------------------------------
    report_lines.append("\n" + "=" * 70)
    report_lines.append("Summary Comparison (in-sample / train window)")
    report_lines.append("=" * 70)
    report_lines.append(f"{'Strategy':<18} {'Return':>10} {'Volatility':>12} {'Sharpe':>10}")
    report_lines.append("-" * 52)
    report_lines.append(f"{'Equal-Weight':<18} {r_eq:>10.4f} {v_eq:>12.4f} {s_eq:>10.4f}")
    report_lines.append(f"{'Min-Variance':<18} {r_mv:>10.4f} {v_mv:>12.4f} {s_mv:>10.4f}")
    report_lines.append(f"{'Sharpe-Max':<18} {r_sharpe:>10.4f} {v_sharpe:>12.4f} {s_sharpe:>10.4f}")
    report_lines.append(f"{'Risk-Parity':<18} {r_rp:>10.4f} {v_rp:>12.4f} {s_rp:>10.4f}")

    # ----------------------------------------------------------
    # Save report
    # ----------------------------------------------------------
    report_text = "\n".join(report_lines)
    report_path = REPORT_DIR / "portfolio_opt_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n{'='*70}")
    print(f"Report saved to: {report_path}")
    print(f"Frontier plot:   {REPORT_DIR / 'efficient_frontier.png'}")
    print(f"Backtest plot:   {REPORT_DIR / 'portfolio_backtest.png'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
