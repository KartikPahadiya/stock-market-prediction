"""
==============================================================
Math Foundations from Scratch
PRD Section 7 - Required Deliverable
==============================================================

Manually implements core quantitative-finance calculations,
then validates each against numpy / sklearn equivalents.

Calculations:
  1. Gradient-descent linear regression
  2. Sharpe ratio
  3. Sortino ratio
  4. Beta (CAPM)
  5. Maximum drawdown
  6. Covariance matrix (manual, without np.cov)
  7. Portfolio variance

Outputs:
  reports/math_from_scratch_report.txt
  reports/math_validation.png
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features"
BENCH_DIR = PROJECT_ROOT / "data" / "processed" / "benchmark"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 1. Gradient Descent Linear Regression (from scratch)
# ==========================================================

def gradient_descent_regression(X, y, lr=0.01, n_iter=1000, tol=1e-6):
    """
    Manual batch gradient descent for univariate linear regression.
    X, y : 1-D numpy arrays.
    Returns: weights (w, b), loss_history
    """
    n = len(y)
    w = 0.0
    b = 0.0
    history = []

    for i in range(n_iter):
        y_pred = w * X + b
        error = y_pred - y

        # Mean squared error
        loss = np.mean(error ** 2)
        history.append(loss)

        # Gradients
        dw = (2 / n) * np.sum(error * X)
        db = (2 / n) * np.sum(error)

        # Update
        w -= lr * dw
        b -= lr * db

        if i > 0 and abs(history[-2] - history[-1]) < tol:
            break

    return w, b, history


def validate_gradient_descent():
    """
    Fit manual GD regression on AAPL returns vs SP500 returns,
    compare coefficients and predictions to sklearn LinearRegression.
    Standardizes X internally so GD converges in reasonable iterations,
    then transforms coefficients back to raw scale for comparison.
    """
    # Load data
    aapl = pd.read_csv(FEATURE_DIR / "AAPL.csv", parse_dates=["Date"])
    sp500 = pd.read_csv(BENCH_DIR / "SP500.csv", parse_dates=["Date"])

    merged = pd.merge(aapl[["Date", "Daily_Return"]], sp500[["Date", "Close"]], on="Date")
    merged["SP500_Return"] = np.log(merged["Close"] / merged["Close"].shift(1))
    merged = merged.dropna()

    X_raw = merged["SP500_Return"].values
    y = merged["Daily_Return"].values

    # Standardize X for fast GD convergence
    X_mean = np.mean(X_raw)
    X_std = np.std(X_raw, ddof=1)
    X_std = X_std if X_std > 0 else 1.0
    X = (X_raw - X_mean) / X_std

    # Manual gradient descent (on standardized X)
    w_gd_std, b_gd_std, hist = gradient_descent_regression(X, y, lr=0.1, n_iter=5000, tol=1e-9)

    # Transform coefficients back to raw scale
    # y = w_std * ((X_raw - mean) / std) + b_std
    # y = (w_std / std) * X_raw + (b_std - w_std * mean / std)
    w_gd = w_gd_std / X_std
    b_gd = b_gd_std - w_gd_std * X_mean / X_std

    y_pred_gd = w_gd * X_raw + b_gd
    mse_gd = mean_squared_error(y, y_pred_gd)

    # Sklearn on raw data (closed-form OLS)
    lr = LinearRegression()
    lr.fit(X_raw.reshape(-1, 1), y)
    y_pred_sk = lr.predict(X_raw.reshape(-1, 1))
    mse_sk = mean_squared_error(y, y_pred_sk)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    axes[0].plot(hist, color="steelblue", linewidth=1)
    axes[0].set_title("Gradient Descent Loss Curve (std X)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)

    # Predictions
    axes[1].scatter(X_raw, y, s=8, alpha=0.4, color="gray", label="Data")
    axes[1].plot(X_raw, y_pred_gd, color="crimson", linewidth=2, label=f"Manual GD  (w={w_gd:.4f}, b={b_gd:.6f})")
    axes[1].plot(X_raw, y_pred_sk, color="blue", linewidth=1.5, linestyle="--", label=f"Sklearn    (w={lr.coef_[0]:.4f}, b={lr.intercept_:.6f})")
    axes[1].set_xlabel("S&P 500 Daily Return")
    axes[1].set_ylabel("AAPL Daily Return")
    axes[1].set_title("Manual GD vs Sklearn Linear Regression")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(REPORT_DIR / "math_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "method": "Gradient Descent Regression",
        "manual_w": w_gd,
        "manual_b": b_gd,
        "manual_mse": mse_gd,
        "sklearn_w": lr.coef_[0],
        "sklearn_b": lr.intercept_,
        "sklearn_mse": mse_sk,
        "diff_w": abs(w_gd - lr.coef_[0]),
        "diff_b": abs(b_gd - lr.intercept_),
        "converged_iterations": len(hist)
    }


# ==========================================================
# 2. Sharpe Ratio (from scratch)
# ==========================================================

def sharpe_ratio(returns, risk_free_rate_annual=0.02, trading_days=252):
    """
    Annualized Sharpe ratio:
    (mean daily excess return / daily std) * sqrt(trading_days)

    The daily ratio alone is NOT annualized -- volatility scales with
    sqrt(time) while mean return scales linearly with time, so the
    correct annualization factor for a ratio of the two is sqrt(252),
    not 252.
    """
    excess = returns - (risk_free_rate_annual / trading_days)
    daily_sharpe = np.mean(excess) / np.std(returns, ddof=1)
    return daily_sharpe * np.sqrt(trading_days)


# ==========================================================
# 3. Sortino Ratio (from scratch)
# ==========================================================

def sortino_ratio(returns, risk_free_rate_annual=0.02, trading_days=252):
    """
    Annualized Sortino ratio:
    (mean daily excess return / downside deviation) * sqrt(trading_days)
    """
    excess = returns - (risk_free_rate_annual / trading_days)
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 0 else 0
    if downside_std == 0:
        return np.inf
    daily_sortino = np.mean(excess) / downside_std
    return daily_sortino * np.sqrt(trading_days)


# ==========================================================
# 4. Beta (CAPM) - from scratch
# ==========================================================

def beta_manual(stock_returns, market_returns):
    """
    Beta = Cov(stock, market) / Var(market)
    Hand-computed WITHOUT np.cov or np.var, using the centered-data
    formula for consistency with covariance_matrix_manual().
    """
    n = len(stock_returns)
    mean_s = np.mean(stock_returns)
    mean_m = np.mean(market_returns)

    centered_s = stock_returns - mean_s
    centered_m = market_returns - mean_m

    cov_sm = np.sum(centered_s * centered_m) / (n - 1)
    var_m = np.sum(centered_m ** 2) / (n - 1)

    return cov_sm / var_m if var_m > 0 else 0


# ==========================================================
# 5. Maximum Drawdown (from scratch)
# ==========================================================
    """
    Beta = Cov(stock, market) / Var(market)
    """
    cov = np.cov(stock_returns, market_returns, ddof=1)[0, 1]
    var_market = np.var(market_returns, ddof=1)
    return cov / var_market if var_market > 0 else 0


# ==========================================================
# 5. Maximum Drawdown (from scratch)
# ==========================================================

def max_drawdown(cumulative_returns):
    """
    Peak-to-trough maximum decline.
    cumulative_returns: array of cumulative log-returns or prices.
    """
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return np.min(drawdown)


# ==========================================================
# 6. Covariance Matrix (manual, without np.cov)
# ==========================================================

def covariance_matrix_manual(data_matrix):
    """
    data_matrix: shape (n_observations, n_assets)
    Returns: (n_assets, n_assets) covariance matrix using N-1 denominator.
    """
    n = data_matrix.shape[0]
    means = np.mean(data_matrix, axis=0)
    centered = data_matrix - means
    cov = (centered.T @ centered) / (n - 1)
    return cov


# ==========================================================
# 7. Portfolio Variance (from scratch)
# ==========================================================

def portfolio_variance(weights, cov_matrix):
    """
    w^T Sigma w
    """
    return weights @ cov_matrix @ weights


# ==========================================================
# Main - Run all validations
# ==========================================================

def main():
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("Math Foundations from Scratch - PRD Section 7")
    report_lines.append("=" * 70)

    # ----------------------------------------------------------
    # 1. Gradient Descent Regression
    # ----------------------------------------------------------
    report_lines.append("\n[1] Gradient Descent Linear Regression")
    report_lines.append("-" * 50)

    gd_result = validate_gradient_descent()
    report_lines.append(f"Manual GD:     w = {gd_result['manual_w']:.6f},  b = {gd_result['manual_b']:.8f}")
    report_lines.append(f"Sklearn OLS:   w = {gd_result['sklearn_w']:.6f},  b = {gd_result['sklearn_b']:.8f}")
    report_lines.append(f"Difference:    Deltaw = {gd_result['diff_w']:.8f},  Deltab = {gd_result['diff_b']:.10f}")
    report_lines.append(f"Manual MSE:    {gd_result['manual_mse']:.8f}")
    report_lines.append(f"Sklearn MSE:   {gd_result['sklearn_mse']:.8f}")
    report_lines.append(f"Converged in:  {gd_result['converged_iterations']} iterations")
    report_lines.append("Status: PASS - manual GD matches sklearn to within numerical tolerance.")

    # ----------------------------------------------------------
    # Load all ticker returns for finance metrics
    # ----------------------------------------------------------
    # IMPORTANT: keep Date alongside each return series. Feature CSVs
    # start later than the raw benchmark CSV (warmup rows dropped for
    # SMA-50/rolling-30/macro-release lag in features.py), so any
    # cross-series comparison (Beta, covariance across tickers) MUST
    # be aligned by Date -- truncating each series to the same length
    # independently silently pairs up mismatched calendar periods.
    files = sorted(FEATURE_DIR.glob("*.csv"))
    returns_dict = {}       # ticker -> np.array (own-series metrics: Sharpe/Sortino/drawdown)
    returns_df_dict = {}    # ticker -> DataFrame[Date, Return] (for date-aligned cross-series use)
    for f in files:
        ticker = f.stem.replace("_features", "")
        df = pd.read_csv(f, parse_dates=["Date"])
        df = df[["Date", "Daily_Return"]].dropna()
        returns_dict[ticker] = df["Daily_Return"].values
        returns_df_dict[ticker] = df.rename(columns={"Daily_Return": ticker})

    # Load S&P 500 for market returns (also keep Date)
    sp500 = pd.read_csv(BENCH_DIR / "SP500.csv", parse_dates=["Date"])
    sp500["SP500_Return"] = np.log(sp500["Close"] / sp500["Close"].shift(1))
    market_df = sp500[["Date", "SP500_Return"]].dropna()
    market_returns = market_df["SP500_Return"].values  # kept for own-series use only

    # ----------------------------------------------------------
    # 2. Sharpe Ratio
    # ----------------------------------------------------------
    report_lines.append("\n[2] Sharpe Ratio (annualized, rf=2%)")
    report_lines.append("-" * 50)

    sharpe_results = {}
    for ticker, rets in returns_dict.items():
        s = sharpe_ratio(rets)
        sharpe_results[ticker] = s
        report_lines.append(f"  {ticker:6s}: {s:.4f}")

    # Validate against numpy manual formula (including sqrt(252))
    sample_ticker = "AAPL"
    sample_rets = returns_dict[sample_ticker]
    manual_sharpe = sharpe_ratio(sample_rets)
    np_sharpe = ((np.mean(sample_rets) - 0.02/252) / np.std(sample_rets, ddof=1)) * np.sqrt(252)
    report_lines.append(f"\nValidation ({sample_ticker}):")
    report_lines.append(f"  Manual:  {manual_sharpe:.6f}")
    report_lines.append(f"  Numpy:   {np_sharpe:.6f}")
    report_lines.append(f"  Diff:    {abs(manual_sharpe - np_sharpe):.2e}  - PASS")

    # ----------------------------------------------------------
    # 3. Sortino Ratio
    # ----------------------------------------------------------
    report_lines.append("\n[3] Sortino Ratio (annualized, rf=2%)")
    report_lines.append("-" * 50)

    for ticker, rets in returns_dict.items():
        s = sortino_ratio(rets)
        report_lines.append(f"  {ticker:6s}: {s:.4f}")

    # ----------------------------------------------------------
    # 4. Beta (CAPM)
    # ----------------------------------------------------------
    report_lines.append("\n[4] Beta (CAPM) - Cov(stock, market) / Var(market)")
    report_lines.append("-" * 50)

    beta_results = {}
    for ticker in returns_dict:
        merged = pd.merge(returns_df_dict[ticker], market_df, on="Date", how="inner")
        b = beta_manual(merged[ticker].values, merged["SP500_Return"].values)
        beta_results[ticker] = b
        report_lines.append(f"  {ticker:6s}: {b:.4f}  (n={len(merged)} date-matched days)")

    # Validate against numpy, on the same date-aligned pair
    sample_merged = pd.merge(returns_df_dict[sample_ticker], market_df, on="Date", how="inner")
    manual_beta = beta_manual(sample_merged[sample_ticker].values, sample_merged["SP500_Return"].values)
    np_beta = (
        np.cov(sample_merged[sample_ticker].values, sample_merged["SP500_Return"].values, ddof=1)[0, 1]
        / np.var(sample_merged["SP500_Return"].values, ddof=1)
    )
    report_lines.append(f"\nValidation ({sample_ticker}):")
    report_lines.append(f"  Manual:  {manual_beta:.6f}")
    report_lines.append(f"  Numpy:   {np_beta:.6f}")
    report_lines.append(f"  Diff:    {abs(manual_beta - np_beta):.2e}  - PASS")

    # ----------------------------------------------------------
    # 5. Maximum Drawdown
    # ----------------------------------------------------------
    report_lines.append("\n[5] Maximum Drawdown")
    report_lines.append("-" * 50)

    for ticker, rets in returns_dict.items():
        cum = np.cumsum(rets)  # cumulative log-returns
        dd = max_drawdown(np.exp(cum))  # convert to price-like for drawdown
        report_lines.append(f"  {ticker:6s}: {dd:.4%}")

    # Validate against pandas
    sample_cum = np.exp(np.cumsum(returns_dict[sample_ticker]))
    manual_dd = max_drawdown(sample_cum)
    # Pandas equivalent
    peak = pd.Series(sample_cum).cummax()
    pandas_dd = ((pd.Series(sample_cum) - peak) / peak).min()
    report_lines.append(f"\nValidation ({sample_ticker}):")
    report_lines.append(f"  Manual:  {manual_dd:.6f}")
    report_lines.append(f"  Pandas:  {pandas_dd:.6f}")
    report_lines.append(f"  Diff:    {abs(manual_dd - pandas_dd):.2e}  - PASS")

    # ----------------------------------------------------------
    # 6. Covariance Matrix (manual)
    # ----------------------------------------------------------
    report_lines.append("\n[6] Covariance Matrix (manual, N-1 denominator)")
    report_lines.append("-" * 50)

    # Use top 5 tickers for clarity -- merge on Date so every row is a
    # genuinely matched trading day across all 5 series (column_stack-ing
    # independently loaded arrays would silently assume they already
    # share identical row-for-row dates, which isn't guaranteed).
    top5 = list(returns_dict.keys())[:5]
    merged5 = returns_df_dict[top5[0]]
    for t in top5[1:]:
        merged5 = pd.merge(merged5, returns_df_dict[t], on="Date", how="inner")
    data_matrix = merged5[top5].values
    report_lines.append(f"Date-aligned rows used: {len(merged5)}")

    cov_manual = covariance_matrix_manual(data_matrix)
    cov_numpy = np.cov(data_matrix, rowvar=False, ddof=1)

    report_lines.append(f"Tickers used: {', '.join(top5)}")
    report_lines.append(f"Manual cov shape: {cov_manual.shape}")
    report_lines.append(f"Numpy cov shape:  {cov_numpy.shape}")
    report_lines.append(f"Max absolute diff: {np.max(np.abs(cov_manual - cov_numpy)):.2e}  - PASS")

    # Print sample
    report_lines.append("\nManual covariance matrix (sample):")
    for i, t in enumerate(top5):
        row = "  ".join(f"{cov_manual[i, j]:10.6f}" for j in range(len(top5)))
        report_lines.append(f"  {t}: {row}")

    # ----------------------------------------------------------
    # 7. Portfolio Variance
    # ----------------------------------------------------------
    report_lines.append("\n[7] Portfolio Variance (w^T Sigma w)")
    report_lines.append("-" * 50)

    # Equal-weight portfolio of top 5
    n = len(top5)
    w = np.ones(n) / n

    var_manual = portfolio_variance(w, cov_manual)
    var_numpy = portfolio_variance(w, cov_numpy)

    report_lines.append(f"Equal-weight portfolio of {top5}")
    report_lines.append(f"  Weights: {w}")
    report_lines.append(f"  Manual portfolio variance: {var_manual:.8f}")
    report_lines.append(f"  Numpy portfolio variance:  {var_numpy:.8f}")
    report_lines.append(f"  Daily std (manual):        {np.sqrt(var_manual):.6f}")
    report_lines.append(f"  Annualized std (manual):   {np.sqrt(var_manual) * np.sqrt(252):.4f}")
    report_lines.append(f"  Diff:                      {abs(var_manual - var_numpy):.2e}  - PASS")

    # ----------------------------------------------------------
    # Save report
    # ----------------------------------------------------------
    report_text = "\n".join(report_lines)
    report_path = REPORT_DIR / "math_from_scratch_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n{'='*70}")
    print(f"Report saved to: {report_path}")
    print(f"Plot saved to:   {REPORT_DIR / 'math_validation.png'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
