"""
==============================================================
Walk-Forward Cross-Validation Demonstration
PRD Section 8.2 - Non-negotiable requirement

Demonstrates TimeSeriesSplit(3) on pooled, calendar-sorted data
for hyperparameter selection, then trains the final model and
evaluates on the hold-out test set.

CRITICAL FIX: data is sorted by Date (not ticker-then-date)
before TimeSeriesSplit, so each fold's train/test boundary is a
genuine calendar cut across all tickers simultaneously.  This
prevents leakage from later-regime AAPL data into early-regime
AMZN validation folds.
==============================================================
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Target_Return"
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
N_SPLITS = 3

# ==========================================================
# Load and pool data
# ==========================================================

def load_all():
    files = sorted(FEATURE_DIR.glob("*.csv"))
    ticker_names = [f.stem for f in files]
    ticker_to_id = {name: i for i, name in enumerate(ticker_names)}

    all_train, all_val, all_test = [], [], []
    for f in files:
        df = pd.read_csv(f, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        df["ticker_id"] = ticker_to_id[f.stem]
        n = len(df)
        tr_end = int(n * (1 - TEST_SIZE - VAL_SIZE))
        va_end = int(n * (1 - TEST_SIZE))
        all_train.append(df.iloc[:tr_end])
        all_val.append(df.iloc[tr_end:va_end])
        all_test.append(df.iloc[va_end:])

    return (
        pd.concat(all_train, ignore_index=True),
        pd.concat(all_val, ignore_index=True),
        pd.concat(all_test, ignore_index=True),
        ticker_to_id
    )


def prepare(df, keep_date=False):
    drop_cols = ["ticker", "Target_Close", "Target_Return", "Target_Direction"]
    if not keep_date:
        drop_cols.append("Date")
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[TARGET]
    mask = X.notnull().all(axis=1)
    return X[mask], y[mask]


# ==========================================================
# Walk-Forward CV
# ==========================================================

def run_walkforward_cv(X_cv, y_cv, n_splits=N_SPLITS):
    """
    Demonstrate walk-forward CV on RandomForest.
    X_cv and y_cv MUST already be calendar-sorted (row order = time order).
    Returns best hyperparameters and per-fold scores.
    """
    print(f"\nWalk-Forward CV: {n_splits} splits on {len(X_cv)} rows")
    print("-" * 50)

    param_grid = [
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 7},
    ]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_params = None
    best_rmse = np.inf
    all_fold_results = []

    for params in param_grid:
        fold_rmses = []
        fold_details = []

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_cv)):
            X_tr = X_cv.iloc[tr_idx]
            y_tr = y_cv.iloc[tr_idx]
            X_va = X_cv.iloc[va_idx]
            y_va = y_cv.iloc[va_idx]

            # Fresh preprocessor per fold
            num_cols = [c for c in X_tr.columns if c != "ticker_id"]
            pre = ColumnTransformer([
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["ticker_id"])
            ])

            X_tr_t = pre.fit_transform(X_tr)
            X_va_t = pre.transform(X_va)

            m = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
            m.fit(X_tr_t, y_tr)
            pred = m.predict(X_va_t)
            rmse = np.sqrt(mean_squared_error(y_va, pred))
            fold_rmses.append(rmse)
            fold_details.append({
                "config": str(params),
                "fold": fold + 1,
                "train_size": len(tr_idx),
                "val_size": len(va_idx),
                "rmse": rmse
            })

        avg_rmse = np.mean(fold_rmses)
        print(f"  Config {params}: avg RMSE={avg_rmse:.6f}  (folds: {[f'{r:.6f}' for r in fold_rmses]})")

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = params

        all_fold_results.extend(fold_details)

    print(f"\n  BEST: {best_params}  (CV RMSE: {best_rmse:.6f})")
    return best_params, pd.DataFrame(all_fold_results)


# ==========================================================
# Final training and test evaluation
# ==========================================================

def train_final_and_evaluate(best_params, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train on train+val, evaluate on test."""
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)

    num_cols = [c for c in X_full.columns if c != "ticker_id"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["ticker_id"])
    ])

    X_full_t = pre.fit_transform(X_full)
    X_test_t = pre.transform(X_test)

    m = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best_params)
    m.fit(X_full_t, y_full)
    pred = m.predict(X_test_t)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = np.mean(np.abs(y_test - pred))
    dir_acc = np.mean(np.sign(y_test) == np.sign(pred)) * 100

    return {"RMSE": rmse, "MAE": mae, "Directional_Accuracy": dir_acc}


# ==========================================================
# Main
# ==========================================================

def main():
    print("=" * 70)
    print("Walk-Forward Cross-Validation Demonstration")
    print("PRD Section 8.2")
    print("=" * 70)

    train_df, val_df, test_df, ticker_to_id = load_all()
    print(f"\nPooled: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Keep Date for CV sorting, drop for test
    X_train, y_train = prepare(train_df, keep_date=True)
    X_val, y_val = prepare(val_df, keep_date=True)
    X_test, y_test = prepare(test_df, keep_date=False)

    # Calendar-sort train+val before TimeSeriesSplit
    Xy_cv = pd.concat([X_train, X_val], ignore_index=True)
    y_cv_raw = pd.concat([y_train, y_val], ignore_index=True)
    Xy_cv["_y_temp_"] = y_cv_raw.values
    Xy_cv = Xy_cv.sort_values("Date").reset_index(drop=True)
    y_cv = Xy_cv["_y_temp_"]
    X_cv = Xy_cv.drop(columns=["_y_temp_", "Date"])

    best_params, fold_df = run_walkforward_cv(X_cv, y_cv)

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation (train+val -> test)")
    print("=" * 70)
    final_metrics = train_final_and_evaluate(
        best_params, X_train.drop(columns=["Date"], errors="ignore"),
        y_train, X_val.drop(columns=["Date"], errors="ignore"),
        y_val, X_test, y_test
    )
    print(f"  Test RMSE:  {final_metrics['RMSE']:.6f}")
    print(f"  Test MAE:   {final_metrics['MAE']:.6f}")
    print(f"  Test DirAcc: {final_metrics['Directional_Accuracy']:.1f}%")

    # Save
    fold_df.to_csv(REPORT_DIR / "walkforward_cv_folds.csv", index=False)
    with open(REPORT_DIR / "walkforward_cv_summary.txt", "w") as f:
        f.write("Walk-Forward Cross-Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"CV splits: {N_SPLITS}\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"Test RMSE: {final_metrics['RMSE']:.6f}\n")
        f.write(f"Test MAE: {final_metrics['MAE']:.6f}\n")
        f.write(f"Test DirAcc: {final_metrics['Directional_Accuracy']:.1f}%\n")

    print(f"\nSaved: {REPORT_DIR / 'walkforward_cv_folds.csv'}")
    print(f"Saved: {REPORT_DIR / 'walkforward_cv_summary.txt'}")


if __name__ == "__main__":
    main()
