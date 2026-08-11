"""
==============================================================
AI Stock Prediction System
Machine Learning Models - Pooled Training

Trains ONE model across all tickers with ticker one-hot encoding.

NOTE on cross-validation (PRD Sec 8.2):
  This script uses a single chronological 70/15/15 split with
  lightweight grid search for hyperparameter selection.  Walk-forward
  CV (TimeSeriesSplit-3) is demonstrated separately in walkforward_cv.py
  for methodology validation; the two pipelines are independent.
==============================================================
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR = PROJECT_ROOT / "data" / "processed" / "predictions"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Config
# ==========================================================

TARGET = "Target_Return"
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# ==========================================================
# Helpers
# ==========================================================

def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred)) * 100


def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        "RMSE": rmse, "MAE": mae, "MAPE": mape,
        "R2": r2, "Directional_Accuracy": directional_accuracy(y_true, y_pred)
    }


# ==========================================================
# Data
# ==========================================================

def load_stock_dataset(feature_file):
    df = pd.read_csv(feature_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def train_test_split_time(df):
    n = len(df)
    train_end = int(n * (1 - TEST_SIZE - VAL_SIZE))
    val_end = int(n * (1 - TEST_SIZE))
    return (
        df.iloc[:train_end].copy().reset_index(drop=True),
        df.iloc[train_end:val_end].copy().reset_index(drop=True),
        df.iloc[val_end:].copy().reset_index(drop=True)
    )


def prepare_features_pooled(train, val, test):
    drop_columns = ["Date", "ticker", "Target_Close", "Target_Return", "Target_Direction"]
    X_train = train.drop(columns=drop_columns, errors="ignore")
    X_val = val.drop(columns=drop_columns, errors="ignore")
    X_test = test.drop(columns=drop_columns, errors="ignore")
    y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]

    for X, y, name in [(X_train, y_train, "train"), (X_val, y_val, "val"), (X_test, y_test, "test")]:
        mask = X.notnull().all(axis=1)
        n_bad = (~mask).sum()
        if n_bad > 0:
            print(f"WARNING: dropping {n_bad} NaN rows in {name}")
        X = X[mask]
        y = y[mask]
        if name == "train":
            X_train, y_train = X, y
        elif name == "val":
            X_val, y_val = X, y
        else:
            X_test, y_test = X, y

    if "ticker_id" not in X_train.columns:
        raise KeyError("ticker_id column missing")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X_train):
    numeric_cols = [c for c in X_train.columns if c != "ticker_id"]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["ticker_id"])
    ])
    return preprocessor, numeric_cols


# ==========================================================
# Tuning (lightweight grids for interactive speed)
# ==========================================================

def tune_and_train(model_name, X_train, y_train, X_val, y_val):
    best_model = None
    best_rmse = np.inf

    def _fit_eval(model_inst):
        nonlocal best_model, best_rmse
        model_inst.fit(X_train, y_train)
        pred = model_inst.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model_inst

    if model_name == "Ridge":
        for alpha in [100, 1000]:
            _fit_eval(Ridge(alpha=alpha))

    elif model_name == "RandomForest":
        for max_depth in [3, 5, 10]:
            for n_est in [100, 200]:
                _fit_eval(RandomForestRegressor(
                    n_estimators=n_est, max_depth=max_depth,
                    random_state=RANDOM_STATE, n_jobs=-1))
        print(f"  Best RF: depth={best_model.max_depth}, n_est={best_model.n_estimators}")

    elif model_name == "XGBoost":
        for max_depth in [2, 3, 5]:
            for lr in [0.01, 0.03]:
                _fit_eval(XGBRegressor(
                    objective="reg:squarederror", n_estimators=300,
                    learning_rate=lr, max_depth=max_depth,
                    subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE))
        print(f"  Best XGB: depth={best_model.max_depth}, lr={best_model.learning_rate}")

    elif model_name == "SVR":
        for C in [1, 10, 100]:
            _fit_eval(SVR(kernel="rbf", C=C, epsilon=0.01))
        print(f"  Best SVR C: {best_model.C}")

    return best_model


# ==========================================================
# Train + Export
# ==========================================================

def train_model_pooled(model_name, X_train, X_val, X_test, y_train, y_val, y_test, preprocessor):
    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    model = tune_and_train(model_name, X_train_t, y_train, X_val_t, y_val)
    predictions = model.predict(X_test_t)

    results = []

    # Overall
    overall = evaluate_model(y_test.values, predictions)
    overall.update({"Ticker": "ALL", "Model": model_name})
    results.append(overall)

    # Zero baseline
    z = evaluate_model(y_test.values, np.zeros_like(y_test.values))
    z.update({"Ticker": "ALL", "Model": "ZeroBaseline"})
    results.append(z)

    # Per-ticker
    ticker_ids = X_test["ticker_id"].values
    for tid in np.unique(ticker_ids):
        mask = ticker_ids == tid
        actual = y_test.values[mask]
        pred = predictions[mask]

        m = evaluate_model(actual, pred)
        m.update({"Ticker": tid, "Model": model_name})
        results.append(m)

        zt = evaluate_model(actual, np.zeros_like(actual))
        zt.update({"Ticker": tid, "Model": "ZeroBaseline"})
        results.append(zt)

    # Export predictions
    pred_df = pd.DataFrame({
        "ticker_id": ticker_ids,
        "actual": y_test.values,
        "predicted": predictions
    })
    pred_df.to_csv(PREDICTION_DIR / f"pooled_{model_name}_predictions.csv", index=False)

    # Save model
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    joblib.dump(pipeline, MODEL_DIR / f"pooled_{model_name}.pkl")

    return results, predictions


# ==========================================================
# Main
# ==========================================================

def main():
    feature_files = sorted(FEATURE_DIR.glob("*.csv"))
    print(f"\nFound {len(feature_files)} datasets.\n")

    ticker_names = [f.stem.replace("_features", "") for f in feature_files]
    ticker_to_id = {name: i for i, name in enumerate(ticker_names)}
    id_to_ticker = {i: name for name, i in ticker_to_id.items()}
    print(f"Tickers: {ticker_names}\n")

    all_train, all_val, all_test = [], [], []
    for file in feature_files:
        df = load_stock_dataset(file)
        ticker_name = file.stem.replace("_features", "")
        df["ticker_id"] = ticker_to_id[ticker_name]
        tr, va, te = train_test_split_time(df)
        all_train.append(tr)
        all_val.append(va)
        all_test.append(te)
        print(f"  {ticker_name}: train={len(tr)}, val={len(va)}, test={len(te)}")

    train_df = pd.concat(all_train, ignore_index=True)
    val_df = pd.concat(all_val, ignore_index=True)
    test_df = pd.concat(all_test, ignore_index=True)
    print(f"\nPooled: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_features_pooled(
        train_df, val_df, test_df)

    preprocessor, numeric_cols = build_preprocessor(X_train)
    print(f"\nNumeric features: {len(numeric_cols)}")
    print(f"Tickers (categorical): {len(ticker_names)}")

    all_results = []
    for model_name in ["Ridge", "RandomForest", "XGBoost", "SVR"]:
        results, _ = train_model_pooled(
            model_name, X_train, X_val, X_test,
            y_train, y_val, y_test, preprocessor)
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    results_df["Ticker"] = results_df["Ticker"].map(lambda x: id_to_ticker.get(x, x))
    results_df.to_csv(REPORT_DIR / "ml_results_pooled.csv", index=False)

    trained_only = results_df[~results_df["Model"].isin(["ZeroBaseline"])]
    best_models = trained_only.sort_values(["Ticker", "RMSE"]).groupby("Ticker", as_index=False).first()
    best_models.to_csv(REPORT_DIR / "best_ml_models_pooled.csv", index=False)

    print("\n" + "=" * 80)
    print("BEST ML MODELS PER TICKER (Pooled Training)")
    print("=" * 80)
    print(best_models[["Ticker", "Model", "RMSE", "R2", "Directional_Accuracy"]])
    print("\nPooled ML Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
