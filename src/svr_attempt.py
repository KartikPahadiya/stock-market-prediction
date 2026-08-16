"""Summarise the standalone SVR attempt for the final report.

The tuned RBF SVR is trained in ``models_ml.py`` with the other classical
models.  This script keeps the PRD/README command stable and emits a compact
SVR-specific report from the pooled ML results.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"
ML_RESULTS = REPORT_DIR / "ml_results_pooled.csv"
OUT = REPORT_DIR / "svr_attempt_report.txt"


def main() -> None:
    if not ML_RESULTS.exists():
        raise FileNotFoundError(f"{ML_RESULTS} missing; run models_ml_quick.py first")

    results = pd.read_csv(ML_RESULTS)
    svr = results[(results["Ticker"] == "ALL") & (results["Model"] == "SVR")]
    if svr.empty:
        raise ValueError("No pooled SVR row found in ml_results_pooled.csv")

    row = svr.iloc[0]
    text = (
        "SVR Attempt Summary\n"
        "===================\n\n"
        "Model: SVR with RBF kernel, trained in the pooled classical-ML pipeline.\n"
        f"RMSE: {row['RMSE']:.6f}\n"
        f"MAE: {row['MAE']:.6f}\n"
        f"MAPE: {row['MAPE']:.4f}\n"
        f"R2: {row['R2']:.6f}\n"
        f"Directional accuracy: {row['Directional_Accuracy']:.2f}%\n\n"
        "Interpretation: SVR is retained for the required four-model classical "
        "benchmark, but it is not the winning model on this noisy daily-return "
        "task. The final recommendation layer therefore uses the best available "
        "cached model outputs rather than privileging SVR.\n"
    )
    print(text)
    try:
        OUT.write_text(text, encoding="utf-8")
    except PermissionError as exc:
        print(f"WARNING: could not write {OUT}: {exc}")
        print("Summary was printed above; rerun from your normal user account to persist it.")
    else:
        print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
