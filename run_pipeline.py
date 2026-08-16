"""End-to-end PRD pipeline runner.

This script gives the project one stable rebuild command while keeping each
module independently runnable.  Use flags to skip slow or already-cached stages
when iterating locally.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


STEPS: list[tuple[str, str, str]] = [
    ("ingest", "src/ingest.py", "Download raw market, macro, and news data"),
    ("clean", "src/clean.py", "Clean and calendar-align raw data"),
    ("vader", "src/sentiment_vader.py", "Score news with VADER"),
    ("finbert", "src/sentiment_finbert.py", "Score news with FinBERT"),
    ("merge_sentiment", "src/merge_sentiment.py", "Merge VADER and FinBERT daily signals"),
    ("features", "src/features.py", "Build no-look-ahead feature panels"),
    ("eda", "src/eda.py", "Generate EDA diagnostics"),
    ("math", "src/math_from_scratch.py", "Validate financial math from scratch"),
    ("ml", "src/models_ml_quick.py", "Train classical ML models"),
    ("dl", "src/models_dl.py", "Train deep-learning sequence models"),
    ("comparison", "src/comparison_table.py", "Create eight-model comparison table"),
    ("walkforward", "src/walkforward_cv.py", "Run walk-forward validation"),
    ("ablation", "src/ablation_vader_finbert.py", "Measure sentiment impact"),
    ("portfolio", "src/portfolio_opt.py", "Optimize and backtest portfolio"),
    ("recommendations", "src/recommendations.py", "Generate recommendations"),
    ("svr", "src/svr_attempt.py", "Write SVR-specific summary"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete stock AI PRD pipeline")
    parser.add_argument(
        "--from-step",
        choices=[name for name, _, _ in STEPS],
        help="Start at this step and run through the end",
    )
    parser.add_argument(
        "--only",
        choices=[name for name, _, _ in STEPS],
        help="Run a single step only",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=[name for name, _, _ in STEPS],
        help="Skip a step; can be supplied multiple times",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip the slow FinBERT and deep-learning stages; use cached outputs",
    )
    return parser.parse_args()


def selected_steps(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    steps = STEPS
    if args.only:
        steps = [step for step in steps if step[0] == args.only]
    elif args.from_step:
        start = next(i for i, step in enumerate(steps) if step[0] == args.from_step)
        steps = steps[start:]

    skips = set(args.skip)
    if args.fast:
        skips.update({"finbert", "dl"})
    return [step for step in steps if step[0] not in skips]


def main() -> int:
    args = parse_args()
    steps = selected_steps(args)
    if not steps:
        print("No steps selected.")
        return 0

    print("=" * 72)
    print("AI Stock Prediction System - PRD Pipeline")
    print("=" * 72)
    print(f"Python: {sys.executable}")
    print(f"Root:   {PROJECT_ROOT}")
    print()

    for index, (name, script, description) in enumerate(steps, start=1):
        path = PROJECT_ROOT / script
        if not path.exists():
            raise FileNotFoundError(f"Pipeline step '{name}' is missing: {path}")

        print("-" * 72)
        print(f"[{index}/{len(steps)}] {name}: {description}")
        print("-" * 72)
        completed = subprocess.run([sys.executable, str(path)], cwd=PROJECT_ROOT)
        if completed.returncode != 0:
            print(f"\nStep failed: {name} ({completed.returncode})")
            return completed.returncode

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
