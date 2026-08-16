"""Sentiment ablation entrypoint.

Runs the unified VADER/FinBERT ablation and writes
``reports/vader_finbert_ablation.csv``.
"""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("ablation_vader_finbert.py")), run_name="__main__")
