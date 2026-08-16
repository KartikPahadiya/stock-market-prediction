"""FinBERT comparison entrypoint.

This is intentionally backed by the same unified ablation script as
``ablation_sentiment.py`` so VADER-only, FinBERT-only, both, and no-sentiment
conditions are evaluated on identical splits.
"""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("ablation_vader_finbert.py")), run_name="__main__")
