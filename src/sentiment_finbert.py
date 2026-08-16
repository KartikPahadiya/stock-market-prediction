"""Compatibility entrypoint for the PRD/README FinBERT step.

The canonical FinBERT implementation lives in ``sentiment.py``.  This wrapper
keeps the documented pipeline command stable without duplicating model code.
"""

from sentiment import *  # noqa: F401,F403
