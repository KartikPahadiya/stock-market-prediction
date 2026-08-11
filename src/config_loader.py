"""Centralized configuration loader.

All scripts should import `cfg` from this module rather than hard-coding
parameters.  This guarantees a single source of truth for:
  • universe, dates, splits
  • paths
  • model hyper-parameters
  • random seeds
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# ------------------------------------------------------------------
# Load YAML once at import time
# ------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
    _RAW: dict[str, Any] = yaml.safe_load(fh)


# ------------------------------------------------------------------
# Thin accessor to avoid scattered dict indexing
# ------------------------------------------------------------------
class _Config:
    """Dot-access wrapper around the raw YAML dict."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def __getattr__(self, name: str) -> Any:
        try:
            return self._raw[name]
        except KeyError as exc:
            raise AttributeError(f"config.yaml has no key '{name}'") from exc

    def get(self, name: str, default: Any = None) -> Any:
        return self._raw.get(name, default)

    def section(self, *keys: str) -> Any:
        """Deep accessor: cfg.section('features', 'sma_windows')"""
        node = self._raw
        for k in keys:
            if not isinstance(node, dict):
                raise KeyError(f"Cannot descend into non-dict at key '{k}'")
            node = node[k]
        return node


cfg = _Config(_RAW)


# ------------------------------------------------------------------
# Convenience helpers (typed)
# ------------------------------------------------------------------
def project_root() -> Path:
    return _PROJECT_ROOT


def data_path(key: str) -> Path:
    """Resolve a path from config relative to project root."""
    rel = cfg.section("paths", key)
    return _PROJECT_ROOT / rel


def tickers() -> list[str]:
    return cfg.section("universe", "tickers")


def benchmark_ticker() -> str:
    return cfg.section("universe", "benchmark")


def split_ratios() -> tuple[float, float, float]:
    s = cfg.splits
    return s["train_ratio"], s["val_ratio"], s["test_ratio"]


def random_seed() -> int:
    return cfg.section("reproducibility", "random_state")


def ml_model_params(model_name: str) -> dict[str, Any]:
    """Return param dict for a quick-run ML model (Ridge, RandomForest, …)."""
    return dict(cfg.section("ml_quick", model_name))


def dl_model_params() -> dict[str, Any]:
    return dict(cfg.dl)


# Back-compat: expose commonly-used constants at module level for one-liners
TICKERS: list[str] = tickers()
BENCHMARK: str = benchmark_ticker()
RANDOM_STATE: int = random_seed()
START_DATE: str = cfg.section("universe", "start_date")
END_DATE: str = cfg.section("universe", "end_date")
