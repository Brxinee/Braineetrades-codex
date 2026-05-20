from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    risk_percent: float = 0.01
    max_concurrent_positions: int = 3
    square_off_time: str = "15:15"
    brokerage_roundtrip: float = 40.0
    slippage_percent: float = 0.0005
    fees: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyResult:
    symbol: str
    strategy: str
    signals: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


class Strategy(Protocol):
    config: StrategyConfig

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a signal dataframe indexed by timestamp."""
        ...


class BaseStrategy:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def validate_input(self, df: pd.DataFrame) -> None:
        required_columns = {"open", "high", "low", "close", "volume"}
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
        if df.empty:
            raise ValueError("Input dataframe is empty.")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Strategy must implement generate_signals")
