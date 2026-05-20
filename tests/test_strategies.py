import pandas as pd
import pytest

from strategies.base import BaseStrategy, StrategyConfig


class DummyStrategy(BaseStrategy):
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        out = df[["close"]].copy()
        out["signal"] = 0
        return out


def test_base_strategy_validation_passes_with_ohlcv() -> None:
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [100],
        }
    )
    strategy = DummyStrategy(StrategyConfig(name="dummy"))
    signals = strategy.generate_signals(df)
    assert "signal" in signals.columns


def test_base_strategy_validation_fails_on_missing_columns() -> None:
    df = pd.DataFrame({"close": [1.0]})
    strategy = DummyStrategy(StrategyConfig(name="dummy"))
    with pytest.raises(ValueError, match="Missing OHLCV columns"):
        strategy.generate_signals(df)
