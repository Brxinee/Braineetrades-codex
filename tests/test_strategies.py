import pandas as pd

from strategies import (
    GapFadeStrategy,
    OpeningRangeBreakoutStrategy,
    RSIDivergenceStrategy,
    SupertrendEMAStrategy,
    VWAPReversalStrategy,
)
from strategies.base import BaseStrategy, StrategyConfig


class DummyStrategy(BaseStrategy):
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        out = df[["close"]].copy()
        out["signal"] = 0
        return out


def build_sample_df(rows: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01 09:15", periods=rows, freq="5min", tz="Asia/Kolkata")
    base = pd.Series(range(rows), dtype=float)
    df = pd.DataFrame(index=idx)
    df["open"] = 100 + base * 0.2
    df["high"] = df["open"] + 0.8
    df["low"] = df["open"] - 0.8
    df["close"] = df["open"] + ((base % 4) - 1.5) * 0.3
    df["volume"] = 1000 + (base * 10).astype(int)
    return df


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
    try:
        strategy.generate_signals(df)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Missing OHLCV columns" in str(exc)


def test_all_strategies_generate_expected_columns() -> None:
    df = build_sample_df()
    strategies = [
        VWAPReversalStrategy(),
        OpeningRangeBreakoutStrategy(),
        SupertrendEMAStrategy(),
        RSIDivergenceStrategy(),
        GapFadeStrategy(),
    ]

    for strategy in strategies:
        out = strategy.generate_signals(df)
        assert "signal" in out.columns
        assert "entry" in out.columns
        assert "stop_loss" in out.columns
        assert "target" in out.columns
        assert len(out) == len(df)
