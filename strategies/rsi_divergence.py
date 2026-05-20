from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy, StrategyConfig


class RSIDivergenceStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(config or StrategyConfig(name="rsi_divergence"))

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        return 100 - (100 / (1 + rs))

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        data = df.copy().sort_index()

        data["rsi14"] = self._rsi(data["close"], 14)

        pivot_window = 5
        data["pivot_low"] = data["low"] == data["low"].rolling(pivot_window, center=True).min()
        data["pivot_high"] = data["high"] == data["high"].rolling(pivot_window, center=True).max()

        prev_pivot_low_price = data["low"].where(data["pivot_low"]).ffill().shift(1)
        prev_pivot_low_rsi = data["rsi14"].where(data["pivot_low"]).ffill().shift(1)

        prev_pivot_high_price = data["high"].where(data["pivot_high"]).ffill().shift(1)
        prev_pivot_high_rsi = data["rsi14"].where(data["pivot_high"]).ffill().shift(1)

        bullish_div = data["pivot_low"] & (data["low"] < prev_pivot_low_price) & (data["rsi14"] > prev_pivot_low_rsi)
        bearish_div = data["pivot_high"] & (data["high"] > prev_pivot_high_price) & (data["rsi14"] < prev_pivot_high_rsi)

        bull_confirm = data["close"] > data["high"].shift(1)
        bear_confirm = data["close"] < data["low"].shift(1)

        long_cond = bullish_div & bull_confirm
        short_cond = bearish_div & bear_confirm

        data["signal"] = 0
        data.loc[long_cond, "signal"] = 1
        data.loc[short_cond, "signal"] = -1

        data["entry"] = data["close"]
        data["stop_loss"] = data["low"].where(data["signal"] == 1)
        data.loc[data["signal"] == -1, "stop_loss"] = data["high"]

        risk = (data["entry"] - data["stop_loss"]).abs()
        data["target"] = data["entry"] + (2 * risk)
        data.loc[data["signal"] == -1, "target"] = data["entry"] - (2 * risk)

        return data[["signal", "entry", "stop_loss", "target", "rsi14", "pivot_low", "pivot_high"]]
