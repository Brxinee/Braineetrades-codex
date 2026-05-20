from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy, StrategyConfig


class VWAPReversalStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(config or StrategyConfig(name="vwap_reversal"))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=period).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        data = df.copy().sort_index()

        tp = (data["high"] + data["low"] + data["close"]) / 3.0
        cum_vp = (tp * data["volume"]).groupby(data.index.date).cumsum()
        cum_vol = data["volume"].groupby(data.index.date).cumsum().replace(0, pd.NA)
        data["vwap"] = cum_vp / cum_vol
        data["atr14"] = self._atr(data, 14)

        data["dist_from_vwap"] = (data["close"] - data["vwap"]) / data["vwap"]

        prev_close = data["close"].shift(1)
        bullish_reversal = (prev_close < data["open"]) & (data["close"] > data["open"])
        bearish_reversal = (prev_close > data["open"]) & (data["close"] < data["open"])

        long_cond = (data["dist_from_vwap"] < -0.01) & bullish_reversal
        short_cond = (data["dist_from_vwap"] > 0.01) & bearish_reversal

        data["signal"] = 0
        data.loc[long_cond, "signal"] = 1
        data.loc[short_cond, "signal"] = -1

        data["entry"] = data["close"]
        data["stop_loss"] = data["close"] - 0.5 * data["atr14"]
        data.loc[data["signal"] == -1, "stop_loss"] = data["close"] + 0.5 * data["atr14"]
        data["target"] = data["vwap"]

        return data[["signal", "entry", "stop_loss", "target", "vwap", "atr14"]]
