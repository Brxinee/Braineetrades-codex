from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategyConfig


class SupertrendEMAStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(config or StrategyConfig(name="supertrend_ema"))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def _supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        atr = self._atr(df, period)
        hl2 = (df["high"] + df["low"]) / 2

        upperband = hl2 + multiplier * atr
        lowerband = hl2 - multiplier * atr

        final_upper = upperband.copy()
        final_lower = lowerband.copy()
        trend = pd.Series(np.ones(len(df), dtype=int), index=df.index)

        for i in range(1, len(df)):
            if df["close"].iloc[i - 1] <= final_upper.iloc[i - 1]:
                final_upper.iloc[i] = min(upperband.iloc[i], final_upper.iloc[i - 1])
            if df["close"].iloc[i - 1] >= final_lower.iloc[i - 1]:
                final_lower.iloc[i] = max(lowerband.iloc[i], final_lower.iloc[i - 1])

            if df["close"].iloc[i] > final_upper.iloc[i - 1]:
                trend.iloc[i] = 1
            elif df["close"].iloc[i] < final_lower.iloc[i - 1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i - 1]

        st = pd.Series(index=df.index, dtype=float)
        st[trend == 1] = final_lower[trend == 1]
        st[trend == -1] = final_upper[trend == -1]

        return pd.DataFrame({"supertrend": st, "trend": trend, "atr10": atr})

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        data = df.copy().sort_index()
        data["ema20"] = data["close"].ewm(span=20, adjust=False).mean()

        st_df = self._supertrend(data, period=10, multiplier=3.0)
        data = data.join(st_df)

        prev_trend = data["trend"].shift(1)
        long_cond = (prev_trend == -1) & (data["trend"] == 1) & (data["close"] > data["ema20"])
        short_cond = (prev_trend == 1) & (data["trend"] == -1) & (data["close"] < data["ema20"])

        data["signal"] = 0
        data.loc[long_cond, "signal"] = 1
        data.loc[short_cond, "signal"] = -1

        data["entry"] = data["close"]
        data["stop_loss"] = data["supertrend"]
        rr = (data["entry"] - data["stop_loss"]).abs()
        data["target"] = data["entry"] + rr
        data.loc[data["signal"] == -1, "target"] = data["entry"] - rr

        return data[["signal", "entry", "stop_loss", "target", "supertrend", "trend", "ema20", "atr10"]]
