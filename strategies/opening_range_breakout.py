from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy, StrategyConfig


class OpeningRangeBreakoutStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(config or StrategyConfig(name="opening_range_breakout"))

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        data = df.copy().sort_index()

        grouped = data.groupby(data.index.date, group_keys=False)
        range_high = grouped["high"].apply(lambda x: x.iloc[:3].max() if len(x) >= 3 else pd.NA)
        range_low = grouped["low"].apply(lambda x: x.iloc[:3].min() if len(x) >= 3 else pd.NA)
        bar_idx = grouped.cumcount()

        data["opening_range_high"] = range_high
        data["opening_range_low"] = range_low
        data["bar_index"] = bar_idx

        vol_avg = data["volume"].rolling(10, min_periods=10).mean().shift(1)
        vol_confirm = data["volume"] > (1.5 * vol_avg)

        eligible = data["bar_index"] >= 3
        long_cond = eligible & (data["close"] > data["opening_range_high"]) & vol_confirm
        short_cond = eligible & (data["close"] < data["opening_range_low"]) & vol_confirm

        data["signal"] = 0
        data.loc[long_cond, "signal"] = 1
        data.loc[short_cond, "signal"] = -1

        data["entry"] = data["close"]
        data["stop_loss"] = data["opening_range_low"]
        data.loc[data["signal"] == -1, "stop_loss"] = data["opening_range_high"]

        range_size = data["opening_range_high"] - data["opening_range_low"]
        data["target"] = data["entry"] + range_size
        data.loc[data["signal"] == -1, "target"] = data["entry"] - range_size

        return data[["signal", "entry", "stop_loss", "target", "opening_range_high", "opening_range_low"]]
