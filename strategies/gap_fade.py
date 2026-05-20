from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy, StrategyConfig


class GapFadeStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig | None = None) -> None:
        super().__init__(config or StrategyConfig(name="gap_fade"))

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        data = df.copy().sort_index()

        day = pd.Series(data.index.date, index=data.index)
        day_open = data.groupby(day)["open"].transform("first")
        prev_close = data.groupby(day)["close"].transform("first").shift(1)
        gap_pct = (day_open - prev_close) / prev_close

        bar_index = data.groupby(day).cumcount()
        first_bar_close = data.groupby(day)["close"].transform("first")

        valid_gap = gap_pct.abs().between(0.008, 0.02)
        fade_short = valid_gap & (gap_pct > 0) & (bar_index == 0)
        fade_long = valid_gap & (gap_pct < 0) & (bar_index == 0)

        data["signal"] = 0
        data.loc[fade_long, "signal"] = 1
        data.loc[fade_short, "signal"] = -1

        data["entry"] = first_bar_close
        session_high = data.groupby(day)["high"].cummax()
        session_low = data.groupby(day)["low"].cummin()
        data["stop_loss"] = session_low
        data.loc[data["signal"] == -1, "stop_loss"] = session_high

        data["target"] = prev_close
        return data[["signal", "entry", "stop_loss", "target", "gap_pct"]]
