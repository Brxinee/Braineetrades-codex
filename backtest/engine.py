from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from strategies.base import Strategy


@dataclass
class EngineConfig:
    initial_capital: float = 1_000_000.0
    risk_percent: float = 0.01
    max_concurrent_positions: int = 3
    square_off_time: time = time(hour=15, minute=15)
    brokerage_roundtrip: float = 40.0
    slippage_percent: float = 0.0005


class BacktestEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def run_for_symbol(self, symbol: str, df: pd.DataFrame, strategy: Strategy) -> pd.DataFrame:
        signals = strategy.generate_signals(df).copy()
        if signals.empty:
            return pd.DataFrame()

        merged = df.join(signals[["signal", "entry", "stop_loss", "target"]], how="left")
        merged["signal"] = merged["signal"].fillna(0).astype(int)

        trades: list[Dict[str, Any]] = []
        open_positions = 0
        capital = self.config.initial_capital

        for ts, row in merged.iterrows():
            if open_positions >= self.config.max_concurrent_positions:
                continue
            signal = int(row["signal"])
            if signal == 0 or pd.isna(row["entry"]) or pd.isna(row["stop_loss"]):
                continue

            risk_amount = capital * self.config.risk_percent
            risk_per_share = abs(float(row["entry"]) - float(row["stop_loss"]))
            if risk_per_share <= 0:
                continue
            qty = max(int(risk_amount / risk_per_share), 1)
            open_positions += 1

            exit_price = float(row["target"]) if not pd.isna(row["target"]) else float(row["close"])
            slippage = float(row["entry"]) * self.config.slippage_percent
            entry_price = float(row["entry"]) + (slippage if signal > 0 else -slippage)
            realized_exit = exit_price - (slippage if signal > 0 else -slippage)

            gross_pnl = (realized_exit - entry_price) * qty * signal
            net_pnl = gross_pnl - self.config.brokerage_roundtrip
            capital += net_pnl

            trades.append(
                {
                    "timestamp": ts.isoformat(),
                    "symbol": symbol,
                    "side": "LONG" if signal > 0 else "SHORT",
                    "entry": round(entry_price, 4),
                    "exit": round(realized_exit, 4),
                    "qty": qty,
                    "gross_pnl": round(float(gross_pnl), 4),
                    "net_pnl": round(float(net_pnl), 4),
                    "capital": round(float(capital), 4),
                }
            )
            open_positions -= 1

        return pd.DataFrame(trades)

    @staticmethod
    def build_equity_curve(trades: pd.DataFrame, initial_capital: float) -> pd.Series:
        if trades.empty:
            return pd.Series([initial_capital], index=[pd.Timestamp.utcnow()])
        sorted_trades = trades.copy()
        sorted_trades["timestamp"] = pd.to_datetime(sorted_trades["timestamp"])
        sorted_trades = sorted_trades.sort_values("timestamp")
        equity = initial_capital + sorted_trades["net_pnl"].cumsum().to_numpy()
        return pd.Series(equity, index=sorted_trades["timestamp"].tolist())
