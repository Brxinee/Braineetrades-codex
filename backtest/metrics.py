from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return (equity_curve - running_max) / running_max.replace(0, np.nan)


def compute_metrics(trades: pd.DataFrame, equity_curve: pd.Series) -> Dict[str, Any]:
    if trades.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_annualized": 0.0,
            "max_drawdown": 0.0,
            "expectancy": 0.0,
            "equity_curve": [{"timestamp": str(ts), "equity": float(val)} for ts, val in equity_curve.items()],
            "per_symbol": {},
        }

    pnl = trades["net_pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = float((pnl > 0).mean() * 100)
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    expectancy = float(pnl.mean())

    returns = equity_curve.pct_change().dropna()
    sharpe = 0.0
    if not returns.empty and returns.std(ddof=0) > 0:
        sharpe = float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252))

    dd = compute_drawdown(equity_curve)
    max_dd = float(dd.min()) if not dd.empty else 0.0

    per_symbol: Dict[str, Any] = {}
    for symbol, sdf in trades.groupby("symbol"):
        spnl = sdf["net_pnl"].astype(float)
        per_symbol[symbol] = {
            "trades": int(len(sdf)),
            "win_rate": float((spnl > 0).mean() * 100),
            "net_pnl": float(spnl.sum()),
        }

    return {
        "total_trades": int(len(trades)),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": float(profit_factor),
        "sharpe_annualized": sharpe,
        "max_drawdown": max_dd,
        "expectancy": expectancy,
        "equity_curve": [{"timestamp": str(ts), "equity": float(val)} for ts, val in equity_curve.items()],
        "per_symbol": per_symbol,
    }
