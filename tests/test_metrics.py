import pandas as pd

from backtest.metrics import compute_drawdown, compute_metrics


def test_compute_drawdown() -> None:
    equity = pd.Series([100.0, 110.0, 105.0, 120.0, 90.0])
    dd = compute_drawdown(equity)
    assert round(float(dd.min()), 4) == -0.25


def test_compute_metrics_empty_trades() -> None:
    equity = pd.Series([1_000_000.0])
    metrics = compute_metrics(pd.DataFrame(), equity)
    assert metrics["total_trades"] == 0
    assert metrics["win_rate"] == 0.0
