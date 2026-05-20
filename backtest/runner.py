from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf

from backtest.engine import BacktestEngine, EngineConfig
from backtest.metrics import compute_metrics
from strategies import (
    GapFadeStrategy,
    OpeningRangeBreakoutStrategy,
    RSIDivergenceStrategy,
    SupertrendEMAStrategy,
    VWAPReversalStrategy,
)

IST = "Asia/Kolkata"
RESULTS_DIR = Path("public/data/results")
LEADERBOARD_PATH = Path("public/data/leaderboard.json")
NIFTY50_PATH = Path("public/data/nifty50.json")


def load_symbols() -> List[str]:
    payload = json.loads(NIFTY50_PATH.read_text())
    return [item["symbol"] for item in payload]


def fetch_5m(symbol: str) -> Tuple[pd.DataFrame, str | None]:
    end = datetime.now()
    start = end - timedelta(days=60)
    try:
        data = yf.download(symbol, start=start, end=end, interval="5m", progress=False, auto_adjust=False, timeout=8)
    except Exception as exc:
        return pd.DataFrame(), f"{symbol}: fetch failed ({exc})"
    if data.empty:
        return pd.DataFrame(), f"{symbol}: empty response"

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0].lower() for c in data.columns]
    else:
        data.columns = [str(c).lower() for c in data.columns]
    data = data.rename(columns={"adj close": "adj_close"})

    req = ["open", "high", "low", "close", "volume"]
    missing = [c for c in req if c not in data.columns]
    if missing:
        return pd.DataFrame(), f"{symbol}: missing columns {missing}"

    data = data[req].dropna()
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    data.index = data.index.tz_convert(IST)
    return data, None


def strategy_map() -> Dict[str, object]:
    return {
        "vwap_reversal": VWAPReversalStrategy(),
        "opening_range_breakout": OpeningRangeBreakoutStrategy(),
        "supertrend_ema": SupertrendEMAStrategy(),
        "rsi_divergence": RSIDivergenceStrategy(),
        "gap_fade": GapFadeStrategy(),
    }


def run_strategy(name: str, symbols: List[str]) -> Dict[str, object]:
    engine = BacktestEngine(EngineConfig())
    strategy = strategy_map()[name]
    all_trades: list[pd.DataFrame] = []
    errors: list[str] = []

    for symbol in symbols:
        data, err = fetch_5m(symbol)
        if err:
            errors.append(err)
            continue
        trades = engine.run_for_symbol(symbol, data, strategy)
        if not trades.empty:
            all_trades.append(trades)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_curve = engine.build_equity_curve(trades_df, engine.config.initial_capital)
    metrics = compute_metrics(trades_df, equity_curve)

    result = {
        "strategy": name,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "timezone": IST,
        "errors": errors,
        "metrics": metrics,
        "trades": trades_df.to_dict(orient="records") if not trades_df.empty else [],
    }
    return result


def write_outputs(strategy_names: List[str]) -> None:
    symbols = load_symbols()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    leaderboard = []
    for name in strategy_names:
        result = run_strategy(name, symbols)
        (RESULTS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2))
        m = result["metrics"]
        leaderboard.append(
            {
                "strategy": name,
                "total_trades": m["total_trades"],
                "win_rate": m["win_rate"],
                "profit_factor": m["profit_factor"],
                "sharpe_annualized": m["sharpe_annualized"],
                "max_drawdown": m["max_drawdown"],
                "expectancy": m["expectancy"],
            }
        )

    LEADERBOARD_PATH.write_text(json.dumps(leaderboard, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Nifty 50 strategy backtests.")
    parser.add_argument("--strategy", default="all", help="Strategy name or 'all'")
    args = parser.parse_args()

    available = list(strategy_map().keys())
    if args.strategy == "all":
        selected = available
    else:
        if args.strategy not in available:
            raise ValueError(f"Unknown strategy '{args.strategy}'. Available: {available}")
        selected = [args.strategy]

    write_outputs(selected)


if __name__ == "__main__":
    main()
