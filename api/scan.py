from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from strategies import (
    GapFadeStrategy,
    OpeningRangeBreakoutStrategy,
    RSIDivergenceStrategy,
    SupertrendEMAStrategy,
    VWAPReversalStrategy,
)
from strategies.base import BaseStrategy

IST = ZoneInfo("Asia/Kolkata")
FETCH_TIMEOUT_SECONDS = 8
NIFTY50_PATH = Path(__file__).resolve().parent.parent / "public" / "data" / "nifty50.json"


STRATEGY_MAP: Dict[str, BaseStrategy] = {
    "vwap_reversal": VWAPReversalStrategy(),
    "opening_range_breakout": OpeningRangeBreakoutStrategy(),
    "supertrend_ema": SupertrendEMAStrategy(),
    "rsi_divergence": RSIDivergenceStrategy(),
    "gap_fade": GapFadeStrategy(),
}


def _json_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _now_iso() -> str:
    return datetime.now(tz=IST).isoformat()


def _load_symbols() -> List[str]:
    data = json.loads(NIFTY50_PATH.read_text())
    return [item["symbol"] for item in data]


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    keep = ["open", "high", "low", "close", "volume"]
    return df[keep].dropna()


def _fetch_intraday_5m(symbol: str) -> Tuple[pd.DataFrame, str | None]:
    try:
        df = yf.download(
            symbol,
            period="5d",
            interval="5m",
            progress=False,
            auto_adjust=False,
            prepost=False,
            timeout=FETCH_TIMEOUT_SECONDS,
            threads=False,
        )
    except Exception as exc:
        return pd.DataFrame(), f"{symbol}: fetch failed ({exc})"

    if df.empty:
        return pd.DataFrame(), f"{symbol}: empty data from Yahoo"

    df = _normalize_df(df)
    if df.empty:
        return pd.DataFrame(), f"{symbol}: no valid OHLCV rows"

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)

    return df, None


def _is_market_hours(ts: datetime) -> bool:
    if ts.weekday() >= 5:
        return False
    open_time = time(hour=9, minute=15)
    close_time = time(hour=15, minute=30)
    return open_time <= ts.timetz().replace(tzinfo=None) <= close_time


def _build_signal_row(symbol: str, signal_row: pd.Series, ltp: float, timestamp: str) -> Dict[str, Any]:
    entry = float(signal_row["entry"])
    stop_loss = float(signal_row["stop_loss"])
    target = float(signal_row["target"])
    risk = abs(entry - stop_loss)
    reward = abs(target - entry)
    rr = (reward / risk) if risk > 0 else 0.0

    signal_int = int(signal_row["signal"])
    signal_type = "LONG" if signal_int > 0 else "SHORT"

    return {
        "symbol": symbol,
        "ltp": ltp,
        "signal": signal_type,
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "risk_reward": rr,
        "timestamp": timestamp,
    }


def handler(request: Any) -> Dict[str, Any]:
    if getattr(request, "method", "POST") != "POST":
        return _json_response(405, {"error": "Method not allowed. Use POST."})

    try:
        payload = request.get_json() if hasattr(request, "get_json") else json.loads(request.body or "{}")
    except Exception:
        return _json_response(400, {"error": "Invalid JSON body."})

    strategy_name = (payload or {}).get("strategy")
    if not strategy_name or strategy_name not in STRATEGY_MAP:
        return _json_response(
            400,
            {
                "error": "Invalid strategy.",
                "allowed": sorted(STRATEGY_MAP.keys()),
            },
        )

    strategy = STRATEGY_MAP[strategy_name]
    symbols = _load_symbols()
    now = datetime.now(tz=IST)

    signals: List[Dict[str, Any]] = []
    errors: List[str] = []

    for symbol in symbols:
        df, err = _fetch_intraday_5m(symbol)
        if err:
            errors.append(err)
            continue

        try:
            signal_df = strategy.generate_signals(df)
        except Exception as exc:
            errors.append(f"{symbol}: strategy error ({exc})")
            continue

        if signal_df.empty or "signal" not in signal_df.columns:
            continue

        active = signal_df[signal_df["signal"].isin([1, -1])]
        if active.empty:
            continue

        latest = active.iloc[-1]
        latest_ts = active.index[-1]
        ltp = float(df["close"].iloc[-1])
        signals.append(_build_signal_row(symbol, latest, ltp, latest_ts.isoformat()))

    if len(signals) == 0 and len(errors) == len(symbols):
        return _json_response(
            503,
            {
                "error": "Data unavailable for all symbols. Scan failed.",
                "strategy": strategy_name,
                "timezone": "Asia/Kolkata",
                "timestamp": _now_iso(),
                "details": errors,
            },
        )

    response = {
        "strategy": strategy_name,
        "timezone": "Asia/Kolkata",
        "timestamp": _now_iso(),
        "market_hours": _is_market_hours(now),
        "total_symbols": len(symbols),
        "signals_count": len(signals),
        "signals": signals,
        "errors": errors,
    }
    return _json_response(200, response)
