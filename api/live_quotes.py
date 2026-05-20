from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
CACHE_TTL_SECONDS = 30
FETCH_TIMEOUT_SECONDS = 8
NIFTY50_PATH = Path(__file__).resolve().parent.parent / "public" / "data" / "nifty50.json"

_CACHE: Dict[str, Any] = {
    "expires_at": 0.0,
    "payload": None,
}


def _json_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _now_ist_iso() -> str:
    return datetime.now(tz=IST).isoformat()


def _load_symbols() -> List[str]:
    data = json.loads(NIFTY50_PATH.read_text())
    return [item["symbol"] for item in data]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]
    return df


def _fetch_symbol_quote(symbol: str) -> Tuple[Dict[str, Any] | None, str | None]:
    try:
        data = yf.download(
            symbol,
            period="1d",
            interval="1m",
            progress=False,
            auto_adjust=False,
            timeout=FETCH_TIMEOUT_SECONDS,
            prepost=False,
            threads=False,
        )
    except Exception as exc:
        return None, f"{symbol}: fetch failed ({exc})"

    if data.empty:
        return None, f"{symbol}: empty data from Yahoo"

    data = _normalize_columns(data)
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        return None, f"{symbol}: missing columns {missing}"

    row = data[required].dropna().iloc[-1] if not data[required].dropna().empty else None
    if row is None:
        return None, f"{symbol}: no valid OHLCV row"

    ts = data.index[-1]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert(IST)

    quote = {
        "symbol": symbol,
        "ltp": float(row["close"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "volume": int(row["volume"]),
        "timestamp": ts.isoformat(),
    }
    return quote, None


def _build_live_quotes_payload() -> Dict[str, Any]:
    symbols = _load_symbols()
    quotes: List[Dict[str, Any]] = []
    errors: List[str] = []

    for symbol in symbols:
        quote, err = _fetch_symbol_quote(symbol)
        if err:
            errors.append(err)
            continue
        quotes.append(quote)

    successful = len(quotes)
    failed = len(errors)
    now_iso = _now_ist_iso()

    payload: Dict[str, Any] = {
        "timezone": "Asia/Kolkata",
        "generated_at": now_iso,
        "stale_at": datetime.fromtimestamp(time.time() + CACHE_TTL_SECONDS, tz=IST).isoformat(),
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "total_symbols": len(symbols),
        "successful_symbols": successful,
        "failed_symbols": failed,
        "quotes": quotes,
        "errors": errors,
    }
    return payload


def handler(request: Any) -> Dict[str, Any]:
    now = time.time()

    if _CACHE.get("payload") and _CACHE.get("expires_at", 0) > now:
        cached_payload = dict(_CACHE["payload"])
        cached_payload["cache"] = "HIT"
        return _json_response(200, cached_payload)

    try:
        payload = _build_live_quotes_payload()
    except Exception as exc:
        return _json_response(
            500,
            {
                "error": "Failed to load live quotes.",
                "details": str(exc),
                "generated_at": _now_ist_iso(),
                "timezone": "Asia/Kolkata",
            },
        )

    if payload["successful_symbols"] == 0:
        payload["cache"] = "MISS"
        return _json_response(
            503,
            {
                "error": "Live quotes unavailable from Yahoo Finance.",
                "details": payload["errors"],
                "generated_at": payload["generated_at"],
                "timezone": payload["timezone"],
            },
        )

    _CACHE["payload"] = payload
    _CACHE["expires_at"] = now + CACHE_TTL_SECONDS

    payload["cache"] = "MISS"
    return _json_response(200, payload)
