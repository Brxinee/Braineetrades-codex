from __future__ import annotations

from http import HTTPStatus
from typing import Any, Dict


def handler(request: Any) -> Dict[str, Any]:
    return {
        "statusCode": HTTPStatus.NOT_IMPLEMENTED,
        "headers": {"Content-Type": "application/json"},
        "body": '{"error":"Phase 1 scaffold: backtest not implemented yet."}',
    }
