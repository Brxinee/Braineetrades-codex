import json
from pathlib import Path


NIFTY50_PATH = Path("public/data/nifty50.json")


def test_nifty50_symbol_count() -> None:
    data = json.loads(NIFTY50_PATH.read_text())
    assert len(data) == 50


def test_nifty50_schema() -> None:
    data = json.loads(NIFTY50_PATH.read_text())
    required_keys = {"symbol", "company", "sector", "lot_size"}
    assert all(required_keys.issubset(item.keys()) for item in data)
    assert all(item["symbol"].endswith(".NS") for item in data)
