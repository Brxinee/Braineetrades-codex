"""Strategy package for Nifty 50 intraday research platform."""

from strategies.base import StrategyConfig, StrategyResult
from strategies.gap_fade import GapFadeStrategy
from strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from strategies.rsi_divergence import RSIDivergenceStrategy
from strategies.supertrend_ema import SupertrendEMAStrategy
from strategies.vwap_reversal import VWAPReversalStrategy

__all__ = [
    "StrategyConfig",
    "StrategyResult",
    "VWAPReversalStrategy",
    "OpeningRangeBreakoutStrategy",
    "SupertrendEMAStrategy",
    "RSIDivergenceStrategy",
    "GapFadeStrategy",
]
