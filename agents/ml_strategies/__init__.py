"""
ML-Based Betting Strategies for Polymarket

This module contains machine learning models and strategies
for identifying profitable betting opportunities.
"""

from .base_strategy import MLBettingStrategy, StrategyResult
from .market_prediction import MarketPredictor
from .edge_detection import EdgeDetector
from .portfolio_optimizer import PortfolioOptimizer
from .backtesting import StrategyBacktester

__all__ = [
    'MLBettingStrategy',
    'StrategyResult',
    'MarketPredictor',
    'EdgeDetector',
    'PortfolioOptimizer',
    'StrategyBacktester'
]
