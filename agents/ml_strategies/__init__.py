"""
ML-Based Betting Strategies for Polymarket

This module contains machine learning models and strategies
for identifying profitable betting opportunities.
"""

from .base_strategy import MLBettingStrategy, StrategyResult
from .market_prediction import MarketPredictor
from .edge_detection import EdgeDetector

__all__ = [
    'MLBettingStrategy',
    'StrategyResult',
    'MarketPredictor',
    'EdgeDetector'
]
