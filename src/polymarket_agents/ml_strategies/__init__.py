"""
ML-Based Betting Strategies for Polymarket

This module contains machine learning models and strategies
for identifying profitable betting opportunities.
"""

from .base_strategy import MLBettingStrategy, StrategyResult

from .market_prediction import MarketPredictor

from .edge_detection import EdgeDetector

from .evaluation import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .neural_net_strategy import NeuralNetStrategy

try:
    from .lstm_probability import lstm_probability_strategy
except ImportError:
    lstm_probability_strategy = None  # type: ignore[assignment]


__all__ = [
    "MLBettingStrategy",
    "StrategyResult",
    "MarketPredictor",
    "EdgeDetector",
    "NeuralNetStrategy",
    "lstm_probability_strategy",
    "classification_report",
    "confusion_matrix",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
]
