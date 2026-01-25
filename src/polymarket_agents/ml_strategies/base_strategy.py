"""
Base ML Betting Strategy Framework

Provides the foundation for machine learning-based betting strategies
that can analyze markets and make predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class StrategyResult:
    """Result of a betting strategy analysis."""

    market_id: str
    market_question: str
    predicted_probability: float
    confidence: float
    edge: float
    recommended_bet: str  # "YES", "NO", or "PASS"
    position_size: float
    expected_value: float
    reasoning: str
    features_used: List[str]
    model_name: str
    timestamp: datetime


class MLBettingStrategy(ABC):
    """
    Base class for machine learning betting strategies.

    Provides common functionality for data preparation, feature engineering,
    and result formatting.
    """

    def __init__(self, name: str):
        self.name = name
        self.trained = False
        self.feature_columns = []

    @abstractmethod
    def train(self, training_data: pd.DataFrame) -> None:
        """Train the ML model on historical data."""
        pass

    @abstractmethod
    def predict(self, market_data: Dict[str, Any]) -> StrategyResult:
        """Make a prediction for a specific market."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores."""
        pass

    def prepare_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare feature vector from market data.

        This is a common method that can be overridden by subclasses
        to implement custom feature engineering.
        """
        # Default features - can be extended by subclasses
        features = []

        # Volume features
        volume = market_data.get("volume", 0)
        features.extend(
            [
                volume,  # raw volume
                np.log(volume + 1),  # log volume
                volume > 1000000,  # high volume flag
            ]
        )

        # Price features
        prices = market_data.get("outcome_prices", ["0.5", "0.5"])
        if isinstance(prices, list) and len(prices) >= 2:
            yes_price = float(prices[0])
            no_price = float(prices[1])
            features.extend(
                [
                    yes_price,  # implied probability
                    abs(yes_price - 0.5),  # distance from fair odds
                    yes_price > 0.5,  # bias towards yes
                ]
            )

        # Category features (one-hot encoded)
        category = market_data.get("category", "unknown").lower()
        category_features = [
            category == "politics",
            category == "sports",
            category == "crypto",
            category == "geopolitics",
            category == "tech",
        ]
        features.extend(category_features)

        # Text features (simplified - could use embeddings)
        question = market_data.get("question", "").lower()
        text_features = [
            len(question.split()),  # word count
            "trump" in question,  # political keywords
            "bitcoin" in question or "crypto" in question,
            "super bowl" in question or "nfl" in question,
            "china" in question or "taiwan" in question,
        ]
        features.extend(text_features)

        # Liquidity features
        liquidity = market_data.get("liquidity", 0)
        features.extend(
            [
                liquidity,
                liquidity / max(volume, 1),  # liquidity ratio
            ]
        )

        self.feature_columns = [
            "volume",
            "log_volume",
            "high_volume",
            "yes_price",
            "price_distance",
            "yes_bias",
            "politics",
            "sports",
            "crypto",
            "geopolitics",
            "tech",
            "word_count",
            "trump_mention",
            "crypto_mention",
            "sports_mention",
            "china_taiwan_mention",
            "liquidity",
            "liquidity_ratio",
        ]

        return np.array(features).reshape(1, -1)

    def calculate_edge(
        self, predicted_prob: float, market_prob: float, commission: float = 0.02
    ) -> float:
        """Calculate the edge of a bet."""
        # Account for commission
        if predicted_prob > 0.5:
            # Betting YES
            cost = market_prob
            payout = 1 - market_prob
            win_prob = predicted_prob
        else:
            # Betting NO
            cost = 1 - market_prob
            payout = market_prob
            win_prob = 1 - predicted_prob

        # Expected value calculation
        ev = (win_prob * payout) - ((1 - win_prob) * cost)

        return ev / cost  # Return as percentage edge

    def kelly_criterion(
        self, edge: float, confidence: float, max_fraction: float = 0.25
    ) -> float:
        """Calculate position size using Kelly Criterion."""
        if edge <= 0:
            return 0

        # Conservative Kelly: edge * confidence * fraction
        kelly = edge * confidence * 0.5  # Half-Kelly for safety

        return min(kelly, max_fraction)  # Cap at max_fraction

    def evaluate_performance(
        self,
        predictions: List[StrategyResult],
        actual_outcomes: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Evaluate strategy performance."""
        if not predictions:
            return {"error": "No predictions to evaluate"}

        # Calculate metrics
        total_predictions = len(predictions)
        profitable_predictions = sum(1 for p in predictions if p.expected_value > 0)
        avg_edge = np.mean([p.edge for p in predictions])
        avg_confidence = np.mean([p.confidence for p in predictions])

        # Sharpe-like ratio (simplified)
        edges = [p.edge for p in predictions]
        if len(edges) > 1:
            sharpe_ratio = (
                np.mean(edges) / (np.std(edges) + 1e-6) * np.sqrt(252)
            )  # Annualized
        else:
            sharpe_ratio = 0

        # Hit rate (if actual outcomes available)
        hit_rate = None
        if actual_outcomes:
            correct_predictions = 0
            for pred in predictions:
                market_id = pred.market_id
                if market_id in actual_outcomes:
                    predicted_outcome = pred.predicted_probability > 0.5
                    actual_outcome = actual_outcomes[market_id]
                    if predicted_outcome == actual_outcome:
                        correct_predictions += 1
            hit_rate = correct_predictions / len(
                [p for p in predictions if p.market_id in actual_outcomes]
            )

        return {
            "total_predictions": total_predictions,
            "profitable_predictions": profitable_predictions,
            "profitable_percentage": profitable_predictions / total_predictions,
            "average_edge": avg_edge,
            "average_confidence": avg_confidence,
            "sharpe_ratio": sharpe_ratio,
            "hit_rate": hit_rate,
            "model_name": self.name,
        }

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        # Implementation depends on the specific ML model used
        pass

    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        # Implementation depends on the specific ML model used
        pass
