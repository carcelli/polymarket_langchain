"""
Market Prediction Strategy using Random Forest

Uses ensemble learning to predict market outcomes based on historical data
and market characteristics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from .base_strategy import MLBettingStrategy, StrategyResult


class MarketPredictor(MLBettingStrategy):
    """
    Random Forest-based market prediction strategy.

    Trains on historical market data to predict the true probability
    of market outcomes, then identifies edges against current market prices.
    """

    def __init__(self, analyzer=None, n_estimators=100, max_depth=10):
        super().__init__("RandomForest_MarketPredictor", analyzer)
        self.regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False

    def train(self, training_data: pd.DataFrame) -> None:
        """
        Train the model on historical market data.

        Expected training_data columns:
        - market_id: unique market identifier
        - question: market question text
        - category: market category
        - volume: trading volume
        - outcome_prices: list of [yes_price, no_price]
        - actual_outcome: 1 for YES resolved, 0 for NO resolved
        - end_date: market end date
        """
        print(f"🏗️ Training {self.name} on {len(training_data)} markets...")

        # Prepare features
        X = []
        y_regression = []  # For predicting true probability
        y_classification = []  # For predicting binary outcome

        for _, row in training_data.iterrows():
            features = self.prepare_features(row.to_dict())
            X.append(features.flatten())

            # For regression: predict the "true" probability (we'll simulate this)
            # In reality, this would be based on historical resolution data
            market_prob = float(row['outcome_prices'][0]) if isinstance(row['outcome_prices'], list) else 0.5
            true_prob = row.get('actual_outcome', market_prob)  # Use actual if available
            y_regression.append(true_prob)

            # For classification: predict binary outcome
            y_classification.append(1 if true_prob > 0.5 else 0)

        X = np.vstack(X)
        y_regression = np.array(y_regression)
        y_classification = np.array(y_classification)

        # Split data
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
            X, y_regression, y_classification, test_size=0.2, random_state=42
        )

        # Train models
        self.regressor.fit(X_train, y_reg_train)
        self.classifier.fit(X_train, y_clf_train)

        # Evaluate
        reg_pred = self.regressor.predict(X_test)
        clf_pred = self.classifier.predict(X_test)

        reg_mse = mean_squared_error(y_reg_test, reg_pred)
        clf_acc = accuracy_score(y_clf_test, clf_pred)

        print(".4f")
        print(".1%")
        self.is_trained = True

    def predict(self, market_data: Dict[str, Any]) -> StrategyResult:
        """Make a prediction for a market."""
        if not self.is_trained:
            return StrategyResult(
                market_id=market_data.get('id', 'unknown'),
                market_question=market_data.get('question', 'Unknown market'),
                predicted_probability=0.5,
                confidence=0.0,
                edge=0.0,
                recommended_bet="PASS",
                position_size=0.0,
                expected_value=0.0,
                reasoning="Model not trained yet",
                features_used=[],
                model_name=self.name,
                timestamp=pd.Timestamp.now()
            )

        # Prepare features
        features = self.prepare_features(market_data)

        # Get predictions
        predicted_true_prob = self.regressor.predict(features)[0]
        predicted_outcome = self.classifier.predict(features)[0]

        # Current market probability
        prices = market_data.get('outcome_prices', ['0.5', '0.5'])
        market_prob = float(prices[0]) if isinstance(prices, list) and len(prices) > 0 else 0.5

        # Calculate edge
        edge = self.calculate_edge(predicted_true_prob, market_prob)

        # Confidence based on prediction certainty
        confidence = abs(predicted_true_prob - 0.5) * 2  # Scale to 0-1

        # Determine recommendation
        if abs(edge) > 0.05 and confidence > 0.6:  # 5% edge threshold, 60% confidence
            recommended_bet = "YES" if edge > 0 else "NO"
        else:
            recommended_bet = "PASS"

        # Position sizing
        position_size = self.kelly_criterion(abs(edge), confidence) if edge != 0 else 0

        # Reasoning
        reasoning = f"""
        ML Model Prediction:
        - Predicted true probability: {predicted_true_prob:.1%}
        - Current market probability: {market_prob:.1%}
        - Edge detected: {edge:.1%}
        - Confidence: {confidence:.1%}
        - Volume: ${market_data.get('volume', 0):,.0f}
        - Category: {market_data.get('category', 'unknown')}

        Recommendation: {recommended_bet} with {position_size:.1%} position size
        """.strip()

        return StrategyResult(
            market_id=market_data.get('id', 'unknown'),
            market_question=market_data.get('question', 'Unknown market'),
            predicted_probability=predicted_true_prob,
            confidence=confidence,
            edge=edge,
            recommended_bet=recommended_bet,
            position_size=position_size,
            expected_value=edge * position_size,
            reasoning=reasoning,
            features_used=self.feature_columns,
            model_name=self.name,
            timestamp=pd.Timestamp.now()
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance from the regressor."""
        if not self.is_trained:
            return {}

        importance = self.regressor.feature_importances_
        return dict(zip(self.feature_columns, importance))


class EnsemblePredictor(MarketPredictor):
    """
    Ensemble version using multiple models for improved predictions.
    """

    def __init__(self, analyzer=None):
        super().__init__(analyzer)
        self.models = {
            'rf_small': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
            'rf_medium': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=43),
            'rf_large': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=44),
        }
        self.is_trained = False

    def train(self, training_data: pd.DataFrame) -> None:
        """Train ensemble of models."""
        print(f"🏗️ Training {self.name} ensemble on {len(training_data)} markets...")

        X = []
        y = []

        for _, row in training_data.iterrows():
            features = self.prepare_features(row.to_dict())
            X.append(features.flatten())

            market_prob = float(row['outcome_prices'][0]) if isinstance(row['outcome_prices'], list) else 0.5
            true_prob = row.get('actual_outcome', market_prob)
            y.append(true_prob)

        X = np.vstack(X)
        y = np.array(y)

        # Train all models
        for name, model in self.models.items():
            model.fit(X, y)
            pred = model.predict(X)
            mse = mean_squared_error(y, pred)
            print(".4f"
        self.is_trained = True

    def predict(self, market_data: Dict[str, Any]) -> StrategyResult:
        """Make ensemble prediction."""
        if not self.is_trained:
            return super().predict(market_data)  # Use base implementation

        # Get predictions from all models
        features = self.prepare_features(market_data)
        predictions = []

        for model in self.models.values():
            pred = model.predict(features)[0]
            predictions.append(pred)

        # Ensemble prediction (weighted average)
        weights = [0.2, 0.3, 0.5]  # Favor larger models
        ensemble_pred = sum(p * w for p, w in zip(predictions, weights))

        # Use ensemble prediction in base result
        base_result = super().predict(market_data)
        base_result.predicted_probability = ensemble_pred
        base_result.model_name = f"{self.name}_Ensemble"

        return base_result
