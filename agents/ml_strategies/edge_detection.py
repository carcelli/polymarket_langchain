"""
Edge Detection Strategy using Neural Networks

Uses deep learning to identify market inefficiencies and edge opportunities
that traditional analysis might miss.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. EdgeDetector will use simplified approach.")

from .base_strategy import MLBettingStrategy, StrategyResult


class EdgeDetector(MLBettingStrategy):
    """
    Neural network-based edge detection strategy.

    Uses deep learning to identify subtle market inefficiencies
    and predict when markets are mispriced.
    """

    def __init__(self, analyzer=None, hidden_layers=[64, 32], learning_rate=0.001):
        super().__init__("NeuralNetwork_EdgeDetector", analyzer)

        if not TENSORFLOW_AVAILABLE:
            print("⚠️ Using simplified edge detection without neural networks")
            self.model = None
        else:
            # Build neural network
            self.model = keras.Sequential([
                keras.layers.Dense(hidden_layers[0], activation='relu', input_shape=(None,)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(hidden_layers[1], activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1, activation='sigmoid')  # Output: edge probability
            ])

            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Enhanced feature engineering for edge detection."""
        # Base features
        base_features = super().prepare_features(market_data).flatten()

        # Additional edge-specific features
        volume = market_data.get('volume', 0)
        prices = market_data.get('outcome_prices', ['0.5', '0.5'])
        yes_price = float(prices[0]) if isinstance(prices, list) and len(prices) > 0 else 0.5

        # Market microstructure features
        edge_features = [
            volume / max(yes_price, 0.01),  # Volume per probability point
            abs(yes_price - 0.5) / max(yes_price, 0.01),  # Relative distance from fair odds
            np.log(volume + 1) / np.log(1000000 + 1),  # Normalized log volume
            yes_price * (1 - yes_price),  # Information entropy
            1 / (abs(yes_price - 0.5) + 0.01),  # Clustering around 0.5 (lower = more clustered)
        ]

        # Combine features
        all_features = np.concatenate([base_features, edge_features])

        # Update feature columns list
        self.feature_columns = [
            'volume', 'log_volume', 'high_volume',
            'yes_price', 'price_distance', 'yes_bias',
            'politics', 'sports', 'crypto', 'geopolitics', 'tech',
            'word_count', 'trump_mention', 'crypto_mention', 'sports_mention', 'china_taiwan_mention',
            'liquidity', 'liquidity_ratio',
            'volume_per_prob', 'relative_distance', 'norm_log_volume', 'entropy', 'clustering'
        ]

        return all_features.reshape(1, -1)

    def train(self, training_data: pd.DataFrame) -> None:
        """
        Train the neural network on edge detection patterns.
        """
        print(f"🧠 Training {self.name} on {len(training_data)} markets...")

        # Prepare data
        X = []
        y = []

        for _, row in training_data.iterrows():
            features = self.prepare_features(row.to_dict())
            X.append(features.flatten())

            # Label: 1 if this market had an edge opportunity, 0 otherwise
            # This is a simplified approach - in practice, you'd have historical edge data
            market_prob = float(row['outcome_prices'][0]) if isinstance(row['outcome_prices'], list) else 0.5
            volume = row.get('volume', 0)

            # Simple heuristic: high volume + extreme probabilities = potential edge
            had_edge = (volume > 500000 and (market_prob < 0.3 or market_prob > 0.7))
            y.append(1 if had_edge else 0)

        X = np.vstack(X)
        y = np.array(y)

        print(f"   Edge opportunities found: {sum(y)}/{len(y)} ({sum(y)/len(y):.1%})")

        if TENSORFLOW_AVAILABLE and self.model is not None:
            # Split and scale data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            history = self.model.fit(
                X_train_scaled, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0,
                callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
            )

            # Evaluate
            test_loss, test_acc = self.model.evaluate(X_test_scaled, y_test, verbose=0)
            print(".4f"        else:
            print("   Using simplified edge detection (no neural network)")

        self.is_trained = True

    def predict(self, market_data: Dict[str, Any]) -> StrategyResult:
        """Predict if this market has an edge opportunity."""
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
        features_scaled = self.scaler.transform(features) if hasattr(self.scaler, 'transform') else features

        # Get edge probability
        if self.model is not None:
            edge_probability = self.model.predict(features_scaled, verbose=0)[0][0]
        else:
            # Simplified approach without neural network
            volume = market_data.get('volume', 0)
            prices = market_data.get('outcome_prices', ['0.5', '0.5'])
            yes_price = float(prices[0]) if isinstance(prices, list) and len(prices) > 0 else 0.5

            # Simple edge detection heuristic
            volume_score = min(volume / 1000000, 1.0)  # Normalize volume
            extremity_score = 1 - (abs(yes_price - 0.5) * 2)  # Higher when closer to 0.5
            edge_probability = (volume_score * 0.7) + (extremity_score * 0.3)

        # Current market data
        prices = market_data.get('outcome_prices', ['0.5', '0.5'])
        market_prob = float(prices[0]) if isinstance(prices, list) and len(prices) > 0 else 0.5
        volume = market_data.get('volume', 0)

        # Determine recommendation based on edge probability
        confidence = edge_probability

        if edge_probability > 0.7 and volume > 100000:
            # High confidence edge - make a directional bet
            recommended_bet = "YES" if market_prob < 0.5 else "NO"
            edge = abs(market_prob - 0.5) * 2  # Scale edge based on distance from fair odds
        elif edge_probability > 0.5:
            # Moderate edge - smaller position
            recommended_bet = "PASS"  # For now, be conservative
            edge = 0
        else:
            # No edge detected
            recommended_bet = "PASS"
            edge = 0

        position_size = self.kelly_criterion(edge, confidence) if edge > 0 else 0

        reasoning = f"""
        Neural Network Edge Detection:
        - Edge probability: {edge_probability:.1%}
        - Market probability: {market_prob:.1%}
        - Volume: ${volume:,.0f}
        - Confidence: {confidence:.1%}

        {'High-confidence edge detected' if edge_probability > 0.7 else 'Moderate edge signal' if edge_probability > 0.5 else 'No significant edge detected'}
        """.strip()

        return StrategyResult(
            market_id=market_data.get('id', 'unknown'),
            market_question=market_data.get('question', 'Unknown market'),
            predicted_probability=market_prob,  # Keep market prob for consistency
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
        """Return feature importance (simplified for neural networks)."""
        if not self.is_trained:
            return {}

        # For neural networks, we can't easily get feature importance
        # Return a placeholder based on feature engineering intuition
        return {
            'volume_per_prob': 0.15,
            'relative_distance': 0.12,
            'entropy': 0.10,
            'clustering': 0.08,
            'volume': 0.10,
            'yes_price': 0.08,
            'liquidity_ratio': 0.06,
            'crypto_mention': 0.05,
            'politics': 0.04,
            'sports': 0.03,
        }
