#!/usr/bin/env python3
"""
Market Prediction Neural Network Example

Demonstrates using neural networks to predict market movements
based on Polymarket data and crowd wisdom features.

This example shows how to:
1. Extract features from Polymarket markets
2. Train a neural network for prediction
3. Evaluate performance
4. Integrate with existing agent workflow
"""

import numpy as np
from typing import List, Dict, Any

from polymarket_agents.ml_foundations import NeuralNetwork, scale_to_01
from polymarket_agents.langchain.agent import extract_market_probability


class MarketPredictor:
    """
    Neural network for predicting market probability movements.

    Uses features extracted from Polymarket data to predict whether
    a market's probability will rise or fall in the next period.
    """

    def __init__(self, n_hidden: int = 8, learning_rate: float = 0.1):
        # Features: [current_prob, volume, spread, age_days, liquidity]
        self.nn = NeuralNetwork(
            n_inputs=5,
            n_hidden=n_hidden,
            n_outputs=1,  # Binary: will rise (1) or fall (0)
            lr=learning_rate,
            use_softmax=False,
            random_seed=42,
        )

    def extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract numerical features from market data.

        Args:
            market_data: Dictionary with market information

        Returns:
            Feature vector [current_prob, volume, spread, age, liquidity]
        """
        # Extract basic features
        current_prob = market_data.get("current_probability", 0.5)
        volume = market_data.get("volume", 0)
        spread = market_data.get("spread", 0.1)
        age_days = market_data.get("age_days", 30)
        liquidity = market_data.get("liquidity", 0.5)

        features = np.array([current_prob, volume, spread, age_days, liquidity])

        # Scale features appropriately
        # Volume: log scale to handle large ranges
        features[1] = np.log1p(features[1]) if features[1] > 0 else 0

        # Age: normalize to [0, 1]
        features[3] = min(features[3] / 365, 1.0)  # Cap at 1 year

        return features

    def train_on_historical_data(
        self, market_history: List[Dict[str, Any]], epochs: int = 1000
    ):
        """
        Train the network on historical market data.

        Args:
            market_history: List of market snapshots with future outcomes
            epochs: Number of training epochs
        """
        features_list = []
        targets_list = []

        for market_snapshot in market_history:
            features = self.extract_features(market_snapshot)
            features_list.append(features)

            # Target: Did probability increase? (1=yes, 0=no)
            future_prob = market_snapshot.get(
                "future_probability", market_snapshot["current_probability"]
            )
            current_prob = market_snapshot["current_probability"]

            # Small threshold to avoid noise
            target = 1.0 if future_prob > current_prob + 0.02 else 0.0
            targets_list.append([target])

        X = np.array(features_list)
        y = np.array(targets_list)

        print(f"Training on {len(X)} market examples...")
        losses = self.nn.batch_train(X, y, epochs=epochs, verbose=True)

        print(f"✅ Training complete. Final loss: {losses[-1]:.4f}")
        return losses

    def predict_movement(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict whether market probability will rise or fall.

        Args:
            market_data: Current market data

        Returns:
            Prediction results with confidence
        """
        features = self.extract_features(market_data)
        prediction = self.nn.predict(features)[0]

        return {
            "prediction": "RISE" if prediction > 0.5 else "FALL",
            "confidence": abs(prediction - 0.5) * 2,  # Scale to [0, 1]
            "raw_probability": prediction,
            "features": features.tolist(),
        }


def create_sample_training_data() -> List[Dict[str, Any]]:
    """
    Create sample training data for demonstration.
    In practice, this would come from historical Polymarket data.
    """
    return [
        # High volume, tight spread, moderate probability -> likely to rise
        {
            "current_probability": 0.55,
            "volume": 10000,
            "spread": 0.02,
            "age_days": 15,
            "liquidity": 0.8,
            "future_probability": 0.62,  # Did rise
        },
        # Low volume, wide spread, high probability -> likely to fall
        {
            "current_probability": 0.75,
            "volume": 500,
            "spread": 0.08,
            "age_days": 5,
            "liquidity": 0.3,
            "future_probability": 0.68,  # Did fall
        },
        # Moderate conditions, low probability -> slight rise
        {
            "current_probability": 0.45,
            "volume": 2000,
            "spread": 0.05,
            "age_days": 25,
            "liquidity": 0.6,
            "future_probability": 0.48,  # Slight rise
        },
        # More training examples...
        {
            "current_probability": 0.62,
            "volume": 8000,
            "spread": 0.03,
            "age_days": 20,
            "liquidity": 0.7,
            "future_probability": 0.65,
        },
        {
            "current_probability": 0.38,
            "volume": 300,
            "spread": 0.12,
            "age_days": 3,
            "liquidity": 0.2,
            "future_probability": 0.35,
        },
    ]


def demo_market_prediction():
    """Demonstrate market prediction neural network."""
    print("🧠 Market Prediction Neural Network Demo")
    print("=" * 50)

    # Create and train predictor
    predictor = MarketPredictor(n_hidden=8, learning_rate=0.1)

    # Get training data
    training_data = create_sample_training_data()

    # Train the network
    print("\n📚 Training neural network on market data...")
    losses = predictor.train_on_historical_data(training_data, epochs=500)

    # Test on new market data
    print("\n🎯 Testing predictions on new markets...")

    test_markets = [
        {
            "current_probability": 0.58,
            "volume": 7500,
            "spread": 0.04,
            "age_days": 18,
            "liquidity": 0.75,
            "description": "Tech stock earnings",
        },
        {
            "current_probability": 0.42,
            "volume": 800,
            "spread": 0.09,
            "age_days": 7,
            "liquidity": 0.4,
            "description": "Political event outcome",
        },
    ]

    for market in test_markets:
        prediction = predictor.predict_movement(market)
        print(f"\n📊 Market: {market['description']}")
        print(f"   Current volume: ${prediction.get('volume', 0):.1f}")
        print(
            f"   Prediction: {prediction['prediction']} ({prediction.get('confidence', 0):.1%} confidence)"
        )
        print(f"   Raw prob: {prediction['raw_probability']:.3f}")
        print(f"   Features: {prediction['features']}")

    print("\n✨ Demo complete! This shows how neural networks can learn from")
    print("   Polymarket data to predict market movements.")


def integrate_with_agents():
    """
    Example of integrating NN predictions with LangChain agents.

    This shows how the trained network can provide signals to
    trading agents or analysis workflows.
    """
    print("\n🤖 Integration with Polymarket Agents")
    print("=" * 40)

    predictor = MarketPredictor()
    training_data = create_sample_training_data()
    predictor.train_on_historical_data(training_data, epochs=200)

    # Example market for analysis
    market_data = {
        "current_probability": 0.51,
        "volume": 5000,
        "spread": 0.06,
        "age_days": 12,
        "liquidity": 0.65,
    }

    # Get NN prediction
    nn_prediction = predictor.predict_movement(market_data)

    print("🧠 Neural Network Prediction:")
    print(f"   Direction: {nn_prediction['prediction']}")
    print(f"   Probability: {nn_prediction.get('raw_probability', 0):.1%}")
    print(f"   Confidence: {nn_prediction['confidence']:.2f}")

    print("\n🔄 Integration Points:")
    print("   • Feed NN signals into trading agents")
    print("   • Combine with crowd wisdom analysis")
    print("   • Use for risk assessment workflows")
    print("   • Provide features for larger ML models")


if __name__ == "__main__":
    demo_market_prediction()
    integrate_with_agents()

    print("\n" + "=" * 50)
    print("🚀 Next Steps:")
    print("1. Train on real Polymarket historical data")
    print("2. Integrate with live agent workflows")
    print("3. Add more sophisticated features")
    print("4. Implement model validation and backtesting")
    print("=" * 50)
