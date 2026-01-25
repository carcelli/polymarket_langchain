import sys
import os
import torch
import numpy as np

# Add src and scripts/workflows to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts/workflows"))
)

from polymarket_agents.ml_strategies.lstm_probability import (
    lstm_probability_strategy,
    PriceLSTM,
)
from polymarket_agents.utils.database import get_price_stream


def test_lstm_strategy():
    print("Testing LSTM Strategy...")

    # Check if torch is available
    if not torch.cuda.is_available():
        print("Running on CPU")

    # Mock market data
    market_id = "test_market_123"
    market_data = {
        "id": market_id,
        "question": "Will LSTM pass the test?",
        "volume": 50000,
        "outcome_prices": ["0.5", "0.5"],
    }

    # Run strategy
    print(f"Running strategy for market {market_id}...")
    result = lstm_probability_strategy(market_data)

    print("\nResult:")
    print(result)

    assert "edge" in result
    assert "recommendation" in result

    # Check if we got a prediction (requires sufficient historical data)
    if result["recommendation"] == "INSUFFICIENT_DATA":
        print("✓ Correctly handled insufficient data case")
        assert "confidence" in result
        assert result["confidence"] == 0.0
        print("LSTM Strategy test passed (insufficient data path)!")
    else:
        # If we have a prediction, validate it
        assert "model_pred" in result
        pred = result["model_pred"]
        print(f"Prediction: {pred:.4f}")

        # Basic sanity checks
        assert 0.0 <= pred <= 1.0, "Prediction out of probability bounds"
        print("LSTM Strategy test passed (prediction path)!")


if __name__ == "__main__":
    test_lstm_strategy()
