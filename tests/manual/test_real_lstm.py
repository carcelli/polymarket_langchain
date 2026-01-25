import sys
import os
import torch
import numpy as np

# Add src and scripts/workflows to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts/workflows"))
)

from polymarket_agents.ml_strategies.lstm_probability import lstm_probability_strategy
from polymarket_agents.utils.database import get_price_stream


def test_real_lstm_strategy():
    print("Testing LSTM Strategy with REAL data...")

    # Use a known active market ID (from my previous inspection)
    # Market ID: 517310
    market_id = "517310"

    print(f"Fetching real data for market {market_id}...")
    stream = get_price_stream(market_id, days_back=30)
    print(f"Got {len(stream)} data points.")

    if not stream:
        print("Warning: No data fetched. API might be down or market inactive.")
        return

    print("Sample data point:", stream[-1])

    # Mock market data structure for the strategy, but pass the ID so it fetches real data
    market_data = {
        "id": market_id,
        "question": "Real Data Test",
        "volume": 0,  # Will be ignored by strategy which fetches its own stream
        "outcome_prices": ["0.5", "0.5"],
    }

    print("Running strategy...")
    result = lstm_probability_strategy(market_data)

    print("\nResult:")
    print(result)

    # We expect either a valid prediction or INSUFFICIENT_DATA
    if result["recommendation"] == "INSUFFICIENT_DATA":
        print("Strategy returned INSUFFICIENT_DATA (expected if history is short)")
    else:
        print(f"Prediction: {result['model_pred']:.4f}")
        assert "edge" in result

    print("Real Data LSTM test passed!")


if __name__ == "__main__":
    test_real_lstm_strategy()
