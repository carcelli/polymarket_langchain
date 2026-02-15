"""
Test the ML Strategy Registry System

Tests the Fluent Python Chapter 6 style strategy registration and discovery.
"""

import pytest
from typing import Dict, Any
from polymarket_agents.ml_strategies.registry import (
    register_strategy,
    get_available_strategies,
    get_strategy,
    best_strategy,
    STRATEGIES,
)


# Test strategies for demonstration
@register_strategy("test_high_edge")
def high_edge_strategy(market_data: Dict[str, Any]) -> Dict[str, float]:
    """Always returns high edge - for testing."""
    return {
        "edge": 0.08,
        "recommendation": "BUY_YES",
        "confidence": 0.9,
        "reasoning": "Test strategy with high edge",
    }


@register_strategy("test_low_edge")
def low_edge_strategy(market_data: Dict[str, Any]) -> Dict[str, float]:
    """Always returns low edge - for testing."""
    return {
        "edge": 0.01,
        "recommendation": "HOLD",
        "confidence": 0.5,
        "reasoning": "Test strategy with low edge",
    }


@register_strategy("test_negative_edge")
def negative_edge_strategy(market_data: Dict[str, Any]) -> Dict[str, float]:
    """Returns negative edge - should be filtered out."""
    return {
        "edge": -0.02,
        "recommendation": "AVOID",
        "confidence": 0.3,
        "reasoning": "Test strategy with negative edge",
    }


class TestStrategyRegistry:
    """Test the strategy registration and selection system."""

    @pytest.fixture(autouse=True)
    def clean_strategies(self):
        """Ensure clean strategy registry for each test."""
        # Backup existing strategies
        original_strategies = STRATEGIES.copy()
        STRATEGIES.clear()

        # Register test strategies
        # Note: We re-register them because we cleared the global registry
        # The decorators ran at import time, but we just wiped their work
        STRATEGIES["test_high_edge"] = high_edge_strategy
        STRATEGIES["test_low_edge"] = low_edge_strategy
        STRATEGIES["test_negative_edge"] = negative_edge_strategy

        yield

        # Restore original strategies
        STRATEGIES.clear()
        STRATEGIES.update(original_strategies)

    def test_strategy_registration(self):
        """Test that strategies are properly registered."""
        available = get_available_strategies()
        assert "test_high_edge" in available
        assert "test_low_edge" in available
        assert "test_negative_edge" in available

    def test_strategy_retrieval(self):
        """Test getting strategies by name."""
        strategy = get_strategy("test_high_edge")
        assert callable(strategy)

        # Test calling the strategy
        result = strategy({"test": "data"})
        assert result["edge"] == 0.08
        assert result["recommendation"] == "BUY_YES"

    def test_best_strategy_selection(self):
        """Test that best_strategy picks the highest edge."""
        market_data = {"volume": 1000000, "outcome_prices": [0.6, 0.4]}

        result = best_strategy(market_data)

        # Should pick high_edge_strategy
        assert result["selected_strategy"] == "test_high_edge"
        assert result["edge"] == 0.08
        assert result["strategies_compared"] == 2  # Only positive edge strategies
        assert result["total_strategies"] == 3

    def test_best_strategy_min_edge_filter(self):
        """Test min_edge filtering."""
        market_data = {"volume": 1000000}

        # With default min_edge=0, should include low_edge
        result = best_strategy(market_data)
        assert result["selected_strategy"] == "test_high_edge"

        # With higher min_edge, should still pick high_edge
        result = best_strategy(market_data, min_edge=0.05)
        assert result["selected_strategy"] == "test_high_edge"

    def test_best_strategy_no_qualifying(self):
        """Test behavior when no strategies meet min_edge."""
        market_data = {"volume": 1000000}

        result = best_strategy(market_data, min_edge=0.1)  # Higher than all strategies

        assert "error" in result
        assert "No qualifying strategies" in result["error"]

    def test_strategy_failure_handling(self):
        """Test that failed strategies don't break the system."""

        @register_strategy("test_failing")
        def failing_strategy(market_data: Dict[str, Any]) -> Dict[str, float]:
            raise ValueError("This strategy always fails")

        market_data = {"volume": 1000000}

        # Should still work despite the failing strategy
        result = best_strategy(market_data)

        # Should still pick the high edge strategy
        assert result["selected_strategy"] == "test_high_edge"
        assert result["edge"] == 0.08


if __name__ == "__main__":
    # Demo the system
    print("🧪 Testing ML Strategy Registry...")

    print(f"📊 Available strategies: {get_available_strategies()}")

    market_data = {"volume": 1000000, "outcome_prices": [0.6, 0.4]}

    result = best_strategy(market_data)
    print(f"🎯 Best strategy result: {result}")

    print("✅ Registry tests completed!")
