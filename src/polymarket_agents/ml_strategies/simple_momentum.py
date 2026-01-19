"""
Simple Momentum Strategy

Function-based strategy that looks for price momentum signals.
Demonstrates the lightweight function approach vs. full class hierarchy.
"""

from typing import Dict, Any
from .registry import register_strategy


@register_strategy("momentum_30d")
def momentum_strategy(market_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Simple 30-day price momentum signal.

    Looks at recent price movement to identify trending markets.
    Positive momentum suggests buying the trending direction.
    """
    history = market_data.get("price_history", [])

    if len(history) < 30:
        return {
            "edge": 0.0,
            "recommendation": "HOLD",
            "confidence": 0.0,
            "reasoning": "Insufficient price history for momentum analysis"
        }

    # Get recent vs older prices
    recent_price = history[-1].get("yes_price", 0.5)
    older_price = history[-30].get("yes_price", 0.5)

    # Calculate momentum (percentage change)
    if older_price == 0:
        momentum = 0.0
    else:
        momentum = (recent_price - older_price) / older_price

    # Convert to edge signal
    edge = abs(momentum) * 5.0  # Scale momentum to edge percentage
    confidence = min(abs(momentum) * 2.0, 1.0)  # Confidence based on momentum strength

    # Determine recommendation
    if edge > 0.02:  # 2% edge threshold
        if momentum > 0:
            recommendation = "BUY_YES"
            reasoning = ".1f"
        else:
            recommendation = "BUY_NO"
            reasoning = ".1f"
    else:
        recommendation = "HOLD"
        reasoning = ".1f"

    return {
        "edge": edge,
        "recommendation": recommendation,
        "confidence": confidence,
        "reasoning": reasoning,
        "momentum": momentum,
        "data_points": len(history)
    }


@register_strategy("volume_spike")
def volume_spike_strategy(market_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Volume spike detector.

    Identifies markets with unusual trading volume that might indicate
    breaking news or significant information flow.
    """
    volume = market_data.get("volume", 0)
    avg_volume = market_data.get("avg_daily_volume", 100000)

    if avg_volume == 0:
        return {
            "edge": 0.0,
            "recommendation": "HOLD",
            "confidence": 0.0,
            "reasoning": "No volume history available"
        }

    # Calculate volume ratio
    volume_ratio = volume / avg_volume

    # Edge based on volume spike
    if volume_ratio > 3.0:
        edge = min(volume_ratio / 10.0, 0.10)  # Cap at 10% edge
        confidence = min(volume_ratio / 5.0, 1.0)
        recommendation = "MONITOR_CLOSELY"
        reasoning = ".1f"
    elif volume_ratio > 1.5:
        edge = volume_ratio / 20.0
        confidence = volume_ratio / 10.0
        recommendation = "MODERATE_INTEREST"
        reasoning = ".1f"
    else:
        edge = 0.0
        confidence = 0.0
        recommendation = "NORMAL_ACTIVITY"
        reasoning = ".1f"

    return {
        "edge": edge,
        "recommendation": recommendation,
        "confidence": confidence,
        "reasoning": reasoning,
        "volume_ratio": volume_ratio,
        "current_volume": volume,
        "avg_volume": avg_volume
    }