"""
Vector Signal Strategy - Demonstrating Chapter 13 Operator Overloading

Uses the enhanced Vector2d class to perform ML-style signal processing with:
- Vector dot products (@ operator) for similarity calculations
- Vector addition (+) for signal combination
- Vector scaling (*) for confidence weighting
- Proper operator overloading with NotImplemented handling

This strategy demonstrates how domain objects with operator overloading
make complex ML operations read like mathematical expressions.
"""

from typing import Dict, Any, List
from .registry import register_strategy
import math
from polymarket_agents.utils.vector import Vector2d


@register_strategy("vector_signal_similarity")
def vector_signal_similarity_strategy(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vector-based signal similarity strategy using dot products.

    Treats market signals as vectors and calculates similarity to
    known profitable patterns using vector dot products.
    """
    # Extract market features as a vector
    volume = market_data.get("volume", 0)
    avg_volume = market_data.get("avg_daily_volume", 100000)
    recent_price = market_data.get("price_history", [{}])[-1].get("yes_price", 0.5)
    price_trend = market_data.get("price_trend", 0.0)  # -1 to 1

    if volume == 0 or avg_volume == 0:
        return {
            "edge": 0.0,
            "recommendation": "HOLD",
            "confidence": 0.0,
            "reasoning": "Insufficient market data for vector analysis",
        }

    # Create market signal vector
    market_vector = Vector2d(
        volume / avg_volume,  # Normalized volume
        price_trend,  # Price momentum (-1 to 1)
    )

    # Define profitable signal patterns (learned from historical data)
    profitable_patterns = [
        Vector2d(2.5, 0.8),  # High volume + upward trend
        Vector2d(1.8, -0.6),  # Moderate volume + downward trend
        Vector2d(3.2, 0.3),  # Very high volume + slight upward
    ]

    # Calculate similarity to each profitable pattern using dot product
    similarities = []
    for pattern in profitable_patterns:
        # Dot product measures alignment between vectors
        similarity = market_vector @ pattern
        similarities.append(similarity)

    # Find best pattern match
    max_similarity = max(similarities)
    best_pattern_idx = similarities.index(max_similarity)

    # Convert similarity to edge signal
    # Normalize similarity and scale to reasonable edge range
    normalized_similarity = max_similarity / math.sqrt(
        abs(market_vector) * abs(profitable_patterns[best_pattern_idx])
    )
    edge = max(0, min(normalized_similarity * 0.05, 0.08))  # Cap at 8% edge

    # Confidence based on similarity strength
    confidence = min(abs(max_similarity) / 10.0, 1.0)

    # Determine recommendation based on best matching pattern
    pattern_descriptions = [
        "high volume upward trend",
        "moderate volume downward trend",
        "very high volume slight upward",
    ]

    if edge > 0.03:
        recommendation = (
            "BUY_YES" if profitable_patterns[best_pattern_idx].y > 0 else "BUY_NO"
        )
        reasoning = f"Strong similarity to profitable {pattern_descriptions[best_pattern_idx]} pattern"
    elif edge > 0.015:
        recommendation = "MODERATE_POSITION"
        reasoning = (
            f"Moderate similarity to {pattern_descriptions[best_pattern_idx]} pattern"
        )
    else:
        recommendation = "HOLD"
        reasoning = f"Weak signal similarity, monitoring for better alignment"

    return {
        "edge": edge,
        "recommendation": recommendation,
        "confidence": confidence,
        "reasoning": reasoning,
        "vector_similarity": max_similarity,
        "best_pattern": pattern_descriptions[best_pattern_idx],
        "market_vector": market_vector,
        "data_points": len(market_data.get("price_history", [])),
    }


@register_strategy("signal_combination")
def signal_combination_strategy(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combines multiple signal vectors using vector addition and scaling.

    Demonstrates how operator overloading makes signal combination
    read like mathematical vector operations.
    """
    # Extract multiple signal components
    momentum_signal = market_data.get("momentum", 0.0)
    volume_signal = market_data.get("volume_ratio", 1.0)
    sentiment_signal = market_data.get("sentiment", 0.0)  # -1 to 1 scale

    # Create individual signal vectors
    momentum_vec = Vector2d(momentum_signal, 0.0)  # Pure momentum
    volume_vec = Vector2d(0.0, volume_signal)  # Pure volume
    sentiment_vec = Vector2d(sentiment_signal, sentiment_signal)  # Diagonal sentiment

    # Combine signals using vector addition
    combined_signal = momentum_vec + volume_vec + sentiment_vec

    # Weight by confidence factors
    confidence_momentum = min(abs(momentum_signal), 1.0)
    confidence_volume = min(volume_signal / 2.0, 1.0)
    confidence_sentiment = market_data.get("sentiment_confidence", 0.5)

    # Scale combined signal by average confidence
    avg_confidence = (
        confidence_momentum + confidence_volume + confidence_sentiment
    ) / 3.0
    final_signal = combined_signal * avg_confidence

    # Calculate edge from signal magnitude and direction
    signal_magnitude = abs(final_signal)
    signal_direction = math.atan2(
        final_signal.y, final_signal.x
    )  # Direction in radians

    # Edge based on magnitude, with directional bias
    edge = min(signal_magnitude * 0.03, 0.06)  # Max 6% edge

    # Direction bias: favor upward trending signals
    if signal_direction > 0:  # Upward bias
        edge *= 1.2
    elif signal_direction < -0.5:  # Strong downward bias
        edge *= 0.8

    edge = min(edge, 0.08)  # Final cap

    # Determine recommendation
    if edge > 0.04:
        recommendation = "STRONG_POSITION"
        reasoning = f"Combined signals show strong {signal_direction:.1f} direction"
    elif edge > 0.02:
        recommendation = "MODERATE_POSITION"
        reasoning = f"Combined signals suggest {signal_direction:.1f} direction"
    else:
        recommendation = "HOLD"
        reasoning = "Signal combination too weak for position"

    return {
        "edge": edge,
        "recommendation": recommendation,
        "confidence": avg_confidence,
        "reasoning": reasoning,
        "combined_signal": final_signal,
        "signal_components": {
            "momentum": momentum_vec,
            "volume": volume_vec,
            "sentiment": sentiment_vec,
        },
        "signal_magnitude": signal_magnitude,
        "signal_direction": signal_direction,
    }
