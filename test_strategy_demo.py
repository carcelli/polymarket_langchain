#!/usr/bin/env python3
"""
Demo script for the ML Strategy Registry system.

Tests the Fluent Python Chapter 6 style strategy registration and best_strategy selection.
"""

import sys
import os

# Set up paths
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts', 'workflows'))

from polymarket_agents.ml_strategies.registry import best_strategy, get_available_strategies

def main():
    print("🧪 Testing ML Strategy Registry System")
    print("=" * 50)

    # Show available strategies
    strategies = get_available_strategies()
    print(f"📊 Available strategies ({len(strategies)}):")
    for strategy in strategies[:5]:  # Show first 5
        print(f"   • {strategy}")
    if len(strategies) > 5:
        print(f"   ... and {len(strategies) - 5} more")
    print()

    # Test with sample market data
    market_data = {
        'id': 'test_market_123',
        'question': 'Will it rain tomorrow?',
        'volume': 500000,
        'avg_daily_volume': 100000,  # For volume spike strategy
        'price_history': [
            {'yes_price': 0.5 + i * 0.01} for i in range(35)  # 35 days of increasing prices
        ],
        'outcome_prices': [0.55, 0.45]
    }

    print("🧪 Testing best_strategy with sample market data...")
    print(f"Market: {market_data['question']}")
    print(f"Volume: ${market_data['volume']:,}")
    print(f"Current price: {market_data['outcome_prices'][0]}")
    print()

    try:
        result = best_strategy(market_data)

        print("🎯 Best Strategy Result:")
        print(f"   Selected Strategy: {result.get('selected_strategy', 'unknown')}")
        print(f"   Edge: {result.get('edge', 0):.3f}")
        print(f"   Recommendation: {result.get('recommendation', 'unknown')}")
        print(f"   Confidence: {result.get('confidence', 0):.2f}")
        print(f"   Strategies Compared: {result.get('strategies_compared', 0)}")
        print(f"   Total Strategies: {result.get('total_strategies', 0)}")

        if 'reasoning' in result:
            print(f"   Reasoning: {result['reasoning']}")

        # Show additional insights if available
        insights = result.get('additional_insights', {})
        if insights:
            print("   Additional Insights:")
            for key, value in insights.items():
                if value is not None:
                    print(f"      {key}: {value}")

    except Exception as e:
        print(f"❌ Error running best_strategy: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ Strategy registry demo completed!")

if __name__ == "__main__":
    main()