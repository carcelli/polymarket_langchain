#!/usr/bin/env python3
"""
Demo script showing the sports market focus functionality.

This demonstrates how to constrain Polymarket agents to sports markets only
using the MARKET_FOCUS environment variable.
"""

import os
from polymarket_agents.langchain.tools import _get_top_volume_markets_impl, _search_markets_db_impl

def demo_sports_focus():
    """Demonstrate sports market filtering."""

    print("🎯 Polymarket Sports Focus Demo")
    print("=" * 50)

    # Test 1: With MARKET_FOCUS=sports
    print("\n📊 Test 1: With MARKET_FOCUS=sports")
    os.environ['MARKET_FOCUS'] = 'sports'

    result = _get_top_volume_markets_impl(limit=3)
    print("Top volume markets (sports only):")
    # Extract just the questions for brevity
    import json
    data = json.loads(result)
    for market in data:
        print(f"  • {market['question']} (volume: ${market['volume']:,.0f})")

    # Test 2: Search for NFL markets
    result = _search_markets_db_impl('NFL', limit=2)
    print("\nNFL search results (sports only):")
    data = json.loads(result)
    for market in data:
        print(f"  • {market['question']}")

    # Test 3: Without MARKET_FOCUS (all categories)
    print("\n📊 Test 2: Without MARKET_FOCUS (all categories)")
    del os.environ['MARKET_FOCUS']

    result = _get_top_volume_markets_impl(limit=3)
    print("Top volume markets (all categories):")
    data = json.loads(result)
    for market in data:
        print(f"  • {market['question']} ({market['category']}, volume: ${market['volume']:,.0f})")

    # Test 4: Search for election markets (should find politics)
    result = _search_markets_db_impl('election', limit=2)
    print("\nElection search results (all categories):")
    data = json.loads(result)
    for market in data:
        print(f"  • {market['question']} ({market['category']})")

    print("\n✅ Demo complete! Use 'export MARKET_FOCUS=sports' to enable sports focus.")

if __name__ == "__main__":
    demo_sports_focus()