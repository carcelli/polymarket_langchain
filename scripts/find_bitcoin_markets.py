#!/usr/bin/env python3
"""Find and display Bitcoin markets from the local database."""

from polymarket_agents.memory.manager import MemoryManager

def main():
    print("\n🔍 Searching for Bitcoin markets...\n")
    
    memory = MemoryManager("data/markets.db")
    markets = memory.search_markets("bitcoin", limit=15)
    
    if not markets:
        print("❌ No Bitcoin markets found in database.")
        print("   Try running: python scripts/python/refresh_markets.py --max-events 500")
        return
    
    print(f"Found {len(markets)} Bitcoin markets:\n")
    print("=" * 100)
    
    for i, market in enumerate(markets, 1):
        question = market['question']
        volume = market.get('volume', 0)
        end_date = market.get('end_date', 'N/A')
        market_id = market['id']
        outcomes = market.get('outcomes', 'N/A')
        prices = market.get('outcome_prices', 'N/A')
        
        print(f"\n{i}. {question}")
        print(f"   ID: {market_id}")
        print(f"   Volume: ${volume:,.2f}")
        print(f"   End Date: {end_date}")
        print(f"   Outcomes: {outcomes}")
        print(f"   Prices: {prices}")
    
    print("\n" + "=" * 100)
    print("\n💡 To bet on a market:")
    print("   1. Copy the market ID from above")
    print("   2. Run: python scripts/bet_on_market.py <market_id>")
    print("\n💡 Or use the planning agent:")
    print('   python -m polymarket_agents.graph.planning_agent "Analyze Bitcoin market <market_id>"')

if __name__ == '__main__':
    main()
