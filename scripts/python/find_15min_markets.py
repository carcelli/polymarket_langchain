"""
Find 15-minute Bitcoin Up or Down market IDs on Polymarket.

Usage:
    python scripts/python/find_15min_markets.py
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime
import json


def fetch_polymarket_markets(
    search: str = "",
    active: bool = True,
    closed: bool = False,
    limit: int = 100,
    offset: int = 0
) -> List[Dict]:
    """Query Gamma API for markets."""
    url = "https://gamma-api.polymarket.com/markets"
    params = {
        "active": str(active).lower(),
        "closed": str(closed).lower(),
        "limit": limit,
        "offset": offset
    }
    
    if search:
        params["search"] = search
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"❌ API error: {e}")
        return []


def find_15min_btc_markets() -> List[Dict]:
    """Find all active 15-minute Bitcoin prediction markets."""
    print("\n" + "="*70)
    print("🔍 SEARCHING FOR 15-MINUTE BITCOIN MARKETS")
    print("="*70 + "\n")
    
    # Try multiple search patterns
    search_patterns = [
        "Bitcoin Up or Down",
        "BTC Up or Down",
        "Bitcoin 15",
        "BTC 15 minute"
    ]
    
    all_markets = []
    seen_ids = set()
    
    for pattern in search_patterns:
        print(f"🔎 Searching: '{pattern}'")
        markets = fetch_polymarket_markets(search=pattern, limit=50)
        
        for market in markets:
            market_id = market.get('id')
            if market_id and market_id not in seen_ids:
                question = market.get('question', '')
                
                # Filter for 15-minute markets
                if any(term in question.lower() for term in ['15m', '15 min', '15-min', 'fifteen min']):
                    all_markets.append(market)
                    seen_ids.add(market_id)
                    print(f"   ✅ Found: {market_id} - {question[:60]}...")
    
    print(f"\n📊 Total 15-min markets found: {len(all_markets)}")
    return all_markets


def display_market_details(markets: List[Dict]):
    """Display detailed info about found markets."""
    if not markets:
        print("\n❌ No 15-minute markets found. They may be expired or not currently listed.")
        print("\n💡 Tip: These markets are time-bound and expire quickly.")
        print("   Try checking Polymarket.com directly for active markets.")
        return
    
    print("\n" + "="*70)
    print("📋 MARKET DETAILS")
    print("="*70 + "\n")
    
    for i, market in enumerate(markets, 1):
        print(f"{i}. Market ID: {market['id']}")
        print(f"   Question: {market['question']}")
        print(f"   Active: {market.get('active', 'N/A')}")
        print(f"   Closed: {market.get('closed', 'N/A')}")
        print(f"   Volume: ${market.get('volume', 0):,.2f}")
        print(f"   End Date: {market.get('end_date_iso', 'N/A')}")
        
        # Show outcome prices if available
        if 'outcomes' in market and market['outcomes']:
            for outcome in market['outcomes']:
                price = outcome.get('price', 0)
                print(f"   {outcome.get('outcome', 'Unknown')}: {price:.3f} ({price*100:.1f}%)")
        
        print()


def export_market_ids(markets: List[Dict], output_file: str = "bitcoin_15min_market_ids.json"):
    """Export market IDs to JSON file."""
    if not markets:
        return
    
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "count": len(markets),
        "markets": [
            {
                "id": m['id'],
                "question": m['question'],
                "active": m.get('active', False),
                "end_date": m.get('end_date_iso'),
                "volume": m.get('volume', 0)
            }
            for m in markets
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Market IDs exported to: {output_file}")


def main():
    markets = find_15min_btc_markets()
    display_market_details(markets)
    
    if markets:
        export_market_ids(markets)
        
        print("\n" + "="*70)
        print("📝 QUICK REFERENCE")
        print("="*70)
        print("\nMarket IDs for use in code:")
        for market in markets[:5]:  # Show top 5
            print(f"  '{market['id']}'  # {market['question'][:50]}...")
    else:
        print("\n" + "="*70)
        print("🔄 ALTERNATIVE APPROACHES")
        print("="*70)
        print("\n1. Check Polymarket.com/events for current 15-min markets")
        print("2. Use broader search: Try 'Bitcoin' and filter manually")
        print("3. Query by date range if markets expired recently")
        print("\nExample code to search all Bitcoin markets:")
        print("\n  markets = fetch_polymarket_markets(search='Bitcoin', limit=200)")
        print("  for m in markets:")
        print("      if '15' in m['question']:")
        print("          print(m['id'], m['question'])")


if __name__ == "__main__":
    main()
