"""
Find Active Markets to Track

Queries Polymarket API to find high-volume active markets suitable for tracking.
Shows market IDs and generates tracker commands.

Usage:
    python scripts/python/find_markets_to_track.py
    python scripts/python/find_markets_to_track.py --category crypto
    python scripts/python/find_markets_to_track.py --min-volume 10000
"""

import argparse
import httpx
from typing import List, Dict


GAMMA_API = "https://gamma-api.polymarket.com"


def find_active_markets(
    category: str = None,
    min_volume: float = 1000.0,
    keywords: List[str] = None,
    limit: int = 20
) -> List[Dict]:
    """
    Find active markets suitable for tracking.
    
    Args:
        category: Optional category filter ('crypto', 'sports', 'politics')
        min_volume: Minimum trading volume
        keywords: Optional list of keywords to filter questions
        limit: Maximum number of markets to return
    
    Returns:
        List of market dictionaries
    """
    print(f"🔍 Searching Polymarket for active markets...")
    if category:
        print(f"   Category: {category}")
    if keywords:
        print(f"   Keywords: {', '.join(keywords)}")
    print(f"   Min volume: ${min_volume:,.0f}")
    print()
    
    # Query API
    params = {
        'active': True,
        'closed': False,
        'limit': 500
    }
    
    response = httpx.get(f"{GAMMA_API}/markets", params=params, timeout=30.0)
    response.raise_for_status()
    markets = response.json()
    
    print(f"📊 Received {len(markets)} markets from API")
    
    # Filter markets
    filtered = []
    for m in markets:
        # Check if has price data
        tokens = m.get('tokens', [])
        if not tokens or not tokens[0].get('price'):
            continue
        
        # Check volume
        volume = float(m.get('volume', 0))
        if volume < min_volume:
            continue
        
        # Check category
        if category:
            m_category = m.get('category', '').lower()
            if category.lower() not in m_category:
                continue
        
        # Check keywords
        if keywords:
            question = m.get('question', '').lower()
            if not any(kw.lower() in question for kw in keywords):
                continue
        
        filtered.append(m)
    
    # Sort by volume
    filtered.sort(key=lambda x: float(x.get('volume', 0)), reverse=True)
    
    return filtered[:limit]


def display_markets(markets: List[Dict]):
    """Display markets in a user-friendly format."""
    if not markets:
        print("❌ No markets found matching criteria")
        print("\n💡 Try:")
        print("   - Lower --min-volume threshold")
        print("   - Remove category filter")
        print("   - Use different keywords")
        return
    
    print(f"\n✅ Found {len(markets)} Active Markets\n")
    print("=" * 100)
    
    for i, m in enumerate(markets, 1):
        tokens = m.get('tokens', [])
        yes_price = tokens[0].get('price') if len(tokens) > 0 else 'N/A'
        no_price = tokens[1].get('price') if len(tokens) > 1 else 'N/A'
        volume = float(m.get('volume', 0))
        liquidity = float(m.get('liquidity', 0))
        category = m.get('category', 'unknown')
        
        print(f"{i}. {m['question']}")
        print(f"   ID: {m['id']}")
        print(f"   Category: {category}")
        print(f"   Volume: ${volume:,.0f} | Liquidity: ${liquidity:,.0f}")
        print(f"   Prices: Yes={yes_price}, No={no_price}")
        print()
    
    print("=" * 100)
    
    # Generate tracker command
    market_ids = ','.join(m['id'] for m in markets)
    
    print(f"\n🚀 To track these markets:")
    print(f"\npython -m polymarket_agents.services.bitcoin_tracker --market-ids {market_ids}\n")
    
    # Also show individual commands
    if len(markets) <= 5:
        print(f"Or track individually:")
        for m in markets:
            print(f"python -m polymarket_agents.services.bitcoin_tracker --market-ids {m['id']}  # {m['question'][:60]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Find active Polymarket markets to track",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find top 20 markets by volume
  python scripts/python/find_markets_to_track.py
  
  # Find crypto markets
  python scripts/python/find_markets_to_track.py --category crypto
  
  # Find Bitcoin markets
  python scripts/python/find_markets_to_track.py --keywords bitcoin btc
  
  # Find high-volume markets (>$50k)
  python scripts/python/find_markets_to_track.py --min-volume 50000
  
  # Find sports markets about specific team
  python scripts/python/find_markets_to_track.py --category sports --keywords "Lakers"
        """
    )
    
    parser.add_argument(
        '--category',
        type=str,
        help='Filter by category (e.g., crypto, sports, politics)'
    )
    
    parser.add_argument(
        '--keywords',
        nargs='+',
        help='Filter by keywords in question (e.g., --keywords bitcoin btc)'
    )
    
    parser.add_argument(
        '--min-volume',
        type=float,
        default=1000.0,
        help='Minimum trading volume (default: 1000)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Maximum number of markets to show (default: 20)'
    )
    
    args = parser.parse_args()
    
    try:
        markets = find_active_markets(
            category=args.category,
            min_volume=args.min_volume,
            keywords=args.keywords,
            limit=args.limit
        )
        
        display_markets(markets)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have internet connection and httpx installed:")
        print("   pip install httpx")


if __name__ == "__main__":
    main()
