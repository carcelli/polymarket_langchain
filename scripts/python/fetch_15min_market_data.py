"""
Fetch 15-minute Bitcoin Up or Down market data from Polymarket.

Based on URL pattern: https://polymarket.com/event/btc-updown-15m-<timestamp>

Usage:
    python scripts/python/fetch_15min_market_data.py
    python scripts/python/fetch_15min_market_data.py --slug btc-updown-15m-1769405400
"""

import argparse
import requests
from typing import Dict, Optional, List
from datetime import datetime, timezone
import json


def extract_timestamp_from_slug(slug: str) -> Optional[int]:
    """Extract Unix timestamp from market slug."""
    if 'btc-updown-15m-' in slug:
        try:
            timestamp_str = slug.split('btc-updown-15m-')[-1]
            return int(timestamp_str)
        except ValueError:
            return None
    return None


def timestamp_to_readable(timestamp: int) -> str:
    """Convert Unix timestamp to readable format."""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S UTC')


def search_market_by_time(timestamp: int) -> Optional[Dict]:
    """Search for market by timestamp in question text."""
    url = "https://gamma-api.polymarket.com/markets"
    
    # Convert timestamp to date for search
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    
    # Try different search patterns
    search_patterns = [
        f"Bitcoin Up or Down January {dt.day}",
        f"BTC Up or Down {dt.strftime('%B')} {dt.day}",
        "Bitcoin Up or Down",
    ]
    
    for pattern in search_patterns:
        params = {
            "search": pattern,
            "limit": 50
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            markets = response.json()
            
            # Filter for markets with matching time in title
            for market in markets:
                question = market.get('question', '')
                # Check if time range matches
                if f"{dt.strftime('%B')} {dt.day}" in question or f"January {dt.day}" in question:
                    # Further filter by time if possible
                    hour_str = dt.strftime('%I:%M%p').lstrip('0').replace(':00', '')
                    if hour_str.lower() in question.lower():
                        return market
            
        except Exception as e:
            print(f"Error searching with pattern '{pattern}': {e}")
            continue
    
    return None


def get_events_endpoint(slug: str) -> Optional[Dict]:
    """Try the events endpoint (may require different API)."""
    # Polymarket may have a separate events endpoint
    url = f"https://gamma-api.polymarket.com/events/{slug}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Events endpoint error: {e}")
        return None


def get_all_recent_updown_markets() -> List[Dict]:
    """Get all recent Up or Down markets."""
    url = "https://gamma-api.polymarket.com/markets"
    params = {
        "search": "Up or Down",
        "limit": 100
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        markets = response.json()
        
        # Filter for Bitcoin and recent
        btc_updown = []
        for m in markets:
            question = m.get('question', '').lower()
            if 'bitcoin' in question and 'up or down' in question:
                # Check if it's a short-duration market (has time range in title)
                if 'am' in question or 'pm' in question:
                    btc_updown.append(m)
        
        return btc_updown
    except Exception as e:
        print(f"Error: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Fetch 15-minute Bitcoin Up or Down market data"
    )
    parser.add_argument(
        '--slug',
        type=str,
        help='Market slug from URL (e.g., btc-updown-15m-1769405400)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Find all recent Up or Down markets'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔍 15-MINUTE BITCOIN MARKET DATA FETCHER")
    print("="*70 + "\n")
    
    if args.all:
        print("📡 Searching for all recent Bitcoin Up or Down markets...\n")
        markets = get_all_recent_updown_markets()
        
        if markets:
            print(f"✅ Found {len(markets)} Bitcoin Up or Down markets:\n")
            for i, m in enumerate(markets, 1):
                volume = float(m.get('volume', 0)) if m.get('volume') else 0
                status = "🟢 ACTIVE" if m.get('active') else "🔴 CLOSED"
                print(f"{i}. {status}")
                print(f"   ID: {m['id']}")
                print(f"   Q: {m['question']}")
                print(f"   Volume: ${volume:,.2f}")
                print(f"   End: {m.get('end_date_iso', 'N/A')}\n")
        else:
            print("❌ No Bitcoin Up or Down markets found")
            print("\n💡 These markets may be:")
            print("   - Not accessible via standard API search")
            print("   - Require event-specific endpoint")
            print("   - Only visible through web UI")
    
    elif args.slug:
        print(f"🔎 Analyzing slug: {args.slug}\n")
        
        # Extract timestamp
        timestamp = extract_timestamp_from_slug(args.slug)
        if timestamp:
            readable_time = timestamp_to_readable(timestamp)
            print(f"📅 Timestamp: {timestamp}")
            print(f"⏰ Time: {readable_time}\n")
        
        # Try events endpoint
        print("🔍 Trying events endpoint...")
        event_data = get_events_endpoint(args.slug)
        if event_data:
            print("✅ Event data found:")
            print(json.dumps(event_data, indent=2))
        else:
            print("❌ Events endpoint not accessible\n")
        
        # Try searching by time
        if timestamp:
            print("🔍 Searching for market by timestamp...")
            market = search_market_by_time(timestamp)
            if market:
                print(f"\n✅ Market found!")
                print(f"   ID: {market['id']}")
                print(f"   Q: {market['question']}")
                volume = float(market.get('volume', 0)) if market.get('volume') else 0
                print(f"   Volume: ${volume:,.2f}")
                print(f"   Active: {market.get('active')}")
            else:
                print("❌ Market not found via search\n")
    
    else:
        # Default: show current info
        print("📊 Analyzing Polymarket 15-minute markets...\n")
        print("From the URL: https://polymarket.com/event/btc-updown-15m-1769405400")
        print("\nURL Pattern Analysis:")
        print("  - Slug format: btc-updown-15m-<unix_timestamp>")
        print("  - Timestamp: 1769405400")
        print(f"  - Date/Time: {timestamp_to_readable(1769405400)}")
        print("  - Duration: 15 minutes")
        print("  - Resolution: Chainlink BTC/USD oracle")
        print("\n" + "-"*70)
        print("\n💡 The web page shows this market exists, but the Gamma API")
        print("   may not expose it through standard /markets endpoint.")
        print("\n🔧 Options to get market data:")
        print("   1. Try event-specific API endpoint")
        print("   2. Web scraping (browser automation)")
        print("   3. Websocket connection for real-time data")
        print("   4. Contact Polymarket for API access")
        
        print("\n📝 Run with --all to search for any accessible markets:")
        print("   python scripts/python/fetch_15min_market_data.py --all")


if __name__ == "__main__":
    main()
