#!/usr/bin/env python3
"""
Find live "Up or Down" crypto markets (5-15 minute duration).

These are ultra-short-term binary options where you predict if the price
will be higher or lower than the starting price at expiry.
"""

import requests
from datetime import datetime, timezone
import json

def find_live_updown_markets():
    """Search for active Up or Down markets."""
    
    print("\n🔍 Searching for live Up or Down markets...\n")
    
    # Query Gamma API for active markets
    url = "https://gamma-api.polymarket.com/markets"
    params = {
        'limit': 500,  # Get more markets to find short-term ones
        'closed': False,  # Only active markets
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        markets = response.json()
        
        now = datetime.now(timezone.utc)
        updown_markets = []
        
        for market in markets:
            question = market.get('question', '')
            
            # Look for "Up or Down" markets
            if 'up or down' in question.lower():
                end_date_str = market.get('end_date_iso')
                
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                        minutes_until_end = (end_date - now).total_seconds() / 60
                        
                        # Only show markets ending in the next hour
                        if 0 < minutes_until_end < 60:
                            updown_markets.append({
                                'id': market.get('condition_id'),
                                'question': question,
                                'minutes_until_end': minutes_until_end,
                                'end_date': end_date_str,
                                'tokens': market.get('tokens', []),
                                'description': market.get('description', ''),
                                'volume': market.get('volume', 0),
                            })
                    except:
                        pass
        
        if not updown_markets:
            print("❌ No live Up or Down markets found in the next hour.")
            print("\n💡 These markets appear periodically throughout the day.")
            print("   Try running this script again in a few minutes.")
            print("\n📊 Example markets from database (expired):")
            print("   - Bitcoin Up or Down - 5-minute windows")
            print("   - Ethereum Up or Down - 15-minute windows")
            print("   - Solana Up or Down")
            print("   - XRP Up or Down")
            return
        
        # Sort by time until expiry
        updown_markets.sort(key=lambda x: x['minutes_until_end'])
        
        print(f"✅ Found {len(updown_markets)} live Up or Down markets:\n")
        print("=" * 100)
        
        for i, market in enumerate(updown_markets, 1):
            mins = market['minutes_until_end']
            
            print(f"\n{i}. {market['question']}")
            print(f"   ID: {market['id']}")
            print(f"   ⏰ Expires in: {mins:.1f} minutes")
            print(f"   💰 Volume: ${market['volume']:,.2f}")
            
            # Extract token IDs if available
            tokens = market.get('tokens', [])
            if tokens and len(tokens) >= 2:
                print(f"   🎯 UP token:   {tokens[0].get('token_id', 'N/A')}")
                print(f"   🎯 DOWN token: {tokens[1].get('token_id', 'N/A')}")
        
        print("\n" + "=" * 100)
        print("\n💡 To bet on a market:")
        print("   python scripts/quick_bet.py <market_id> <0=UP|1=DOWN> <amount>")
        print("\n⏰ These markets are time-sensitive!")
        print("   Check prices and bet quickly before expiry.")
        
    except requests.RequestException as e:
        print(f"❌ Error fetching markets: {e}")
        print("\n💡 Make sure you have internet access.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == '__main__':
    find_live_updown_markets()
