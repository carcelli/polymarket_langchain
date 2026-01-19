#!/usr/bin/env python3
"""
Continuously monitor for new Up or Down markets.

Run this script to watch for new ultra-short-term crypto markets and
get notified when they appear.
"""

import requests
import time
from datetime import datetime, timezone
import json

def get_updown_markets():
    """Fetch current Up or Down markets."""
    url = "https://gamma-api.polymarket.com/markets"
    params = {'limit': 500, 'closed': False}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        markets = response.json()
        
        now = datetime.now(timezone.utc)
        updown_markets = []
        
        for market in markets:
            question = market.get('question', '')
            
            if 'up or down' in question.lower():
                end_date_str = market.get('end_date_iso')
                
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                        minutes_until_end = (end_date - now).total_seconds() / 60
                        
                        if 0 < minutes_until_end < 60:
                            updown_markets.append({
                                'id': market.get('condition_id'),
                                'question': question,
                                'minutes_until_end': minutes_until_end,
                                'end_date': end_date_str,
                                'tokens': market.get('tokens', []),
                                'volume': market.get('volume', 0),
                            })
                    except:
                        pass
        
        return updown_markets
        
    except Exception as e:
        print(f"Error fetching markets: {e}")
        return []

def monitor_markets(interval_seconds=30):
    """Monitor for new Up or Down markets."""
    
    print("=" * 80)
    print("🔍 UP/DOWN MARKET MONITOR")
    print("=" * 80)
    print(f"\n⏰ Checking every {interval_seconds} seconds for new markets...")
    print("💡 Press Ctrl+C to stop\n")
    
    seen_markets = set()
    check_count = 0
    
    try:
        while True:
            check_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            markets = get_updown_markets()
            
            if markets:
                # Check for new markets
                new_markets = [m for m in markets if m['id'] not in seen_markets]
                
                if new_markets:
                    print(f"\n🚨 [{current_time}] NEW MARKETS FOUND!\n")
                    
                    for market in new_markets:
                        print(f"✅ {market['question']}")
                        print(f"   ID: {market['id']}")
                        print(f"   ⏰ Expires in: {market['minutes_until_end']:.1f} minutes")
                        print(f"   💰 Volume: ${market['volume']:,.2f}")
                        
                        tokens = market.get('tokens', [])
                        if tokens and len(tokens) >= 2:
                            print(f"   🎯 UP:   {tokens[0].get('token_id', 'N/A')}")
                            print(f"   🎯 DOWN: {tokens[1].get('token_id', 'N/A')}")
                        print()
                        
                        seen_markets.add(market['id'])
                    
                    print(f"💡 To bet: python scripts/quick_bet_updown.py <market_id> <UP|DOWN> <amount>\n")
                
                elif check_count % 10 == 0:  # Status update every 10 checks
                    print(f"[{current_time}] Watching... ({len(markets)} active markets)")
            
            else:
                if check_count % 10 == 0:
                    print(f"[{current_time}] No Up/Down markets currently available")
            
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.")
        print(f"📊 Watched {check_count} times, found {len(seen_markets)} unique markets")

if __name__ == '__main__':
    import sys
    
    interval = 30  # Default: check every 30 seconds
    
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except:
            print("Usage: python scripts/monitor_updown_markets.py [interval_seconds]")
            sys.exit(1)
    
    monitor_markets(interval)
