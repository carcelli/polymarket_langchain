"""
Scrape 15-minute Bitcoin Up or Down market data from Polymarket web page.

Since the Gamma API doesn't expose these markets, we need to scrape the page.

Usage:
    python scripts/python/scrape_15min_market.py
    python scripts/python/scrape_15min_market.py --url "https://polymarket.com/event/btc-updown-15m-1769405400"
"""

import argparse
import requests
import json
import re
from typing import Dict, Optional, List
from bs4 import BeautifulSoup


def scrape_polymarket_page(url: str) -> Optional[Dict]:
    """
    Scrape market data from Polymarket page.
    
    The page likely contains JSON-LD structured data or embedded market info.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        print(f"🌐 Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Method 1: Look for JSON-LD structured data
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        if json_ld_scripts:
            print(f"   Found {len(json_ld_scripts)} JSON-LD scripts")
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    print(f"   JSON-LD data: {json.dumps(data, indent=2)[:200]}...")
                except:
                    pass
        
        # Method 2: Look for __NEXT_DATA__ (Next.js apps often use this)
        next_data_script = soup.find('script', id='__NEXT_DATA__')
        if next_data_script:
            print("   ✅ Found __NEXT_DATA__ script")
            try:
                next_data = json.loads(next_data_script.string)
                
                # Extract market data from Next.js props
                page_props = next_data.get('props', {}).get('pageProps', {})
                
                if page_props:
                    print(f"   📦 Page props keys: {list(page_props.keys())}")
                    
                    # Look for market/event data
                    market_data = page_props.get('market') or page_props.get('event') or page_props.get('data')
                    
                    if market_data:
                        return market_data
                    else:
                        # Save full data for inspection
                        with open('polymarket_page_data.json', 'w') as f:
                            json.dump(page_props, f, indent=2)
                        print("   💾 Full page props saved to: polymarket_page_data.json")
                        return page_props
            
            except json.JSONDecodeError as e:
                print(f"   ❌ Error parsing __NEXT_DATA__: {e}")
        
        # Method 3: Look for inline script with market ID
        all_scripts = soup.find_all('script')
        print(f"   Found {len(all_scripts)} total script tags")
        
        for script in all_scripts:
            if script.string and 'market' in script.string.lower():
                # Look for market ID patterns
                market_id_matches = re.findall(r'"id"\s*:\s*"(\d+)"', script.string)
                if market_id_matches:
                    print(f"   🎯 Found potential market IDs: {market_id_matches}")
                
                # Look for condition_id (CLOB identifier)
                condition_id_matches = re.findall(r'"condition_id"\s*:\s*"(0x[a-fA-F0-9]+)"', script.string)
                if condition_id_matches:
                    print(f"   🎯 Found condition IDs: {condition_id_matches}")
        
        # Method 4: Extract from meta tags
        og_url = soup.find('meta', property='og:url')
        if og_url:
            print(f"   🔗 OG URL: {og_url.get('content')}")
        
        og_title = soup.find('meta', property='og:title')
        if og_title:
            print(f"   📰 Title: {og_title.get('content')}")
        
        return None
        
    except Exception as e:
        print(f"❌ Error scraping page: {e}")
        return None


def extract_market_from_clob_api(condition_id: str) -> Optional[Dict]:
    """
    Try to fetch market data from CLOB API using condition_id.
    
    The CLOB (Central Limit Order Book) is Polymarket's trading API.
    """
    base_url = "https://clob.polymarket.com"
    
    try:
        # Try markets endpoint
        url = f"{base_url}/markets/{condition_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"CLOB API error: {e}")
        return None


def get_current_15min_markets() -> List[str]:
    """
    Generate URLs for current 15-minute market windows.
    
    Markets are created every 15 minutes, so we can predict the slugs.
    """
    from datetime import datetime, timedelta, timezone
    
    now = datetime.now(timezone.utc)
    
    # Round down to nearest 15-minute interval
    minute = (now.minute // 15) * 15
    current_slot = now.replace(minute=minute, second=0, microsecond=0)
    
    # Generate slugs for current and next few intervals
    urls = []
    for i in range(5):  # Current + next 4 intervals
        slot_time = current_slot + timedelta(minutes=15 * i)
        timestamp = int(slot_time.timestamp())
        slug = f"btc-updown-15m-{timestamp}"
        url = f"https://polymarket.com/event/{slug}"
        urls.append(url)
    
    return urls


def main():
    parser = argparse.ArgumentParser(
        description="Scrape 15-minute Bitcoin market data from Polymarket"
    )
    parser.add_argument(
        '--url',
        type=str,
        help='Full Polymarket event URL'
    )
    parser.add_argument(
        '--current',
        action='store_true',
        help='Try to scrape current 15-minute market'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🕷️  POLYMARKET 15-MINUTE MARKET SCRAPER")
    print("="*70 + "\n")
    
    if args.current:
        print("⏰ Generating URLs for current 15-minute markets...\n")
        urls = get_current_15min_markets()
        
        for i, url in enumerate(urls, 1):
            print(f"\n{i}. {url}")
            market_data = scrape_polymarket_page(url)
            
            if market_data:
                print(f"\n✅ Successfully scraped market {i}")
                break
            else:
                print(f"❌ Failed to scrape market {i}")
    
    elif args.url:
        market_data = scrape_polymarket_page(args.url)
        
        if market_data:
            print("\n✅ Market data extracted!")
            
            # Try to find market ID
            if isinstance(market_data, dict):
                market_id = market_data.get('id') or market_data.get('market_id')
                if market_id:
                    print(f"\n🎯 MARKET ID: {market_id}")
                    print(f"\n💾 Use this in your code:")
                    print(f"   MARKET_ID = '{market_id}'")
                
                # Show other useful fields
                for key in ['question', 'volume', 'end_date_iso', 'condition_id']:
                    if key in market_data:
                        print(f"   {key}: {market_data[key]}")
        else:
            print("\n❌ Could not extract market data")
            print("\n💡 The page data has been saved to polymarket_page_data.json")
            print("   Inspect this file to find the market ID manually")
    
    else:
        # Default: Show instructions
        print("📝 Usage examples:")
        print("\n1. Scrape a specific market:")
        print('   python scripts/python/scrape_15min_market.py --url "https://polymarket.com/event/btc-updown-15m-1769405400"')
        print("\n2. Scrape current 15-minute market:")
        print("   python scripts/python/scrape_15min_market.py --current")
        print("\n" + "="*70)
        print("\n📊 Known 15-minute market pattern:")
        print("   URL: https://polymarket.com/event/btc-updown-15m-<timestamp>")
        print("   Timestamp: Unix time at start of 15-min interval")
        print("   Duration: 15 minutes")
        print("   Resolution: Chainlink BTC/USD oracle")


if __name__ == "__main__":
    main()
