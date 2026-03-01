# Bitcoin 15-Minute Market Data - FOUND!

**Source:** https://polymarket.com/event/btc-updown-15m-1769405400

## Market Identifiers

```python
# Primary market identifiers for API calls
EVENT_ID = "187013"
MARKET_ID = "1263088"
CONDITION_ID = "0x0b456f836e5d83ef4a1c88802725d28efecb9f5b95525c814f2ec4b3bf0a8a68"
SLUG = "btc-updown-15m-1769405400"
QUESTION_ID = "0x2f77f592e05795f0debf7675de04c44276246b3d90215c33ddcf252ffd03f17c"
```

## Market Details

**Question:** "Bitcoin Up or Down - January 26, 12:30AM-12:45AM ET"

**Description:** This market will resolve to "Up" if the Bitcoin price at the end of the time range is greater than or equal to the price at the beginning. Otherwise, it will resolve to "Down".

**Resolution Source:** Chainlink BTC/USD data stream  
https://data.chain.link/streams/btc-usd

**Time Window:**
- Start: 2026-01-26 05:30:00 UTC (12:30 AM ET)
- End: 2026-01-26 05:45:00 UTC (12:45 AM ET)
- Duration: 15 minutes

## Market Statistics

```python
{
    "volume": 142318.18,  # $142,318
    "liquidity": 13092.72,  # $13,093
    "active": True,
    "closed": False,
    "outcomes": ["Up", "Down"],
    "outcome_prices": {
        "Up": 0.115,    # 11.5%
        "Down": 0.885   # 88.5%
    }
}
```

## API Access

### Method 1: Gamma API (Direct Market Access)

```python
import requests

def get_15min_market_data(market_id: str = "1263088"):
    """Fetch market data from Gamma API."""
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"
    response = requests.get(url)
    return response.json()

market = get_15min_market_data()
print(f"Question: {market['question']}")
print(f"Volume: ${market['volume']:,.2f}")
print(f"Outcome prices: {market.get('outcomePrices', [])}")
```

### Method 2: CLOB API (Order Book Data)

```python
def get_clob_market_data(condition_id: str):
    """Fetch order book data from CLOB API."""
    url = f"https://clob.polymarket.com/markets/{condition_id}"
    response = requests.get(url)
    return response.json()

# Use the condition_id
clob_data = get_clob_market_data("0x0b456f836e5d83ef4a1c88802725d28efecb9f5b95525c814f2ec4b3bf0a8a68")
```

### Method 3: Web Scraping (Current Approach)

```python
import requests
from bs4 import BeautifulSoup
import json

def scrape_current_15min_market():
    """Scrape the latest 15-minute market from web page."""
    from datetime import datetime, timezone, timedelta
    
    now = datetime.now(timezone.utc)
    minute = (now.minute // 15) * 15
    current_slot = now.replace(minute=minute, second=0, microsecond=0)
    timestamp = int(current_slot.timestamp())
    
    url = f"https://polymarket.com/event/btc-updown-15m-{timestamp}"
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract __NEXT_DATA__
    next_data = soup.find('script', id='__NEXT_DATA__')
    if next_data:
        data = json.loads(next_data.string)
        return data['props']['pageProps']
    
    return None
```

## Finding Active 15-Min Markets

### Pattern: Time-Based Slugs

Markets are created every 15 minutes with predictable slugs:

```python
from datetime import datetime, timezone, timedelta

def generate_15min_market_urls(num_intervals: int = 10):
    """Generate URLs for upcoming 15-minute markets."""
    now = datetime.now(timezone.utc)
    minute = (now.minute // 15) * 15
    current_slot = now.replace(minute=minute, second=0, microsecond=0)
    
    urls = []
    for i in range(num_intervals):
        slot_time = current_slot + timedelta(minutes=15 * i)
        timestamp = int(slot_time.timestamp())
        url = f"https://polymarket.com/event/btc-updown-15m-{timestamp}"
        urls.append(url)
    
    return urls

# Example: Get next 10 market URLs
upcoming_markets = generate_15min_market_urls(10)
for url in upcoming_markets:
    print(url)
```

## Key Insights

### Why These Markets Don't Show in Standard API

1. **Not in `/markets` search** - Standard search endpoint doesn't return them
2. **Event-based structure** - Organized as "events" with nested markets
3. **Short-lived** - Expire quickly and are removed from listings
4. **High-frequency** - New market every 15 minutes

### Accessing the Data

**Best approach:** Web scraping with predictable URL pattern

```python
# This works because slugs follow timestamp pattern
timestamp = 1769405400  # Start of 15-min interval
url = f"https://polymarket.com/event/btc-updown-15m-{timestamp}"
```

**Alternative:** Monitor Gamma API directly with market IDs once found

```python
# After finding market ID via scraping
CURRENT_MARKET_ID = "1263088"
api_url = f"https://gamma-api.polymarket.com/markets/{CURRENT_MARKET_ID}"
```

## Integration with Bitcoin Tracker

### Update `bitcoin_tracker.py`

```python
class BitcoinMarketTracker:
    def __init__(self):
        # Use current 15-min market
        self.market_id = self.get_current_15min_market_id()
    
    def get_current_15min_market_id(self) -> str:
        """Scrape current 15-min market ID."""
        from datetime import datetime, timezone
        import requests
        from bs4 import BeautifulSoup
        import json
        
        now = datetime.now(timezone.utc)
        minute = (now.minute // 15) * 15
        slot = now.replace(minute=minute, second=0, microsecond=0)
        timestamp = int(slot.timestamp())
        
        url = f"https://polymarket.com/event/btc-updown-15m-{timestamp}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            next_data = soup.find('script', id='__NEXT_DATA__')
            if next_data:
                data = json.loads(next_data.string)
                page_props = data['props']['pageProps']
                
                # Navigate to market data
                queries = page_props['dehydratedState']['queries']
                for query in queries:
                    if 'data' in query['state'] and 'markets' in query['state']['data']:
                        markets = query['state']['data']['markets']
                        if markets:
                            return markets[0]['id']
        except Exception as e:
            print(f"Error fetching 15-min market: {e}")
            return "574073"  # Fallback to long-term BTC market
        
        return "574073"  # Fallback
```

## Example: Fetch Current Market

```python
import requests
from datetime import datetime, timezone

def fetch_current_15min_market():
    """Fetch the current active 15-minute Bitcoin market."""
    now = datetime.now(timezone.utc)
    minute = (now.minute // 15) * 15
    slot = now.replace(minute=minute, second=0, microsecond=0)
    
    # Try current slot
    market_id = scrape_market_id_from_timestamp(int(slot.timestamp()))
    
    if market_id:
        # Fetch data from Gamma API
        url = f"https://gamma-api.polymarket.com/markets/{market_id}"
        response = requests.get(url)
        data = response.json()
        
        return {
            'market_id': market_id,
            'question': data['question'],
            'volume': float(data['volume']),
            'yes_price': float(data.get('outcomePrices', [0.5, 0.5])[0]),
            'no_price': float(data.get('outcomePrices', [0.5, 0.5])[1]),
            'active': data['active'],
            'end_date': data['endDate']
        }
    
    return None

# Usage
market = fetch_current_15min_market()
if market:
    print(f"Market ID: {market['market_id']}")
    print(f"Volume: ${market['volume']:,.2f}")
    print(f"Up probability: {market['yes_price']*100:.1f}%")
```

## Summary

✅ **Market ID Found:** 1263088  
✅ **Condition ID Found:** 0x0b456f836e5d83ef4a1c88802725d28efecb9f5b95525c814f2ec4b3bf0a8a68  
✅ **Access Method:** Web scraping + predictable URL pattern  
✅ **Volume:** $142,318 (highly liquid)  
✅ **Update Frequency:** New market every 15 minutes  

**Next Steps:**
1. Implement market ID scraper in `bitcoin_tracker.py`
2. Poll every 15 minutes for new market
3. Store historical market IDs for backtesting
4. Use Gamma API for real-time price updates once ID is known
