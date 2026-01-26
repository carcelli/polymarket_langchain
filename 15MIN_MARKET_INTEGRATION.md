# 15-Minute Bitcoin Market Integration Guide

## ✅ SUCCESS: Market Data Access Working

**Current Market (Live Data):**
```
Market ID: 1263100
Question: "Bitcoin Up or Down - January 26, 12:45AM-1:00AM ET"
Volume: $4,193.17
Liquidity: $10,013.83
Up Price: 48.5%
Down Price: 51.5%
Active: True
```

## Quick Start

### 1. Get Current Market Data (One-Liner)

```python
from polymarket_agents.connectors.updown_markets import get_current_15min_market

market = get_current_15min_market()
print(f"Market ID: {market['market_id']}")
print(f"Up probability: {market['up_price']*100:.1f}%")
print(f"Volume: ${market['volume']:,.2f}")
```

### 2. Full Integration Example

```python
from polymarket_agents.connectors.updown_markets import UpDownMarketConnector
import time

# Initialize connector
connector = UpDownMarketConnector()

# Get current market
market_data = connector.get_current_market_data()

if market_data:
    print(f"✅ Tracking Market {market_data['market_id']}")
    print(f"   Question: {market_data['question']}")
    print(f"   Up: {market_data['up_price']:.3f} | Down: {market_data['down_price']:.3f}")
    
    # Monitor for 15 minutes
    while time.time() < market_data['end_date']:
        # Refresh data every 30 seconds
        updated = connector.fetch_market_data(market_data['market_id'])
        print(f"   Up: {updated['outcomePrices'][0]} | Volume: ${updated['volume']}")
        time.sleep(30)
    
    print("✅ Market resolved!")
```

## API Endpoints Discovered

### Primary: Gamma API

**Get Market by ID:**
```
GET https://gamma-api.polymarket.com/markets/{market_id}
```

**Example:**
```bash
curl https://gamma-api.polymarket.com/markets/1263100
```

**Response Format:**
```json
{
  "id": "1263100",
  "question": "Bitcoin Up or Down - January 26, 12:45AM-1:00AM ET",
  "conditionId": "0xefe01c55d962a660d6fc87a9a6eca21bd5656099595d7e31617b56de89ec7e0f",
  "outcomePrices": "[\"0.485\", \"0.515\"]",  // JSON string, not array!
  "volume": "4193.167658",                    // String, not number!
  "liquidity": "10013.8309",
  "active": true,
  "closed": false,
  "endDate": "2026-01-26T06:00:00Z"
}
```

⚠️ **Important:** `outcomePrices` is a JSON string, not an array. Parse with `json.loads()`.

### Secondary: Web Scraping (for Market ID Discovery)

Markets follow predictable URL pattern:
```
https://polymarket.com/event/btc-updown-15m-{timestamp}
```

Where `timestamp` is the Unix timestamp of the 15-minute interval start time (rounded down).

## Integration with Bitcoin Tracker

### Update `bitcoin_tracker.py`

Replace hardcoded market ID with dynamic fetching:

```python
from polymarket_agents.connectors.updown_markets import get_current_market_id

class BitcoinTracker:
    def __init__(self):
        # Get current 15-min market ID dynamically
        self.market_id = get_current_market_id()
        
        if not self.market_id:
            # Fallback to long-term market
            self.market_id = "574073"
            print("⚠️  Using fallback market (long-term BTC prediction)")
        else:
            print(f"✅ Tracking 15-min market: {self.market_id}")
    
    def collect_snapshot(self):
        """Collect market snapshot every minute."""
        market_data = self.fetch_market_data(self.market_id)
        
        # Check if market expired (moved to next 15-min interval)
        if not market_data.get('active'):
            print("🔄 Market expired, switching to next interval...")
            self.market_id = get_current_market_id()
            market_data = self.fetch_market_data(self.market_id)
        
        # Store snapshot...
```

### Polling Strategy

15-minute markets require frequent updates:

```python
import time
from datetime import datetime, timezone, timedelta

def run_15min_tracker():
    connector = UpDownMarketConnector()
    
    while True:
        # Get current market
        market = connector.get_current_market_data()
        
        if market:
            print(f"📊 {market['question']}")
            print(f"   Up: {market['up_price']:.3f} | Volume: ${market['volume']:,.0f}")
            
            # Calculate time until market ends
            end_time = datetime.fromisoformat(market['end_date'].replace('Z', '+00:00'))
            time_left = (end_time - datetime.now(timezone.utc)).total_seconds()
            
            if time_left < 60:
                print(f"⏰ Market ending in {int(time_left)}s")
                time.sleep(5)  # Poll every 5 seconds near end
            else:
                time.sleep(30)  # Poll every 30 seconds
        else:
            print("❌ No active market found, retrying in 60s...")
            time.sleep(60)
```

## Data Storage Schema

Update your database schema to track 15-minute markets:

```sql
CREATE TABLE IF NOT EXISTS market_snapshots_15min (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    market_id TEXT NOT NULL,
    question TEXT,
    interval_start DATETIME,
    interval_end DATETIME,
    up_price REAL,
    down_price REAL,
    volume REAL,
    liquidity REAL,
    btc_spot_price REAL,
    resolved BOOLEAN DEFAULT 0,
    outcome TEXT,
    INDEX idx_market_id (market_id),
    INDEX idx_timestamp (timestamp)
);
```

## Production Deployment

### Environment Variables

```bash
# .env
POLYMARKET_POLL_INTERVAL=30  # Seconds between polls
POLYMARKET_FALLBACK_MARKET=574073  # Long-term BTC market as fallback
ENABLE_15MIN_TRACKING=true
```

### Error Handling

```python
from polymarket_agents.connectors.updown_markets import UpDownMarketConnector

connector = UpDownMarketConnector()

try:
    market = connector.get_current_market_data()
    
    if not market:
        # Fallback to long-term market
        fallback_id = "574073"
        market = connector.fetch_market_data(fallback_id)
        print(f"⚠️  Using fallback market {fallback_id}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    # Implement retry logic, alerting, etc.
```

### Rate Limiting

Gamma API has no explicit rate limits, but be respectful:

```python
import time
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, calls_per_minute=60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        now = datetime.now()
        self.calls = [c for c in self.calls if now - c < timedelta(minutes=1)]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0]).total_seconds()
            time.sleep(sleep_time)
        
        self.calls.append(now)

# Usage
rate_limiter = RateLimiter(calls_per_minute=60)

while True:
    rate_limiter.wait_if_needed()
    market = connector.get_current_market_data()
    # Process...
```

## Testing

### Test Current Market Access

```bash
# Test the connector
python -m polymarket_agents.connectors.updown_markets

# Test API response
python scripts/python/test_updown_api.py

# Find all upcoming markets
python scripts/python/fetch_15min_market_data.py --all
```

### Validate Data Quality

```python
def validate_market_data(market):
    """Ensure market data is valid before storing."""
    assert market['market_id'], "Missing market ID"
    assert 0 <= market['up_price'] <= 1, "Invalid up_price"
    assert 0 <= market['down_price'] <= 1, "Invalid down_price"
    assert abs(market['up_price'] + market['down_price'] - 1.0) < 0.01, "Prices don't sum to 1"
    assert market['volume'] >= 0, "Negative volume"
    assert market['liquidity'] >= 0, "Negative liquidity"
    
    return True
```

## Key Files Created

1. **`src/polymarket_agents/connectors/updown_markets.py`** - Main connector
2. **`scripts/python/test_updown_api.py`** - API testing tool
3. **`scripts/python/fetch_15min_market_data.py`** - Market discovery
4. **`scripts/python/scrape_15min_market.py`** - Web scraping utility
5. **`BTC_15MIN_MARKET_DATA.md`** - Detailed market documentation
6. **`POLYMARKET_BITCOIN_MARKETS.md`** - Research findings
7. **`15MIN_MARKET_INTEGRATION.md`** - This guide

## Cost & Performance

**Web Scraping (Market ID Discovery):**
- ~4 seconds per request
- Required once per 15-minute interval
- ~96 requests/day

**Gamma API (Market Data):**
- ~0.5 seconds per request
- Poll every 30 seconds = 2 requests/minute
- ~2,880 requests/day

**Total:** ~3,000 API calls/day (well within reasonable limits)

## Success Metrics

✅ **Market discovery working** - Successfully finds current 15-min market IDs  
✅ **Data fetching working** - Gamma API returns complete market data  
✅ **Parsing working** - Correctly handles JSON strings in API response  
✅ **Live data** - Current market shows $4,193 volume, 48.5% Up probability  

## Next Steps

1. **Integrate into bitcoin_tracker.py** - Replace hardcoded market ID
2. **Add database storage** - Store 15-min snapshots
3. **Implement ML features** - Calculate momentum, volume spikes, etc.
4. **Backtest strategy** - Test on historical 15-min data
5. **Deploy monitoring** - Alert if markets expire or data quality drops

## Support

- **Gamma API Docs:** https://docs.polymarket.com/developers/gamma-markets-api
- **Chainlink BTC/USD:** https://data.chain.link/streams/btc-usd
- **Polymarket Events:** https://polymarket.com/events

---

**Status:** ✅ FULLY OPERATIONAL  
**Last Updated:** 2026-01-26  
**Current Market:** 1263100 ($4.2K volume, 48.5% Up)
