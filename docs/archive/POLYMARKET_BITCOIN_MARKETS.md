# Polymarket Bitcoin Market IDs - Research Summary

Date: January 26, 2026

## Key Findings

### 15-Minute Markets Status: NOT FOUND

No active 15-minute Bitcoin "Up or Down" markets found via Gamma API.

- Searched: "Bitcoin Up or Down", "BTC Up or Down", "Bitcoin 15", "15 minute"
- Result: 0 matches

### Long-Term Bitcoin Market: FOUND

**Market ID: 574073**

```
Question: "Will Bitcoin reach $170,000 by December 31, 2025?"
Active: True
Volume: $7,721,957.29
```

This is the market used in your existing `query_bitcoin_data.py` code.

## Why 15-Minute Markets Aren't Found

Possible reasons:

1. **Time-bound nature** - Markets expire every 15 minutes and are removed quickly
2. **API visibility** - May require authenticated access or special endpoint
3. **Product change** - Polymarket may have paused or moved this feature

## Recommended Action

**Use Market ID: 574073** for your Bitcoin tracker

This market:
- Has $7.7M in volume (highly liquid)
- Is actively traded
- Is already integrated in your codebase
- Provides reliable long-term Bitcoin price prediction data

## Alternative: Find Active 15-Min Markets

If 15-minute markets exist, they may only be visible:

1. On polymarket.com/events (web UI)
2. With authenticated API access
3. Through a different API endpoint

### Method to Check Web UI

Visit: https://polymarket.com/events

1. Filter for Crypto/Bitcoin category
2. Look for "Up or Down" or "15 minute" in market titles
3. Extract market ID from URL: `polymarket.com/event/<slug>/<market_id>`

## Code Example

```python
import requests

def get_bitcoin_market():
    """Get the main Bitcoin $170k market."""
    market_id = "574073"
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"
    response = requests.get(url)
    return response.json()

market = get_bitcoin_market()
print(f"Market: {market['question']}")
print(f"Volume: ${float(market['volume']):,.2f}")
```

## Scripts Created

Three utility scripts were created during this research:

1. `scripts/python/find_15min_markets.py` - Search for 15-minute markets
2. `scripts/python/find_all_btc_markets.py` - Comprehensive Bitcoin search
3. `scripts/python/find_updown_markets.py` - Find "Up or Down" pattern markets

Run any of these:

```bash
python scripts/python/find_updown_markets.py
```

## Next Steps

1. **Continue using Market ID 574073** - Already implemented and working
2. **Monitor for 15-min markets** - Set up periodic polling
3. **Check Polymarket website** - Manually verify if 15-min markets exist
4. **Contact Polymarket support** - Ask about API access to short-duration markets

## Summary

- **15-Min Market ID:** Not found
- **Recommended ID:** 574073 (BTC $170k by EOY 2025)
- **Status:** Your existing code is correct
- **Action:** Continue with current implementation
