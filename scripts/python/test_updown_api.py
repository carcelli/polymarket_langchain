"""Test script to debug the UpDown market API response."""

import requests
import json
from datetime import datetime, timezone


def test_gamma_api(market_id: str = "1263100"):
    """Test fetching market data from Gamma API."""
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"

    print(f"🔍 Fetching: {url}\n")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        print("✅ API Response:")
        print(json.dumps(data, indent=2))

        print("\n" + "=" * 70)
        print("KEY FIELDS")
        print("=" * 70)
        print(f"ID: {data.get('id')}")
        print(f"Question: {data.get('question')}")
        print(f"Volume: {data.get('volume')} (type: {type(data.get('volume'))})")
        print(
            f"Liquidity: {data.get('liquidity')} (type: {type(data.get('liquidity'))})"
        )
        print(
            f"Outcome Prices: {data.get('outcomePrices')} (type: {type(data.get('outcomePrices'))})"
        )
        print(f"Active: {data.get('active')}")
        print(f"Closed: {data.get('closed')}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

    # Get current market ID
    from polymarket_agents.connectors.updown_markets import UpDownMarketConnector

    connector = UpDownMarketConnector()
    market_id = connector.get_current_market_id()

    print(f"Current 15-min slot: {connector.get_current_15min_slot()}")
    print(f"Current market ID: {market_id}\n")
    print("=" * 70 + "\n")

    if market_id:
        test_gamma_api(market_id)
    else:
        print("❌ Could not find current market ID")
